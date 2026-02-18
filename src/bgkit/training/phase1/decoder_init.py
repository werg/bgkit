"""Phase 1, Step 1: Decoder initialization on uncompressed output.

Train the reconstruction decoder to generate text from BgKIT's full
(uncompressed) output representations, using Qwen3's native chat template
with tool-call format for in-distribution agentic conversation.

The BgKIT backbone is called directly (bypassing flag embeddings) since
there's no compression in Step 1 — flag embeddings are untrained at this
point and would corrupt the input. The prompt separator embedding is
zero-initialized and frozen alongside the rest of BgKIT.

Loss is computed only on the file content tokens inside the chat template's
code fence (via loss_mask), so the decoder learns pure reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.auto_repro_dataset import AutoReproDataset
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm, enable_gradient_checkpointing
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss

logger = structlog.get_logger()


class DecoderInitTrainer(BaseTrainer):
    """Step 1: Initialize decoder on uncompressed BgKIT output."""

    def setup(self) -> None:
        """Load frozen BgKIT backbone and trainable decoder."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- BgKIT backbone (frozen) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        logger.info("loading_bgkit_backbone", model=backbone_name, revision=backbone_revision)
        bgkit_backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=backbone_revision,
            attn_implementation="sdpa",
        )
        self.bgkit_model = BgKITCompressor(bgkit_backbone, hidden_dim=hidden_dim)
        self.bgkit_model.to(device)
        self.bgkit_model.requires_grad_(False)
        self.bgkit_model.eval()

        # Load BgKIT from auto-repro checkpoint if available.
        # Auto-repro checkpoints save under key "model"; step 1 checkpoints
        # save under "bgkit_model". This path handles the auto-repro format.
        bgkit_checkpoint = self.cfg.get("bgkit_checkpoint", None)
        if bgkit_checkpoint is not None:
            logger.info("loading_bgkit_checkpoint", path=bgkit_checkpoint)
            _, state_dicts = load_checkpoint(Path(bgkit_checkpoint))
            self.bgkit_model.load_state_dict(state_dicts["model"])

        # --- Decoder (trainable) ---
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        decoder_revision = decoder_cfg.get("backbone_revision", None)
        logger.info("loading_decoder", model=decoder_name, revision=decoder_revision)
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation="sdpa",
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
        self.decoder.to(device)

        enable_gradient_checkpointing(self.decoder.backbone)

        # torch.compile: ~15-30% speedup on Blackwell. If compile fails on sm_121,
        # remove this line — training is correct without it.
        try:
            self.decoder.backbone = torch.compile(self.decoder.backbone)
            logger.info("torch_compile_enabled")
        except Exception:
            logger.warning("torch_compile_failed", exc_info=True)

        # BaseTrainer logging/device logic uses self.model
        self.model = self.decoder

        # --- Tokenizer (for chat template) ---
        tokenizer_name = decoder_cfg.backbone_name
        tokenizer_revision = decoder_cfg.get("backbone_revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )

        # --- Dataset ---
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        variant_bank_path = self.cfg.data.get(
            "variant_bank_path", "data/prompt_variants/file_read_repro.json",
        )

        inner_dataset = AutoReproDataset(data_dir, max_seq_len=max_seq_len)
        full_dataset = ChatReproDataset(
            inner_dataset,
            tokenizer=self.tokenizer,
            variant_bank_path=variant_bank_path,
            seed=self.cfg.get("seed", 42),
        )

        max_eval_samples = tcfg.get("max_eval_samples", 10000)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {len(full_dataset)} samples, "
                "need at least 2)"
            )
        self.chat_dataset = full_dataset
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )
        self._eval_count = 0

        # Token-budget batching (based on chat-formatted lengths)
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = [full_dataset.lengths[i] for i in self.train_dataset.indices]
        eval_lengths = [full_dataset.lengths[i] for i in self.eval_dataset.indices]

        self.train_sampler = TokenBudgetBatchSampler(
            train_lengths, max_batch_tokens, shuffle=True, seed=self.cfg.get("seed", 42),
        )
        eval_sampler = TokenBudgetBatchSampler(
            eval_lengths, max_batch_tokens, shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # --- Optimizer ---
        # 8-bit AdamW: reduces optimizer memory. If bnb causes issues,
        # fall back to torch.optim.AdamW — correctness is unaffected.
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.decoder.parameters(), lr=tcfg.lr)
            logger.info("using_adamw8bit")
        except ImportError:
            self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=tcfg.lr)
            logger.info("using_adamw_fp32", reason="bitsandbytes not available")

        logger.info(
            "decoder_init_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
        )

    def train_step(self, batch) -> dict[str, float]:
        self.decoder.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)

        with torch.no_grad():
            survivors = self._compute_survivors(batch)

        # BF16 autocast for decoder forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.decoder(
                survivor_embeddings=survivors,
                target_ids=token_ids,
                target_attention_mask=attention_mask,
                survivor_attention_mask=content_attention_mask,
            )
            loss = data_reconstruction_loss(
                logits, token_ids, attention_mask, loss_mask=loss_mask,
            )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(
            [p for p in self.decoder.parameters() if p.requires_grad]
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
        }

    def _compute_survivors(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run content through frozen BgKIT and return survivor embeddings."""
        content_token_ids = batch["content_token_ids"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)
        compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
        compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

        bgkit_embed = self.bgkit_model.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_token_ids)
        prompt_emb = bgkit_embed(compression_prompt_ids)

        bgkit_out = self.bgkit_model.backbone(
            inputs_embeds=torch.cat([
                prompt_emb,
                self.bgkit_model.prompt_separator_embedding.unsqueeze(0).unsqueeze(0).expand(
                    content_emb.size(0), 1, -1,
                ),
                content_emb,
            ], dim=1),
            attention_mask=torch.cat([
                compression_prompt_mask,
                torch.ones(
                    content_emb.size(0), 1,
                    dtype=torch.bool, device=self.device,
                ),
                content_attention_mask,
            ], dim=1),
        )
        prompt_len = compression_prompt_ids.size(1) + 1
        return bgkit_out.last_hidden_state[:, prompt_len:, :]

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.decoder.eval()
        self._eval_count += 1

        total_loss = 0.0
        total_content_tokens = 0.0

        num_batches = len(self.eval_dataloader)
        for batch_idx, batch in enumerate(self.eval_dataloader):
            if batch_idx % 100 == 0:
                logger.info("eval_progress", batch=batch_idx, total=num_batches)
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            content_attention_mask = batch["content_attention_mask"].to(self.device)

            survivors = self._compute_survivors(batch)

            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                logits = self.decoder(
                    survivor_embeddings=survivors,
                    target_ids=token_ids,
                    target_attention_mask=attention_mask,
                    survivor_attention_mask=content_attention_mask,
                )
                loss = data_reconstruction_loss(
                    logits, token_ids, attention_mask, loss_mask=loss_mask,
                )

            # Weight by content tokens only (loss_mask[:, 1:] for shift alignment)
            batch_content_tokens = loss_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_content_tokens
            total_content_tokens += batch_content_tokens

        avg_loss = total_loss / max(total_content_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics: dict[str, float] = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

        # Generation metrics (expensive — only every Nth eval)
        tcfg = self.cfg.training
        gen_every = tcfg.get("eval", {}).get("generation_eval_every", 4)
        if self._eval_count % gen_every == 0:
            gen_metrics = self._run_generation_eval()
            metrics.update(gen_metrics)

        return metrics

    @torch.no_grad()
    def _run_generation_eval(self) -> dict[str, float]:
        """Run generation-based eval metrics on a small subset."""
        tcfg = self.cfg.training
        max_gen_samples = tcfg.get("eval", {}).get("generation_samples", 50)

        gen_metrics: dict[str, float] = {}
        generated_texts: list[str] = []
        generated_languages: list[str] = []
        all_survivors: list[torch.Tensor] = []
        samples_seen = 0

        suffix_ids = self.chat_dataset.suffix_ids.to(self.device)

        for batch in self.eval_dataloader:
            if samples_seen >= max_gen_samples:
                break

            content_attention_mask = batch["content_attention_mask"].to(self.device)
            prefix_ids = batch["prefix_ids"].to(self.device)
            prefix_attention_mask = batch["prefix_attention_mask"].to(self.device)

            survivors = self._compute_survivors(batch)

            # Collect survivors for embedding health (subsample to ~512 vectors)
            total_vecs = sum(s.size(0) for s in all_survivors)
            if total_vecs < 512:
                flat = survivors[content_attention_mask].detach()
                remaining = 512 - total_vecs
                all_survivors.append(flat[:remaining])

            # Generate
            gen_output = self.decoder.generate(
                survivor_embeddings=survivors,
                survivor_attention_mask=content_attention_mask,
                prefix_ids=prefix_ids,
                prefix_attention_mask=prefix_attention_mask,
                suffix_ids=suffix_ids,
                tokenizer=self.tokenizer,
                max_new_tokens=2048,
                temperature=0.0,
            )
            generated_texts.extend(gen_output.content_text)
            generated_languages.extend(batch["languages"])
            samples_seen += survivors.size(0)

        # Parse success rate
        if generated_texts:
            gen_metrics["parse_success_rate"] = parse_success_rate(
                generated_texts, languages=generated_languages,
            )

        # Embedding health
        if all_survivors:
            combined_survivors = torch.cat(all_survivors, dim=0)
            token_emb = self.bgkit_model.backbone.get_input_embeddings().weight.detach()
            health = embedding_drift_metrics(combined_survivors, token_emb)
            gen_metrics["mean_max_cosine_sim"] = health["mean_max_cosine_sim"]
            gen_metrics["std_max_cosine_sim"] = health["std_max_cosine_sim"]

        return gen_metrics

    def save_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Save both BgKIT and decoder models."""
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            bgkit_model=self.bgkit_model.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load both BgKIT and decoder models."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        if "bgkit_model" in state_dicts:
            self.bgkit_model.load_state_dict(state_dicts["bgkit_model"])
        self.decoder.load_state_dict(state_dicts["decoder"])
        if "optimizer" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        logger.info("restored_from_checkpoint", step=self.global_step)
