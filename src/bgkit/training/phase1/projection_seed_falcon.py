"""Falcon-H1 dense pair projection seed/adaptation trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM

from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder, normalize_decoder_family
from bgkit.models.encoder import BgKITEncoder, EncoderOutput
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.utils.attention_backend import (
    resolve_attention_implementation,
    resolve_decoder_attention_implementation,
)
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


class _FalconSeedModel(nn.Module):
    def __init__(self, encoder: BgKITEncoder, decoder: ReconstructionDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class _ValidCompanionSubset(Dataset):
    def __init__(
        self,
        dataset: MmapTokenDataset,
        max_sample_length: int | None = None,
        min_sample_length: int | None = None,
        max_chunks_per_repo: int | None = None,
        per_repo_subsample_seed: int = 17,
        exclude_files_metadata_path: str | None = None,
    ):
        valid = dataset.companion_valid_indices
        if valid is None:
            raise ValueError("Falcon dense seed requires MmapTokenDataset(companion_dir=...)")
        self.dataset = dataset
        lengths = dataset.lengths[valid]
        if max_sample_length or min_sample_length:
            keep = np.ones(valid.size, dtype=bool)
            if max_sample_length:
                keep &= lengths <= int(max_sample_length)
            if min_sample_length:
                keep &= lengths >= int(min_sample_length)
            valid = valid[keep]
            lengths = lengths[keep]
        # Exclude (repo_path, file_path) tuples that appear in a
        # previously-trained dataset's metadata.parquet. Used to filter
        # out already-seen content when switching to a superset corpus —
        # avoids re-training on the heavily-cycled overlap. Match is
        # exact on the (repo, file) key; chunks within a file are all
        # kept-or-dropped together.
        if exclude_files_metadata_path:
            import pyarrow.parquet as pq
            exclude_table = pq.read_table(
                str(exclude_files_metadata_path),
                columns=["repo_path", "file_path"],
            )
            exclude_keys = set(zip(
                exclude_table.column("repo_path").to_pylist(),
                exclude_table.column("file_path").to_pylist(),
                strict=True,
            ))
            meta = dataset.get_metadata_table()
            chunk_file_idx_full = dataset.chunk_file_indices
            if chunk_file_idx_full is None:
                raise ValueError(
                    "exclude_files_metadata_path requires "
                    "MmapTokenDataset(include_metadata=True)"
                )
            repo_per_file = meta.column("repo_path").to_pylist()
            file_per_file = meta.column("file_path").to_pylist()
            valid_file_idx = chunk_file_idx_full[valid]
            keep = np.fromiter(
                (
                    (repo_per_file[i], file_per_file[i]) not in exclude_keys
                    for i in valid_file_idx.tolist()
                ),
                dtype=bool,
                count=valid_file_idx.size,
            )
            kept_count = int(keep.sum())
            logger.info(
                "exclude_files_filter_applied",
                excluded_keys=len(exclude_keys),
                kept_chunks=kept_count,
                dropped_chunks=int(keep.size - kept_count),
            )
            valid = valid[keep]
            lengths = lengths[keep]
        # Per-repo subsampling to break heavy-tail overfitting. The raw
        # corpus is power-law skewed (top 1% of repos contribute ~17% of
        # chunks, top repo has 6,151 chunks vs median 31 — a 198x upweight
        # under uniform sampling). Cap each repo to ``max_chunks_per_repo``
        # via a deterministic per-repo random subset.
        #
        # Indexing: metadata.parquet has ONE row per file; chunks (which is
        # what ``valid`` and ``dataset.lengths`` are indexed over) can have
        # multiple per file. ``chunk_file_indices`` maps chunk -> file row.
        if max_chunks_per_repo is not None and int(max_chunks_per_repo) > 0:
            cap = int(max_chunks_per_repo)
            meta = dataset.get_metadata_table()
            repo_per_file = meta.column("repo_path").to_numpy(zero_copy_only=False)
            chunk_file_idx = dataset.chunk_file_indices
            if chunk_file_idx is None:
                raise ValueError(
                    "max_chunks_per_repo requires dataset.chunk_file_indices "
                    "(MmapTokenDataset must be built with include_metadata=True)."
                )
            repo_for_chunk = repo_per_file[chunk_file_idx[valid]]
            rng = np.random.default_rng(int(per_repo_subsample_seed))
            keep_idx_in_valid: list[int] = []
            unique_repos, inverse = np.unique(repo_for_chunk, return_inverse=True)
            for repo_id in range(unique_repos.size):
                in_repo = np.flatnonzero(inverse == repo_id)
                if in_repo.size <= cap:
                    keep_idx_in_valid.extend(in_repo.tolist())
                else:
                    chosen = rng.choice(in_repo, size=cap, replace=False)
                    keep_idx_in_valid.extend(chosen.tolist())
            keep_idx_in_valid = np.array(sorted(keep_idx_in_valid), dtype=np.int64)
            valid = valid[keep_idx_in_valid]
            lengths = lengths[keep_idx_in_valid]
            logger.info(
                "per_repo_cap_applied",
                cap=cap,
                unique_repos=int(unique_repos.size),
                kept_chunks=int(valid.size),
            )
        self.indices = valid
        if self.indices.size == 0:
            raise ValueError(
                "Falcon companion stream has no non-degenerate chunks "
                "(or all were filtered by min/max_sample_length / per-repo cap)"
            )
        self.lengths = lengths

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[int(self.indices[idx])]


def collate_falcon_dense_seed(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("collate_falcon_dense_seed() received an empty batch")

    content = [b["token_ids"] for b in batch]
    lengths = [int(t.numel()) for t in content]
    content_cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0, out=content_cu[1:])
    total = int(content_cu[-1])

    forced_mask = torch.zeros(total, dtype=torch.bool)
    cursor = 0
    pair_ids: list[torch.Tensor] = []
    pair_masks: list[torch.Tensor] = []
    alignment_scores: list[torch.Tensor] = []
    forced_counts: list[int] = []
    for sample, length in zip(batch, lengths, strict=True):
        forced = sample["forced_survivor_indices"].to(torch.int64)
        if forced.numel() == 0:
            raise ValueError("Degenerate Falcon companion sample reached collator")
        if int(forced.max().item()) >= length or int(forced.min().item()) < 0:
            raise ValueError("forced_survivor_indices must be chunk-local Qwen positions")
        forced_mask[cursor + forced] = True
        cursor += length
        pair_ids.append(sample["target_falcon_pair_ids"].to(torch.int64))
        pair_masks.append(sample["target_pair_loss_mask"].to(torch.bool))
        alignment_scores.append(sample["alignment_scores"].to(torch.float32))
        forced_counts.append(int(forced.numel()))

    target_pair_ids = torch.cat(pair_ids, dim=0)
    target_pair_mask = torch.cat(pair_masks, dim=0)
    target_ids = target_pair_ids.reshape(-1)
    target_loss_mask = target_pair_mask.reshape(-1)

    return {
        "content_token_ids": torch.cat(content, dim=0).to(torch.long),
        "content_cu_seqlens": content_cu,
        "content_position_ids": position_ids_from_cu(content_cu, total),
        "content_max_seqlen": max(lengths) if lengths else 0,
        "forced_survivor_mask_l0": forced_mask,
        "target_falcon_pair_ids": target_pair_ids,
        "target_pair_loss_mask": target_pair_mask,
        "target_falcon_ids": target_ids,
        "target_loss_mask": target_loss_mask,
        "alignment_scores": torch.cat(alignment_scores, dim=0),
        "forced_counts": torch.tensor(forced_counts, dtype=torch.int64),
    }


class FalconProjectionSeedTrainer(BaseTrainer):
    """Train ``projection_blocks.falcon_h1`` against dense Falcon embeddings."""

    def setup(self) -> None:
        tcfg = self.cfg.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        bgkit_cfg = self.cfg.model.bgkit
        hidden_dim = int(bgkit_cfg.get("hidden_dim", 1024))
        src_ckpt = self._resolve_source_checkpoint()
        _meta, state_dicts = load_checkpoint(Path(src_ckpt))
        if "encoder" not in state_dicts:
            raise ValueError(f"checkpoint {src_ckpt} missing 'encoder' key")

        projection_num_layers = int(bgkit_cfg.get("projection_num_layers", 1))
        # Per-training-config threshold_controller override so Falcon configs
        # can cap anchors at 6 (0.60) without colliding with the encoder
        # default of 7 anchors. Mirrors the same fix made in compression.py
        # / decoder_init.py / kr_kb_trainer.py / build_dense_seed_cache.py.
        step_model_cfg = tcfg.get("model", {}) or {}
        threshold_cfg = step_model_cfg.get(
            "threshold_controller",
            self.cfg.model.get("threshold_controller", {}),
        )
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            bgkit_cfg.backbone_name,
            state_dicts["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            active_decoder_family="falcon_h1",
            projection_num_layers=projection_num_layers,
            threshold_controller_cfg=dict(threshold_cfg) if threshold_cfg else None,
        ).to(self.device)
        self.encoder.set_active_decoder_family("falcon_h1")

        decoder_cfg = tcfg.get("model", {}).get("decoder", self.cfg.model.decoder)
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "falcon_h1"))
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
        )
        decoder_name = decoder_cfg.get(
            "backbone_name",
            "tiiuae/Falcon-H1-Tiny-90M-Instruct",
        )
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            revision=decoder_cfg.get("backbone_revision", None),
            attn_implementation=decoder_attention_impl,
        ).to(self.device)
        decoder_hidden = int(decoder_backbone.get_input_embeddings().weight.shape[1])
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_hidden,
            decoder_family=decoder_family,
        )
        self.decoder.requires_grad_(False)
        self.decoder.eval()
        inner, _lm_head = self.decoder._get_inner_model_and_head()
        self._falcon_embed = inner.get_input_embeddings()

        self.model = _FalconSeedModel(self.encoder, self.decoder)

        self._adapt_encoder = bool(tcfg.get("adapt_encoder", False))
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder.projection_blocks["qwen35"].requires_grad_(False)
        self.encoder.projection_blocks["falcon_h1"].requires_grad_(True)
        self._encoder_trainable_params: list[torch.nn.Parameter] = []
        if self._adapt_encoder:
            self.encoder.l0.requires_grad_(True)
            self._encoder_trainable_params = [
                p for p in self.encoder.l0.parameters() if p.requires_grad
            ]
        self._set_training_modes()

        self._anchor_weight = float(tcfg.get("anchor_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 0.5))
        self._norm_weight = float(tcfg.get("norm_weight", 0.2))
        self._survivor_bce_weight = float(tcfg.get("survivor_bce_weight", 0.0))

        # Prefer training-scoped data overrides (consistent with the l0/l1
        # CompressionTrainer); fall back to the global data section. This
        # lets the dense_seed YAML self-contain its data wiring without
        # mutating the shared data/tokens.yaml defaults.
        training_data = tcfg.get("data", None) or {}
        training_tokens = training_data.get("tokens", None) or {}
        data_dir = (
            training_tokens.get("input_dir", None)
            or training_data.get("file_tokens_path", None)
            or self.cfg.data.tokens.input_dir
        )
        companion_dir = (
            training_data.get("falcon_companion_dir", None)
            or self.cfg.data.get("falcon_companion_dir", None)
        )
        # Deprecated locations — accept with a warning so old configs
        # still resolve, but encourage migration to data.falcon_companion_dir.
        if not companion_dir:
            legacy = tcfg.get("companion_dir", None)
            if legacy:
                logger.warning(
                    "deprecated_companion_dir_key",
                    found="training.companion_dir",
                    canonical="data.falcon_companion_dir",
                )
                companion_dir = legacy
        if not companion_dir:
            legacy = self.cfg.data.tokens.get("companion_dir", None)
            if legacy:
                logger.warning(
                    "deprecated_companion_dir_key",
                    found="data.tokens.companion_dir",
                    canonical="data.falcon_companion_dir",
                )
                companion_dir = legacy
        if not companion_dir:
            raise ValueError(
                "Falcon dense seed requires data.falcon_companion_dir to "
                "point at the output of "
                "scripts/convert_tokens_to_falcon_mmap.py"
            )
        # max_seq_len must match the value the Falcon companion was built
        # at — the companion is chunk-aligned at that length and
        # MmapTokenDataset's strict count check otherwise rejects load.
        max_seq_len = int(
            training_tokens.get("max_seq_len", None)
            or self.cfg.data.tokens.get("max_seq_len", 8192)
        )
        inner_dataset = MmapTokenDataset(
            data_dir,
            max_seq_len=max_seq_len,
            include_metadata=True,
            companion_dir=str(companion_dir),
        )
        # forced_adapt unfreezes encoder.l0, so the full Qwen3.5 bidi backbone
        # runs with autograd at every step. Memory then scales with content
        # length; long-tail chunks (p95 ~4600 content tokens) push the 65%
        # per-process CUDA budget over its cap and OOM. Filter both ends:
        # max_sample_length caps the worst-case batch memory, min_sample_length
        # drops degenerate short chunks. Both default to None (no filtering).
        max_sample_length = tcfg.get("max_sample_length", None)
        min_sample_length = tcfg.get("min_sample_length", None)
        max_chunks_per_repo = tcfg.get("max_chunks_per_repo", None)
        exclude_files_metadata_path = tcfg.get("exclude_files_metadata_path", None)
        full_dataset = _ValidCompanionSubset(
            inner_dataset,
            max_sample_length=max_sample_length,
            min_sample_length=min_sample_length,
            max_chunks_per_repo=max_chunks_per_repo,
            per_repo_subsample_seed=int(self.cfg.get("seed", 42)),
            exclude_files_metadata_path=exclude_files_metadata_path,
        )

        max_eval_samples = int(tcfg.get("max_eval_samples", 1000))
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=generator,
        )

        train_lengths = full_dataset.lengths[np.asarray(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.asarray(self.eval_dataset.indices)]
        max_batch_tokens = int(tcfg.get("max_batch_tokens", 32768))
        eval_budget = int(tcfg.get("max_batch_tokens_eval", max_batch_tokens * 2))
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=int(self.cfg.get("seed", 42)),
        )
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=eval_budget,
            shuffle=False,
        )
        num_workers = int(self.cfg.compute.get("num_workers", 4))
        pin_memory = bool(self.cfg.compute.get("pin_memory", False))
        # persistent_workers=True keeps the worker fork alive across
        # epochs — important on unified memory because the per-fork CoW
        # cost would otherwise repeat at every epoch boundary. Only
        # meaningful when num_workers > 0.
        dataloader_kwargs = {
            "collate_fn": collate_falcon_dense_seed,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            **dataloader_kwargs,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            **dataloader_kwargs,
        )

        proj_params = [
            p for p in self.encoder.projection_blocks["falcon_h1"].parameters()
            if p.requires_grad
        ]
        proj_lr = float(tcfg.get("projection_lr", tcfg.lr))
        param_groups = [{"params": proj_params, "lr": proj_lr, "base_lr": proj_lr}]
        if self._encoder_trainable_params:
            enc_lr = float(tcfg.get("encoder_lr", proj_lr * 0.1))
            param_groups.append({
                "params": self._encoder_trainable_params,
                "lr": enc_lr,
                "base_lr": enc_lr,
            })
        self.optimizer = self._create_optimizer(
            param_groups,
            proj_lr,
            exclude_from_muon=frozenset(),
        )

        logger.info(
            "falcon_projection_seed_setup",
            source_checkpoint=src_ckpt,
            train_samples=train_size,
            eval_samples=eval_size,
            projection_params=sum(p.numel() for p in proj_params),
            adapt_encoder=self._adapt_encoder,
            companion_dir=str(companion_dir),
        )

    def _resolve_source_checkpoint(self) -> str:
        src = self.cfg.get("bgkit_checkpoint", None)
        if src and src != "auto":
            self._input_sources = {"bgkit": Path(str(src)).name}
            return str(src)
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        training_phase = self.cfg.get("training", {}).get("phase", "")
        # Prefer the latest split-L0/L1 layout encoder (Step 5 / Step 6, post
        # 2026-05-03 rebuild) before falling back to legacy step2p5 / step2
        # which predate the split. `from_pretrained_with_state_dict` auto-
        # migrates the legacy `projection_block.*` keys present in step5/step6
        # to `projection_blocks.qwen35.*` and clones a starting `falcon_h1`
        # projection block from qwen35 with a loud warning.
        # phase1_step5 is preferred over phase1_step6 because (as of
        # 2026-05-11) the most recent split-L0/L1 encoder state lives in
        # phase1_step5 step6000 (eval/loss≈0.70). The available step6
        # checkpoints are older (April) with worse loss; resolve_checkpoint
        # picks by metric within a phase, so listing step6 first would
        # silently down-grade to the old checkpoint.
        if training_phase == "phase1_falcon_forced_adapt":
            candidate_phases = (
                "phase1_falcon_dense_seed",
                "phase1_step5",
                "phase1_step6",
                "phase1_step2p5",
                "phase1_step2",
            )
        else:
            candidate_phases = (
                "phase1_step5",
                "phase1_step6",
                "phase1_step2p5",
                "phase1_step2",
            )

        errors: list[str] = []
        for phase in candidate_phases:
            try:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase=phase,
                    metric="eval/loss",
                    label="bgkit_checkpoint",
                )
                self._input_sources = {"bgkit": resolved.name}
                return str(resolved)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(
            "bgkit_checkpoint=auto could not resolve any candidate phase "
            f"for {training_phase or 'Falcon projection seed'}: "
            f"{', '.join(candidate_phases)}. "
            + " | ".join(errors)
        )

    def trainable_parameters(self) -> list:
        params = [
            p for p in self.encoder.projection_blocks["falcon_h1"].parameters()
            if p.requires_grad
        ]
        params.extend(self._encoder_trainable_params)
        return params

    def _named_parameters_for_optimizer(self):
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param

    def _encoder_forward(self, batch: dict) -> EncoderOutput:
        content_ids = batch["content_token_ids"].to(self.device)
        content_cu = batch["content_cu_seqlens"].to(self.device)
        content_pos = batch["content_position_ids"].to(self.device)
        forced = batch["forced_survivor_mask_l0"].to(self.device)
        embed = self.encoder.l0.backbone.get_input_embeddings()
        # target_ratio_l0 is dead under forced_survivor_mask_l0 (LevelCompressor
        # overrides selection); pass None so a future reader doesn't assume 0.5
        # is load-bearing.
        return self.encoder(
            content_embeddings=embed(content_ids),
            content_cu_seqlens=content_cu,
            content_position_ids=content_pos,
            target_ratio_l0=None,
            target_ratio_l1=None,
            forced_survivor_mask_l0=forced,
        )

    def _set_training_modes(self) -> None:
        """Train only the explicitly enabled Falcon adaptation modules."""

        self.decoder.eval()
        self.encoder.eval()
        self.encoder.projection_blocks["falcon_h1"].train()
        if self._adapt_encoder:
            self.encoder.l0.train()

    def _set_evaluation_modes(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def _losses(
        self,
        projected: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_proj = projected[mask]
        valid_target = target[mask]
        n = max(int(valid_proj.shape[0]), 1)
        mse = (valid_proj - valid_target).pow(2).sum() / (n * projected.shape[-1])
        cos = F.cosine_similarity(valid_proj, valid_target, dim=-1)
        cos_loss = 1.0 - cos.mean()
        proj_norm = valid_proj.norm(dim=-1).clamp(min=1e-6)
        target_norm = valid_target.norm(dim=-1).clamp(min=1e-6)
        norm_loss = (torch.log(proj_norm) - torch.log(target_norm)).pow(2).mean()
        total = (
            self._anchor_weight * mse
            + self._cos_weight * cos_loss
            + self._norm_weight * norm_loss
        )
        return total, {
            "anchor_mse": mse.detach(),
            "anchor_cos": cos_loss.detach(),
            "anchor_norm_log": norm_loss.detach(),
            "cos_sim": cos.mean().detach(),
            "norm_ratio": (proj_norm / target_norm).mean().detach(),
        }

    def _forward_backward(self, batch) -> dict[str, float]:
        self._set_training_modes()
        enc_out = self._encoder_forward(batch)
        target_ids = batch["target_falcon_ids"].to(self.device)
        target_mask = batch["target_loss_mask"].to(self.device).to(torch.bool)
        target_emb = self._falcon_embed(target_ids).detach()
        if enc_out.survivor_embeddings.shape[0] != target_ids.shape[0]:
            raise RuntimeError(
                "Falcon projection output length does not match dense pair targets: "
                f"{enc_out.survivor_embeddings.shape[0]} vs {target_ids.shape[0]}"
            )
        total, stats = self._losses(enc_out.survivor_embeddings, target_emb, target_mask)
        metrics: dict[str, torch.Tensor] = {
            "loss": total,
            "loss/anchor_mse": stats["anchor_mse"],
            "loss/anchor_cos": stats["anchor_cos"],
            "loss/anchor_norm_log": stats["anchor_norm_log"],
            "diag/cos_sim": stats["cos_sim"],
            "diag/norm_ratio": stats["norm_ratio"],
            "diag/forced_survivors": batch["forced_counts"].float().mean(),
            "diag/alignment_score": batch["alignment_scores"].float().mean(),
        }
        # Asymmetric "doom-the-bogus" head BCE (ported 2026-05-23 from
        # bgkit.training.survivorship_helpers): supervise the head to
        # predict 0 at non-forced positions (where projection_block's
        # 2x expansion produces unreliable embeddings). Forced positions
        # are unsupervised so utility-grad / aggregate-ratio pressure in
        # l0_align / l0 can rank them. Uses base_raw (pre-tanh) so
        # gradient stays alive — the prior logits_for_op path
        # (= tanh(base_raw/T)) saturated and killed the head's gradient.
        if self._survivor_bce_weight > 0.0 and enc_out.l0.base_raw is not None:
            base_raw = enc_out.l0.base_raw
            forced = batch["forced_survivor_mask_l0"].to(
                device=base_raw.device, dtype=torch.bool,
            )
            if forced.shape[0] != base_raw.shape[0]:
                raise ValueError(
                    "forced_survivor_mask_l0 shape "
                    f"{tuple(forced.shape)} does not match base_raw shape "
                    f"{tuple(base_raw.shape)}"
                )
            neg = (~forced).to(dtype=torch.float32)
            target = torch.zeros_like(base_raw, dtype=torch.float32)
            bce_per_pos = F.binary_cross_entropy_with_logits(
                base_raw.float(), target, reduction="none",
            )
            denom = neg.sum().clamp(min=1.0)
            forced_bce = (bce_per_pos * neg).sum() / denom
            total = total + self._survivor_bce_weight * forced_bce
            metrics["loss"] = total
            metrics["loss/forced_survivor_bce"] = forced_bce.detach()
            metrics["loss/forced_survivor_pos_fraction"] = (
                forced.float().mean().detach()
            )

        (total / self._accum_steps).backward()
        enc_out.release()
        return {k: float(v.item()) for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self._set_evaluation_modes()
        totals: dict[str, float] = {}
        count = 0
        try:
            for batch in self.eval_dataloader:
                enc_out = self._encoder_forward(batch)
                target_ids = batch["target_falcon_ids"].to(self.device)
                target_mask = batch["target_loss_mask"].to(self.device).to(torch.bool)
                target_emb = self._falcon_embed(target_ids)
                total, stats = self._losses(
                    enc_out.survivor_embeddings,
                    target_emb,
                    target_mask,
                )
                totals["loss"] = totals.get("loss", 0.0) + float(total.item())
                for name, value in stats.items():
                    totals[name] = totals.get(name, 0.0) + float(value.item())
                count += 1
                # Leak fix: drop the EncoderOutput payload (release wipes the L0
                # ``content_embeddings`` + survivor tensors) AND hard-clear the
                # loop-locals before the dataloader advances.  Without this the
                # previous-iter encoder activations + Falcon target_emb stay
                # GPU-resident until Python rebinds locals at the NEXT
                # iteration, overlapping with the prefetcher's next batch and
                # the next encoder forward — eval runs at 2x batch tokens *and*
                # without gradient checkpointing (``self.training=False``
                # disables it), so the overlap balloons the reserved pool by
                # ~20 GB per eval.
                enc_out.release()
                del enc_out, target_ids, target_mask, target_emb, total, stats, batch
            if count:
                totals = {k: v / count for k, v in totals.items()}
            return totals
        finally:
            self._set_training_modes()
            # Eval ran with bigger budget + no gradient checkpointing, so its
            # peak reserved pool is larger than the training steady-state.
            # Hand the headroom back before training resumes; otherwise the
            # next train step has to fight the eval-sized pool for allocation
            # and pace collapses (60 s/step vs 1.6 s/step observed).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
            optimizer_type=self._optimizer_type,
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            encoder=self.encoder.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
