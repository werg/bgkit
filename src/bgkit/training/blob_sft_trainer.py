"""Family-A blob-format compaction SFT trainer (capability-packaging §4).

Lean trainer over :class:`~bgkit.data.datasets.blob_sft_dataset.BlobSFTDataset`:
each sample is a chat-template-rendered trajectory whose compacted history is
represented by ``<bgkit_blob>`` blocks. At forward time every blob's raw
content tokens are encoded live (query-free L0 → L1 → projection, exact_topk
at both levels) and the projected survivors replace the blob's splice sentinel
inside the decoder sequence (:class:`EmbeddingSegment` between
:class:`TokenSegment` runs). CE covers only the target assistant turn
(``loss_mask`` from :func:`bgkit.data.blob_tokenize.tokenize_blob_sample`).

Deliberately small: single Qwen decoder, no browse trees, no tool grammar,
no round-robin. Model weights start from a Phase-2 wide-net checkpoint
(``training.init_checkpoint``) so the encoder/decoder pair already speaks the
splice contract; this trainer teaches the *compaction* reading of it.

Eval reports teacher-forced loss/token-accuracy per qtype (continuation vs
recall_probe), recall-probe exact match, and a zeroed-rep ablation gap —
the "are the reps load-bearing" gate that the zero-rep incident
(2026-08-22) made mandatory for every new splice path.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from bgkit.training.base_trainer import BaseTrainer
from bgkit.utils.packing import position_ids_from_cu

try:  # project structlog wrapper
    from bgkit.utils.logging import get_logger

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover
    import structlog

    logger = structlog.get_logger(__name__)


class _BlobModel(nn.Module):
    """Container so BaseTrainer checkpointing round-trips encoder + decoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def build_blob_segments(
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    sentinel_spans: list[tuple[int, int]],
    blob_survivors: list[torch.Tensor],
    *,
    zero_reps: bool = False,
) -> list:
    """Interleave token runs with survivor embeddings at the sentinel spans.

    The sentinel tokens themselves are DROPPED (replaced by the embedding
    run); ``loss_mask`` slices travel with their token runs. ``zero_reps``
    swaps every survivor run for zeros of the same shape — the
    reps-absent ablation arm, sequence length preserved.
    """
    from bgkit.models.decoder import EmbeddingSegment, TokenSegment

    if len(sentinel_spans) != len(blob_survivors):
        raise ValueError(
            f"{len(sentinel_spans)} sentinel spans but "
            f"{len(blob_survivors)} survivor runs"
        )
    segments: list = []
    cursor = 0
    for (a, b), surv in zip(sentinel_spans, blob_survivors, strict=True):
        if a < cursor or b <= a:
            raise ValueError(f"bad/overlapping sentinel span ({a}, {b}) at cursor {cursor}")
        if a > cursor:
            segments.append(
                TokenSegment(token_ids[cursor:a], loss_mask=loss_mask[cursor:a])
            )
        reps = torch.zeros_like(surv) if zero_reps else surv
        segments.append(EmbeddingSegment(embeddings=reps))
        cursor = b
    if cursor < int(token_ids.shape[0]):
        segments.append(
            TokenSegment(token_ids[cursor:], loss_mask=loss_mask[cursor:])
        )
    return segments


def _prefixed_state(model_state: dict, prefix: str) -> dict:
    dotted = f"{prefix}."
    return {
        k[len(dotted):]: v for k, v in model_state.items() if k.startswith(dotted)
    }


def _resolve_shards(patterns: list[str], exclude: set[str] | None = None) -> list[str]:
    out: list[str] = []
    for pat in patterns or []:
        matches = sorted(_glob.glob(str(pat)))
        out.extend(matches if matches else [str(pat)])
    seen: set[str] = set()
    uniq = []
    for p in out:
        if p not in seen and (exclude is None or p not in exclude):
            seen.add(p)
            uniq.append(p)
    return uniq


class BlobSFTTrainer(BaseTrainer):
    """See module docstring."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.step_cfg = cfg.training
        self.device = torch.device("cpu")
        self.encoder: nn.Module | None = None
        self.decoder = None
        self.tokenizer = None
        # Fixed held-out text slice for the plain-LM health metric (loaded
        # once on first eval); see bgkit.training.lm_health.
        self._lm_health_chunks: list | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset
        from bgkit.models.decoder import (
            ReconstructionDecoder,
            normalize_decoder_family,
        )
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.training.checkpointing import load_checkpoint
        from bgkit.training.gradient_utils import (
            maybe_enable_decoder_gradient_checkpointing,
            maybe_enable_gradient_checkpointing,
        )
        from bgkit.utils.attention_backend import (
            resolve_decoder_attention_implementation,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- init checkpoint (REQUIRED: a Phase-2 wide-net ckpt) ---
        init_ckpt = self.step_cfg.get("init_checkpoint", None)
        if not init_ckpt:
            raise ValueError(
                "blob_sft requires training.init_checkpoint — a Phase-2 "
                "checkpoint carrying encoder + qwen decoder state (the splice "
                "contract is co-trained; cold HF weights cannot read blobs)."
            )
        _meta, state_dicts = load_checkpoint(Path(str(init_ckpt)))
        model_state = state_dicts.get("model", {}) or {}

        # --- decoder (Qwen only) ---
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        if family != "qwen35":
            raise ValueError(f"blob_sft is Qwen-only for now; got family {family!r}")
        attn_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=family,
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).to(self.device)
        hidden = backbone.get_input_embeddings().weight.shape[1]
        self.decoder = ReconstructionDecoder(
            backbone, hidden_dim=hidden, decoder_family=family,
        )
        self.decoder.set_lm_ce_impl(
            self.step_cfg.get(
                "decoder_ce_impl", self.cfg.compute.get("decoder_ce_impl", None),
            )
        )
        self.decoder.train()
        maybe_enable_decoder_gradient_checkpointing(backbone, self.cfg)
        # Force per-layer reentrant GC inside the interleaved decode (the OOM
        # path) — mirrors KRKBTrainer FIX 1b.
        gc_req = self.step_cfg.get(
            "decoder_gradient_checkpointing",
            self.cfg.compute.get("decoder_gradient_checkpointing", None),
        )
        self.decoder._interleaved_gc_mode = (
            "reentrant" if gc_req not in (None, False, "false", "off", "0", "") else None
        )
        self.decoder._decode_gc_min_seqlen = int(
            self.step_cfg.get("decode_gc_min_seqlen", 4096)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True)

        dec_state = (
            _prefixed_state(model_state, f"decoders.{family}")
            or _prefixed_state(model_state, "decoder")
            or state_dicts.get("decoder_qwen", {})
        )
        if not dec_state:
            raise ValueError(f"init checkpoint {init_ckpt} carries no {family} decoder state")
        d_missing, d_unexpected = self.decoder.load_state_dict(dec_state, strict=False)
        if d_missing or d_unexpected:
            raise RuntimeError(
                f"incomplete decoder init from {init_ckpt}: "
                f"missing={list(d_missing)[:10]}, unexpected={list(d_unexpected)[:10]}"
            )

        # --- encoder ---
        encoder_state = state_dicts.get("encoder") or _prefixed_state(model_state, "encoder")
        if not encoder_state:
            raise ValueError(f"init checkpoint {init_ckpt} carries no encoder state")
        encoder_cfg = self.cfg.model.get("encoder", {})
        threshold_cfg = dict(self.cfg.model.get("threshold_controller", {}) or {})
        saved_anchors = encoder_state.get("l0.threshold.anchor_ratios")
        if saved_anchors is not None:
            threshold_cfg["anchor_ratios"] = saved_anchors.tolist()
        # Drop learnable per-task L0 prompts from the source run (dataset-keyed;
        # blob encodes are query-free and the keys wouldn't resolve here).
        encoder_state = {
            k: v for k, v in encoder_state.items() if not k.startswith("l0_task_prompts.")
        }
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            encoder_cfg.get("backbone_name", "Qwen/Qwen3.5-0.8B-Base"),
            encoder_state,
            hidden_dim=int(encoder_cfg.get("hidden_dim", 1024)),
            active_decoder_family=family,
            threshold_controller_cfg=threshold_cfg or None,
        ).to(self.device)
        self.encoder.set_active_decoder_family(family)
        maybe_enable_gradient_checkpointing(self.encoder.l0.backbone, self.cfg)
        if getattr(self.encoder, "l1", None) is not None:
            maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)

        # Blob content is tokenized with the DECODER tokenizer (BlobSFTDataset)
        # but embedded by the ENCODER — fail closed on any vocab-size mismatch.
        enc_vocab = self.encoder.l0.backbone.get_input_embeddings().num_embeddings
        dec_vocab = backbone.get_input_embeddings().weight.shape[0]
        if enc_vocab < dec_vocab:
            raise RuntimeError(
                f"encoder vocab ({enc_vocab}) smaller than decoder vocab "
                f"({dec_vocab}) — blob content ids would index out of range"
            )

        if bool(self.step_cfg.get("freeze_encoder", False)):
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            logger.info("blob_sft_encoder_frozen")

        self.register_checkpoint_source("encoder", str(init_ckpt))
        self.register_checkpoint_source("decoder", str(init_ckpt))

        self.model = _BlobModel(self.encoder, self.decoder)
        self.model.train()

        # --- data ---
        data_cfg = self.step_cfg.get("data", {}) or {}
        eval_shards = _resolve_shards(list(data_cfg.get("eval_shards", []) or []))
        train_shards = _resolve_shards(
            list(data_cfg.get("train_shards", []) or []), exclude=set(eval_shards),
        )
        if not train_shards:
            raise ValueError("training.data.train_shards resolved to no files")
        ds_kwargs = dict(
            draws_per_trajectory=int(self.step_cfg.get("draws_per_trajectory", 4)),
            max_rows_per_shard=self.step_cfg.get("max_rows_per_shard", None),
            max_blob_content_tokens=int(self.step_cfg.get("max_blob_content_tokens", 45_000)),
            min_blob_content_tokens=int(self.step_cfg.get("min_blob_content_tokens", 64)),
            seed=int(self.cfg.get("seed", 17)),
        )
        # Preferred split: REPO-level holdout over the SAME shard list (no
        # repo can leak between train and eval). Explicit eval_shards remain
        # supported as the legacy shard-level split.
        holdout = float(data_cfg.get("repo_holdout_fraction", 0.0) or 0.0)
        train_ds = BlobSFTDataset(
            train_shards, self.tokenizer, **ds_kwargs,
            repo_holdout_fraction=holdout, split="train",
        )
        num_workers = int(self.step_cfg.get("num_workers", 2))
        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )
        if holdout > 0.0:
            eval_ds = BlobSFTDataset(
                train_shards, self.tokenizer, **ds_kwargs,
                repo_holdout_fraction=holdout, split="eval",
            )
        elif eval_shards:
            eval_ds = BlobSFTDataset(eval_shards, self.tokenizer, **ds_kwargs)
        else:
            eval_ds = None
        if eval_ds is not None:
            self.eval_dataloader = DataLoader(
                eval_ds,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda x: x,
            )
        else:
            self.eval_dataloader = None
        logger.info(
            "blob_sft_data_loaded",
            train_shards=len(train_shards),
            eval_shards=len(eval_shards),
            train_samples=len(train_ds),
        )

        # --- optimizer ---
        # Freeze the decoder's tied embedding / LM head BEFORE collecting
        # trainable params. HYGIENE, NOT A FIX for the Phase-2 language
        # collapse — see KRKBTrainer._freeze_decoder_embeddings for the
        # measurements that refuted that diagnosis (the embedding is
        # essentially identical in the healthy base and the destroyed
        # checkpoints). Pinning a 248,320-row tied matrix against ~107
        # loss-bearing tokens per sample is still the right default.
        if bool(self.step_cfg.get("freeze_decoder_embeddings", True)):
            frozen = 0
            bb = getattr(self.decoder, "backbone", self.decoder)
            for mod in filter(None, (
                bb.get_input_embeddings(), bb.get_output_embeddings(),
            )):
                for prm in mod.parameters():
                    if prm.requires_grad:
                        prm.requires_grad_(False)
                        frozen += prm.numel()
            logger.info("decoder_embeddings_frozen", params=frozen)
        else:
            logger.warning("decoder_embeddings_trainable_opt_out")
        lr = float(self.step_cfg.lr)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(
            [{"params": trainable, "lr": lr, "base_lr": lr}], lr,
        )

        self._l0_ratio = float(self.step_cfg.get("l0_ratio", 0.35))
        self._l1_ratio = float(self.step_cfg.get("l1_ratio", 0.5))
        self._max_decode_tokens = int(self.step_cfg.get("max_decode_tokens", 40_000))
        logger.info(
            "blob_sft_setup_complete",
            l0_ratio=self._l0_ratio,
            l1_ratio=self._l1_ratio,
            trainable_params=sum(p.numel() for p in trainable),
        )

    def _create_dataloader_iter(self, *, use_prefetch: bool | None = None):
        # Samples are lists of nested dicts (variable-length tensor lists);
        # the device prefetcher only understands flat tensor batches. Tensors
        # move to device inside _forward_backward instead.
        return super()._create_dataloader_iter(use_prefetch=False)

    def _wrap_dataloader_iter(self, iterator, *, use_prefetch: bool | None = None):
        # The RESUME-replay path in BaseTrainer.train() wraps its raw iterator
        # via this method directly (not _create_dataloader_iter), so the
        # prefetch bypass must live here too — the first resumed launch
        # crashed with AttributeError('list' has no 'items') inside
        # _DevicePrefetcher (2026-08-25).
        return super()._wrap_dataloader_iter(iterator, use_prefetch=False)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _encode_blobs(self, blob_content_ids: list[torch.Tensor]) -> list[torch.Tensor]:
        """One packed query-free encoder forward over all blobs of a sample."""
        embed = self.encoder.l0.backbone.get_input_embeddings()
        ids_flat = torch.cat([c.to(self.device) for c in blob_content_ids], dim=0)
        lengths = [int(c.shape[0]) for c in blob_content_ids]
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.device)
        cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=self.device).cumsum(0)
        pos = position_ids_from_cu(cu, int(ids_flat.shape[0]))
        enc_out = self.encoder(
            content_embeddings=embed(ids_flat),
            content_cu_seqlens=cu,
            content_position_ids=pos,
            target_ratio_l0=self._l0_ratio,
            target_ratio_l1=self._l1_ratio,
            selection_mode_l0="exact_topk",
            selection_mode_l1="exact_topk",
        )
        out_cu = enc_out.survivor_cu_seqlens.to(torch.int64).tolist()
        survivors = []
        for i in range(len(lengths)):
            s, e = out_cu[i], out_cu[i + 1]
            if e <= s:
                survivors.append(
                    torch.zeros(
                        1,
                        enc_out.survivor_embeddings.shape[-1],
                        device=self.device,
                        dtype=enc_out.survivor_embeddings.dtype,
                    )
                )
            else:
                survivors.append(enc_out.survivor_embeddings[s:e])
        return survivors

    def _sample_loss(
        self, sample: dict, *, zero_reps: bool = False, return_output: bool = False,
    ):
        token_ids = sample["token_ids"].to(self.device)
        loss_mask = sample["loss_mask"].to(self.device)
        survivors = self._encode_blobs(sample["blob_content_ids"])
        segments = build_blob_segments(
            token_ids, loss_mask, sample["sentinel_spans"], survivors,
            zero_reps=zero_reps,
        )
        # The decoder's spliced-rep norm guard escalates to RuntimeError after
        # a streak of degenerate splices. Under ``zero_reps`` the degenerate
        # reps ARE the experiment, so the guard must stand down for exactly
        # that forward — set here, the single choke point through which every
        # ablation forward passes, so the flag cannot drift out of sync.
        # (2026-08-25: the standalone 256-sample eval crashed here; the
        # in-training pass only ever ran short enough streaks to survive.)
        prev = getattr(self.decoder, "_rep_norm_guard_expect_degenerate", False)
        self.decoder._rep_norm_guard_expect_degenerate = bool(zero_reps)
        try:
            result = self.decoder.forward_interleaved_with_loss(
                segments, return_hidden_states=return_output,
            )
        finally:
            self.decoder._rep_norm_guard_expect_degenerate = prev
        n_reps = sum(int(s.shape[0]) for s in survivors)
        return result, n_reps

    def _forward_backward(self, batch) -> dict[str, float]:
        sample = batch[0] if isinstance(batch, list) else batch
        if sample is None:
            # Dataset draw produced no valid sample (short trajectory /
            # over-long blob); contribute nothing to this optimizer step.
            return {"loss": 0.0, "skipped_samples": 1.0}
        from bgkit.data.datasets.blob_sft_dataset import spliced_length

        decode_len = spliced_length(
            sample, l0_ratio=self._l0_ratio, l1_ratio=self._l1_ratio,
        )
        if decode_len > self._max_decode_tokens:
            logger.warning(
                "blob_sft_sample_too_long", decode_len=decode_len,
                cap=self._max_decode_tokens,
            )
            return {"loss": 0.0, "skipped_samples": 1.0}

        loss, n_reps = self._sample_loss(sample)
        accum = max(1, int(getattr(self, "_accum_steps", 1)))
        (loss / accum).backward()

        qtype = str(sample.get("meta", {}).get("qtype", "?"))
        return {
            "loss": float(loss.detach().item()),
            f"loss/{qtype}": float(loss.detach().item()),
            "n_blobs": float(len(sample["blob_content_ids"])),
            "n_reps": float(n_reps),
            "decode_len": float(decode_len),
            "skipped_samples": 0.0,
        }

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            return {"eval/loss": float("nan"), "eval/n_samples": 0.0}
        max_samples = int(self.step_cfg.get("max_eval_samples", 128))
        do_zeroed = bool(self.step_cfg.get("eval_ablation_zeroed", True))

        tot_loss = tot_tokens = 0.0
        tot_correct = 0.0
        zero_loss = zero_tokens = 0.0
        zero_correct = 0.0
        zero_probe_hits = zero_probe_n = 0
        # Per-qtype zeroed loss + verdict agreement (2026-08-25). EM is
        # all-or-nothing over the whole answer, so equal EM counts do not
        # prove the reps changed nothing: probe LOSS moves continuously, and
        # agreement says whether the very same probes were right both ways.
        zero_per_qtype: dict[str, list[float]] = {}
        probe_agree = 0
        per_qtype: dict[str, list[float]] = {}
        probe_hits = probe_n = 0
        n_seen = 0
        for batch in self.eval_dataloader:
            if n_seen >= max_samples:
                break
            sample = batch[0]
            if sample is None:
                continue
            from bgkit.data.datasets.blob_sft_dataset import spliced_length

            if (
                spliced_length(sample, l0_ratio=self._l0_ratio, l1_ratio=self._l1_ratio)
                > self._max_decode_tokens
            ):
                continue
            out, _ = self._sample_loss(sample, return_output=True)
            n_tokens = int(out.loss_mask.sum().item())
            if n_tokens == 0:
                continue
            n_seen += 1
            tot_loss += float(out.loss.item()) * n_tokens
            tot_tokens += n_tokens
            # Next-token shift: prediction at shifted index i targets token i+1.
            preds = out.argmax_predictions()  # (B, S-1)
            targets = out.token_ids[:, 1:]
            mask = out.loss_mask[:, 1:]
            correct = (preds == targets) & mask
            tot_correct += float(correct.sum().item())
            qtype = str(sample.get("meta", {}).get("qtype", "?"))
            per_qtype.setdefault(qtype, []).append(float(out.loss.item()))
            if qtype == "recall_probe":
                probe_n += 1
                if bool(correct[mask].all().item()):
                    probe_hits += 1
            if do_zeroed:
                # Rich zeroed arm (2026-08-25): pooled loss alone cannot
                # answer "are probe answers read from the reps or guessed
                # from priors?" (probes are ~2% of eval tokens; SWE-Zero
                # paths are templated and partially guessable). The metric
                # that decides it is probe EM WITH the reps zeroed.
                z_out, _ = self._sample_loss(
                    sample, zero_reps=True, return_output=True,
                )
                zero_loss += float(z_out.loss.item()) * n_tokens
                zero_tokens += n_tokens
                z_correct = (
                    (z_out.argmax_predictions() == z_out.token_ids[:, 1:])
                    & z_out.loss_mask[:, 1:]
                )
                zero_correct += float(z_correct.sum().item())
                zero_per_qtype.setdefault(qtype, []).append(float(z_out.loss.item()))
                if qtype == "recall_probe":
                    zero_probe_n += 1
                    z_hit = bool(z_correct[z_out.loss_mask[:, 1:]].all().item())
                    zero_probe_hits += int(z_hit)
                    probe_agree += int(z_hit == bool(correct[mask].all().item()))

        if tot_tokens == 0:
            return {"eval/loss": float("nan"), "eval/n_samples": 0.0}
        metrics = {
            "eval/loss": tot_loss / tot_tokens,
            "eval/token_accuracy": tot_correct / tot_tokens,
            "eval/n_samples": float(n_seen),
        }
        for qtype, losses in per_qtype.items():
            metrics[f"eval/{qtype}/loss"] = sum(losses) / len(losses)
            metrics[f"eval/{qtype}/n"] = float(len(losses))
        if probe_n:
            metrics["eval/probe_exact_match"] = probe_hits / probe_n
            metrics["eval/probe_n"] = float(probe_n)
        if do_zeroed and zero_tokens:
            metrics["eval/ablation/zeroed/loss"] = zero_loss / zero_tokens
            metrics["eval/zeroed_gap"] = metrics["eval/ablation/zeroed/loss"] - metrics["eval/loss"]
            metrics["eval/ablation/zeroed/token_accuracy"] = zero_correct / zero_tokens
            for qtype, losses in zero_per_qtype.items():
                zl = sum(losses) / len(losses)
                metrics[f"eval/ablation/zeroed/{qtype}/loss"] = zl
                base = metrics.get(f"eval/{qtype}/loss")
                if base is not None:
                    metrics[f"eval/zeroed_gap/{qtype}"] = zl - base
            if zero_probe_n:
                metrics["eval/ablation/zeroed/probe_exact_match"] = (
                    zero_probe_hits / zero_probe_n
                )
                # THE Family-A gate: recall that survives only when the reps
                # are present. > 0 means the blob carries load-bearing detail.
                metrics["eval/probe_zeroed_gap"] = (
                    metrics.get("eval/probe_exact_match", 0.0)
                    - metrics["eval/ablation/zeroed/probe_exact_match"]
                )
                # Equal EM counts can still hide churn (different probes right
                # each way). 1.0 = the reps flipped no probe at all.
                metrics["eval/probe_zeroed_agreement"] = probe_agree / zero_probe_n
        # Plain held-out-text CE: the only metric here that can see the
        # decoder's general language ability collapsing. Every other number
        # above is measured on the distribution being fitted.
        n_health = int(self.step_cfg.get("lm_health_samples", 0) or 0)
        if n_health > 0:
            from bgkit.training.lm_health import lm_health_metrics, load_health_chunks

            try:
                if getattr(self, "_lm_health_chunks", None) is None:
                    self._lm_health_chunks = load_health_chunks(
                        self.step_cfg.get("lm_health_tokens_path", None)
                        or self.cfg.training.data.file_tokens_path,
                        n_docs=n_health,
                        seq_len=int(self.step_cfg.get("lm_health_seq_len", 1024)),
                    )
                metrics.update(
                    lm_health_metrics(self.decoder, self._lm_health_chunks, self.device)
                )
            except Exception as exc:  # diagnostics must never kill a run
                logger.warning("lm_health_failed", error=str(exc))
        return metrics
