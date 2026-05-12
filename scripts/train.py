#!/usr/bin/env python
"""Main Hydra entry point for training."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from bgkit.utils.deltanet_patch import (
    patch_fused_rms_norm_gated_for_sm121,
    patch_gated_delta_rule_numerics,
)
from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.step_watchdog import install_step_watchdog
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner


def _create_trainer(cfg: DictConfig):
    """Create the appropriate trainer for the configured phase."""
    phase = cfg.get("training", {}).get("phase", None)
    if phase is None:
        raise ValueError("No training phase specified. Use a training config override.")

    if phase == "joint_block_pretrain":
        from bgkit.training.joint_block_trainer import JointBlockTrainer

        return JointBlockTrainer(cfg)
    elif phase == "phase1_step1":
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    elif phase == "phase1_step2":
        from bgkit.training.distillation.pruning_distill import PruningDistillTrainer

        return PruningDistillTrainer(cfg)
    elif phase == "phase1_step2p5":
        # Projection-only embed-anchor repair. Loads a Step 2 checkpoint,
        # freezes compressor + decoder, re-trains only the projection
        # block against decoder.embed_tokens(content_ids).
        from bgkit.training.phase1.projection_repair import ProjectionRepairTrainer

        return ProjectionRepairTrainer(cfg)
    elif phase == "phase1_falcon_dense_seed":
        # Cache-based: the encoder is frozen and the forced-survivor mask is
        # fixed, so pre-projection survivor embeddings are deterministic and
        # are pre-computed once by scripts/build_dense_seed_cache.py. Training
        # is then pure projection_block forward+backward over the cache,
        # ~50x faster than running the encoder every step.
        from bgkit.training.phase1.projection_seed_falcon_cached import (
            FalconProjectionCachedTrainer,
        )

        return FalconProjectionCachedTrainer(cfg)
    elif phase == "phase1_falcon_forced_adapt":
        # forced_adapt unfreezes encoder.l0 (adapter learning), so we can't
        # use the cache there — the encoder forward is part of the trainable
        # graph. The original trainer drives that phase.
        from bgkit.training.phase1.projection_seed_falcon import (
            FalconProjectionSeedTrainer,
        )

        return FalconProjectionSeedTrainer(cfg)
    elif phase == "phase1_step3":
        # Pruned reconstruction with compression curriculum (post Step 2.5).
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    elif phase == "phase1_step4":
        # QA-conditioned head supervision. Reuses DecoderInitTrainer because
        # the data flow is identical (encoder + decoder + chat template);
        # the QA-position loss + question-as-compression-prompt wiring lives
        # in the config + survivorship_helpers.
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    elif phase in (
        "phase1_step4p7",
        "phase1_step4p7_v2",
        "phase1_step4p7_v3",
    ):
        # Bridge distillation: teacher = frozen reference encoder; student =
        # full L0->L1 with bridge + last L0 block + first 2 L1 blocks +
        # projection_block trainable. v1 (phase1_step4p7) uses Step 4 as
        # both teacher and student. v2 (phase1_step4p7_v2) loads teacher
        # and student from separate checkpoints (typically teacher =
        # phase1_step4p7, student = current phase1_step5) and trains
        # across an extended ratio range. See plans/bridge-distill-step.md
        # for the v1 design.
        from bgkit.training.phase1.bridge_distill import BridgeDistillTrainer

        return BridgeDistillTrainer(cfg)
    elif phase == "phase1_step5":
        from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

        return CommitEncodingTrainer(cfg)
    elif phase in (
        "phase1_step6",
        "phase1_falcon_l0_align",
        "phase1_falcon_l0",
        "phase1_falcon_l1",
    ):
        # phase1_falcon_l0_align runs the CompressionTrainer at target_ratio=1.0
        # for end-to-end no-compression alignment; phase1_falcon_l0 is then the
        # slow compression ramp 1.0 → 0.10. The resolver chain in
        # compression.py (`_resolve_step1_checkpoint`) walks
        # l0 → l0_align → forced_adapt → dense_seed so each phase resumes from
        # the most recent prior stage automatically.
        from bgkit.training.phase1.compression import CompressionTrainer

        return CompressionTrainer(cfg)
    elif phase in ("phase2", "phase2_kb"):
        # Phase 2 is unified: a single trainer handles every dataset via
        # the trajectory framework. Flat datasets (NewsQA, MS MARCO,
        # SearchQA, git history, memory) emit single-bgkit trajectories;
        # hierarchical datasets (KILT, PubMedQA via MeSH, NarrativeQA
        # per book) emit browse + bgkit trajectories. The legacy
        # ``phase2`` route is kept as an alias so existing checkpoints
        # / configs still resolve.
        from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

        return KRKBTrainer(cfg)
    elif phase == "phase3":
        from bgkit.training.phase3.distillation_trainer import DistillationTrainer

        return DistillationTrainer(cfg)
    else:
        raise NotImplementedError(f"Training phase '{phase}' not yet implemented")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_triton_allocator()
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()
    patch_fused_rms_norm_gated_for_sm121()
    install_step_watchdog(
        timeout_seconds=float(os.environ.get("BGKIT_STEP_TIMEOUT", "60.0")),
        poll_seconds=5.0,
    )
    setup_logging()
    set_seed(cfg.seed)

    # Hard-cap the PyTorch CUDA allocator on unified-memory systems. Docker's
    # mem_limit only accounts for cgroup CPU memory; CUDA allocations via
    # nvidia-container-runtime bypass it, so on DGX Spark a runaway alloc can
    # drag the 128 GB unified pool into page-thrash and stall the host before
    # any OOM-killer fires. The fraction-of-total cap lets PyTorch raise a
    # clean OOMError instead. Override with BGKIT_CUDA_MEM_FRACTION.
    import os as _os

    import torch as _torch

    if _torch.cuda.is_available():
        _frac = float(_os.environ.get("BGKIT_CUDA_MEM_FRACTION", "0.65"))

        # Peer-container safety check: on unified-memory systems,
        # ``set_per_process_memory_fraction`` is per-process, not per-host.
        # Two containers each claiming 65% of the 128 GB pool can push total
        # allocation past the pool, triggering kernel page-thrash and host
        # death — this bit us twice already. Query current CUDA usage before
        # configuring our fraction; if existing usage + our ask would exceed
        # 90% of the pool (leaving 10% for OS + driver), refuse to start.
        # Set BGKIT_ALLOW_PEER_CUDA=1 to bypass (auto-shrink instead).
        #
        # On unified memory, ``mem_get_info`` returns the host-wide pool, so
        # ``total - free`` includes reclaimable page cache (mmap'd dataset
        # offsets, prior model weights still cached, HF cache files). Those
        # pages get evicted instantly under memory pressure, so they are not
        # real contention. Subtract them out to get an accurate estimate of
        # what's actually pinned by peer processes.
        def _reclaimable_bytes() -> int:
            """Page cache + buffers + reclaimable slab from /proc/meminfo.

            Linux can evict these instantly under memory pressure; counting
            them as "held" inflates the peer-CUDA estimate by 30+ GB on a
            system that has just finished a CUDA-heavy run.
            """
            try:
                with open("/proc/meminfo") as f:
                    info: dict[str, int] = {}
                    for line in f:
                        k, _, rest = line.partition(":")
                        # values like "12345678 kB"
                        parts = rest.strip().split()
                        if parts:
                            info[k.strip()] = int(parts[0]) * 1024
                return (
                    info.get("Cached", 0)
                    + info.get("Buffers", 0)
                    + info.get("SReclaimable", 0)
                )
            except (OSError, ValueError):
                return 0

        _free_bytes, _total_bytes = _torch.cuda.mem_get_info()
        _total_gb = _total_bytes / 1e9
        _raw_used_bytes = _total_bytes - _free_bytes
        _reclaimable_bytes_val = _reclaimable_bytes()
        _used_by_peers_bytes = max(0, _raw_used_bytes - _reclaimable_bytes_val)
        _used_by_peers_gb = _used_by_peers_bytes / 1e9
        _raw_used_gb = _raw_used_bytes / 1e9
        _reclaimable_gb = _reclaimable_bytes_val / 1e9
        print(
            f"[cuda-mem-guard] raw used = {_raw_used_gb:.1f} GB "
            f"(reclaimable cache = {_reclaimable_gb:.1f} GB) "
            f"=> peer CUDA = {_used_by_peers_gb:.1f} GB / {_total_gb:.1f} GB pool",
            flush=True,
        )
        _our_ask_gb = _frac * _total_gb
        _safe_ceiling_gb = 0.90 * _total_gb
        _allow_peer = _os.environ.get("BGKIT_ALLOW_PEER_CUDA", "0") == "1"

        if _used_by_peers_gb + _our_ask_gb > _safe_ceiling_gb:
            _max_safe_frac = max(0.05, (_safe_ceiling_gb - _used_by_peers_gb) / _total_gb)
            _msg = (
                f"[cuda-mem-guard] peer CUDA usage = {_used_by_peers_gb:.1f} GB / "
                f"{_total_gb:.1f} GB pool. Our requested fraction {_frac:.2f} "
                f"({_our_ask_gb:.1f} GB) would push total past the 90% host "
                f"safety ceiling ({_safe_ceiling_gb:.1f} GB). "
            )
            if _allow_peer:
                print(
                    f"{_msg}BGKIT_ALLOW_PEER_CUDA=1 set — auto-shrinking to "
                    f"fraction {_max_safe_frac:.3f} ({_max_safe_frac * _total_gb:.1f} GB).",
                    flush=True,
                )
                _frac = _max_safe_frac
            else:
                raise SystemExit(
                    f"{_msg}Refusing to start to avoid host OOM. "
                    f"Options: (a) stop peer CUDA containers, (b) set "
                    f"BGKIT_CUDA_MEM_FRACTION={_max_safe_frac:.2f} or lower, "
                    f"(c) set BGKIT_ALLOW_PEER_CUDA=1 to auto-shrink."
                )

        _torch.cuda.set_per_process_memory_fraction(_frac)

    print(OmegaConf.to_yaml(cfg))

    retry_cfg = cfg.get("retry", {})
    retry_enabled = retry_cfg.get("enabled", False) if retry_cfg else False

    # Validate: keep_latest >= 1 when retry is enabled
    if retry_enabled:
        prune_cfg = cfg.get("training", {}).get("checkpoint_pruning", {})
        if prune_cfg and prune_cfg.get("enabled", False):
            keep_latest = prune_cfg.get("keep_latest", 2)
            if keep_latest < 1:
                raise ValueError(
                    "checkpoint_pruning.keep_latest must be >= 1 when retry is enabled, "
                    f"got {keep_latest}"
                )

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    original_resume = cfg.get("resume_checkpoint", None)

    if retry_enabled:
        from bgkit.training.retry import retry_training

        last_ckpt_file = checkpoint_dir / ".last_checkpoint"

        # Stale file guard: delete .last_checkpoint before first attempt
        if last_ckpt_file.exists():
            last_ckpt_file.unlink()

        def _train_attempt():
            # Resolve resume path: .last_checkpoint > original > auto-resolve
            resume_path = None
            if last_ckpt_file.exists():
                candidate = last_ckpt_file.read_text().strip()
                if candidate and Path(candidate).exists():
                    resume_path = candidate
            if resume_path is None:
                resume_path = original_resume

            # Update config with resolved resume path (None triggers auto-resolve
            # inside the trainer's train() method)
            with _open_dict(cfg):
                cfg.resume_checkpoint = resume_path

            trainer = _create_trainer(cfg)
            trainer.train()

        retry_training(
            _train_attempt,
            max_retries=retry_cfg.get("max_retries", 3),
            base_delay=retry_cfg.get("base_delay", 30.0),
            max_delay=retry_cfg.get("max_delay", 300.0),
        )
    else:
        # Auto-resume is handled inside trainer.train() when resume_checkpoint is None
        trainer = _create_trainer(cfg)
        trainer.train()


def _open_dict(cfg):
    """Context manager to allow OmegaConf struct modification."""
    from contextlib import contextmanager

    from omegaconf import OmegaConf

    @contextmanager
    def _ctx():
        was_struct = OmegaConf.is_struct(cfg)
        OmegaConf.set_struct(cfg, False)
        try:
            yield cfg
        finally:
            OmegaConf.set_struct(cfg, was_struct)

    return _ctx()


if __name__ == "__main__":
    main()
