#!/usr/bin/env python3
"""Pre-compute pre-projection survivor embeddings for Falcon dense-seed training.

The Falcon dense-seed objective only needs (survivor_embedding, falcon_target_pair_ids)
pairs. The encoder is frozen and the forced-survivor mask is fixed by the companion
mmap, so the survivor embeddings are a deterministic function of the source checkpoint.
Running the encoder forward inside the training loop wastes ~95% of wall-clock on
recomputing the same vectors every step.

This script:
  1. Loads the latest split-L0/L1 encoder checkpoint via the project's resolver.
  2. Iterates a (subsampled) view of MmapTokenDataset + Falcon companion
     sequentially with no_grad, batched for GPU throughput.
  3. Runs encoder.l0(...) with forced_survivor_mask, captures
     l0_out.survivor_embeddings — exactly what feeds into projection_block.
  4. Writes a compact cache to disk: survivor_embeddings (bf16) + per-chunk
     CSR offsets + the companion's pair targets + loss masks + alignment
     scores + manifest.

The training counterpart consumes this cache and runs the projection block as
a pure feedforward problem (no encoder forward, no decoder layers). See
``bgkit/training/phase1/projection_seed_falcon_cached.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset  # noqa: E402
from bgkit.models.encoder import BgKITEncoder  # noqa: E402
from bgkit.training.checkpoint_registry import resolve_checkpoint  # noqa: E402
from bgkit.training.checkpointing import load_checkpoint  # noqa: E402
from bgkit.utils.attention_backend import resolve_attention_implementation  # noqa: E402
from bgkit.utils.packing import position_ids_from_cu  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_encoder_checkpoint(
    checkpoint_dir: Path,
    explicit: str | None = None,
) -> Path:
    """Resolve the source bgkit encoder, preferring latest split-L0/L1 phases."""
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = checkpoint_dir / path
        return path
    # Mirror FalconProjectionSeedTrainer._resolve_source_checkpoint priority.
    for phase in ("phase1_step5", "phase1_step6", "phase1_step2p5", "phase1_step2"):
        try:
            return resolve_checkpoint(
                checkpoint_dir,
                phase=phase,
                metric="eval/loss",
                label="bgkit_checkpoint",
            )
        except ValueError:
            continue
    raise ValueError(
        f"No split-L0/L1 encoder checkpoint found under {checkpoint_dir} in any "
        "of phase1_step5 / phase1_step6 / phase1_step2p5 / phase1_step2."
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path(os.environ["DATA_DIR"]) / "processed_v2" / "tokens",
        help="Qwen base mmap directory (with tokens.npy + offsets.npy + manifest.json).",
    )
    parser.add_argument(
        "--companion-dir",
        type=Path,
        default=Path(os.environ["DATA_DIR"]) / "processed_v2" / "tokens_falcon_h1",
        help="Falcon companion mmap directory (with falcon_tokens.npy etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ["DATA_DIR"]) / "processed" / "falcon_dense_seed_cache",
        help="Where to write the cache mmap files + manifest.json.",
    )
    parser.add_argument(
        "--bgkit-checkpoint",
        type=str,
        default=None,
        help="Source bgkit encoder checkpoint (absolute path or registry name); "
        "defaults to the latest phase1_step5 → step6 → step2p5 → step2.",
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HF backbone name for encoder construction.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=30_000,
        help="Subsample to at most this many chunks from the (valid) companion "
        "view. Set to 0 for the full ~1.3M chunks (~260 GB cache).",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=8192,
        help="Token budget per encoder forward batch. Smaller batches are more "
        "memory-conservative; larger improve throughput.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Must match the companion build (chunk-count alignment is strict). "
        "Default 8192 matches the existing companion.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Subsample RNG seed."
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader num_workers for content-token prefetch.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ env
    ckpt_env = os.environ.get("CHECKPOINT_DIR")
    if not ckpt_env:
        raise EnvironmentError(
            "CHECKPOINT_DIR is not set in the environment (the Docker image "
            "sets it to /workspace/checkpoints; on the host, your .env must "
            "export it)."
        )
    checkpoint_dir = Path(ckpt_env)
    src_path = _resolve_encoder_checkpoint(checkpoint_dir, args.bgkit_checkpoint)
    print(f"[cache] source encoder checkpoint: {src_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache] device: {device}", flush=True)

    # ------------------------------------------------------------------ encoder
    _meta, state_dicts = load_checkpoint(src_path)
    if "encoder" not in state_dicts:
        raise ValueError(f"Source checkpoint {src_path} has no 'encoder' state dict")

    attn_impl = resolve_attention_implementation("auto")
    # Falcon Phase 1 checkpoints cap anchors at 0.60 (6 anchors); the encoder
    # default schedule is 7. Pass the Falcon-shaped threshold config so the
    # state_dict load doesn't fail with size mismatch on l{0,1}.threshold.*.
    falcon_threshold_cfg = {
        "anchor_ratios": [0.02, 0.04, 0.08, 0.16, 0.32, 0.60],
    }
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        args.backbone_name,
        state_dicts["encoder"],
        hidden_dim=1024,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        bidi_warmup_steps=0,
        active_decoder_family="falcon_h1",
        threshold_controller_cfg=falcon_threshold_cfg,
    ).to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    embed = encoder.l0.backbone.get_input_embeddings()
    hidden_dim = int(encoder.l0.hidden_dim)
    print(f"[cache] encoder loaded; hidden_dim={hidden_dim}", flush=True)

    # ------------------------------------------------------------------ dataset
    base_dataset = MmapTokenDataset(
        args.tokens_dir,
        max_seq_len=args.max_seq_len,
        include_metadata=False,
        companion_dir=str(args.companion_dir),
    )
    valid = base_dataset.companion_valid_indices
    if valid is None or valid.size == 0:
        raise ValueError(
            f"Falcon companion at {args.companion_dir} exposes no valid chunks"
        )
    # Subsample (random, deterministic).
    if args.max_chunks and args.max_chunks < valid.size:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(valid.size, size=args.max_chunks, replace=False)
        idx.sort()  # sequential disk access
        selected = valid[idx]
    else:
        selected = np.sort(valid)
    n_chunks = int(selected.size)
    print(
        f"[cache] selected {n_chunks:,} chunks "
        f"(out of {valid.size:,} valid; companion total "
        f"{base_dataset.lengths.size:,})",
        flush=True,
    )

    # ------------------------------------------------------------------ output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Predict total survivors so we can allocate output mmaps in one pass.
    forced_offsets = base_dataset._companion_arrays["forced_survivor_offsets"]
    per_chunk_surv = (forced_offsets[selected + 1] - forced_offsets[selected]).astype(
        np.int64
    )
    survivor_offsets = np.zeros(n_chunks + 1, dtype=np.int64)
    np.cumsum(per_chunk_surv, out=survivor_offsets[1:])
    total_survivors = int(survivor_offsets[-1])
    print(
        f"[cache] total survivors to cache: {total_survivors:,} "
        f"(avg {total_survivors / max(n_chunks, 1):.1f} / chunk)",
        flush=True,
    )

    survivor_emb_bytes = total_survivors * hidden_dim * 2  # bf16 = 2 bytes
    print(
        f"[cache] survivor_embeddings.bin will be "
        f"{survivor_emb_bytes / 1e9:.1f} GB on disk",
        flush=True,
    )

    emb_path = args.output_dir / "survivor_embeddings.bin"
    # Preallocate as a sparse file; write in chunks via np.memmap.
    emb_mmap = np.memmap(
        emb_path,
        mode="w+",
        dtype=np.dtype("<f2"),  # signed bf16 is not a numpy dtype; use float16
        shape=(total_survivors, hidden_dim),
    )
    # NOTE: we write bf16 *bytes* through a uint16 view to avoid float16 cast.
    # bf16 occupies the same 2 bytes as float16 for numpy memmap addressing.
    emb_mmap_u16 = np.memmap(
        emb_path,
        mode="r+",
        dtype=np.uint16,
        shape=(total_survivors, hidden_dim),
    )
    del emb_mmap  # close the float16 view

    # ------------------------------------------------------------------ pair targets
    # Companion stores pair_ids / pair_loss_mask as ``(N_survivors, 2)`` —
    # each surviving Qwen position decodes into 2 Falcon-vocab token ids
    # (matches projection_block.output_split_factor=2 for falcon_h1). We keep
    # the (N_survivors, 2) layout in the cache so the training collator can
    # ``.reshape(-1)`` exactly like the existing dense-pair collator did.
    target_pair_ids = np.empty((total_survivors, 2), dtype=np.int32)
    pair_loss_mask = np.empty((total_survivors, 2), dtype=bool)
    alignment_scores = np.empty(total_survivors, dtype=np.float32)

    src_pair_ids = base_dataset._companion_arrays["target_falcon_pair_ids"]
    src_pair_mask = base_dataset._companion_arrays["target_pair_loss_mask"]
    src_align = base_dataset._companion_arrays["alignment_scores"]

    print("[cache] copying companion pair targets...", flush=True)
    for i in range(n_chunks):
        chunk = int(selected[i])
        s0 = int(forced_offsets[chunk])
        s1 = int(forced_offsets[chunk + 1])
        out_lo = int(survivor_offsets[i])
        out_hi = int(survivor_offsets[i + 1])
        target_pair_ids[out_lo:out_hi] = src_pair_ids[s0:s1]
        pair_loss_mask[out_lo:out_hi] = src_pair_mask[s0:s1]
        alignment_scores[out_lo:out_hi] = src_align[s0:s1]
        if (i + 1) % 5000 == 0:
            print(f"[cache]   ...{i + 1:,}/{n_chunks:,} chunks copied", flush=True)

    np.save(args.output_dir / "survivor_offsets.npy", survivor_offsets)
    np.save(args.output_dir / "target_pair_ids.npy", target_pair_ids)
    np.save(args.output_dir / "pair_loss_mask.npy", pair_loss_mask)
    np.save(args.output_dir / "alignment_scores.npy", alignment_scores)
    np.save(args.output_dir / "selected_chunks.npy", selected.astype(np.int64))
    print("[cache] pair targets written", flush=True)

    # ------------------------------------------------------------------ encode
    # Greedy length-bucketed batching: maximize tokens per batch under
    # max_batch_tokens, preserving deterministic order.
    chunk_lens = base_dataset.lengths[selected].astype(np.int64)
    max_batch_tokens = int(args.max_batch_tokens)

    def _build_batches() -> list[list[int]]:
        # Sequential greedy packing — order preserved so disk reads are
        # monotonic. Each chunk-local index maps to its position in ``selected``.
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_tokens = 0
        for local_i in range(n_chunks):
            L = int(chunk_lens[local_i])
            if cur and cur_tokens + L > max_batch_tokens:
                batches.append(cur)
                cur = [local_i]
                cur_tokens = L
            else:
                cur.append(local_i)
                cur_tokens += L
        if cur:
            batches.append(cur)
        return batches

    batches = _build_batches()
    print(f"[cache] {len(batches):,} encoder batches (budget={max_batch_tokens})", flush=True)

    forced_indices_arr = base_dataset._companion_arrays["forced_survivor_indices"]

    completed_survivors = 0
    last_log = 0
    import time as _time
    t0 = _time.monotonic()

    with torch.inference_mode():
        for batch_idx, batch_local in enumerate(batches):
            # Gather content tokens for each chunk in the batch.
            tokens_list: list[torch.Tensor] = []
            forced_mask_pieces: list[torch.Tensor] = []
            sample_lengths: list[int] = []
            for local_i in batch_local:
                global_chunk = int(selected[local_i])
                L = int(base_dataset.lengths[global_chunk])
                start = int(base_dataset._chunk_offset[global_chunk])
                tokens = (
                    base_dataset._tokens[start : start + L]
                    .astype(np.int64, copy=False)
                )
                tokens_list.append(torch.from_numpy(tokens.copy()))
                # Build per-chunk forced mask.
                s0 = int(forced_offsets[global_chunk])
                s1 = int(forced_offsets[global_chunk + 1])
                forced = forced_indices_arr[s0:s1].astype(np.int64, copy=False)
                mask = np.zeros(L, dtype=bool)
                mask[forced] = True
                forced_mask_pieces.append(torch.from_numpy(mask.copy()))
                sample_lengths.append(L)

            content_ids = torch.cat(tokens_list, dim=0).to(device, non_blocking=True)
            forced_mask = torch.cat(forced_mask_pieces, dim=0).to(
                device, non_blocking=True
            )
            cu = torch.zeros(len(sample_lengths) + 1, dtype=torch.int32)
            torch.cumsum(
                torch.tensor(sample_lengths, dtype=torch.int32), 0, out=cu[1:]
            )
            cu = cu.to(device, non_blocking=True)
            total_content = int(cu[-1].item())
            pos = position_ids_from_cu(cu, total_content)

            content_emb = embed(content_ids)
            l0_out = encoder.l0(
                content_embeddings=content_emb,
                content_cu_seqlens=cu,
                content_position_ids=pos,
                target_ratio=None,
                forced_survivor_mask=forced_mask,
            )
            survivor_embeddings = l0_out.survivor_embeddings  # (N_surv, hidden_dim)
            # Convert bf16 GPU tensor → uint16 numpy view → write to mmap.
            sv_cpu = survivor_embeddings.to(torch.bfloat16).contiguous().cpu()
            # bfloat16 tensor → uint16 view of the same bytes.
            sv_u16 = sv_cpu.view(torch.uint16).numpy()

            # Slice survivor_embeddings by per-sample survivor counts (== forced
            # counts; the L0 head is overridden by the forced mask).
            sv_cu = l0_out.survivor_cu_seqlens.cpu().numpy().astype(np.int64)
            for j, local_i in enumerate(batch_local):
                lo = int(sv_cu[j])
                hi = int(sv_cu[j + 1])
                out_lo = int(survivor_offsets[local_i])
                out_hi = int(survivor_offsets[local_i + 1])
                if hi - lo != out_hi - out_lo:
                    raise RuntimeError(
                        "Survivor count mismatch on chunk "
                        f"{int(selected[local_i])}: encoder produced {hi - lo}, "
                        f"companion expected {out_hi - out_lo}"
                    )
                emb_mmap_u16[out_lo:out_hi] = sv_u16[lo:hi]

            completed_survivors = int(survivor_offsets[batch_local[-1] + 1])
            if completed_survivors - last_log >= 200_000 or batch_idx == len(batches) - 1:
                elapsed = _time.monotonic() - t0
                rate = completed_survivors / max(elapsed, 1e-6)
                done_frac = completed_survivors / max(total_survivors, 1)
                eta = (total_survivors - completed_survivors) / max(rate, 1e-6)
                print(
                    f"[cache] batch {batch_idx + 1:,}/{len(batches):,} "
                    f"survivors={completed_survivors:,}/{total_survivors:,} "
                    f"({done_frac:.1%}) rate={rate:,.0f} surv/s "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
                last_log = completed_survivors

    emb_mmap_u16.flush()
    del emb_mmap_u16

    # ------------------------------------------------------------------ manifest
    src_companion_manifest = args.companion_dir / "manifest.json"
    companion_sha = (
        _sha256_file(src_companion_manifest)
        if src_companion_manifest.exists()
        else None
    )
    src_base_manifest = args.tokens_dir / "manifest.json"
    base_sha = (
        _sha256_file(src_base_manifest) if src_base_manifest.exists() else None
    )

    manifest = {
        "schema_version": 1,
        "n_chunks": n_chunks,
        "total_survivors": total_survivors,
        "hidden_dim": hidden_dim,
        "embedding_dtype": "bfloat16",
        "max_seq_len_at_build": int(args.max_seq_len),
        "max_chunks_arg": int(args.max_chunks),
        "max_batch_tokens": int(args.max_batch_tokens),
        "subsample_seed": int(args.seed),
        "source_checkpoint": str(src_path),
        "source_backbone_name": args.backbone_name,
        "source_companion_manifest_sha256": companion_sha,
        "source_base_manifest_sha256": base_sha,
        "decoder_family": "falcon_h1",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[cache] manifest written to {args.output_dir / 'manifest.json'}",
        flush=True,
    )
    print("[cache] done", flush=True)


if __name__ == "__main__":
    main()
