#!/usr/bin/env python3
"""Offline L0 pre-computation: encode documents through BgKIT encoder + ICE.

Produces sharded mmap survivors for fast training at Steps 3-4.
Uses the Phase 2 Step 2 checkpoint (where L0 is already frozen).

Output format per shard:
  survivors.npy   - bfloat16 mmap, shape (total_survivors_in_shard, hidden_dim)
  offsets.npy     - int64 CSR offsets, shape (docs_in_shard + 1,)
  ice_scores.npy  - float32 mmap, shape (total_survivors_in_shard,)

Global index:
  index.parquet   - (document_id, shard_id, row_index) mapping
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def _load_encoder_and_ice(checkpoint_path: str, device: torch.device):
    """Load BgKIT encoder and ICE from a Phase 2 checkpoint."""
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.models.ice import ICE
    from bgkit.training.checkpointing import load_checkpoint

    _metadata, state_dicts = load_checkpoint(Path(checkpoint_path))
    model_state = state_dicts.get("model", {})

    # Extract encoder state
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in model_state.items() if k.startswith("encoder.")
    }
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        "Qwen/Qwen3.5-0.8B-Base",
        encoder_state,
        hidden_dim=1024,
    )
    encoder.to(device).eval()
    encoder.requires_grad_(False)

    # Extract ICE state
    ice_state = {
        k.replace("ice.", "", 1): v
        for k, v in model_state.items() if k.startswith("ice.")
    }
    ice = ICE(input_dim=1024, hidden_dim=128, num_layers=3)
    if ice_state:
        ice.load_state_dict(ice_state, strict=False)
    ice.to(device).eval()
    ice.requires_grad_(False)

    return encoder, ice


def _encode_batch(
    encoder,
    ice,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    retention_ratio: float,
    device: torch.device,
) -> list[dict]:
    """Encode a batch and return per-document survivors + scores."""
    embed_tokens = encoder.compressor.backbone.get_input_embeddings()
    input_embeddings = embed_tokens(token_ids)

    # ICE scoring
    ice_scores = ice(input_embeddings)  # (B, L)

    # Select survivors per document
    results = []
    batch_size = token_ids.size(0)
    for i in range(batch_size):
        length = int(attention_mask[i].sum())
        if length == 0:
            results.append({
                "survivors": np.zeros((0, 1024), dtype=np.float16),
                "scores": np.array([], dtype=np.float32),
            })
            continue

        keep = max(1, math.ceil(length * retention_ratio))
        scores_i = ice_scores[i, :length]
        _, topk_indices = torch.topk(scores_i, min(keep, length))
        topk_indices, _ = topk_indices.sort()

        # Build survivor mask
        survivor_mask = torch.zeros(1, token_ids.size(1), dtype=torch.bool, device=device)
        survivor_mask[0, topk_indices] = True

        # Run encoder with survivor mask
        output = encoder(
            input_embeddings=input_embeddings[i : i + 1],
            survivor_mask=survivor_mask,
            attention_mask=attention_mask[i : i + 1],
        )

        survivors = output.survivor_embeddings[0].cpu().to(torch.float16).numpy()
        selected_scores = scores_i[topk_indices].cpu().numpy().astype(np.float32)
        results.append({"survivors": survivors, "scores": selected_scores})

    return results


def precompute_l0(
    checkpoint_path: str,
    token_dir: str,
    output_dir: str,
    retention_ratio: float = 0.05,
    shard_size: int = 100_000,
    batch_size: int = 4,
    max_seq_len: int = 8192,
) -> None:
    """Run L0 pre-computation over a token dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}")
    encoder, ice = _load_encoder_and_ice(checkpoint_path, device)

    # Load token dataset
    tokens = np.load(Path(token_dir) / "tokens.npy", mmap_mode="r")
    offsets = np.load(Path(token_dir) / "offsets.npy")

    # Load metadata for document IDs
    metadata_path = Path(token_dir) / "metadata.parquet"
    doc_ids = []
    if metadata_path.exists():
        meta = pq.read_table(metadata_path)
        for col_name in ("document_id", "id"):
            if col_name in meta.column_names:
                doc_ids = meta.column(col_name).to_pylist()
                break
    if not doc_ids:
        doc_ids = [str(i) for i in range(len(offsets) - 1)]

    total_docs = len(offsets) - 1
    print(f"Processing {total_docs} documents at {retention_ratio} retention")

    # Process in shards
    index_rows = []
    shard_id = 0
    shard_survivors: list[np.ndarray] = []
    shard_scores: list[np.ndarray] = []
    shard_lengths: list[int] = []
    shard_doc_count = 0

    def _flush_shard():
        nonlocal shard_id, shard_survivors, shard_scores, shard_lengths, shard_doc_count
        if not shard_survivors:
            return

        shard_name = f"shard_{shard_id:05d}"
        shard_dir = output_path / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)

        all_survivors = np.concatenate(shard_survivors, axis=0)
        all_scores = np.concatenate(shard_scores, axis=0)
        shard_offsets = np.zeros(len(shard_lengths) + 1, dtype=np.int64)
        np.cumsum(shard_lengths, out=shard_offsets[1:])

        np.save(shard_dir / "survivors.npy", all_survivors)
        np.save(shard_dir / "offsets.npy", shard_offsets)
        np.save(shard_dir / "ice_scores.npy", all_scores)

        print(f"  Shard {shard_name}: {shard_doc_count} docs, {len(all_survivors)} survivors")
        shard_id += 1
        shard_survivors = []
        shard_scores = []
        shard_lengths = []
        shard_doc_count = 0

    start_time = time.time()
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_token_lists = []
        batch_doc_indices = []

        for doc_idx in range(batch_start, batch_end):
            start = int(offsets[doc_idx])
            end = int(offsets[doc_idx + 1])
            doc_tokens = tokens[start:end].astype(np.int64)
            if len(doc_tokens) == 0:
                continue
            if len(doc_tokens) > max_seq_len:
                doc_tokens = doc_tokens[:max_seq_len]
            batch_token_lists.append(doc_tokens)
            batch_doc_indices.append(doc_idx)

        if not batch_token_lists:
            continue

        # Pad batch
        max_len = max(len(t) for t in batch_token_lists)
        padded = np.zeros((len(batch_token_lists), max_len), dtype=np.int64)
        masks = np.zeros((len(batch_token_lists), max_len), dtype=np.bool_)
        for i, t in enumerate(batch_token_lists):
            padded[i, : len(t)] = t
            masks[i, : len(t)] = True

        token_tensor = torch.from_numpy(padded).to(device)
        mask_tensor = torch.from_numpy(masks).to(device)

        with torch.no_grad():
            results = _encode_batch(
                encoder, ice, token_tensor, mask_tensor, retention_ratio, device,
            )

        for i, (doc_idx, result) in enumerate(
            zip(batch_doc_indices, results, strict=False),
        ):
            survivors = result["survivors"]
            scores = result["scores"]

            index_rows.append({
                "document_id": str(doc_ids[doc_idx]),
                "shard_id": f"shard_{shard_id:05d}",
                "row_index": shard_doc_count,
            })

            shard_survivors.append(survivors)
            shard_scores.append(scores)
            shard_lengths.append(len(survivors))
            shard_doc_count += 1

            if shard_doc_count >= shard_size:
                _flush_shard()

        # Progress
        if (batch_start // batch_size) % 100 == 0:
            elapsed = time.time() - start_time
            docs_done = batch_end
            rate = docs_done / max(elapsed, 1)
            eta = (total_docs - docs_done) / max(rate, 1)
            print(
                f"  Progress: {docs_done}/{total_docs} docs "
                f"({rate:.0f} docs/s, ETA {eta / 3600:.1f}h)",
            )

    # Flush final shard
    _flush_shard()

    # Write global index
    index_table = pa.table({
        "document_id": pa.array([r["document_id"] for r in index_rows], type=pa.string()),
        "shard_id": pa.array([r["shard_id"] for r in index_rows], type=pa.string()),
        "row_index": pa.array([r["row_index"] for r in index_rows], type=pa.int64()),
    })
    pq.write_table(index_table, output_path / "index.parquet")

    elapsed = time.time() - start_time
    print(
        f"\nDone: {total_docs} documents in {elapsed / 3600:.1f}h, "
        f"{shard_id} shards, {sum(r['row_index'] + 1 for r in index_rows[-1:])} total entries",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Phase 2 Step 2 checkpoint path")
    parser.add_argument("--token-dir", required=True, help="Mmap token dataset directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for sharded cache")
    parser.add_argument("--retention-ratio", type=float, default=0.05)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    args = parser.parse_args()

    precompute_l0(
        checkpoint_path=args.checkpoint,
        token_dir=args.token_dir,
        output_dir=args.output_dir,
        retention_ratio=args.retention_ratio,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
