"""Generate per-token cross-entropy labels for the full tokenized corpus.

Reads token shards from corpus processing, runs Qwen3-0.6B forward passes,
and writes matching Parquet shards with CE values as ICE training targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from bgkit.data.ice_labels import compute_per_token_cross_entropy
from bgkit.data.tokenization import chunk_tokens

log = logging.getLogger(__name__)


def generate_ce_for_sequence(
    token_ids: list[int],
    model: torch.nn.Module,
    max_seq_len: int = 8192,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute per-token cross-entropy for a token sequence.

    Chunks the sequence if it exceeds max_seq_len, runs the model in
    inference mode (no_grad, bf16), and returns float16 CE values.

    Args:
        token_ids: Token IDs for a single file.
        model: Causal LM model (e.g. Qwen3-0.6B).
        max_seq_len: Maximum sequence length per forward pass.
        device: Device to run on. If None, uses model's device.

    Returns:
        numpy float16 array of per-token CE values (length = len(token_ids) - 1).
    """
    if device is None:
        device = next(model.parameters()).device

    if len(token_ids) < 2:
        return np.array([], dtype=np.float16)

    chunks = chunk_tokens(token_ids, max_seq_len)
    all_ce: list[np.ndarray] = []

    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids)

        # logits: (1, seq_len, vocab_size)
        # Shift: predict token[i+1] from logits[i]
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        ce = compute_per_token_cross_entropy(logits, targets)  # (1, seq_len-1)

        all_ce.append(ce[0].float().cpu().numpy())

    # For overlapping chunks, keep only non-overlapping portion of each
    # Since chunk_tokens uses overlap=0 by default, just concatenate
    result = np.concatenate(all_ce).astype(np.float16)
    return result


def generate_labels_for_corpus(
    token_shards_dir: str,
    model_name: str,
    output_dir: str,
    max_seq_len: int = 8192,
    device: str = "cuda",
) -> None:
    """Generate CE labels for all token shards.

    Args:
        token_shards_dir: Directory containing token shard Parquet files.
        model_name: HuggingFace model name (e.g. Qwen/Qwen3-0.6B).
        output_dir: Output directory for CE label shards.
        max_seq_len: Max sequence length per forward pass.
        device: Device string (cuda or cpu).
    """
    from transformers import AutoModelForCausalLM

    shards_path = Path(token_shards_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)

    log.info("Loading model %s on %s", model_name, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()

    shard_files = sorted(shards_path.glob("shard_*.parquet"))
    log.info("Found %d token shards", len(shard_files))

    for shard_file in tqdm(shard_files, desc="Shards", unit="shard"):
        table = pq.read_table(shard_file)
        repo_paths = table.column("repo_path").to_pylist()
        file_paths = table.column("file_path").to_pylist()
        languages = table.column("language").to_pylist()
        token_ids_col = table.column("token_ids").to_pylist()

        out_rows: list[dict] = []

        file_pbar = tqdm(
            enumerate(token_ids_col),
            total=len(token_ids_col),
            desc=f"  {shard_file.name}",
            unit="file",
            leave=False,
        )
        for j, token_ids in file_pbar:
            token_ids = list(token_ids)
            if len(token_ids) < 2:
                continue

            chunks = chunk_tokens(token_ids, max_seq_len)

            for chunk_idx, chunk in enumerate(chunks):
                ce_values = generate_ce_for_sequence(
                    chunk, model, max_seq_len, torch_device
                )
                out_rows.append({
                    "repo_path": repo_paths[j],
                    "file_path": file_paths[j],
                    "language": languages[j],
                    "chunk_idx": chunk_idx,
                    "token_ids": np.array(chunk, dtype=np.int32),
                    "ce_values": ce_values,
                })

        if out_rows:
            out_table = pa.table({
                "repo_path": pa.array([r["repo_path"] for r in out_rows], type=pa.string()),
                "file_path": pa.array([r["file_path"] for r in out_rows], type=pa.string()),
                "language": pa.array([r["language"] for r in out_rows], type=pa.string()),
                "chunk_idx": pa.array(
                    [r["chunk_idx"] for r in out_rows], type=pa.int32()
                ),
                "token_ids": pa.array(
                    [r["token_ids"] for r in out_rows], type=pa.list_(pa.int32())
                ),
                "ce_values": pa.array(
                    [r["ce_values"] for r in out_rows], type=pa.list_(pa.float16())
                ),
            })
            out_shard = out_path / shard_file.name
            pq.write_table(out_table, out_shard, compression="zstd")
            log.info(
                "Wrote %s (%d chunks from %d files)",
                out_shard.name, len(out_rows), len(token_ids_col),
            )
