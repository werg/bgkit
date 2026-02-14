"""Generate per-token cross-entropy labels for the full tokenized corpus.

Reads token shards from corpus processing, runs Qwen3-0.6B forward passes,
and writes matching Parquet shards with CE values as ICE training targets.

Uses batched inference: sequences are sorted by length and packed into
padded batches to maximize GPU utilization.
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

# Pad token ID used for batching — must not collide with real vocab.
# We use 0 (typically a special token), and mask it out via ignore_index.
PAD_TOKEN_ID = 0


def generate_ce_for_sequence(
    token_ids: list[int],
    model: torch.nn.Module,
    max_seq_len: int = 8192,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute per-token cross-entropy for a single token sequence.

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

        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        ce = compute_per_token_cross_entropy(logits, targets)

        all_ce.append(ce[0].float().cpu().numpy())

    result = np.concatenate(all_ce).astype(np.float16)
    return result


def _batched_ce_inference(
    sequences: list[list[int]],
    model: torch.nn.Module,
    device: torch.device,
    max_batch_tokens: int = 65536,
) -> list[np.ndarray]:
    """Run batched CE inference over multiple sequences.

    Groups sequences into batches where total padded tokens stay under
    max_batch_tokens, pads them, runs a single forward pass per batch,
    and extracts per-sequence CE values.

    Args:
        sequences: List of token ID lists (already chunked to max_seq_len).
        model: Causal LM model.
        device: Torch device.
        max_batch_tokens: Max total tokens (batch_size * max_len) per batch.

    Returns:
        List of float16 numpy arrays, one per input sequence.
    """
    if not sequences:
        return []

    # Sort by length for efficient packing, track original order
    indexed = sorted(enumerate(sequences), key=lambda x: len(x[1]))
    results: list[tuple[int, np.ndarray]] = []

    # Build batches greedily
    batch_indices: list[int] = []
    batch_seqs: list[list[int]] = []

    def _flush_batch():
        if not batch_seqs:
            return

        max_len = max(len(s) for s in batch_seqs)

        # Pad input_ids
        padded = [s + [PAD_TOKEN_ID] * (max_len - len(s)) for s in batch_seqs]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.tensor(
            [[1] * len(s) + [0] * (max_len - len(s)) for s in batch_seqs],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)

        # Shift for causal LM: predict token[i+1] from logits[i]
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:].clone()

        # Mask padding positions in targets so CE ignores them
        target_mask = attention_mask[:, 1:]
        targets[target_mask == 0] = -100

        ce = compute_per_token_cross_entropy(logits, targets)  # (batch, seq_len-1)

        for i, (orig_idx, seq) in enumerate(zip(batch_indices, batch_seqs, strict=True)):
            real_len = len(seq) - 1  # CE has one fewer value than tokens
            ce_values = ce[i, :real_len].float().cpu().numpy().astype(np.float16)
            results.append((orig_idx, ce_values))

    for orig_idx, seq in indexed:
        seq_len = len(seq)
        if seq_len < 2:
            results.append((orig_idx, np.array([], dtype=np.float16)))
            continue

        # Check if adding this seq would exceed budget
        new_max_len = max(max(len(s) for s in batch_seqs), seq_len) if batch_seqs else seq_len
        new_batch_size = len(batch_seqs) + 1
        if batch_seqs and new_max_len * new_batch_size > max_batch_tokens:
            _flush_batch()
            batch_indices = []
            batch_seqs = []

        batch_indices.append(orig_idx)
        batch_seqs.append(seq)

    _flush_batch()

    # Restore original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def generate_labels_for_corpus(
    token_shards_dir: str,
    model_name: str,
    output_dir: str,
    max_seq_len: int = 8192,
    device: str = "cuda",
    max_batch_tokens: int = 65536,
    files_per_slice: int = 1000,
) -> None:
    """Generate CE labels for all token shards using batched inference.

    Args:
        token_shards_dir: Directory containing token shard Parquet files.
        model_name: HuggingFace model name (e.g. Qwen/Qwen3-0.6B).
        output_dir: Output directory for CE label shards.
        max_seq_len: Max sequence length per forward pass.
        device: Device string (cuda or cpu).
        max_batch_tokens: Max total tokens per batch for GPU utilization.
        files_per_slice: Number of files to process per sub-batch (controls
            memory usage and progress granularity).
    """
    import sys

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
    log.info("Found %d token shards (max_batch_tokens=%d)", len(shard_files), max_batch_tokens)
    sys.stderr.flush()

    for shard_file in tqdm(shard_files, desc="Shards", unit="shard"):
        out_shard = out_path / shard_file.name
        if out_shard.exists():
            log.info("Skipping %s (already exists)", out_shard.name)
            continue

        table = pq.read_table(shard_file)
        n_rows = table.num_rows
        repo_paths_col = table.column("repo_path")
        file_paths_col = table.column("file_path")
        languages_col = table.column("language")
        token_ids_col = table.column("token_ids")

        # Accumulate output columns across slices
        out_repo_paths: list[str] = []
        out_file_paths: list[str] = []
        out_languages: list[str] = []
        out_chunk_indices: list[int] = []
        out_token_ids: list[np.ndarray] = []
        out_ce_values: list[np.ndarray] = []

        n_slices = (n_rows + files_per_slice - 1) // files_per_slice
        for start in tqdm(
            range(0, n_rows, files_per_slice),
            desc=shard_file.stem,
            unit="slice",
            total=n_slices,
            leave=False,
        ):
            end = min(start + files_per_slice, n_rows)

            # Convert only this slice to Python lists (avoids huge to_pylist)
            slice_token_ids = token_ids_col[start:end].to_pylist()

            # Chunk and collect sequences for batched inference
            chunks: list[list[int]] = []
            chunk_meta: list[tuple[int, int]] = []  # (absolute_file_idx, chunk_idx)

            for j_rel, toks in enumerate(slice_token_ids):
                if len(toks) < 2:
                    continue
                file_chunks = chunk_tokens(list(toks), max_seq_len)
                for ci, chunk in enumerate(file_chunks):
                    chunks.append(chunk)
                    chunk_meta.append((start + j_rel, ci))

            if not chunks:
                continue

            # Batched inference for this slice
            ce_results = _batched_ce_inference(
                chunks, model, torch_device, max_batch_tokens
            )

            # Collect results
            for chunk, (file_idx, chunk_idx), ce_vals in zip(
                chunks, chunk_meta, ce_results, strict=True
            ):
                out_repo_paths.append(repo_paths_col[file_idx].as_py())
                out_file_paths.append(file_paths_col[file_idx].as_py())
                out_languages.append(languages_col[file_idx].as_py())
                out_chunk_indices.append(chunk_idx)
                out_token_ids.append(np.array(chunk, dtype=np.int32))
                out_ce_values.append(ce_vals)

        if not out_repo_paths:
            continue

        out_table = pa.table({
            "repo_path": pa.array(out_repo_paths, type=pa.string()),
            "file_path": pa.array(out_file_paths, type=pa.string()),
            "language": pa.array(out_languages, type=pa.string()),
            "chunk_idx": pa.array(out_chunk_indices, type=pa.int32()),
            "token_ids": pa.array(out_token_ids, type=pa.list_(pa.int32())),
            "ce_values": pa.array(out_ce_values, type=pa.list_(pa.float16())),
        })
        pq.write_table(out_table, out_shard, compression="zstd")
        log.info("Wrote %s (%d chunks)", out_shard.name, len(out_repo_paths))
