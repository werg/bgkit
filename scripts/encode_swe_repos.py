#!/usr/bin/env python3
"""Pre-compute BgKIT embeddings for SWE-bench repos at each trajectory's base_commit.

Uses incremental blob-SHA keyed caching:
1. Group trajectories by repo (1,800 unique repos, avg 18 trajectories each)
2. Cache L0 survivors by git blob SHA (content-addressed dedup)
3. For each (repo, base_commit): list files, encode only new blobs
4. Assemble per-trajectory context from blob cache

Estimated compute: ~24h total (vs ~200h naive).

Usage:
    python scripts/encode_swe_repos.py \
        --checkpoint checkpoints/phase2_best \
        --trajectories data/trajectories/openhands_filtered.jsonl \
        --repos-dir data/swe_repos \
        --output-dir data/swe_embeddings \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bgkit.data.repo_processing import looks_minified
from bgkit.utils.packing import position_ids_from_cu


def _make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Build a cumulative sequence lengths tensor from a list of lengths."""
    t = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0, out=t[1:])
    return t


def _load_encoder(checkpoint_path: str, device: torch.device):
    """Load BgKIT encoder from checkpoint. Survivorship head is inside the encoder."""
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.training.checkpointing import load_checkpoint

    _metadata, state_dicts = load_checkpoint(Path(checkpoint_path))
    model_state = state_dicts.get("model", {})

    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in model_state.items() if k.startswith("encoder.")
    }
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        "Qwen/Qwen3.5-0.8B-Base", encoder_state, hidden_dim=1024,
    )
    encoder.to(device).eval()
    encoder.requires_grad_(False)

    return encoder


def _get_blob_sha(content: bytes) -> str:
    """Compute git blob SHA for content (same as git's content-addressing)."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _list_files_at_commit(repo_path: Path, commit_sha: str) -> dict[str, str]:
    """List files and their blob SHAs at a specific commit.

    Returns dict mapping file_path -> blob_sha.
    """
    try:
        import pygit2

        repo = pygit2.Repository(str(repo_path))
        commit = repo.get(commit_sha)
        if commit is None:
            return {}

        tree = commit.tree
        files = {}

        def _walk_tree(tree, prefix=""):
            for entry in tree:
                full_path = f"{prefix}{entry.name}" if not prefix else f"{prefix}/{entry.name}"
                if entry.type_str == "blob":
                    files[full_path] = str(entry.id)
                elif entry.type_str == "tree":
                    subtree = repo.get(entry.id)
                    if subtree:
                        _walk_tree(subtree, full_path)

        _walk_tree(tree)
        return files
    except Exception:
        return {}


def _read_blob(repo_path: Path, blob_sha: str) -> bytes | None:
    """Read blob content by SHA."""
    try:
        import pygit2

        repo = pygit2.Repository(str(repo_path))
        blob = repo.get(blob_sha)
        if blob and blob.type_str == "blob":
            return blob.data
    except Exception:
        pass
    return None


def encode_swe_repos(
    checkpoint_path: str,
    trajectories_path: str,
    repos_dir: str,
    output_dir: str,
    batch_size: int = 4,
    retention_ratio: float = 0.05,
    max_file_tokens: int = 4096,
) -> None:
    """Pre-compute BgKIT embeddings with blob-SHA dedup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    encoder = _load_encoder(checkpoint_path, device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", trust_remote_code=True)

    # Load trajectories and group by repo
    print(f"Loading trajectories from {trajectories_path}")
    repo_commits: dict[str, set[str]] = defaultdict(set)
    with open(trajectories_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            repo = record.get("repo", "")
            base_commit = record.get("base_commit", "")
            if repo and base_commit:
                repo_commits[repo].add(base_commit)

    print(f"Found {len(repo_commits)} unique repos, "
          f"{sum(len(c) for c in repo_commits.values())} (repo, commit) pairs")

    # Blob SHA cache: blob_sha -> survivor embeddings (numpy)
    blob_cache: dict[str, np.ndarray] = {}
    cache_path = output_path / "blob_cache"
    cache_path.mkdir(exist_ok=True)

    # Per-trajectory manifest
    manifest_rows = []
    repos_dir_path = Path(repos_dir)
    start_time = time.time()
    total_blobs_encoded = 0
    total_blobs_cached = 0

    for repo_idx, (repo, commits) in enumerate(sorted(repo_commits.items())):
        repo_path = repos_dir_path / repo.replace("/", "_")
        if not repo_path.exists():
            # Try with slash
            repo_path = repos_dir_path / repo
        if not repo_path.exists():
            continue

        for commit_sha in sorted(commits):
            files = _list_files_at_commit(repo_path, commit_sha)
            if not files:
                continue

            # Identify new blobs to encode
            new_blobs: dict[str, str] = {}  # blob_sha -> file_path (for logging)
            for file_path, blob_sha in files.items():
                if blob_sha not in blob_cache:
                    new_blobs[blob_sha] = file_path

            # Batch-encode new blobs
            if new_blobs:
                blob_shas = list(new_blobs.keys())
                for batch_start in range(0, len(blob_shas), batch_size):
                    batch_shas = blob_shas[batch_start : batch_start + batch_size]
                    batch_texts = []
                    valid_shas = []

                    for sha in batch_shas:
                        content = _read_blob(repo_path, sha)
                        if content is None:
                            continue
                        try:
                            text = content.decode("utf-8", errors="replace")
                        except Exception:
                            continue
                        # Reject minified / bundled files from the Phase 3
                        # context cache. Same heuristic as Phase 1 data-prep.
                        if looks_minified(text):
                            continue
                        tokens = tokenizer.encode(text, add_special_tokens=False)[:max_file_tokens]
                        if tokens:
                            batch_texts.append(tokens)
                            valid_shas.append(sha)

                    if not batch_texts:
                        continue

                    # Encode each blob independently using packed (varlen) format.
                    embed_tokens = encoder.l0.backbone.get_input_embeddings()

                    with torch.no_grad():
                        for sha, tokens in zip(valid_shas, batch_texts, strict=True):
                            length = len(tokens)
                            token_tensor = torch.tensor(
                                tokens, dtype=torch.long, device=device,
                            )
                            content_emb = embed_tokens(token_tensor)  # (L, D)

                            # Build single-sample packed metadata (B=1).
                            cu = _make_cu_seqlens([length]).to(device)
                            pos = position_ids_from_cu(cu, length).to(device)

                            output = encoder(
                                content_embeddings=content_emb,
                                content_cu_seqlens=cu,
                                content_position_ids=pos,
                                target_ratio_l0=retention_ratio,
                            )
                            # survivor_embeddings is flat (K, D); no mask needed.
                            survivors = (
                                output.survivor_embeddings.cpu().to(torch.float16).numpy()
                            )
                            blob_cache[sha] = survivors
                            total_blobs_encoded += 1

                total_blobs_cached += len(files) - len(new_blobs)

            # Assemble per-trajectory context from blob cache
            file_survivors = []
            for file_path in sorted(files):
                blob_sha = files[file_path]
                if blob_sha in blob_cache:
                    file_survivors.append(blob_cache[blob_sha])

            if file_survivors:
                combined = np.concatenate(file_survivors, axis=0)
                # Save per-(repo, commit)
                safe_repo = repo.replace("/", "_")
                ctx_dir = output_path / safe_repo / commit_sha[:8]
                ctx_dir.mkdir(parents=True, exist_ok=True)
                np.save(ctx_dir / "survivors.npy", combined)

                manifest_rows.append({
                    "repo": repo,
                    "base_commit": commit_sha,
                    "num_files": len(files),
                    "num_survivors": len(combined),
                    "path": str(ctx_dir / "survivors.npy"),
                })

        if (repo_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"  {repo_idx + 1}/{len(repo_commits)} repos, "
                f"{total_blobs_encoded} encoded, {total_blobs_cached} cached, "
                f"{elapsed:.0f}s elapsed",
            )

    # Write manifest
    if manifest_rows:
        manifest_table = pa.table({
            "repo": pa.array([r["repo"] for r in manifest_rows], type=pa.string()),
            "base_commit": pa.array([r["base_commit"] for r in manifest_rows], type=pa.string()),
            "num_files": pa.array([r["num_files"] for r in manifest_rows], type=pa.int64()),
            "num_survivors": pa.array([r["num_survivors"] for r in manifest_rows], type=pa.int64()),
            "path": pa.array([r["path"] for r in manifest_rows], type=pa.string()),
        })
        pq.write_table(manifest_table, output_path / "manifest.parquet")

    elapsed = time.time() - start_time
    print(
        f"\nDone: {len(manifest_rows)} (repo, commit) pairs in {elapsed / 3600:.1f}h, "
        f"{total_blobs_encoded} blobs encoded, {total_blobs_cached} cache hits",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--repos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retention-ratio", type=float, default=0.05)
    parser.add_argument("--max-file-tokens", type=int, default=4096)
    args = parser.parse_args()

    encode_swe_repos(
        checkpoint_path=args.checkpoint,
        trajectories_path=args.trajectories,
        repos_dir=args.repos_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        retention_ratio=args.retention_ratio,
        max_file_tokens=args.max_file_tokens,
    )


if __name__ == "__main__":
    main()
