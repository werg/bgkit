#!/usr/bin/env python
"""Build a small token mmap from deterministic synthetic source text."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SNIPPETS: tuple[tuple[str, str, str], ...] = (
    (
        "python",
        "src/engine/planner.py",
        """
def plan_training_step(batch, model, optimizer, scaler):
    metrics = {}
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward_compressed(batch)
    loss = outputs.loss + 0.03 * outputs.ratio_penalty
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    metrics["loss"] = float(loss.detach())
    metrics["tokens"] = int(batch["content_token_ids"].numel())
    return metrics
""",
    ),
    (
        "typescript",
        "ui/components/InspectorPanel.tsx",
        """
export function InspectorPanel({ selection, metrics, onPin }: Props) {
  const rows = selection.items.map((item) => ({
    id: item.id,
    label: item.path.split("/").slice(-2).join("/"),
    score: metrics[item.id]?.utility ?? 0,
  }));
  return (
    <section aria-label="Inspector">
      {rows.map((row) => (
        <button key={row.id} onClick={() => onPin(row.id)}>
          <span>{row.label}</span>
          <strong>{row.score.toFixed(3)}</strong>
        </button>
      ))}
    </section>
  );
}
""",
    ),
    (
        "markdown",
        "docs/kernel_profile_notes.md",
        """
# Kernel Profile Notes

The benchmark should keep data movement explicit. We report decoder time,
state-space mixer time, convolution time, cross entropy time, and optimizer
time separately. A useful synthetic profile repeats enough realistic source
tokens to exercise survivor selection, projection, Falcon reconstruction, and
the loss path without depending on a full corpus conversion.
""",
    ),
    (
        "rust",
        "crates/store/src/cache.rs",
        """
pub fn insert_with_deadline<K, V>(map: &mut Cache<K, V>, key: K, value: V, ttl_ms: u64)
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    let expires_at = Instant::now() + Duration::from_millis(ttl_ms);
    map.entries.insert(key.clone(), CacheEntry { key, value, expires_at });
    while map.entries.len() > map.capacity {
        if let Some(oldest) = map.oldest_key() {
            map.entries.remove(&oldest);
        } else {
            break;
        }
    }
}
""",
    ),
    (
        "text",
        "notes/debugging_session.txt",
        """
During the profiling pass we need stable batches, stable sequence lengths, and
clear kernel accounting. The synthetic documents intentionally mix prose and
source-code syntax so tokenization contains identifiers, punctuation, comments,
and natural language spans. The goal is timing fidelity, not benchmark quality
claims about the corpus.
""",
    ),
)


def _sha256_bytes(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _make_text(seed: int, target_tokens: int, tokenizer) -> tuple[str, list[int], str, str]:
    language, path, snippet = SNIPPETS[seed % len(SNIPPETS)]
    header = (
        f"// synthetic_id={seed} language={language} path={path}\n"
        f"// This file is generated for Falcon-H1 training-profile smoke tests.\n"
    )
    body_parts: list[str] = []
    ids: list[int] = []
    repeat = 0
    while len(ids) < target_tokens:
        body_parts.append(header if repeat == 0 else f"\n// continuation block {repeat}\n")
        body_parts.append(snippet.strip())
        body_parts.append(
            "\nThe surrounding training objective should reconstruct this content "
            "from compressed survivors while preserving local ordering and names.\n"
        )
        text = "\n".join(body_parts)
        ids = tokenizer.encode(text, add_special_tokens=False)
        repeat += 1
    max_tokens = target_tokens + 48
    ids = ids[:max_tokens]
    text = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text, ids, language, path


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        revision=args.revision,
    )

    all_ids: list[int] = []
    offsets = [0]
    rows: list[dict[str, str]] = []
    source_hash_parts: list[bytes] = []
    for i in range(int(args.samples)):
        target_tokens = int(args.min_tokens) + (
            i * 37 % max(1, int(args.max_tokens) - int(args.min_tokens) + 1)
        )
        text, ids, language, path = _make_text(i, target_tokens, tokenizer)
        all_ids.extend(ids)
        offsets.append(len(all_ids))
        commit_sha = hashlib.sha1(f"synthetic-falcon-{i}".encode()).hexdigest()
        rows.append(
            {
                "file_path": path,
                "language": language,
                "repo_path": "synthetic/falcon_profile",
                "commit_sha": commit_sha,
            }
        )
        source_hash_parts.append(text.encode("utf-8"))

    tokens = np.asarray(all_ids, dtype=np.int32)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    np.save(output_dir / "tokens.npy", tokens)
    np.save(output_dir / "offsets.npy", offsets_arr)
    metadata = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(metadata, output_dir / "metadata.parquet")

    manifest = {
        "schema_version": 1,
        "row_count": len(rows),
        "total_tokens": int(tokens.shape[0]),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.revision,
        "synthetic": True,
        "source_sha256": _sha256_bytes(source_hash_parts),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    event = {"event": "synthetic_mmap_written", "output_dir": str(output_dir), **manifest}
    print(json.dumps(event, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
