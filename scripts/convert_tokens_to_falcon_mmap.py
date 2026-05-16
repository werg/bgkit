#!/usr/bin/env python
"""Build Falcon-H1 companion mmap artifacts for an existing Qwen token mmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

ALIGNMENT_QUALITY_REAL_DP = 0
ALIGNMENT_QUALITY_ARTIFICIAL_FAST = 1
ALIGNMENT_QUALITY_ARTIFICIAL_FALLBACK = 2
ALIGNMENT_QUALITY_PATHOLOGICAL = 3
ALIGNMENT_QUALITY_SKIPPED = 4


@dataclass
class DensePairAlignment:
    forced_survivor_indices: np.ndarray
    target_falcon_pair_ids: np.ndarray
    target_pair_loss_mask: np.ndarray
    alignment_scores: np.ndarray
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class PathologyConfig:
    enabled: bool = True
    max_decoded_chars: int = 256 * 1024
    max_line_chars: int = 32 * 1024
    min_printable_ratio: float = 0.80
    max_replacement_ratio: float = 0.01
    min_base64ish_chars: int = 8192
    max_base64ish_ratio: float = 0.98
    max_falcon_tokens: int = 32 * 1024
    max_falcon_expansion: float = 4.0


class RawArrayWriter:
    """Append arrays without keeping the full output in memory.

    The final artifact is a normal ``.npy`` file; appends go to a temporary raw
    binary file and are copied into a correctly-headered numpy file at finalize.
    """

    def __init__(
        self,
        path: Path,
        dtype: np.dtype | str,
        *,
        tail_shape: tuple[int, ...] = (),
    ):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.tail_shape = tuple(tail_shape)
        self.rows = 0
        self._raw_path = path.with_suffix(path.suffix + ".raw.tmp")
        self._npy_tmp_path = path.with_suffix(path.suffix + ".tmp")
        self._fh = self._raw_path.open("wb")

    def append(self, array: np.ndarray | list[int]) -> None:
        arr = np.asarray(array, dtype=self.dtype)
        if self.tail_shape:
            arr = arr.reshape((-1, *self.tail_shape))
            self.rows += int(arr.shape[0])
        else:
            arr = arr.reshape(-1)
            self.rows += int(arr.shape[0])
        arr.tofile(self._fh)

    def finalize(self, *, copy_chunk_bytes: int = 64 * 1024 * 1024) -> None:
        self._fh.close()
        shape = (self.rows, *self.tail_shape)
        header = {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": shape,
        }
        with self._npy_tmp_path.open("wb") as out_fh:
            np.lib.format.write_array_header_2_0(out_fh, header)
            if self.rows:
                with self._raw_path.open("rb") as raw_fh:
                    while True:
                        chunk = raw_fh.read(copy_chunk_bytes)
                        if not chunk:
                            break
                        out_fh.write(chunk)
        os.replace(self._npy_tmp_path, self.path)
        self._raw_path.unlink()


def _span_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(int(a[1]), int(b[1])) - max(int(a[0]), int(b[0])))


def _pair_targets(
    falcon_ids: list[int],
    falcon_offsets: list[tuple[int, int]],
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    n_pairs = (len(falcon_ids) + 1) // 2
    pair_ids = np.full((n_pairs, 2), int(pad_id), dtype=np.int64)
    loss_mask = np.zeros((n_pairs, 2), dtype=np.bool_)
    pair_spans: list[tuple[int, int]] = []
    for k in range(n_pairs):
        j0 = 2 * k
        j1 = j0 + 1
        pair_ids[k, 0] = int(falcon_ids[j0])
        loss_mask[k, 0] = True
        if j1 < len(falcon_ids):
            pair_ids[k, 1] = int(falcon_ids[j1])
            loss_mask[k, 1] = True
            pair_spans.append((
                min(int(falcon_offsets[j0][0]), int(falcon_offsets[j1][0])),
                max(int(falcon_offsets[j0][1]), int(falcon_offsets[j1][1])),
            ))
        else:
            pair_spans.append((int(falcon_offsets[j0][0]), int(falcon_offsets[j0][1])))
    return pair_ids, loss_mask, pair_spans


def monotone_dense_pair_alignment(
    qwen_offsets: list[tuple[int, int]],
    falcon_ids: list[int],
    falcon_offsets: list[tuple[int, int]],
    *,
    falcon_pad_id: int = 0,
) -> DensePairAlignment:
    """Select Qwen survivor positions for dense Falcon token pairs.

    The dynamic program chooses exactly ``ceil(len(falcon_ids) / 2)``
    monotonically increasing Qwen positions.  Scores are character overlap
    between a Qwen token span and a two-token Falcon pair span, with small
    deterministic tie-breakers for center distance and index drift.
    """

    n_qwen = len(qwen_offsets)
    pair_ids, loss_mask, pair_spans = _pair_targets(
        falcon_ids,
        falcon_offsets,
        falcon_pad_id,
    )
    n_pairs = len(pair_spans)
    if n_pairs == 0:
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=pair_ids,
            target_pair_loss_mask=loss_mask,
            alignment_scores=np.zeros(0, dtype=np.float32),
        )
    if n_pairs > n_qwen:
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=np.zeros((0, 2), dtype=np.int64),
            target_pair_loss_mask=np.zeros((0, 2), dtype=np.bool_),
            alignment_scores=np.zeros(0, dtype=np.float32),
            skipped=True,
            skip_reason="ceil_falcon_pairs_gt_qwen_tokens",
        )

    scores = np.full((n_pairs, n_qwen), -np.inf, dtype=np.float64)
    raw_overlap = np.zeros((n_pairs, n_qwen), dtype=np.float32)
    for k, p_span in enumerate(pair_spans):
        p_center = 0.5 * (p_span[0] + p_span[1])
        linear_i = k * n_qwen / max(n_pairs, 1)
        for i, q_span in enumerate(qwen_offsets):
            overlap = _span_overlap(q_span, p_span)
            raw_overlap[k, i] = float(overlap)
            q_center = 0.5 * (q_span[0] + q_span[1])
            center_penalty = 1e-3 * abs(q_center - p_center)
            drift_penalty = 1e-6 * abs(i - linear_i)
            scores[k, i] = float(overlap) - center_penalty - drift_penalty

    prev = np.full(n_qwen, -np.inf, dtype=np.float64)
    backptr = np.full((n_pairs, n_qwen), -1, dtype=np.int64)
    for k in range(n_pairs):
        cur = np.full(n_qwen, -np.inf, dtype=np.float64)
        prefix_best_val = -np.inf
        prefix_best_idx = -1
        min_i = k
        max_i = n_qwen - (n_pairs - k)
        for i in range(min_i, max_i + 1):
            if k == 0:
                prev_val = 0.0
                prev_idx = -1
            else:
                h = i - 1
                if prev[h] > prefix_best_val:
                    prefix_best_val = float(prev[h])
                    prefix_best_idx = h
                prev_val = prefix_best_val
                prev_idx = prefix_best_idx
            if np.isfinite(prev_val):
                cur[i] = prev_val + scores[k, i]
                backptr[k, i] = prev_idx
        prev = cur

    best_i = int(np.nanargmax(prev))
    if not np.isfinite(prev[best_i]):
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=np.zeros((0, 2), dtype=np.int64),
            target_pair_loss_mask=np.zeros((0, 2), dtype=np.bool_),
            alignment_scores=np.zeros(0, dtype=np.float32),
            skipped=True,
            skip_reason="no_finite_dp_path",
        )

    indices = np.zeros(n_pairs, dtype=np.int64)
    i = best_i
    for k in range(n_pairs - 1, -1, -1):
        indices[k] = i
        i = int(backptr[k, i])
    selected_scores = raw_overlap[np.arange(n_pairs), indices].astype(np.float32)
    return DensePairAlignment(
        forced_survivor_indices=indices,
        target_falcon_pair_ids=pair_ids,
        target_pair_loss_mask=loss_mask,
        alignment_scores=selected_scores,
    )


def fast_monotone_dense_pair_alignment(
    qwen_offsets: list[tuple[int, int]] | np.ndarray,
    falcon_ids: list[int],
    falcon_offsets: list[tuple[int, int]],
    *,
    falcon_pad_id: int = 0,
) -> DensePairAlignment:
    """Vectorized monotone alignment suitable for full-corpus conversion.

    The exact DP is useful for small fixtures and diagnostics but is too slow
    for millions of chunks.  This path maps Falcon token pairs to nearby Qwen
    positions by span center and then enforces a strictly increasing survivor
    sequence.
    """

    n_qwen = len(qwen_offsets)
    f_ids = np.asarray(falcon_ids, dtype=np.int64)
    n_falcon = int(f_ids.shape[0])
    n_pairs = (n_falcon + 1) // 2

    pair_ids = np.full((n_pairs, 2), int(falcon_pad_id), dtype=np.int64)
    loss_mask = np.zeros((n_pairs, 2), dtype=np.bool_)
    if n_pairs:
        pair_ids[:, 0] = f_ids[0::2]
        loss_mask[:, 0] = True
        second_ids = f_ids[1::2]
        if second_ids.size:
            pair_ids[: second_ids.size, 1] = second_ids
            loss_mask[: second_ids.size, 1] = True

    if n_pairs == 0:
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=pair_ids,
            target_pair_loss_mask=loss_mask,
            alignment_scores=np.zeros(0, dtype=np.float32),
        )
    if n_pairs > n_qwen:
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=np.zeros((0, 2), dtype=np.int64),
            target_pair_loss_mask=np.zeros((0, 2), dtype=np.bool_),
            alignment_scores=np.zeros(0, dtype=np.float32),
            skipped=True,
            skip_reason="ceil_falcon_pairs_gt_qwen_tokens",
        )

    q = np.asarray(qwen_offsets, dtype=np.int64)
    f = np.asarray(falcon_offsets, dtype=np.int64)

    pair_start = f[0::2, 0].copy()
    pair_end = f[0::2, 1].copy()
    if n_falcon > 1:
        second = f[1::2]
        n_second = second.shape[0]
        pair_start[:n_second] = np.minimum(pair_start[:n_second], second[:, 0])
        pair_end[:n_second] = np.maximum(pair_end[:n_second], second[:, 1])

    q_centers = 0.5 * (q[:, 0] + q[:, 1])
    pair_centers = 0.5 * (pair_start + pair_end)
    raw = np.searchsorted(q_centers, pair_centers, side="left").astype(np.int64)
    raw = np.clip(raw, 0, n_qwen - 1)

    # Enforce strict monotonicity while preserving the center-based choices as
    # much as possible.  If pathological offsets still overflow, fall back to an
    # evenly-spaced monotone map.
    lower = np.arange(n_pairs, dtype=np.int64)
    indices = np.maximum.accumulate(np.maximum(raw, lower))
    overflow = int(indices[-1] - (n_qwen - 1))
    if overflow > 0:
        indices = np.maximum.accumulate(np.maximum(indices - overflow, lower))
    if indices[-1] >= n_qwen or np.any(np.diff(indices) <= 0):
        indices = np.floor(
            (np.arange(n_pairs, dtype=np.float64) + 0.5) * n_qwen / n_pairs,
        ).astype(np.int64)
        indices = np.clip(indices, 0, n_qwen - 1)

    selected_q = q[indices]
    overlaps = np.maximum(
        0,
        np.minimum(selected_q[:, 1], pair_end) - np.maximum(selected_q[:, 0], pair_start),
    ).astype(np.float32)

    return DensePairAlignment(
        forced_survivor_indices=indices.astype(np.int64),
        target_falcon_pair_ids=pair_ids,
        target_pair_loss_mask=loss_mask,
        alignment_scores=overlaps,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_index(offsets: np.ndarray, max_seq_len: int) -> list[tuple[int, int, int, int]]:
    entries: list[tuple[int, int, int, int]] = []
    for file_idx, (start, end) in enumerate(pairwise(offsets)):
        file_start = int(start)
        file_len = int(end - start)
        if file_len <= 0:
            continue
        n_chunks = (file_len - 1) // max_seq_len + 1
        for chunk_idx in range(n_chunks):
            rel_start = chunk_idx * max_seq_len
            rel_end = min(file_len, rel_start + max_seq_len)
            entries.append((file_idx, file_start + rel_start, rel_start, rel_end))
    return entries


def _linear_offsets(text_len: int, n_tokens: int) -> np.ndarray:
    if n_tokens <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    boundaries = np.linspace(0, max(int(text_len), 0), n_tokens + 1)
    boundaries = np.rint(boundaries).astype(np.int64)
    return np.stack([boundaries[:-1], boundaries[1:]], axis=1)


_BASE64ISH_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-"
)


def _text_pathology_reason(text: str, cfg: PathologyConfig) -> str | None:
    if not cfg.enabled:
        return None
    if not text:
        return "empty_decoded_text"
    if len(text) > cfg.max_decoded_chars:
        return "decoded_chars_gt_limit"

    printable = 0
    replacement = 0
    non_ws = 0
    base64ish = 0
    line_len = 0
    max_line_len = 0
    for ch in text:
        if ch == "\n" or ch == "\r":
            max_line_len = max(max_line_len, line_len)
            line_len = 0
        else:
            line_len += 1
        if ch.isspace() or ch.isprintable():
            printable += 1
        if ch == "\ufffd":
            replacement += 1
        if not ch.isspace():
            non_ws += 1
            if ch in _BASE64ISH_CHARS:
                base64ish += 1
    max_line_len = max(max_line_len, line_len)

    if max_line_len > cfg.max_line_chars:
        return "line_chars_gt_limit"
    if printable / len(text) < cfg.min_printable_ratio:
        return "printable_ratio_lt_limit"
    if replacement / len(text) > cfg.max_replacement_ratio:
        return "replacement_ratio_gt_limit"
    if (
        non_ws >= cfg.min_base64ish_chars
        and base64ish / non_ws > cfg.max_base64ish_ratio
    ):
        return "base64ish_ratio_gt_limit"
    return None


def _falcon_pathology_reason(
    *,
    qwen_tokens: int,
    falcon_tokens: int,
    cfg: PathologyConfig,
) -> str | None:
    if not cfg.enabled:
        return None
    if falcon_tokens == 0:
        return "empty_falcon_tokens"
    if cfg.max_falcon_tokens > 0 and falcon_tokens > cfg.max_falcon_tokens:
        return "falcon_tokens_gt_limit"
    if (
        qwen_tokens > 0
        and cfg.max_falcon_expansion > 0
        and falcon_tokens / qwen_tokens > cfg.max_falcon_expansion
    ):
        return "falcon_expansion_gt_limit"
    return None


def _artificial_offsets_for_ids(tokenizer, ids: np.ndarray) -> tuple[str, np.ndarray]:
    text = tokenizer.decode(
        [int(tok) for tok in ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text, _linear_offsets(len(text), len(ids))


def _resolve_source_path(source_root: Path | None, row: dict) -> Path | None:
    if source_root is None:
        return None
    file_path = row.get("file_path")
    if not file_path:
        return None
    candidates = []
    repo_path = row.get("repo_path")
    if repo_path:
        candidates.append(source_root / str(repo_path) / str(file_path))
    candidates.append(source_root / str(file_path))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _chunk_text_and_qwen_offsets(
    *,
    qwen_tokenizer,
    qwen_ids: np.ndarray,
    file_rel_start: int,
    file_rel_end: int,
    source_path: Path | None,
) -> tuple[str, np.ndarray]:
    if source_path is None:
        return _artificial_offsets_for_ids(qwen_tokenizer, qwen_ids)

    try:
        text = source_path.read_text(errors="replace")
    except OSError:
        return _artificial_offsets_for_ids(qwen_tokenizer, qwen_ids)

    encoded = qwen_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    if file_rel_end > len(offsets):
        return _artificial_offsets_for_ids(qwen_tokenizer, qwen_ids)

    chunk_offsets_abs = offsets[file_rel_start:file_rel_end]
    if chunk_offsets_abs.shape[0] == 0:
        return "", np.zeros((0, 2), dtype=np.int64)
    char_start = int(chunk_offsets_abs[:, 0].min())
    char_end = int(chunk_offsets_abs[:, 1].max())
    chunk_text = text[char_start:char_end]
    chunk_offsets = chunk_offsets_abs - char_start
    return chunk_text, chunk_offsets


def containment_pair_alignment(
    qwen_offsets: list[tuple[int, int]] | np.ndarray,
    falcon_ids: list[int],
    falcon_offsets: list[tuple[int, int]],
    *,
    falcon_pad_id: int = 0,
) -> DensePairAlignment:
    """Strict containment-based alignment (no heuristic overlap scoring).

    For each Qwen position ``i`` with char range ``[s_q, e_q]``, find the
    Falcon tokens whose char range falls ENTIRELY inside ``[s_q, e_q]``
    (i.e. ``falcon_start[j] >= s_q`` AND ``falcon_end[j] <= e_q``). Keep
    Qwen positions where EXACTLY 2 Falcon tokens are contained; those
    become forced_survivors with ``pair_ids = (a, b)`` where a, b are
    the two contained Falcon token IDs.

    Why strict count==2: ``projection_block.output_split_factor=2`` means
    each Qwen survivor's projection output is exactly 2 Falcon embeddings.
    Keeping only count==2 Qwen positions ensures every forced_survivor has
    a clean, fully-determined target with no padding or partial-loss
    masking — every projected vector is supervised against a real
    Falcon-embedding-space target.

    Result: fewer forced_survivors than the monotone-DP alignment, but
    every kept position has a target that's unambiguous from the bytes.
    No char-overlap penalty: the heuristic "best-guess" pair assignment
    that capped cos_sim at ~0.6 is replaced with exact byte-level
    containment.
    """
    q = np.asarray(qwen_offsets, dtype=np.int64)
    if q.ndim == 1:
        q = q.reshape(-1, 2)
    n_qwen = int(q.shape[0])
    n_falcon = len(falcon_ids)
    if n_falcon == 0 or n_qwen == 0:
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=np.zeros((0, 2), dtype=np.int64),
            target_pair_loss_mask=np.zeros((0, 2), dtype=np.bool_),
            alignment_scores=np.zeros(0, dtype=np.float32),
        )

    f = np.asarray(falcon_offsets, dtype=np.int64)
    if f.ndim == 1:
        f = f.reshape(-1, 2)
    falcon_starts = f[:, 0]
    falcon_ends = f[:, 1]
    falcon_ids_arr = np.asarray(falcon_ids, dtype=np.int64)
    qwen_starts = q[:, 0]
    qwen_ends = q[:, 1]

    # Vectorized: per Qwen i, find the index range [j_start, j_end] of
    # Falcon tokens with falcon_start >= qwen_start AND falcon_end <= qwen_end.
    # Tokens are in order, so falcon_starts is non-decreasing.
    j_start = np.searchsorted(falcon_starts, qwen_starts, side="left")
    # searchsorted on falcon_ends (also non-decreasing for normal tokenizers)
    j_end_exclusive = np.searchsorted(falcon_ends, qwen_ends, side="right")
    counts = j_end_exclusive - j_start
    keep_mask = counts == 2
    if not np.any(keep_mask):
        return DensePairAlignment(
            forced_survivor_indices=np.zeros(0, dtype=np.int64),
            target_falcon_pair_ids=np.zeros((0, 2), dtype=np.int64),
            target_pair_loss_mask=np.zeros((0, 2), dtype=np.bool_),
            alignment_scores=np.zeros(0, dtype=np.float32),
            skipped=True,
            skip_reason="no_qwen_position_contains_exactly_2_falcon_tokens",
        )

    selected_qwen = np.flatnonzero(keep_mask).astype(np.int64)
    selected_j_start = j_start[keep_mask]
    a_ids = falcon_ids_arr[selected_j_start]
    b_ids = falcon_ids_arr[selected_j_start + 1]
    pair_ids = np.stack([a_ids, b_ids], axis=1)
    # alignment_score = char span covered by the pair (analogous to old
    # raw_overlap so downstream metrics keep meaning).
    a_starts = falcon_starts[selected_j_start]
    b_ends = falcon_ends[selected_j_start + 1]
    scores = (b_ends - a_starts).astype(np.float32)
    return DensePairAlignment(
        forced_survivor_indices=selected_qwen,
        target_falcon_pair_ids=pair_ids,
        target_pair_loss_mask=np.ones_like(pair_ids, dtype=np.bool_),
        alignment_scores=scores,
    )


def _select_alignment(
    mode: str,
    qwen_offsets: np.ndarray,
    falcon_ids: list[int],
    falcon_offsets: list[tuple[int, int]],
    *,
    falcon_pad_id: int,
) -> DensePairAlignment:
    if mode == "dp":
        return monotone_dense_pair_alignment(
            qwen_offsets,
            falcon_ids,
            falcon_offsets,
            falcon_pad_id=falcon_pad_id,
        )
    if mode == "containment":
        return containment_pair_alignment(
            qwen_offsets,
            falcon_ids,
            falcon_offsets,
            falcon_pad_id=falcon_pad_id,
        )
    return fast_monotone_dense_pair_alignment(
        qwen_offsets,
        falcon_ids,
        falcon_offsets,
        falcon_pad_id=falcon_pad_id,
    )


def _write_skip_record(
    fh,
    *,
    kind: str,
    reason: str,
    chunk_idx: int,
    file_idx: int,
    rel_start: int,
    rel_end: int,
    qwen_tokens: int,
    falcon_tokens: int | None,
    decoded_chars: int | None,
    row: dict,
) -> None:
    record = {
        "kind": kind,
        "reason": reason,
        "chunk_idx": int(chunk_idx),
        "file_idx": int(file_idx),
        "rel_start": int(rel_start),
        "rel_end": int(rel_end),
        "qwen_tokens": int(qwen_tokens),
        "falcon_tokens": None if falcon_tokens is None else int(falcon_tokens),
        "decoded_chars": None if decoded_chars is None else int(decoded_chars),
        "repo_path": row.get("repo_path", ""),
        "file_path": row.get("file_path", ""),
        "commit_sha": row.get("commit_sha", ""),
    }
    fh.write(json.dumps(record, sort_keys=True) + "\n")


def convert(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.load(input_dir / "tokens.npy", mmap_mode="r")
    offsets = np.load(input_dir / "offsets.npy")
    metadata = pq.read_table(input_dir / "metadata.parquet").to_pylist()
    chunks = _chunk_index(offsets, int(args.max_seq_len))

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_tokenizer,
        trust_remote_code=True,
        revision=args.qwen_revision,
    )
    falcon_tokenizer = AutoTokenizer.from_pretrained(
        args.falcon_tokenizer,
        trust_remote_code=True,
        revision=args.falcon_revision,
    )
    falcon_pad_id = falcon_tokenizer.pad_token_id
    if falcon_pad_id is None:
        falcon_pad_id = falcon_tokenizer.eos_token_id or 0
    pathology_cfg = PathologyConfig(
        enabled=not args.disable_pathology_filter,
        max_decoded_chars=int(args.max_decoded_chars),
        max_line_chars=int(args.max_line_chars),
        min_printable_ratio=float(args.min_printable_ratio),
        max_replacement_ratio=float(args.max_replacement_ratio),
        min_base64ish_chars=int(args.min_base64ish_chars),
        max_base64ish_ratio=float(args.max_base64ish_ratio),
        max_falcon_tokens=int(args.max_falcon_tokens),
        max_falcon_expansion=float(args.max_falcon_expansion),
    )

    writers = {
        "falcon_tokens": RawArrayWriter(
            output_dir / "falcon_tokens.npy",
            np.int32,
        ),
        "forced_survivor_indices": RawArrayWriter(
            output_dir / "forced_survivor_indices.npy",
            np.int32,
        ),
        "target_falcon_pair_ids": RawArrayWriter(
            output_dir / "target_falcon_pair_ids.npy",
            np.int32,
            tail_shape=(2,),
        ),
        "target_pair_loss_mask": RawArrayWriter(
            output_dir / "target_pair_loss_mask.npy",
            np.bool_,
            tail_shape=(2,),
        ),
        "alignment_scores": RawArrayWriter(
            output_dir / "alignment_scores.npy",
            np.float32,
        ),
    }
    falcon_offsets = [0]
    forced_offsets = [0]
    # Per-chunk alignment quality: one byte per chunk so consumers can filter.
    # 0 = real DP alignment from source-root offsets
    # 1 = artificial offsets in fast mode (no source root) — alignment_scores
    #     are meaningful only as a rough monotone-pairing signal
    # 2 = source-root requested but path/encode fell back to artificial
    # 3 = pathological chunk (skipped — zero rows)
    # 4 = alignment skipped (zero forced rows kept; falcon tokens present)
    alignment_quality: list[int] = []
    skipped = 0
    skipped_pathological = 0
    odd_final_pairs = 0
    pathology_reason_counts: dict[str, int] = {}
    alignment_skip_counts: dict[str, int] = {}

    source_root = Path(args.source_root) if args.source_root else None
    use_source_root = source_root is not None and args.alignment_mode == "dp"
    if source_root is not None and not use_source_root:
        print(
            "NOTE: --source-root is ignored in fast alignment mode to avoid "
            "full-file offset-mapping tokenization.",
        )

    skipped_chunks_path = output_dir / "skipped_chunks.jsonl"

    def append_chunk(
        *,
        chunk_idx: int,
        file_idx: int,
        rel_start: int,
        rel_end: int,
        row: dict,
        qwen_offsets: np.ndarray,
        falcon_ids: list[int],
        falcon_offsets_for_chunk: list[tuple[int, int]] | np.ndarray,
        skipped_fh,
        quality: int,
    ) -> None:
        nonlocal skipped, odd_final_pairs
        f_ids_array = np.asarray(falcon_ids, dtype=np.int32)
        writers["falcon_tokens"].append(f_ids_array)
        falcon_offsets.append(falcon_offsets[-1] + int(f_ids_array.shape[0]))

        if f_ids_array.shape[0] % 2 == 1:
            odd_final_pairs += 1
        alignment = _select_alignment(
            args.alignment_mode,
            qwen_offsets,
            falcon_ids,
            falcon_offsets_for_chunk,
            falcon_pad_id=int(falcon_pad_id),
        )
        if alignment.skipped:
            skipped += 1
            reason = alignment.skip_reason or "alignment_skipped"
            alignment_skip_counts[reason] = alignment_skip_counts.get(reason, 0) + 1
            quality = ALIGNMENT_QUALITY_SKIPPED
            _write_skip_record(
                skipped_fh,
                kind="alignment",
                reason=reason,
                chunk_idx=chunk_idx,
                file_idx=file_idx,
                rel_start=rel_start,
                rel_end=rel_end,
                qwen_tokens=int(rel_end - rel_start),
                falcon_tokens=int(f_ids_array.shape[0]),
                decoded_chars=None,
                row=row,
            )
        writers["forced_survivor_indices"].append(alignment.forced_survivor_indices)
        forced_offsets.append(
            forced_offsets[-1] + int(alignment.forced_survivor_indices.shape[0])
        )
        writers["target_falcon_pair_ids"].append(alignment.target_falcon_pair_ids)
        writers["target_pair_loss_mask"].append(alignment.target_pair_loss_mask)
        writers["alignment_scores"].append(alignment.alignment_scores)
        alignment_quality.append(quality)

    def skip_pathological_chunk(
        *,
        skipped_fh,
        reason: str,
        chunk_idx: int,
        file_idx: int,
        rel_start: int,
        rel_end: int,
        row: dict,
        decoded_chars: int | None = None,
        falcon_tokens: int | None = None,
    ) -> None:
        nonlocal skipped_pathological
        skipped_pathological += 1
        pathology_reason_counts[reason] = pathology_reason_counts.get(reason, 0) + 1
        falcon_offsets.append(falcon_offsets[-1])
        forced_offsets.append(forced_offsets[-1])
        alignment_quality.append(ALIGNMENT_QUALITY_PATHOLOGICAL)
        _write_skip_record(
            skipped_fh,
            kind="pathology",
            reason=reason,
            chunk_idx=chunk_idx,
            file_idx=file_idx,
            rel_start=rel_start,
            rel_end=rel_end,
            qwen_tokens=int(rel_end - rel_start),
            falcon_tokens=falcon_tokens,
            decoded_chars=decoded_chars,
            row=row,
        )

    with skipped_chunks_path.open("w", encoding="utf-8") as skipped_fh:
        cached_source_path: Path | None = None
        cached_text: str | None = None
        cached_qwen_offsets: np.ndarray | None = None
        for chunk_idx, (file_idx, abs_start, rel_start, rel_end) in enumerate(tqdm(
            chunks,
            desc="falcon companions",
        )):
            qwen_ids = tokens[abs_start : abs_start + (rel_end - rel_start)].astype(
                np.int64
            )
            row = metadata[file_idx] if file_idx < len(metadata) else {}
            chunk_quality = ALIGNMENT_QUALITY_REAL_DP
            if not use_source_root:
                chunk_text, qwen_offsets = _artificial_offsets_for_ids(
                    qwen_tokenizer,
                    qwen_ids,
                )
                chunk_quality = ALIGNMENT_QUALITY_ARTIFICIAL_FAST
            else:
                source_path = _resolve_source_path(source_root, row)
                if source_path is None:
                    chunk_text, qwen_offsets = _artificial_offsets_for_ids(
                        qwen_tokenizer,
                        qwen_ids,
                    )
                    chunk_quality = ALIGNMENT_QUALITY_ARTIFICIAL_FALLBACK
                elif source_path != cached_source_path:
                    try:
                        cached_text = source_path.read_text(errors="replace")
                        encoded = qwen_tokenizer(
                            cached_text,
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                        )
                        cached_qwen_offsets = np.asarray(
                            encoded["offset_mapping"],
                            dtype=np.int64,
                        )
                        cached_source_path = source_path
                    except OSError:
                        cached_source_path = None
                        cached_text = None
                        cached_qwen_offsets = None
                if source_path is None:
                    pass
                elif (
                    cached_text is None
                    or cached_qwen_offsets is None
                    or rel_end > cached_qwen_offsets.shape[0]
                ):
                    chunk_text, qwen_offsets = _artificial_offsets_for_ids(
                        qwen_tokenizer,
                        qwen_ids,
                    )
                    chunk_quality = ALIGNMENT_QUALITY_ARTIFICIAL_FALLBACK
                else:
                    chunk_offsets_abs = cached_qwen_offsets[rel_start:rel_end]
                    if chunk_offsets_abs.shape[0] == 0:
                        chunk_text = ""
                        qwen_offsets = np.zeros((0, 2), dtype=np.int64)
                    else:
                        char_start = int(chunk_offsets_abs[:, 0].min())
                        char_end = int(chunk_offsets_abs[:, 1].max())
                        chunk_text = cached_text[char_start:char_end]
                        qwen_offsets = chunk_offsets_abs - char_start

            reason = _text_pathology_reason(chunk_text, pathology_cfg)
            if reason is not None:
                skip_pathological_chunk(
                    skipped_fh=skipped_fh,
                    reason=reason,
                    chunk_idx=chunk_idx,
                    file_idx=file_idx,
                    rel_start=rel_start,
                    rel_end=rel_end,
                    row=row,
                    decoded_chars=len(chunk_text),
                )
                continue

            if args.alignment_mode == "dp":
                falcon_encoded = falcon_tokenizer(
                    chunk_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                falcon_ids = [int(x) for x in falcon_encoded["input_ids"]]
                falcon_offsets_for_chunk: list[tuple[int, int]] | np.ndarray = [
                    (int(a), int(b)) for a, b in falcon_encoded["offset_mapping"]
                ]
            else:
                falcon_ids = [
                    int(x)
                    for x in falcon_tokenizer.encode(
                        chunk_text,
                        add_special_tokens=False,
                    )
                ]
                falcon_offsets_for_chunk = _linear_offsets(
                    len(chunk_text),
                    len(falcon_ids),
                )

            reason = _falcon_pathology_reason(
                qwen_tokens=int(rel_end - rel_start),
                falcon_tokens=len(falcon_ids),
                cfg=pathology_cfg,
            )
            if reason is not None:
                skip_pathological_chunk(
                    skipped_fh=skipped_fh,
                    reason=reason,
                    chunk_idx=chunk_idx,
                    file_idx=file_idx,
                    rel_start=rel_start,
                    rel_end=rel_end,
                    row=row,
                    decoded_chars=len(chunk_text),
                    falcon_tokens=len(falcon_ids),
                )
                continue

            append_chunk(
                chunk_idx=chunk_idx,
                file_idx=file_idx,
                rel_start=rel_start,
                rel_end=rel_end,
                row=row,
                qwen_offsets=qwen_offsets,
                falcon_ids=falcon_ids,
                falcon_offsets_for_chunk=falcon_offsets_for_chunk,
                skipped_fh=skipped_fh,
                quality=chunk_quality,
            )

    for writer in writers.values():
        writer.finalize()
    np.save(output_dir / "falcon_offsets.npy", np.asarray(falcon_offsets, dtype=np.int64))
    np.save(
        output_dir / "forced_survivor_offsets.npy",
        np.asarray(forced_offsets, dtype=np.int64),
    )
    alignment_quality_arr = np.asarray(alignment_quality, dtype=np.int8)
    np.save(output_dir / "alignment_quality.npy", alignment_quality_arr)
    quality_summary = {
        "real_dp": int((alignment_quality_arr == ALIGNMENT_QUALITY_REAL_DP).sum()),
        "artificial_fast": int(
            (alignment_quality_arr == ALIGNMENT_QUALITY_ARTIFICIAL_FAST).sum()
        ),
        "artificial_fallback": int(
            (alignment_quality_arr == ALIGNMENT_QUALITY_ARTIFICIAL_FALLBACK).sum()
        ),
        "pathological_skip": int(
            (alignment_quality_arr == ALIGNMENT_QUALITY_PATHOLOGICAL).sum()
        ),
        "alignment_skip": int(
            (alignment_quality_arr == ALIGNMENT_QUALITY_SKIPPED).sum()
        ),
    }
    artificial_total = (
        quality_summary["artificial_fast"] + quality_summary["artificial_fallback"]
    )
    if artificial_total > 0:
        print(
            f"WARNING: {artificial_total}/{len(alignment_quality_arr)} chunks were "
            "aligned via re-decoded text (artificial offsets). Their "
            "`alignment_scores` are approximate monotone-pairing signals only "
            "— do not filter by absolute score. Set --alignment-mode dp with "
            "--source-root for byte-stable alignment.",
            flush=True,
        )
    if args.strict_alignment and artificial_total > 0:
        raise SystemExit(
            f"--strict-alignment refused: {artificial_total} chunks would have "
            "artificial alignment. Re-run with dp mode + --source-root."
        )

    manifest = {
        "source_qwen_mmap": str(input_dir),
        "source_manifest_sha256": _sha256(input_dir / "manifest.json"),
        "qwen_tokenizer": args.qwen_tokenizer,
        "qwen_revision": args.qwen_revision,
        "falcon_tokenizer": args.falcon_tokenizer,
        "falcon_revision": args.falcon_revision,
        "max_seq_len": int(args.max_seq_len),
        "n_chunks": len(chunks),
        "skipped_degenerate_chunks": skipped,
        "skipped_degenerate_rate": skipped / max(len(chunks), 1),
        "skipped_pathological_chunks": skipped_pathological,
        "skipped_pathological_rate": skipped_pathological / max(len(chunks), 1),
        "pathology_reason_counts": pathology_reason_counts,
        "alignment_skip_counts": alignment_skip_counts,
        "alignment_quality_summary": quality_summary,
        "skipped_chunks_jsonl": str(skipped_chunks_path),
        "odd_final_pair_chunks": odd_final_pairs,
        "odd_final_pair_rate": odd_final_pairs / max(len(chunks), 1),
        "source_root": str(source_root) if use_source_root else None,
        "alignment_mode": args.alignment_mode,
        "memory_bounded_chunkwise": True,
        "pathology_filter": {
            "enabled": pathology_cfg.enabled,
            "max_decoded_chars": pathology_cfg.max_decoded_chars,
            "max_line_chars": pathology_cfg.max_line_chars,
            "min_printable_ratio": pathology_cfg.min_printable_ratio,
            "max_replacement_ratio": pathology_cfg.max_replacement_ratio,
            "min_base64ish_chars": pathology_cfg.min_base64ish_chars,
            "max_base64ish_ratio": pathology_cfg.max_base64ish_ratio,
            "max_falcon_tokens": pathology_cfg.max_falcon_tokens,
            "max_falcon_expansion": pathology_cfg.max_falcon_expansion,
        },
        "array_dtypes": {
            "falcon_tokens": "int32",
            "forced_survivor_indices": "int32",
            "target_falcon_pair_ids": "int32",
            "target_pair_loss_mask": "bool",
            "alignment_scores": "float32",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--qwen-tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--qwen-revision", default=None)
    parser.add_argument("--falcon-tokenizer", default="tiiuae/Falcon-H1-Tiny-90M-Instruct")
    parser.add_argument("--falcon-revision", default=None)
    parser.add_argument(
        "--alignment-mode",
        choices=["fast", "dp", "containment"],
        default="fast",
        help=(
            "fast: vectorized monotone-DP best-overlap heuristic (default). "
            "dp: exact DP, slower, for diagnostics. "
            "containment: strict — only Qwen positions whose char range "
            "contains EXACTLY 2 Falcon tokens; targets become unambiguous. "
            "Fewer forced_survivors but every target is byte-clean."
        ),
    )
    parser.add_argument(
        "--strict-alignment",
        action="store_true",
        help=(
            "Refuse to write a companion if any chunk would use artificial "
            "(re-decoded) offsets. Use with --alignment-mode dp + --source-root."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Deprecated compatibility flag; conversion is chunkwise for bounded memory.",
    )
    parser.add_argument("--disable-pathology-filter", action="store_true")
    parser.add_argument("--max-decoded-chars", type=int, default=256 * 1024)
    parser.add_argument("--max-line-chars", type=int, default=32 * 1024)
    parser.add_argument("--min-printable-ratio", type=float, default=0.80)
    parser.add_argument("--max-replacement-ratio", type=float, default=0.01)
    parser.add_argument("--min-base64ish-chars", type=int, default=8192)
    parser.add_argument("--max-base64ish-ratio", type=float, default=0.98)
    parser.add_argument("--max-falcon-tokens", type=int, default=32 * 1024)
    parser.add_argument("--max-falcon-expansion", type=float, default=4.0)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
