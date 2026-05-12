"""Packed data collators for FA4 varlen training.

All collators produce flat ``(N,)`` tensors over samples with ``N = sum(L_i)``,
``cu_seqlens`` of shape ``(B+1,)`` int32, and ``position_ids`` of shape
``(N,)`` int64 that restart to 0 at each sample boundary.  No ``attention_mask``
tensors are produced at the attention boundary -- segmentation lives in
``cu_seqlens``.  Semantic masks (``loss_mask``, ``answer_position_mask``) remain
as ``(N,)`` flat tensors.
"""

from __future__ import annotations

import torch

from bgkit.utils.packing import position_ids_from_cu

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Build a cumulative sequence lengths tensor from a list of lengths.

    Returns
    -------
    Tensor
        Shape ``(B+1,)`` int32 with ``cu[0] == 0`` and ``cu[-1] == sum(lengths)``.
    """
    t = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0, out=t[1:])
    return t


def _cat_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Flat-concatenate a list of 1-D tensors."""
    return torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# Public collators
# ---------------------------------------------------------------------------


def collate_token_ids(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length token ID samples into a packed batch.

    Args:
        batch: List of dicts with ``"token_ids"`` key holding a ``(L,)`` int64
               tensor.

    Returns:
        Dict with keys:
        - ``"input_ids"``: ``(N,)`` int64 flat concatenation.
        - ``"position_ids"``: ``(N,)`` int64, per-sample restart.
        - ``"cu_seqlens"``: ``(B+1,)`` int32.
        - ``"max_seqlen"``: int.
    """
    seqs = [s["token_ids"] for s in batch]
    lengths = [int(s.size(0)) for s in seqs]
    cu = _make_cu_seqlens(lengths)
    total = int(cu[-1])
    pos = position_ids_from_cu(cu, total)
    return {
        "input_ids": _cat_tensors(seqs),
        "position_ids": pos,
        "cu_seqlens": cu,
        "max_seqlen": max(lengths) if lengths else 0,
    }


def collate_chat_repro(batch: list[dict]) -> dict:
    """Collate chat-formatted reproduction samples into a packed batch.

    Args:
        batch: List of dicts with keys:
            - ``"token_ids"`` (L,) -- full chat-formatted sequence.
            - ``"loss_mask"`` (L,) -- 1 for content tokens, 0 elsewhere.
            - ``"content_token_ids"`` (C,) -- raw file tokens for BgKIT input.
            - ``"compression_prompt_ids"`` (P,) -- tokenised compression prompt.
            - ``"prefix_ids"`` (X,) -- chat template prefix tokens.
            - ``"language"`` str -- source language label.
            - ``"bgkit_splice_start"`` int (optional).
            - ``"bgkit_splice_len"`` int (optional).
            - ``"answer_position_mask"`` (C,) bool (optional, QA-mode batches only).

    Returns:
        Dict with packed flat tensors, per-sequence ``cu_seqlens``/``position_ids``,
        per-sample scalars, and ``languages`` list.
    """
    # --- main sequence ---
    tok_seqs = [s["token_ids"] for s in batch]
    tok_lengths = [int(t.size(0)) for t in tok_seqs]
    tok_cu = _make_cu_seqlens(tok_lengths)
    tok_total = int(tok_cu[-1])

    # --- loss_mask (same segmentation as token_ids) ---
    loss_seqs = [s["loss_mask"] for s in batch]

    # --- content ---
    content_seqs = [s["content_token_ids"] for s in batch]
    content_lengths = [int(t.size(0)) for t in content_seqs]
    content_cu = _make_cu_seqlens(content_lengths)
    content_total = int(content_cu[-1])

    # --- compression prompt ---
    prompt_seqs = [s["compression_prompt_ids"] for s in batch]
    prompt_lengths = [int(t.size(0)) for t in prompt_seqs]
    prompt_cu = _make_cu_seqlens(prompt_lengths)

    # --- prefix ---
    prefix_seqs = [s["prefix_ids"] for s in batch]
    prefix_lengths = [int(t.size(0)) for t in prefix_seqs]
    prefix_cu = _make_cu_seqlens(prefix_lengths)

    out = {
        "token_ids": _cat_tensors(tok_seqs),
        "position_ids": position_ids_from_cu(tok_cu, tok_total),
        "cu_seqlens": tok_cu,
        "max_seqlen": max(tok_lengths) if tok_lengths else 0,
        "loss_mask": _cat_tensors(loss_seqs),
        "encoder_content_token_ids": _cat_tensors(content_seqs),
        "content_token_ids": _cat_tensors(content_seqs),
        "content_position_ids": position_ids_from_cu(content_cu, content_total),
        "content_cu_seqlens": content_cu,
        "content_max_seqlen": max(content_lengths) if content_lengths else 0,
        "compression_prompt_ids": _cat_tensors(prompt_seqs),
        "compression_prompt_cu_seqlens": prompt_cu,
        "compression_prompt_max_seqlen": max(prompt_lengths) if prompt_lengths else 0,
        "prefix_ids": _cat_tensors(prefix_seqs),
        "prefix_cu_seqlens": prefix_cu,
        "prefix_max_seqlen": max(prefix_lengths) if prefix_lengths else 0,
        "bgkit_splice_start": torch.tensor(
            [int(s.get("bgkit_splice_start", -1)) for s in batch], dtype=torch.long,
        ),
        "bgkit_splice_len": torch.tensor(
            [int(s.get("bgkit_splice_len", 0)) for s in batch], dtype=torch.long,
        ),
        "languages": [s["language"] for s in batch],
    }

    # Optional QA answer-position mask -- all-or-nothing within a batch.
    has_pos = ["answer_position_mask" in s for s in batch]
    if any(has_pos):
        if not all(has_pos):
            raise ValueError(
                "collate_chat_repro: batch mixes samples with and without "
                "answer_position_mask. Interleaving must operate at batch "
                "granularity, not within a batch.",
            )
        # Flat over content positions; segmentation via content_cu_seqlens.
        out["answer_position_mask"] = _cat_tensors(
            [s["answer_position_mask"] for s in batch]
        )

    return out


def collate_compression(batch: list) -> dict:
    """Collate compression samples into a packed batch.

    Dispatches to :func:`_collate_file_samples` or
    :func:`_collate_repo_samples` based on sample type.  Mixed batches
    are split and returned under the ``"mixed"`` key.
    """
    from bgkit.data.datasets.compression_dataset import (
        FileCompressionSample,
        RepoCompressionSample,
    )

    file_samples = [s for s in batch if isinstance(s, FileCompressionSample)]
    repo_samples = [s for s in batch if isinstance(s, RepoCompressionSample)]

    if file_samples and repo_samples:
        return {
            "mixed": True,
            "file_batch": _collate_file_samples(file_samples),
            "repo_batch": _collate_repo_samples(repo_samples),
        }

    if file_samples:
        return _collate_file_samples(file_samples)
    return _collate_repo_samples(repo_samples)


def _collate_file_samples(samples: list) -> dict:
    """Collate FileCompressionSample list into a packed batch.

    Returns a dict with keys:
    - ``"sample_type"``: ``"file"``
    - ``"content_token_ids"``: ``(N_content,)`` flat.
    - ``"content_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"content_position_ids"``: ``(N_content,)`` int64.
    - ``"content_max_seqlen"``: int.
    - ``"target_token_ids"``: ``(N_target,)`` flat.
    - ``"target_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"target_loss_mask"``: ``(N_target,)`` flat.
    - ``"prefix_ids"``: ``(N_prefix,)`` flat.
    - ``"prefix_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"compression_prompt_ids"``: ``(N_prompt,)`` flat.
    - ``"prompt_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"compression_ratios"``: ``(B,)`` float.
    - ``"compression_levels"``: ``(B,)`` long.
    - ``"bgkit_splice_start"``: ``(B,)`` long.
    - ``"bgkit_splice_len"``: ``(B,)`` long.
    - ``"objectives"``: list[str].
    """
    content_seqs = [s.content_token_ids for s in samples]
    content_lengths = [int(t.size(0)) for t in content_seqs]
    content_cu = _make_cu_seqlens(content_lengths)
    content_total = int(content_cu[-1])

    target_seqs = [s.target_token_ids for s in samples]
    target_lengths = [int(t.size(0)) for t in target_seqs]
    target_cu = _make_cu_seqlens(target_lengths)

    loss_seqs = [s.target_loss_mask for s in samples]

    prefix_seqs = [s.prefix_ids for s in samples]
    prefix_lengths = [int(t.size(0)) for t in prefix_seqs]
    prefix_cu = _make_cu_seqlens(prefix_lengths)

    prompt_seqs = [s.compression_prompt_ids for s in samples]
    prompt_lengths = [int(t.size(0)) for t in prompt_seqs]
    prompt_cu = _make_cu_seqlens(prompt_lengths)

    # Forced-survivor mask propagation: only emit a packed mask if every
    # sample in the batch has one (Falcon companion is loaded), otherwise
    # set to None so downstream survivorship_losses skips the forced BCE
    # term cleanly. The packed shape is ``(N_content,)`` bool — same flat
    # axis as ``content_token_ids``.
    if all(getattr(s, "forced_survivor_mask", None) is not None for s in samples):
        forced_seqs = [s.forced_survivor_mask for s in samples]
        forced_packed = _cat_tensors(forced_seqs).to(torch.bool)
    else:
        forced_packed = None

    return {
        "sample_type": "file",
        "objectives": [s.objective for s in samples],
        "encoder_content_token_ids": _cat_tensors(content_seqs),
        "content_token_ids": _cat_tensors(content_seqs),
        "content_cu_seqlens": content_cu,
        "content_position_ids": position_ids_from_cu(content_cu, content_total),
        "content_max_seqlen": max(content_lengths) if content_lengths else 0,
        "target_token_ids": _cat_tensors(target_seqs),
        "target_cu_seqlens": target_cu,
        "target_max_seqlen": max(target_lengths) if target_lengths else 0,
        "target_loss_mask": _cat_tensors(loss_seqs),
        "prefix_ids": _cat_tensors(prefix_seqs),
        "prefix_cu_seqlens": prefix_cu,
        "compression_prompt_ids": _cat_tensors(prompt_seqs),
        "prompt_cu_seqlens": prompt_cu,
        "compression_ratios": torch.tensor([s.compression_ratio for s in samples]),
        "compression_levels": torch.tensor(
            [s.compression_level for s in samples], dtype=torch.long,
        ),
        "bgkit_splice_start": torch.tensor(
            [getattr(s, "bgkit_splice_start", -1) for s in samples], dtype=torch.long,
        ),
        "bgkit_splice_len": torch.tensor(
            [getattr(s, "bgkit_splice_len", 0) for s in samples], dtype=torch.long,
        ),
        "forced_survivor_mask_l0": forced_packed,
    }


def _collate_repo_samples(samples: list) -> dict:
    """Collate RepoCompressionSample list into a two-level packed batch.

    Two-level packing: each *repo* is a group of packed file-segments.

    Fields
    ------
    ``content_token_ids`` : ``(N_content,)``
        Flat over all files in all repos.
    ``cu_file_seqlens`` : ``(total_files + 1,)`` int32
        One cumulative-seqlen segment per file across all repos.
    ``content_position_ids`` : ``(N_content,)`` int64
        Per-file position restart; computed from ``cu_file_seqlens``.
    ``content_max_seqlen`` : int
    ``cu_repo_seqlens`` : ``(B + 1,)`` int32
        Indices **into** ``cu_file_seqlens`` marking where each repo's files
        end.  E.g. if repo 0 has 3 files, repo 1 has 2 files, repo 2 has 4
        files: ``cu_repo_seqlens = [0, 3, 5, 9]``.
    ``prompt_token_ids`` : ``(N_prompt,)``
        Each repo's prompt is tiled ``n_files_i`` times; the flat buffer
        contains ``n_files_i * prompt_len_i`` tokens for repo ``i``.
    ``prompt_cu_seqlens`` : ``(total_files + 1,)`` int32
        One prompt segment per file, aligned 1:1 with ``cu_file_seqlens``.
    ``target_token_ids`` : ``(N_target,)``
    ``target_cu_seqlens`` : ``(B + 1,)``
    ``target_loss_mask`` : ``(N_target,)``
    ``prefix_ids`` : ``(N_prefix,)``
    ``prefix_cu_seqlens`` : ``(B + 1,)``
    ``compression_ratios`` : ``(B,)``
    ``compression_levels`` : ``(B,)``
    ``bgkit_splice_start`` : ``(B,)``
    ``bgkit_splice_len`` : ``(B,)``
    ``objectives`` : list[str]
    """
    # --- two-level file packing ---
    # For each repo collect its per-file token tensors.
    all_file_seqs: list[torch.Tensor] = []
    all_prompt_seqs: list[torch.Tensor] = []
    file_lengths: list[int] = []
    # cu_repo_seqlens values: index into cu_file_seqlens (i.e. file count boundaries)
    repo_file_boundaries: list[int] = [0]

    for s in samples:
        n_files = len(s.file_token_ids)
        for f_ids in s.file_token_ids:
            all_file_seqs.append(f_ids)
            file_lengths.append(int(f_ids.size(0)))
            # tile the prompt once per file
            all_prompt_seqs.append(s.compression_prompt_ids)
        repo_file_boundaries.append(repo_file_boundaries[-1] + n_files)

    cu_file = _make_cu_seqlens(file_lengths)
    content_total = int(cu_file[-1])

    cu_repo = torch.tensor(repo_file_boundaries, dtype=torch.int32)

    prompt_lengths = [int(t.size(0)) for t in all_prompt_seqs]
    prompt_cu = _make_cu_seqlens(prompt_lengths)

    # --- per-repo target / prefix (one per repo, not per file) ---
    target_seqs = [s.target_token_ids for s in samples]
    target_lengths_list = [int(t.size(0)) for t in target_seqs]
    target_cu = _make_cu_seqlens(target_lengths_list)
    loss_seqs = [s.target_loss_mask for s in samples]

    prefix_seqs = [s.prefix_ids for s in samples]
    prefix_lengths_list = [int(t.size(0)) for t in prefix_seqs]
    prefix_cu = _make_cu_seqlens(prefix_lengths_list)

    empty_long = torch.zeros(0, dtype=torch.long)
    return {
        "sample_type": "repo",
        "objectives": [s.objective for s in samples],
        "encoder_content_token_ids": _cat_tensors(all_file_seqs) if all_file_seqs else empty_long,
        "content_token_ids": _cat_tensors(all_file_seqs) if all_file_seqs else empty_long,
        "cu_file_seqlens": cu_file,
        "content_position_ids": position_ids_from_cu(cu_file, content_total),
        "content_max_seqlen": max(file_lengths) if file_lengths else 0,
        "cu_repo_seqlens": cu_repo,
        "prompt_token_ids": _cat_tensors(all_prompt_seqs) if all_prompt_seqs else empty_long,
        "prompt_cu_seqlens": prompt_cu,
        "target_token_ids": _cat_tensors(target_seqs),
        "target_cu_seqlens": target_cu,
        "target_max_seqlen": max(target_lengths_list) if target_lengths_list else 0,
        "target_loss_mask": _cat_tensors(loss_seqs),
        "prefix_ids": _cat_tensors(prefix_seqs),
        "prefix_cu_seqlens": prefix_cu,
        "compression_ratios": torch.tensor([s.compression_ratio for s in samples]),
        "compression_levels": torch.tensor(
            [s.compression_level for s in samples], dtype=torch.long,
        ),
        "bgkit_splice_start": torch.tensor(
            [getattr(s, "bgkit_splice_start", -1) for s in samples], dtype=torch.long,
        ),
        "bgkit_splice_len": torch.tensor(
            [getattr(s, "bgkit_splice_len", 0) for s in samples], dtype=torch.long,
        ),
    }


def collate_qa(batch: list) -> dict:
    """Collate ``QASample`` objects into a packed Phase 2 batch.

    Returns all fields from :func:`_collate_file_samples` plus:
    - ``"question_token_ids"``: ``(N_q,)`` flat.
    - ``"question_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"answer_token_ids"``: ``(N_a,)`` flat.
    - ``"answer_cu_seqlens"``: ``(B+1,)`` int32.
    - ``"objectives"``, ``"dataset_names"``, ``"sample_ids"``, ``"document_ids"``,
      ``"tags"``, ``"metadata"``.
    """
    from bgkit.data.datasets.qa_sample import QASample

    if not batch:
        raise ValueError("collate_qa() received an empty batch")
    if not all(isinstance(sample, QASample) for sample in batch):
        types = sorted({type(sample).__name__ for sample in batch})
        raise TypeError(f"collate_qa() expects QASample items, got {types}")

    # Base file-compression fields (content / target / prefix / prompt).
    base = _collate_file_samples(batch)

    question_seqs = [s.question_token_ids for s in batch]
    question_lengths = [int(t.size(0)) for t in question_seqs]
    question_cu = _make_cu_seqlens(question_lengths)

    answer_seqs = [s.answer_token_ids for s in batch]
    answer_lengths = [int(t.size(0)) for t in answer_seqs]
    answer_cu = _make_cu_seqlens(answer_lengths)

    return {
        **base,
        "sample_type": "qa",
        "question_token_ids": _cat_tensors(question_seqs),
        "question_cu_seqlens": question_cu,
        "question_max_seqlen": max(question_lengths) if question_lengths else 0,
        "answer_token_ids": _cat_tensors(answer_seqs),
        "answer_cu_seqlens": answer_cu,
        "answer_max_seqlen": max(answer_lengths) if answer_lengths else 0,
        "dataset_names": [s.dataset_name for s in batch],
        "sample_ids": [s.sample_id for s in batch],
        "document_ids": [s.document_id for s in batch],
        "tags": [list(s.tags) for s in batch],
        "metadata": [dict(s.metadata) for s in batch],
    }
