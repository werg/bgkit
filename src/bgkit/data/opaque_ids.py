"""Opaque, token-friendly node/article ids for tree-based Phase 2 datasets.

The recursive drill-down trainer teaches the decoder to read a child's id out
of a parent node's compressed representation and emit it to navigate. For that
signal to be real the ids must be **opaque** — unguessable from a node's ordinal
position (a positional ``cm_0007`` id lets the decoder cheat by counting instead
of reading the rep) — and **token-friendly** (short, ideally one BPE token per
word, so emitting an id is cheap and the discrimination is clean).

This module provides the two shared primitives every tree dataset's id scheme is
built on:

- :func:`bip39_id` — deterministic ``sha256(hash_input)`` → ``n_words`` words
  from the BIP-39 English wordlist, joined by ``-``. Same input always yields the
  same id; different inputs collide only with cryptographic (birthday)
  probability over the ``2048**n_words`` space.
- :func:`single_token_bip39_wordlist` — filter BIP-39 to the words that encode to
  exactly one token under *every* supplied tokenizer (e.g. Qwen3.5 + Falcon-H1),
  so ids built from the filtered list stay single-token across the whole decoder
  fleet. Degrades gracefully to the full list if no tokenizer is usable.

The id *scheme* (what a commit / chunk / leaf hashes) is dataset-local; this
module is the dataset-agnostic plumbing shared by all of them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from bgkit.data._bip39_wordlist import BIP39_ENGLISH

__all__ = ["BIP39_ENGLISH", "bip39_id", "single_token_bip39_wordlist"]


def bip39_id(
    hash_input: str,
    n_words: int = 2,
    wordlist: Sequence[str] | None = None,
) -> str:
    """Deterministically map ``hash_input`` to ``n_words`` hyphen-joined words.

    ``sha256(hash_input)`` is consumed as a big integer; each word index is the
    next base-``len(wordlist)`` digit. Same string in → same id out, on any
    machine, forever (the id is baked into browse-tree / mmap / trajectory
    artifacts, so it MUST be stable).

    Args:
        hash_input: the stable string that identifies the node (e.g. a commit
            sha, or the concatenation of a chunk's descendant shas). Never pass a
            node's ordinal position — the whole point is that the id is *not*
            derivable from position.
        n_words: how many words the id is composed of. ``2`` (the default) gives
            ``2048**2 ≈ 4.2M`` combinations — ample for per-scope uniqueness at
            the ~hundreds-of-siblings scale while staying short. Use ``1`` only
            when the uniqueness scope is tiny (< ~64 items) and the wordlist is
            single-token-filtered.
        wordlist: word source; defaults to the full 2048-word BIP-39 list. Pass
            a :func:`single_token_bip39_wordlist` result for maximal token
            efficiency (but then every producer of a given id space must use the
            *same* filtered list, or ids will diverge).

    Returns:
        ``"word"`` for ``n_words == 1``, else ``"word-word[-word...]"``.
    """
    if n_words < 1:
        raise ValueError(f"n_words must be >= 1, got {n_words}")
    words = list(wordlist) if wordlist is not None else list(BIP39_ENGLISH)
    n = len(words)
    if n == 0:
        raise ValueError("wordlist must be non-empty")
    # 256-bit digest → integer. 11 bits per BIP-39 word, so 256 bits covers
    # n_words up to ~23 before running out of entropy.
    value = int.from_bytes(hashlib.sha256(hash_input.encode("utf-8")).digest(), "big")
    out: list[str] = []
    for _ in range(n_words):
        out.append(words[value % n])
        value //= n
    return "-".join(out)


def single_token_bip39_wordlist(
    tokenizers: Sequence[Any],
    *,
    wordlist: Sequence[str] | None = None,
    add_leading_space: bool = True,
    stats: dict[str, Any] | None = None,
) -> list[str]:
    """Filter BIP-39 to words that are exactly ONE token under all tokenizers.

    A word survives only if ``tokenizer.encode(word)`` (optionally with a leading
    space, matching how ids are spliced mid-text) yields a single id for *every*
    tokenizer supplied. This keeps ids emitted during navigation single-token per
    word across the whole decoder fleet (Qwen3.5-0.8B + Falcon-H1).

    Robustness: if ``tokenizers`` is empty, or any tokenizer raises while
    encoding, the function **degrades to the full wordlist** rather than
    returning a silently-truncated set — a build must never produce a tiny id
    space because a tokenizer failed to load.

    Reporting: pass a mutable ``stats`` dict to receive per-tokenizer survivor
    counts and the intersection size::

        {"total": 2048, "add_leading_space": True, "degraded": False,
         "per_tokenizer": [{"index": 0, "survivors": N0}, ...],
         "intersection": M}

    If the intersection is too small for the uniqueness scale you need at
    ``n_words=1`` (i.e. ``M`` items but you need collision-free ids for more than
    ~``sqrt(M)`` siblings), callers should use ``n_words=2`` with the filtered
    list — ``M**2`` is almost always plenty.

    Returns the surviving words in the original BIP-39 order.
    """
    words = list(wordlist) if wordlist is not None else list(BIP39_ENGLISH)
    report: dict[str, Any] = {
        "total": len(words),
        "add_leading_space": add_leading_space,
        "degraded": False,
        "per_tokenizer": [],
        "intersection": len(words),
    }

    def _degrade(reason: str) -> list[str]:
        report["degraded"] = True
        report["degrade_reason"] = reason
        report["intersection"] = len(words)
        if stats is not None:
            stats.clear()
            stats.update(report)
        return list(words)

    if not tokenizers:
        return _degrade("no tokenizers supplied")

    per_tok_survivors: list[set[str]] = []
    for idx, tok in enumerate(tokenizers):
        survivors: set[str] = set()
        for w in words:
            text = f" {w}" if add_leading_space else w
            try:
                ids = tok.encode(text, add_special_tokens=False)
            except TypeError:
                # Tokenizers that don't accept add_special_tokens kwarg.
                try:
                    ids = tok.encode(text)
                except Exception as exc:
                    return _degrade(f"tokenizer {idx} encode failed: {exc!r}")
            except Exception as exc:
                return _degrade(f"tokenizer {idx} encode failed: {exc!r}")
            if len(ids) == 1:
                survivors.add(w)
        per_tok_survivors.append(survivors)
        report["per_tokenizer"].append({"index": idx, "survivors": len(survivors)})

    intersection: set[str] = set(words)
    for s in per_tok_survivors:
        intersection &= s
    result = [w for w in words if w in intersection]
    report["intersection"] = len(result)
    if stats is not None:
        stats.clear()
        stats.update(report)
    return result
