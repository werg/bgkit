"""Commit serialization for reproduction training.

Converts ExtractedCommit objects into plaintext documents with structured tags,
suitable for training a model to reconstruct commits from compressed representations.

Chat template integration (TODO for Phase 1 Step 2):
    The serialized commit text produced here is raw content — it does not include
    Qwen3's chat template wrapping. By Step 2, the decoder expects chat-formatted
    input (trained on it in Step 1 via ChatReproDataset). The commit reproduction
    objective needs a chat-wrapping dataset analogous to ChatReproDataset:
    - New template YAML: configs/templates/commit_repro.yaml
    - Different tool name / prompt fields, but same sentinel-based loss masking
    - The _build_messages() pattern in chat_repro_dataset.py should be extracted
      into a shared template assembler parameterized by task-specific fields
    See also: scripts/generate_prompt_variants.py for variant generation.
"""

from __future__ import annotations

import string

from bgkit.data.commit_extraction import ExtractedCommit

_PRINTABLE = set(string.printable)
_BASE64ISH = set(string.ascii_letters + string.digits + "+/=_-")


def _pathology_reason(
    text: str,
    *,
    max_chars: int,
    max_line_chars: int = 32 * 1024,
    min_printable_ratio: float = 0.80,
    min_base64ish_chars: int = 8192,
    max_base64ish_ratio: float = 0.98,
) -> str | None:
    if len(text) > max_chars:
        return "chars_gt_limit"

    lines = text.splitlines() or [text]
    if max((len(line) for line in lines), default=0) > max_line_chars:
        return "line_chars_gt_limit"

    if text:
        printable = sum(1 for ch in text if ch in _PRINTABLE or ch.isspace())
        if printable / len(text) < min_printable_ratio:
            return "printable_ratio_lt_limit"

        compact = "".join(ch for ch in text if not ch.isspace())
        if len(compact) >= min_base64ish_chars:
            base64ish = sum(1 for ch in compact if ch in _BASE64ISH)
            if base64ish / len(compact) > max_base64ish_ratio:
                return "base64ish_ratio_gt_limit"

    return None


def serialize_commit(commit: ExtractedCommit) -> str:
    """Serialize an extracted commit into a tagged plaintext document.

    Format:
        <commit>
        <message>
        {commit message}
        </message>
        <files>
        {path1}
        {path2}
        </files>
        <diff>
        --- {path1}
        {hunks for path1}
        --- {path2}
        {hunks for path2}
        </diff>
        </commit>
    """
    lines = ["<commit>", "<message>", commit.message, "</message>", "<files>"]

    for path in commit.diff_paths:
        lines.append(path)

    lines.append("</files>")
    lines.append("<diff>")

    for path, file_hunks in zip(commit.diff_paths, commit.diff_hunks, strict=True):
        lines.append(f"--- {path}")
        for hunk in file_hunks:
            lines.append(hunk)

    lines.append("</diff>")
    lines.append("</commit>")

    return "\n".join(lines)


def serialize_and_tokenize_commit(
    commit: ExtractedCommit,
    tokenizer,
    max_tokens: int,
) -> list[int] | None:
    """Serialize a commit and tokenize it, returning None if over the token limit.

    Truncated diffs are misleading training signal, so we discard (not truncate)
    commits that exceed max_tokens.

    Args:
        commit: The extracted commit to serialize.
        tokenizer: HuggingFace tokenizer instance.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        List of token IDs, or None if the serialized commit exceeds max_tokens.
    """
    text = serialize_commit(commit)
    if _pathology_reason(text, max_chars=max_tokens * 64) is not None:
        return None

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) > max_tokens:
        return None

    return token_ids
