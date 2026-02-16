"""Commit serialization for reproduction training.

Converts ExtractedCommit objects into plaintext documents with structured tags,
suitable for training a model to reconstruct commits from compressed representations.
"""

from __future__ import annotations

from bgkit.data.commit_extraction import ExtractedCommit


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
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) > max_tokens:
        return None

    return token_ids
