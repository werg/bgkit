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
