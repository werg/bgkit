"""Git checkout and diff parsing utilities using pygit2."""

from __future__ import annotations

import re

import pygit2


def get_file_at_commit(repo_path: str, commit_sha: str, file_path: str) -> str | None:
    """Get file contents at a specific commit.

    Args:
        repo_path: Path to git repo (working dir or bare).
        commit_sha: Git commit SHA (full or abbreviated).
        file_path: Path relative to repo root.

    Returns:
        File contents as string, or None if file doesn't exist at that commit
        or is binary.
    """
    repo = pygit2.Repository(repo_path)
    try:
        commit = repo.revparse_single(commit_sha).peel(pygit2.Commit)
    except (KeyError, ValueError):
        return None

    try:
        entry = commit.tree[file_path]
    except KeyError:
        return None

    if entry.type_str != "blob":
        return None

    blob = repo.get(entry.id)
    if blob.is_binary:
        return None

    try:
        return blob.data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def parse_diff_hunks(diff_text: str) -> list[dict]:
    """Parse a unified diff into structured hunks.

    Args:
        diff_text: Raw unified diff text.

    Returns:
        List of hunk dicts with keys:
            file_path: str - path of the affected file
            old_start: int - start line in old file
            new_start: int - start line in new file
            old_count: int - number of lines in old file
            new_count: int - number of lines in new file
            lines: list[str] - raw hunk lines (with +/-/space prefixes)
    """
    hunks = []
    current_file = None

    # Match file headers: --- a/path and +++ b/path
    file_re = re.compile(r"^\+\+\+ b/(.+)$")
    # Match hunk headers: @@ -old_start,old_count +new_start,new_count @@
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    lines = diff_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track current file
        file_match = file_re.match(line)
        if file_match:
            current_file = file_match.group(1)
            i += 1
            continue

        # Parse hunk header
        hunk_match = hunk_re.match(line)
        if hunk_match and current_file:
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or 1)
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or 1)

            # Collect hunk lines
            hunk_lines = []
            i += 1
            while i < len(lines):
                hline = lines[i]
                if hline.startswith(("+", "-", " ")):
                    hunk_lines.append(hline)
                    i += 1
                elif hline.startswith("\\"):
                    # "\ No newline at end of file"
                    i += 1
                else:
                    break

            hunks.append({
                "file_path": current_file,
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "lines": hunk_lines,
            })
            continue

        i += 1

    return hunks
