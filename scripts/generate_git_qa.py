#!/usr/bin/env python3
"""Generate question-answer pairs from git commit history for Phase 2 Track B.

For each repo, extracts meaningful commits (filtering merges, empty diffs,
oversized diffs, trivial changes) and uses an LLM to generate developer-style
recall questions and answers grounded in the commit context.

Idempotent and resumable -- skips repos with existing .jsonl output files,
writes .partial.jsonl atomically.

Usage:
    python scripts/generate_git_qa.py \
        --repos-dir $DATA_DIR/repos/ \
        --output-dir $DATA_DIR/mmap/phase2/git_qa/ \
        --server-url http://localhost:8090/v1 \
        --workers 8

    # Quick test on 5 repos
    python scripts/generate_git_qa.py \
        --repos-dir $DATA_DIR/repos/ \
        --output-dir $DATA_DIR/mmap/phase2/git_qa/ \
        --max-repos 5 --max-commits-per-repo 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """\
Generate {num_questions} questions a developer might ask about this commit \
when returning to the codebase later. Questions should be natural recall \
queries, NOT metadata lookups.

GOOD: "Why did we change the authentication approach?"
BAD: "What files were modified?" (trivial metadata)

Each question must fall into one of these types:
- factual_recall: retrieving a specific fact from the commit
- rationale: understanding why a change was made
- diff_grounded: requires reading the actual diff to answer
- cross_commit: relates this change to other parts of the codebase
- temporal: about when/ordering of changes
- state_recall: about the state of code before or after the commit

Commit message: {message}
Files changed: {diff_paths}
Diff:
```
{diff_content}
```

Return ONLY a JSON array of objects with "question" and "type" keys.
Valid types: factual_recall, rationale, diff_grounded, cross_commit, temporal, state_recall.
Do not include any other text."""

ANSWER_GENERATION_PROMPT = """\
Answer the following question about a git commit. Be specific -- reference \
actual function names, variable names, and logic from the diff and file \
contents. Target 50-300 tokens. Do not reproduce entire files.

Commit message: {message}
Files changed: {diff_paths}

Diff:
```
{diff_content}
```

{file_context_section}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Commit filtering constants (tighter than general extract_commits defaults)
# ---------------------------------------------------------------------------

# Maximum diff size in characters to include in the LLM prompt
_MAX_DIFF_CHARS = 6000

# Maximum number of file contents to include as context
_MAX_FILE_CONTEXT_CHARS = 4000

# Minimum commit message length (single-word messages rarely produce good QA)
_MIN_MESSAGE_LENGTH = 10


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_questions_json(text: str) -> list[dict] | None:
    """Parse question JSON from model output, handling common issues."""
    # Handle markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        return None

    json_str = text[start:end]

    # Try standard json first
    try:
        questions = json.loads(json_str)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, dict) and q.get("question")]
    except json.JSONDecodeError:
        pass

    # Try json_repair if available
    try:
        import json_repair  # type: ignore[import-untyped]
        questions = json_repair.loads(json_str)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, dict) and q.get("question")]
    except (ImportError, Exception):
        pass

    # Manual cleanup: trailing commas, single quotes
    cleaned = json_str.replace("'", '"')
    # Remove trailing commas before ] or }
    import re
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        questions = json.loads(cleaned)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, dict) and q.get("question")]
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Diff formatting
# ---------------------------------------------------------------------------

def _format_diff(commit) -> str:
    """Format commit diff hunks into a single string for the prompt.

    Args:
        commit: An ExtractedCommit instance.

    Returns:
        Formatted diff string, truncated to _MAX_DIFF_CHARS.
    """
    parts = []
    for path, file_hunks in zip(commit.diff_paths, commit.diff_hunks, strict=True):
        parts.append(f"--- a/{path}")
        parts.append(f"+++ b/{path}")
        for hunk in file_hunks:
            parts.append(hunk)
        parts.append("")

    full_diff = "\n".join(parts)
    if len(full_diff) > _MAX_DIFF_CHARS:
        full_diff = full_diff[:_MAX_DIFF_CHARS] + "\n... (truncated)"
    return full_diff


def _get_file_context(
    repo_path: str,
    commit,
    max_chars: int = _MAX_FILE_CONTEXT_CHARS,
) -> str:
    """Get file contents at the parent commit for context.

    Retrieves the state of changed files *before* the commit so the LLM
    can understand what was modified.
    """
    from bgkit.utils.git_utils import get_file_at_commit

    if not commit.parent_sha:
        return ""

    parts = []
    chars_used = 0

    for path in commit.diff_paths:
        if chars_used >= max_chars:
            break

        content = get_file_at_commit(repo_path, commit.parent_sha, path)
        if content is None:
            continue

        remaining = max_chars - chars_used
        if len(content) > remaining:
            content = content[:remaining] + "\n... (truncated)"

        parts.append(f"File state before commit ({path}):\n```\n{content}\n```")
        chars_used += len(content)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def _is_good_question(q: str) -> bool:
    """Check if a question meets minimum quality standards."""
    if len(q) < 10:
        return False
    # Must end with a question mark (or close to it)
    stripped = q.rstrip()
    return stripped.endswith("?")


def _is_good_answer(a: str) -> bool:
    """Check if an answer meets minimum quality standards."""
    words = a.split()
    return len(words) >= 15


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

async def generate_questions_for_commit(
    client,
    commit,
    num_questions: int,
) -> list[dict]:
    """Generate questions for a single commit via LLM."""
    diff_content = _format_diff(commit)
    diff_paths = ", ".join(commit.diff_paths)

    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=num_questions,
        message=commit.message[:2000],
        diff_paths=diff_paths[:1000],
        diff_content=diff_content,
    )

    try:
        text = await client.generate(prompt, max_tokens=1024, temperature=0.7)
        questions = _parse_questions_json((text or "").strip())
    except Exception:
        questions = None

    # Retry once on failure with slightly higher temperature
    if not questions:
        try:
            text = await client.generate(prompt, max_tokens=1024, temperature=0.8)
            questions = _parse_questions_json((text or "").strip())
        except Exception:
            return []

    if not questions:
        return []

    # Filter and normalize
    valid = []
    for q in questions[:num_questions]:
        question_text = q.get("question", "")
        q_type = q.get("type", "unknown")
        if _is_good_question(question_text):
            valid.append({"question": question_text, "type": q_type})

    return valid


async def generate_answer_for_question(
    client,
    commit,
    question: str,
    file_context: str,
) -> str | None:
    """Generate an answer for a single question via LLM."""
    diff_content = _format_diff(commit)
    diff_paths = ", ".join(commit.diff_paths)

    file_context_section = ""
    if file_context:
        file_context_section = f"File contents before commit:\n{file_context}"

    prompt = ANSWER_GENERATION_PROMPT.format(
        message=commit.message[:2000],
        diff_paths=diff_paths[:1000],
        diff_content=diff_content,
        file_context_section=file_context_section,
        question=question,
    )

    try:
        answer = (await client.generate(
            prompt, max_tokens=512, temperature=0.3,
        ) or "").strip()
    except Exception:
        return None

    if not answer or not _is_good_answer(answer):
        return None

    return answer


async def process_commit(
    client,
    commit,
    num_questions: int,
    repo_name: str,
    model_id: str,
) -> list[dict]:
    """Generate QA pairs for a single commit."""
    # Generate questions
    questions = await generate_questions_for_commit(client, commit, num_questions)
    if not questions:
        return []

    # Get file context once for all questions on this commit
    file_context = _get_file_context(commit.repo_path, commit)

    # Generate answers
    results = []
    for q in questions:
        answer = await generate_answer_for_question(
            client, commit, q["question"], file_context,
        )
        if answer is None:
            continue

        results.append({
            "repo_path": repo_name,
            "commit_sha": commit.sha,
            "parent_sha": commit.parent_sha,
            "commit_message": commit.message,
            "diff_paths": commit.diff_paths,
            "question": q["question"],
            "answer": answer,
            "question_type": q["type"],
            "question_words": len(q["question"].split()),
            "answer_words": len(answer.split()),
            "additions": commit.additions,
            "deletions": commit.deletions,
            "is_cross_file": commit.is_cross_file,
            "model_id": model_id,
            "prompt_version": 1,
        })

    return results


# ---------------------------------------------------------------------------
# Repo-level processing
# ---------------------------------------------------------------------------

async def process_repo(
    repo_dir: Path,
    output_dir: Path,
    client,
    max_commits: int,
    questions_per_commit: int,
    model_id: str,
) -> dict:
    """Process a single repo: extract commits, generate QA pairs, write JSONL."""
    from bgkit.data.commit_extraction import extract_commits
    from bgkit.data.commit_filters import CommitFilterConfig

    repo_name = f"{repo_dir.parent.name}/{repo_dir.name}"
    out_path = output_dir / repo_dir.parent.name / repo_dir.name / "git_qa.jsonl"

    # Skip if already done
    if out_path.exists():
        return {"repo": repo_name, "status": "skipped", "pairs": 0, "commits": 0}

    # Extract commits with stricter filtering for QA generation
    config = CommitFilterConfig(
        exclude_merges=True,
        max_files_changed=20,
        max_diff_lines=2000,
        min_diff_lines=5,
        exclude_patterns=["*.lock", "*.generated.*", "*.min.js", "*.min.css"],
    )

    try:
        commits = extract_commits(
            str(repo_dir),
            max_commits=max_commits,
            config=config,
        )
    except Exception as e:
        return {"repo": repo_name, "status": f"extract_error: {e}", "pairs": 0, "commits": 0}

    if not commits:
        return {"repo": repo_name, "status": "no_commits", "pairs": 0, "commits": 0}

    # Additional filtering: skip commits with very short messages
    commits = [
        c for c in commits
        if len(c.message.strip()) >= _MIN_MESSAGE_LENGTH
    ]

    if not commits:
        return {"repo": repo_name, "status": "no_eligible_commits", "pairs": 0, "commits": 0}

    # Generate QA pairs for each commit
    total_pairs = 0
    records = []

    for commit in commits:
        pairs = await process_commit(
            client, commit, questions_per_commit, repo_name, model_id,
        )
        records.extend(pairs)
        total_pairs += len(pairs)

    if records:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(".partial.jsonl")
        with open(partial, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        partial.rename(out_path)

    return {
        "repo": repo_name,
        "status": "done",
        "pairs": total_pairs,
        "commits": len(commits),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_generation(args):
    """Main generation loop."""
    from bgkit.inference import InferenceConfig, LlamaClient
    from bgkit.utils.git_utils import collect_repo_paths

    repos_dir = args.repos_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect repos, shuffled for diversity on partial runs
    all_repos = collect_repo_paths(
        repos_dir,
        max_repos=args.max_repos,
        shuffle_seed=42,
    )

    repo_dirs_done = []
    repo_dirs_todo = []
    for repo_dir in all_repos:
        qa_path = output_dir / repo_dir.parent.name / repo_dir.name / "git_qa.jsonl"
        if qa_path.exists():
            repo_dirs_done.append(repo_dir)
        else:
            repo_dirs_todo.append(repo_dir)

    # Process unfinished repos first
    repo_dirs = repo_dirs_todo + repo_dirs_done
    print(
        f"Found {len(repo_dirs)} repos in {repos_dir} "
        f"({len(repo_dirs_todo)} pending, {len(repo_dirs_done)} already done)"
    )

    if not repo_dirs_todo:
        print("All repos already processed. Nothing to do.")
        return

    # Initialize client
    client = LlamaClient(InferenceConfig(
        base_url=args.server_url, max_concurrent=args.workers,
    ))
    await client.wait_ready()
    print(f"Server ready: {args.server_url}")

    model_id = os.environ.get("VLLM_MODEL_PRIMARY", "gpt-oss-20b")

    # Process repos with concurrency control
    start_time = time.time()
    total_pairs = 0
    total_commits = 0
    skipped = 0
    errors = 0

    sem = asyncio.Semaphore(args.workers)

    async def process_with_sem(repo_dir):
        async with sem:
            return await process_repo(
                repo_dir, output_dir,
                client,
                args.max_commits_per_repo,
                args.questions_per_commit,
                model_id,
            )

    # Process in batches for progress reporting
    batch_size = 50
    for batch_start in range(0, len(repo_dirs), batch_size):
        batch = repo_dirs[batch_start:batch_start + batch_size]
        tasks = [process_with_sem(rd) for rd in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                errors += 1
                print(f"  ERROR: {r}")
                continue
            if r["status"] == "skipped":
                skipped += 1
            elif r["status"] == "done":
                total_pairs += r["pairs"]
                total_commits += r["commits"]
            else:
                errors += 1
                if not r["status"].startswith("no_"):
                    print(f"  {r['repo']}: {r['status']}")

        elapsed = time.time() - start_time
        processed = batch_start + len(batch)
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{processed}/{len(repo_dirs)}] "
            f"{total_pairs} pairs from {total_commits} commits, "
            f"{skipped} skipped, {errors} errors "
            f"({rate:.1f} repos/s)"
        )

    elapsed = time.time() - start_time
    print(
        f"\nDone: {total_pairs} QA pairs from {total_commits} commits "
        f"across {len(repo_dirs)} repos "
        f"({skipped} skipped, {errors} errors) in {elapsed:.0f}s"
    )

    await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from git commit history (Phase 2 Track B)"
    )
    parser.add_argument(
        "--repos-dir", type=Path, required=True,
        help="Root directory containing owner/repo git clones",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for JSONL files (one per repo)",
    )
    parser.add_argument(
        "--server-url", type=str,
        default=os.environ.get("VLLM_URL", "http://localhost:8090/v1"),
        help="vLLM/llama-server endpoint URL",
    )
    parser.add_argument(
        "--max-repos", type=int, default=None,
        help="Maximum number of repos to process (for testing)",
    )
    parser.add_argument(
        "--max-commits-per-repo", type=int, default=50,
        help="Maximum commits to extract per repo",
    )
    parser.add_argument(
        "--questions-per-commit", type=int, default=3,
        help="Number of questions to generate per commit",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Maximum concurrent LLM requests",
    )
    args = parser.parse_args()

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
