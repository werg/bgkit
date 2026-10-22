#!/usr/bin/env python3
"""Generate question-answer pairs for the query_conditioned training objective.

For each file in the training corpus, generates N (question, answer) pairs
using vLLM inference. Stored as per-repo JSONL alongside existing descriptions.

Two-tier routing: primary model for complex files, fast model for simple files.
Idempotent and resumable — skips repos with existing output files.

Usage:
    python scripts/generate_qa_pairs.py \
        --repos-dir $DATA_DIR/repos/ \
        --output-dir $DATA_DIR/qa_pairs/ \
        --server-url-primary http://localhost:8090/v1 \
        --server-url-fast http://localhost:8091/v1 \
        --workers 8
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


QUESTION_GENERATION_PROMPT = """\
You are generating training data for a code comprehension system that \
learns to compress source code relative to a question. Your job is to \
generate diverse, specific questions about the code below.

CRITICAL: Maximize diversity. Each question should focus on a DIFFERENT \
aspect of the code. Ask about specific functions, variables, types, \
algorithms, edge cases, assumptions, security properties, performance \
characteristics, or anything else you find interesting. Be creative — \
ask unexpected questions that require deep understanding of the code. \
Name specific identifiers from the code in your questions when possible. \
Vary the granularity (single line → whole module) and the expected \
answer format (prose explanation, structured list, specific value, \
rewritten code, etc.).

Do not repeat the same question type. If you already asked about \
purpose, ask about something completely different next.

Generate {num_questions} questions.

File: {file_path}
```
{content}
```

Return ONLY a JSON array: [{{"question": "...", "category": "..."}}]
The "category" is a short label you choose (freeform, not from a fixed list).
Do not include any other text."""

ANSWER_GENERATION_PROMPT = """\
Answer the following question about a source code file.
Be specific — reference actual function names, class names, and logic \
from the code. Target 100-400 tokens. Do not reproduce the entire file.

File: {file_path}
```
{content}
```

Question: {question}

Answer:"""

# Files that should be skipped for QA generation
_SKIP_EXTENSIONS = {
    ".lock", ".min.js", ".min.css", ".map", ".svg", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".pyc", ".pyo",
    ".so", ".o", ".a", ".dylib", ".exe", ".bin", ".pb", ".onnx", ".pkl",
    ".npy", ".npz", ".h5", ".hdf5", ".parquet", ".arrow", ".db", ".sqlite",
}

_SKIP_NAMES = {
    "package-lock.json", "yarn.lock", "Cargo.lock", "go.sum",
    "poetry.lock", "composer.lock", "pnpm-lock.yaml",
}

# Conservative chars-per-token estimate for filtering. Lower value = more
# aggressive filtering, avoiding QA generation on files that will exceed
# max_seq_len tokens and get multi-chunked (discarded by training join).
_CHARS_PER_TOKEN = 3.0


_TINY_ROUTE_LANGUAGES = {
    "json", "yaml", "yml", "toml", "ini", "cfg", "conf",
    "sql", "shell", "bash", "sh", "makefile", "dockerfile",
    "markdown", "md", "txt", "rst", "csv", "tsv", "xml",
}
_TINY_ROUTE_NAMES = {
    "setup.py", "setup.cfg", "requirements.txt", "pyproject.toml",
    "package.json", "tsconfig.json", "Makefile", "Dockerfile",
    "LICENSE", "MANIFEST.in", ".gitignore", ".dockerignore",
}
_TINY_ROUTE_STEMS = {"__init__", "setup", "conftest", "__main__"}


def _is_tiny_routable(path: str, language: str, size_bytes: int) -> bool:
    """Check if file should be routed to fast tier (same logic as descriptions)."""
    p = Path(path)
    if size_bytes <= 1500:
        return True
    if language.lower() in _TINY_ROUTE_LANGUAGES:
        return True
    if p.name in _TINY_ROUTE_NAMES:
        return True
    if p.stem in _TINY_ROUTE_STEMS:
        return True
    return (
        "test" in p.stem.lower()
        or p.parent.name in ("tests", "test", "__tests__")
    )


def _should_skip_file(
    file_path: str, content: str, max_seq_len: int,
) -> str | None:
    """Return skip reason or None if file should be processed."""
    p = Path(file_path)

    if p.suffix in _SKIP_EXTENSIONS:
        return "skip_extension"
    if p.name in _SKIP_NAMES:
        return "skip_name"

    lines = content.strip().split("\n")
    if len(lines) < 10:
        return "too_short"

    # Skip files that would exceed max_seq_len tokens
    estimated_tokens = len(content) / _CHARS_PER_TOKEN
    if estimated_tokens > max_seq_len:
        return "too_long"

    # Skip __init__.py with <5 lines of real code
    if p.name == "__init__.py" and len([x for x in lines if x.strip()]) < 5:
        return "trivial_init"

    return None


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

    try:
        questions = json.loads(text[start:end])
        if not isinstance(questions, list):
            return None
        return [q for q in questions if isinstance(q, dict) and q.get("question")]
    except json.JSONDecodeError:
        return None


async def process_single_file(
    client, file_path: str, content: str, num_questions: int,
) -> list[dict]:
    """Generate QA pairs for a single file."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        file_path=file_path,
        content=content[:8000],
        num_questions=num_questions,
    )

    try:
        text = await client.generate(prompt, max_tokens=1024, temperature=0.7)
        questions = _parse_questions_json((text or "").strip())
    except Exception:
        questions = None

    # Retry once on failure
    if not questions:
        try:
            text = await client.generate(prompt, max_tokens=1024, temperature=0.8)
            questions = _parse_questions_json((text or "").strip())
        except Exception:
            return []

    if not questions:
        return []

    # Generate answers
    results = []
    for q in questions[:num_questions]:
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        if not question_text or len(question_text) < 8:
            continue

        answer_prompt = ANSWER_GENERATION_PROMPT.format(
            file_path=file_path,
            content=content[:8000],
            question=question_text,
        )

        try:
            answer = (await client.generate(
                answer_prompt, max_tokens=512, temperature=0.3,
            ) or "").strip()
        except Exception:
            continue

        if not answer or len(answer.split()) < 10:
            continue

        results.append({
            "question": question_text,
            "answer": answer,
            "category": category,
        })

    return results


async def process_repo(
    repo_dir: Path,
    output_dir: Path,
    client_primary,
    client_fast,
    max_files: int,
    max_qa_per_file: int,
    max_seq_len: int,
    model_primary: str,
    model_fast: str,
) -> dict:
    """Process a single repo: extract files, generate QA pairs, write JSONL."""
    from bgkit.data.repo_processing import extract_repo_snapshot

    repo_name = f"{repo_dir.parent.name}/{repo_dir.name}"
    out_path = output_dir / repo_dir.parent.name / repo_dir.name / "qa_pairs.jsonl"

    # Skip if already done
    if out_path.exists():
        return {"repo": repo_name, "status": "skipped", "pairs": 0}

    try:
        snapshot = extract_repo_snapshot(repo_dir)
    except Exception as e:
        return {"repo": repo_name, "status": f"extract_error: {e}", "pairs": 0}

    commit_sha = snapshot.commit_sha
    files = snapshot.files

    # Filter and select files
    eligible = []
    for f in files:
        fp = f.path
        content = f.content
        if not content:
            continue
        skip = _should_skip_file(fp, content, max_seq_len)
        if skip:
            continue
        eligible.append(f)

    if not eligible:
        return {"repo": repo_name, "status": "no_eligible_files", "pairs": 0}

    # Limit files per repo
    if len(eligible) > max_files:
        import random
        rng = random.Random(hash(repo_name))
        eligible = rng.sample(eligible, max_files)

    # Generate QA pairs
    total_pairs = 0
    records = []

    for f in eligible:
        fp = f.path
        content = f.content
        language = f.language or Path(fp).suffix.lstrip(".")
        size = len(content.encode("utf-8", errors="replace"))

        # Route to appropriate tier
        if _is_tiny_routable(fp, language, size) and client_fast is not None:
            client = client_fast
            model_id = model_fast
            tier = "fast"
        else:
            client = client_primary
            model_id = model_primary
            tier = "primary"

        qa_pairs = await process_single_file(
            client, fp, content, max_qa_per_file,
        )

        for qa in qa_pairs:
            records.append({
                "repo_path": repo_name,
                "file_path": fp,
                "commit_sha": commit_sha,
                "question": qa["question"],
                "answer": qa["answer"],
                "category": qa["category"],
                "question_words": len(qa["question"].split()),
                "answer_words": len(qa["answer"].split()),
                "model_id": model_id,
                "generation_tier": tier,
                "prompt_version": 1,
            })
            total_pairs += 1

    if records:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(".partial.jsonl")
        with open(partial, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        partial.rename(out_path)

    return {"repo": repo_name, "status": "done", "pairs": total_pairs}


async def run_generation(args):
    """Main generation loop."""
    from bgkit.inference import InferenceConfig, LlamaClient

    repos_dir = args.repos_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect repos, shuffled to avoid alphabetical bias on partial runs
    from bgkit.utils.git_utils import collect_repo_paths

    all_repos = collect_repo_paths(repos_dir, shuffle_seed=42)
    repo_dirs_done = []
    repo_dirs_todo = []
    for repo_dir in all_repos:
        qa_path = output_dir / repo_dir.parent.name / repo_dir.name / "qa_pairs.jsonl"
        if qa_path.exists():
            repo_dirs_done.append(repo_dir)
        else:
            repo_dirs_todo.append(repo_dir)

    # Process unfinished repos first for better diversity on partial runs
    repo_dirs = repo_dirs_todo + repo_dirs_done
    print(
        f"Found {len(repo_dirs)} repos in {repos_dir} "
        f"({len(repo_dirs_todo)} pending, {len(repo_dirs_done)} already done)"
    )

    # Initialize clients
    client_primary = LlamaClient(InferenceConfig(
        base_url=args.server_url_primary, max_concurrent=args.workers,
    ))
    await client_primary.wait_ready()
    print(f"Primary server ready: {args.server_url_primary}")

    client_fast = None
    if args.server_url_fast:
        try:
            client_fast = LlamaClient(InferenceConfig(
                base_url=args.server_url_fast, max_concurrent=args.workers,
            ))
            await client_fast.wait_ready()
            print(f"Fast server ready: {args.server_url_fast}")
        except Exception as e:
            print(f"Fast server not available ({e}), using primary only")

    model_primary = os.environ.get("VLLM_MODEL_PRIMARY", "gpt-oss-20b")
    model_fast = os.environ.get("VLLM_MODEL_FAST", "qwen3.5-0.8b")

    # Process repos
    start_time = time.time()
    total_pairs = 0
    skipped = 0
    errors = 0

    sem = asyncio.Semaphore(args.workers)

    async def process_with_sem(repo_dir):
        async with sem:
            return await process_repo(
                repo_dir, output_dir,
                client_primary, client_fast,
                args.max_files_per_repo, args.max_qa_per_file,
                args.max_seq_len,
                model_primary, model_fast,
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
                continue
            if r["status"] == "skipped":
                skipped += 1
            elif r["status"] == "done":
                total_pairs += r["pairs"]
            else:
                errors += 1

        elapsed = time.time() - start_time
        processed = batch_start + len(batch)
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{processed}/{len(repo_dirs)}] "
            f"{total_pairs} pairs, {skipped} skipped, {errors} errors "
            f"({rate:.1f} repos/s)"
        )

    elapsed = time.time() - start_time
    print(f"\nDone: {total_pairs} QA pairs from {len(repo_dirs)} repos "
          f"({skipped} skipped, {errors} errors) in {elapsed:.0f}s")

    await client_primary.close()
    if client_fast:
        await client_fast.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs for query-conditioned compression objective"
    )
    parser.add_argument("--repos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files-per-repo", type=int, default=30)
    parser.add_argument("--max-qa-per-file", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--server-url-primary", type=str,
        default=os.environ.get("VLLM_URL", "http://localhost:8090"),
    )
    parser.add_argument(
        "--server-url-fast", type=str,
        default=os.environ.get("VLLM_URL_FAST", "http://localhost:8091"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
