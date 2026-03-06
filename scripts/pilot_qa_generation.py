#!/usr/bin/env python3
"""Pilot QA pair generation for prompt diversity validation.

Generates (question, answer) pairs for ~50 diverse source files using vLLM.
Outputs a single JSONL for manual review of question diversity and answer quality.

Usage:
    python scripts/pilot_qa_generation.py \
        --repos-dir $DATA_DIR/repos/ \
        --output pilot_qa_output.jsonl \
        --num-files 50 \
        --server-url http://localhost:8090/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
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

Generate 3-5 questions.

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


def select_diverse_files(repos_dir: Path, num_files: int, seed: int = 42) -> list[dict]:
    """Select diverse files from available repos for pilot evaluation."""
    rng = random.Random(seed)

    # Collect repo dirs
    repo_dirs = []
    for owner_dir in sorted(repos_dir.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            repo_dirs.append(repo_dir)

    if not repo_dirs:
        print("ERROR: No repos found", file=sys.stderr)
        sys.exit(1)

    # Sample repos
    sample_repos = rng.sample(repo_dirs, min(num_files * 2, len(repo_dirs)))

    # Collect candidate files
    candidates = []
    skip_extensions = {".lock", ".min.js", ".min.css", ".map", ".svg", ".png", ".jpg",
                       ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".pyc",
                       ".pyo", ".so", ".o", ".a", ".dylib", ".exe", ".bin"}
    skip_names = {"package-lock.json", "yarn.lock", "Cargo.lock", "go.sum",
                  "poetry.lock", "composer.lock"}

    for repo_dir in sample_repos:
        for fpath in repo_dir.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix in skip_extensions:
                continue
            if fpath.name in skip_names:
                continue
            if ".git" in fpath.parts:
                continue

            # Skip tiny files
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            if size < 200 or size > 50000:
                continue

            # Read and check for binary
            try:
                content = fpath.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue

            lines = content.strip().split("\n")
            if len(lines) < 10:
                continue

            rel_path = fpath.relative_to(repo_dir)
            repo_name = f"{repo_dir.parent.name}/{repo_dir.name}"
            candidates.append({
                "repo_path": repo_name,
                "file_path": str(rel_path),
                "content": content,
                "language": fpath.suffix.lstrip(".") or "text",
                "num_lines": len(lines),
            })

        if len(candidates) >= num_files * 3:
            break

    # Sample diverse selection
    selected = rng.sample(candidates, min(num_files, len(candidates)))
    print(f"Selected {len(selected)} files from {len(sample_repos)} repos")
    return selected


async def generate_questions(client, file_info: dict) -> list[dict] | None:
    """Generate questions for a single file using vLLM."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        file_path=file_info["file_path"],
        content=file_info["content"][:8000],  # Truncate very long files
    )

    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()

        # Try to parse JSON from response
        # Handle common issues: markdown fences, extra text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Find the JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        questions = json.loads(text)
        if not isinstance(questions, list):
            return None
        return questions
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Question generation failed for {file_info['file_path']}: {e}")
        return None


async def generate_answer(client, file_info: dict, question: str) -> str | None:
    """Generate answer for a question about a file."""
    prompt = ANSWER_GENERATION_PROMPT.format(
        file_path=file_info["file_path"],
        content=file_info["content"][:8000],
        question=question,
    )

    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Answer generation failed: {e}")
        return None


async def run_pilot(files: list[dict], server_url: str, output_path: Path):
    """Run the pilot QA generation."""
    from bgkit.inference import LlamaClient

    client = LlamaClient(base_url=server_url, max_concurrent=4)
    await client.wait_for_ready()

    results = []
    for i, file_info in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {file_info['repo_path']}/{file_info['file_path']}")

        questions = await generate_questions(client, file_info)
        if not questions:
            print("  No questions generated (parse failure)")
            continue

        print(f"  Generated {len(questions)} questions")

        for q in questions:
            question_text = q.get("question", "")
            category = q.get("category", "unknown")
            if not question_text:
                continue

            answer = await generate_answer(client, file_info, question_text)
            if not answer:
                continue

            results.append({
                "repo_path": file_info["repo_path"],
                "file_path": file_info["file_path"],
                "language": file_info["language"],
                "num_lines": file_info["num_lines"],
                "question": question_text,
                "category": category,
                "answer": answer,
                "answer_tokens": len(answer.split()),  # rough estimate
                "content_preview": file_info["content"][:500],
            })

    # Write output
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(results)} QA pairs to {output_path}")

    # Summary stats
    categories = {}
    for r in results:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory diversity: {len(categories)} unique categories")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cat}: {count}")

    await client.close()


def main():
    parser = argparse.ArgumentParser(description="Pilot QA pair generation")
    parser.add_argument("--repos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pilot_qa_output.jsonl"))
    parser.add_argument("--num-files", type=int, default=50)
    parser.add_argument("--server-url", type=str, default="http://localhost:8090/v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = select_diverse_files(args.repos_dir, args.num_files, args.seed)
    asyncio.run(run_pilot(files, args.server_url, args.output))


if __name__ == "__main__":
    main()
