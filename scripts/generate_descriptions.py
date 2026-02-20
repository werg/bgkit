#!/usr/bin/env python3
"""Batch description generation for repo files, modules, and repos.

Dual-backend description generation using Claude CLI (haiku) and/or a local
Qwen model. Produces per-repo JSONL files with file-level, module-level,
and repo-level descriptions.

Usage:
    python scripts/generate_descriptions.py \
        --repos-dir data/repos/ \
        --output-dir data/descriptions/ \
        --backend mixed \
        --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from threading import Lock, Semaphore

import structlog

from bgkit.data.repo_processing import extract_repo_snapshot
from bgkit.utils.git_utils import is_git_repo

logger = structlog.get_logger()

# Rate-limit haiku calls
_HAIKU_SEMAPHORE = Semaphore(10)

# Maximum file content length to include in prompts (characters)
MAX_CONTENT_CHARS = 12000

# Maximum number of file descriptions to include in module/repo prompts
MAX_FILE_DESCS_IN_PROMPT = 30


# ---------------------------------------------------------------------------
# Claude CLI (haiku) backend
# ---------------------------------------------------------------------------

def call_claude(prompt: str, max_retries: int = 3) -> str | None:
    """Call Claude CLI and return the text response."""
    for attempt in range(max_retries):
        try:
            _HAIKU_SEMAPHORE.acquire()
            try:
                result = subprocess.run(
                    ["claude", "-p", "--model", "haiku", "--output-format", "json"],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            finally:
                _HAIKU_SEMAPHORE.release()

            if result.returncode != 0:
                logger.warning(
                    "claude_cli_error",
                    attempt=attempt + 1,
                    stderr=result.stderr[:200],
                )
                _backoff(attempt)
                continue

            response = json.loads(result.stdout)
            if isinstance(response, dict) and "result" in response:
                return response["result"]
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
                return str(content)
            return result.stdout

        except subprocess.TimeoutExpired:
            logger.warning("claude_cli_timeout", attempt=attempt + 1)
            _backoff(attempt)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("claude_parse_error", attempt=attempt + 1, error=str(e))
            _backoff(attempt)

    return None


def _backoff(attempt: int) -> None:
    """Exponential backoff: 2^attempt seconds."""
    time.sleep(min(2 ** attempt, 30))


# ---------------------------------------------------------------------------
# Local model backend
# ---------------------------------------------------------------------------

_local_model = None
_local_tokenizer = None


def load_local_model(model_name: str) -> None:
    """Load the local model and tokenizer into module-level globals."""
    global _local_model, _local_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading_local_model", model=model_name)
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _local_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _local_model.eval()
    logger.info("local_model_loaded", model=model_name)


def call_local(prompt: str, max_new_tokens: int = 512) -> str | None:
    """Generate a description using the local model."""
    import torch

    if _local_model is None or _local_tokenizer is None:
        return None

    try:
        messages = [{"role": "user", "content": prompt}]
        text = _local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _local_tokenizer(text, return_tensors="pt").to(_local_model.device)

        with torch.no_grad():
            outputs = _local_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = _local_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result.strip() if result else None

    except Exception as e:
        logger.warning("local_model_error", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

_CYCLE_LOCK = Lock()


def generate_description(
    prompt: str,
    backend: str,
    backend_cycle: itertools.cycle | None = None,
) -> tuple[str | None, str]:
    """Generate a description using the specified backend.

    Returns (description_text, backend_used).
    """
    if backend == "mixed" and backend_cycle is not None:
        with _CYCLE_LOCK:
            chosen = next(backend_cycle)
    else:
        chosen = backend

    if chosen == "haiku":
        result = call_claude(prompt)
        return (result, "haiku")
    elif chosen == "local":
        result = call_local(prompt)
        return (result, "local")
    else:
        return (None, chosen)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_file_prompt(file_path: str, content: str, language: str | None) -> str:
    """Build a prompt asking for a file-level description."""
    # Truncate content if too long
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n... (truncated)"

    lang_note = f" ({language})" if language else ""
    return (
        f"Describe what the following source file does in 1-3 sentences. "
        f"Be specific about its purpose, key functions, and how it fits into "
        f"a larger project.\n\n"
        f"File: {file_path}{lang_note}\n\n"
        f"```\n{content}\n```\n\n"
        f"Description:"
    )


def build_module_prompt(
    module_path: str,
    file_descriptions: list[dict],
    skeleton_text: str | None = None,
) -> str:
    """Build a prompt asking for a module-level description."""
    parts = [
        f"Describe what the directory/module '{module_path}' provides in 2-4 sentences. "
        f"Focus on the module's purpose, its public API, and how its files work together.\n\n"
        f"Files in this module:\n"
    ]

    for fd in file_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
        parts.append(f"- {fd['file_path']}: {fd['description']}")

    if skeleton_text:
        parts.append(f"\nStructural skeleton:\n{skeleton_text}")

    parts.append("\nModule description:")
    return "\n".join(parts)


def build_repo_prompt(
    repo_path: str,
    module_descriptions: list[dict],
    file_descriptions: list[dict],
) -> str:
    """Build a prompt asking for a repo-level description."""
    parts = [
        f"Describe what the project at '{repo_path}' does in 2-5 sentences. "
        f"Focus on its purpose, main functionality, technology stack, and target users.\n\n"
    ]

    if module_descriptions:
        parts.append("Modules:")
        for md in module_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
            parts.append(f"- {md['module_path']}: {md['description']}")
        parts.append("")

    if file_descriptions:
        parts.append("Key files:")
        for fd in file_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
            parts.append(f"- {fd['file_path']}: {fd['description']}")
        parts.append("")

    parts.append("Project description:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Repo-level processing
# ---------------------------------------------------------------------------

def collect_repo_paths(repos_dir: Path, max_repos: int | None = None) -> list[Path]:
    """Collect all repo paths sorted deterministically."""
    repo_paths: list[Path] = []
    for owner_dir in sorted(repos_dir.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if is_git_repo(repo_dir):
                repo_paths.append(repo_dir)

    if max_repos is not None:
        repo_paths = repo_paths[:max_repos]

    return repo_paths


def _load_structural_skeletons(structural_dir: Path, rel_key: str) -> dict[str, str]:
    """Load skeleton texts from structural JSONL if available.

    Returns a dict mapping module_path -> skeleton text.
    """
    jsonl_path = structural_dir / f"{rel_key}.jsonl"
    skeletons: dict[str, str] = {}
    if not jsonl_path.exists():
        return skeletons

    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                mod_text = rec.get("module_summary_text", "")
                if mod_text:
                    skeletons[rec.get("file_path", "")] = mod_text
    except (json.JSONDecodeError, OSError):
        pass

    return skeletons


def process_single_repo(
    repo_path: Path,
    output_dir: Path,
    backend: str,
    scope: str,
    structural_dir: Path | None,
    backend_cycle: itertools.cycle | None,
) -> dict:
    """Process a single repo and write descriptions JSONL.

    Returns stats dict.
    """
    owner = repo_path.parent.name
    repo_name = repo_path.name
    rel_key = f"{owner}/{repo_name}"

    output_file = output_dir / f"{rel_key}.jsonl"

    # Skip if already done
    if output_file.exists():
        return {"repo": rel_key, "status": "skipped"}

    try:
        snapshot = extract_repo_snapshot(str(repo_path))
        if not snapshot.files:
            return {"repo": rel_key, "status": "empty"}

        commit_sha = snapshot.commit_sha
        records: list[dict] = []

        # Load structural skeletons if available
        module_skeletons: dict[str, str] = {}
        if structural_dir is not None:
            module_skeletons = _load_structural_skeletons(structural_dir, rel_key)

        # --- File-level descriptions ---
        file_descriptions: list[dict] = []

        if scope in ("file", "all"):
            for fr in snapshot.files:
                prompt = build_file_prompt(fr.path, fr.content, fr.language)
                desc, used_backend = generate_description(
                    prompt, backend, backend_cycle
                )
                if desc:
                    rec = {
                        "scope": "file",
                        "file_path": fr.path,
                        "commit_sha": commit_sha,
                        "description": desc,
                        "language": fr.language or "",
                        "backend": used_backend,
                    }
                    records.append(rec)
                    file_descriptions.append(rec)

        # --- Module-level descriptions ---
        module_descriptions: list[dict] = []

        if scope in ("module", "all"):
            # Group files by parent directory
            modules: dict[str, list[dict]] = {}
            # Use file_descriptions if we have them, otherwise build minimal ones
            descs_for_modules = file_descriptions
            if not descs_for_modules:
                # Create simple file descriptions for module prompts
                descs_for_modules = [
                    {
                        "file_path": fr.path,
                        "description": f"{fr.language or 'text'} file, {fr.size_bytes} bytes",
                    }
                    for fr in snapshot.files
                ]

            for fd in descs_for_modules:
                parent = str(PurePosixPath(fd["file_path"]).parent)
                if parent and parent != ".":
                    modules.setdefault(parent, []).append(fd)

            for mod_path, mod_files in sorted(modules.items()):
                skeleton_text = module_skeletons.get(mod_path)
                prompt = build_module_prompt(mod_path, mod_files, skeleton_text)
                desc, used_backend = generate_description(
                    prompt, backend, backend_cycle
                )
                if desc:
                    # Detect primary language in module
                    langs = [
                        f.get("language", "") for f in mod_files if f.get("language")
                    ]
                    primary_lang = max(set(langs), key=langs.count) if langs else ""

                    rec = {
                        "scope": "module",
                        "module_path": mod_path,
                        "commit_sha": commit_sha,
                        "description": desc,
                        "language": primary_lang,
                        "backend": used_backend,
                    }
                    records.append(rec)
                    module_descriptions.append(rec)

        # --- Repo-level description ---
        if scope in ("repo", "all"):
            # Use module descriptions if available, else file descriptions
            descs_for_repo = file_descriptions
            if not descs_for_repo:
                descs_for_repo = [
                    {
                        "file_path": fr.path,
                        "description": f"{fr.language or 'text'} file, {fr.size_bytes} bytes",
                    }
                    for fr in snapshot.files[:MAX_FILE_DESCS_IN_PROMPT]
                ]

            prompt = build_repo_prompt(rel_key, module_descriptions, descs_for_repo)
            desc, used_backend = generate_description(
                prompt, backend, backend_cycle
            )
            if desc:
                records.append({
                    "scope": "repo",
                    "commit_sha": commit_sha,
                    "description": desc,
                    "backend": used_backend,
                })

        if not records:
            return {"repo": rel_key, "status": "no_descriptions"}

        # Atomic write
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fd_num, tmp_path = tempfile.mkstemp(
            dir=str(output_file.parent), suffix=".jsonl.tmp"
        )
        try:
            with os.fdopen(fd_num, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.rename(tmp_path, str(output_file))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        return {
            "repo": rel_key,
            "status": "ok",
            "file_descs": sum(1 for r in records if r["scope"] == "file"),
            "module_descs": sum(1 for r in records if r["scope"] == "module"),
            "repo_descs": sum(1 for r in records if r["scope"] == "repo"),
        }

    except Exception:
        tb = traceback.format_exc()
        return {"repo": rel_key, "status": "error", "error": tb}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate file/module/repo descriptions using Claude and/or local model."
    )
    parser.add_argument(
        "--repos-dir", type=Path, default=Path("data/repos"),
        help="Directory containing owner/repo/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/descriptions"),
        help="Output directory for per-repo JSONL files",
    )
    parser.add_argument(
        "--backend", type=str, choices=["haiku", "local", "mixed"], default="mixed",
        help="Description generation backend (default: mixed)",
    )
    parser.add_argument(
        "--local-model", type=str, default="Qwen/Qwen3-Coder-0.6B",
        help="Local model name for the 'local' or 'mixed' backend",
    )
    parser.add_argument(
        "--structural-dir", type=Path, default=None,
        help="Directory with structural JSONL files (optional, improves module/repo prompts)",
    )
    parser.add_argument(
        "--scope", type=str, choices=["file", "module", "repo", "all"], default="all",
        help="Which description scopes to generate (default: all)",
    )
    parser.add_argument(
        "--max-repos", type=int, default=None,
        help="Process at most this many repos",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    if not args.repos_dir.is_dir():
        print(f"ERROR: {args.repos_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load local model if needed
    if args.backend in ("local", "mixed"):
        load_local_model(args.local_model)

    # Set up backend cycle for mixed mode
    backend_cycle: itertools.cycle | None = None
    if args.backend == "mixed":
        backend_cycle = itertools.cycle(["haiku", "local"])

    repo_paths = collect_repo_paths(args.repos_dir, args.max_repos)
    logger.info("collected_repos", count=len(repo_paths))

    total_ok = 0
    total_skipped = 0
    total_errors = 0
    total_file_descs = 0
    total_module_descs = 0
    total_repo_descs = 0

    # Use ThreadPoolExecutor since the work is I/O-bound (Claude calls, model inference)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_repo,
                rp,
                args.output_dir,
                args.backend,
                args.scope,
                args.structural_dir,
                backend_cycle,
            ): rp
            for rp in repo_paths
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            status = result["status"]

            if status == "ok":
                total_ok += 1
                total_file_descs += result.get("file_descs", 0)
                total_module_descs += result.get("module_descs", 0)
                total_repo_descs += result.get("repo_descs", 0)
            elif status == "skipped":
                total_skipped += 1
            elif status == "error":
                total_errors += 1
                logger.error(
                    "repo_failed", repo=result["repo"],
                    error=result.get("error", "")[:200],
                )

            if (i + 1) % 50 == 0 or (i + 1) == len(repo_paths):
                logger.info(
                    "progress",
                    done=i + 1,
                    total=len(repo_paths),
                    ok=total_ok,
                    skipped=total_skipped,
                    errors=total_errors,
                )

    logger.info(
        "generation_complete",
        repos_processed=total_ok,
        repos_skipped=total_skipped,
        repos_errored=total_errors,
        file_descriptions=total_file_descs,
        module_descriptions=total_module_descs,
        repo_descriptions=total_repo_descs,
    )


if __name__ == "__main__":
    main()
