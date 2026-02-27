#!/usr/bin/env python3
"""Batch description generation for repo files, modules, and repos.

Dual-backend description generation using Claude CLI (haiku) and/or local
llama-server instances. Produces per-repo JSONL files with file-level,
module-level, and repo-level descriptions.

Usage:
    python scripts/generate_descriptions.py \
        --repos-dir data/repos/ \
        --output-dir data/descriptions/ \
        --backend mixed \
        --workers 12
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
from bgkit.inference import InferenceConfig, LlamaClient
from bgkit.inference.models import resolve_profile
from bgkit.utils.git_utils import is_git_repo

logger = structlog.get_logger()

# Rate-limit haiku calls
_HAIKU_SEMAPHORE = Semaphore(10)

# Maximum file content length to include in prompts (characters)
MAX_CONTENT_CHARS = 8000

# Maximum number of file descriptions to include in module/repo prompts
MAX_FILE_DESCS_IN_PROMPT = 30

# Maximum characters for embedded descriptions in module/repo prompts
MAX_DESC_CHARS_IN_PROMPT = 200

# Maximum files per repo (sort by path, take first N)
MAX_FILES_PER_REPO = 1000

# Prompt format version — increment when prompt builders change
PROMPT_VERSION = 4

# Llama-server clients (initialized by init_local_client)
_llama_client_large: LlamaClient | None = None
_llama_client_small: LlamaClient | None = None
_llama_client_tiny: LlamaClient | None = None

# Model IDs per tier, populated by init_local_client after auto-detection
_tier_models: dict[str, str] = {}

# Files above this char count go to the small model (unless routed to tiny)
SMALL_MODEL_CHAR_THRESHOLD = 3000


# Languages where a tiny model suffices (config, data, markup, scripting — low complexity)
_TINY_LANGUAGES: frozenset[str] = frozenset({
    "JSON", "YAML", "TOML", "XML", "Markdown", "reStructuredText",
    "HTML", "CSS", "SQL", "Dockerfile", "Terraform", "Nix",
    "Protocol Buffers", "CMake", "Gradle",
    "Shell", "Batchfile", "PowerShell", "Makefile",
    "INI", "Properties", "Dotenv",
    "Plain Text", "CSV", "TSV",
})

# Path patterns that indicate easy-to-describe files (test, config, boilerplate, generated)
_TINY_PATH_STEMS: frozenset[str] = frozenset({
    "__init__", "setup", "conftest",
})
_TINY_PATH_NAMES: frozenset[str] = frozenset({
    "setup.cfg", "pyproject.toml", "package.json", "package-lock.json",
    "tsconfig.json", "tslint.json", ".eslintrc", ".eslintrc.json", ".eslintrc.js",
    ".prettierrc", ".prettierrc.json", ".babelrc", ".editorconfig",
    "requirements.txt", "Pipfile", "Pipfile.lock", "Cargo.lock", "go.sum",
    "yarn.lock", "pnpm-lock.yaml", "composer.lock", "Gemfile.lock",
    "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE",
    "CHANGELOG.md", "CHANGELOG", "CHANGES.md",
    "Makefile", "Rakefile", "Justfile", "justfile",
    "Dockerfile", ".dockerignore", ".gitignore", ".gitattributes",
    ".env.example", ".env.sample", ".env.template",
    "README.md", "README.rst", "README.txt", "README",
})

# Short files are trivial to describe regardless of language
TINY_SIZE_THRESHOLD = 1500


def _is_tiny_routable(path: str, language: str | None, size_bytes: int) -> bool:
    """Return True if a file is simple enough for the tiny model."""
    if size_bytes <= TINY_SIZE_THRESHOLD:
        return True
    if language and language in _TINY_LANGUAGES:
        return True
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    if name in _TINY_PATH_NAMES:
        return True
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if stem in _TINY_PATH_STEMS:
        return True
    # test files, migrations, generated code
    if stem.startswith("test_") or stem.endswith("_test") or stem.endswith(".spec"):
        return True
    lower_path = path.lower()
    if "/test/" in lower_path or "/tests/" in lower_path or "/spec/" in lower_path:
        return True
    if "/migrations/" in lower_path or ".generated." in lower_path:
        return True
    return name.endswith(".lock") or ".lock." in name


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
# Local llama-server backend
# ---------------------------------------------------------------------------

def _make_client(
    base_url: str,
    max_concurrent: int,
    readiness_timeout: float,
    label: str,
) -> LlamaClient:
    """Create, verify, and auto-configure a single LlamaClient.

    max_concurrent should match the server's --parallel slot count to avoid
    sending more requests than the server can handle.

    After the server is healthy, queries /v1/models to detect the loaded model
    and resolves the appropriate ModelProfile (thinking-tag stripping, etc.).
    """
    config = InferenceConfig(
        base_url=base_url,
        max_concurrent=max_concurrent,
    )
    client = LlamaClient(config)
    logger.info("waiting_for_llama_server", label=label, url=base_url)
    if not client.wait_ready_sync(timeout=readiness_timeout):
        print(
            f"ERROR: llama-server ({label}) at {base_url} not ready after {readiness_timeout}s. "
            f"Start it with: make llama-server",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-detect model and apply the correct profile
    model_id = client.detect_model_sync()
    if not model_id:
        print(
            f"ERROR: could not detect model from llama-server ({label}) at {base_url}. "
            f"Is /v1/models endpoint available?",
            file=sys.stderr,
        )
        sys.exit(1)
    profile = resolve_profile(model_id)
    client.apply_profile(profile)
    _tier_models[label] = model_id
    logger.info("auto_detected_model", label=label, model=model_id, profile=profile.name)

    client.warmup_sync()
    logger.info("llama_server_ready", label=label, url=base_url)
    return client


def init_local_client(
    url_large: str | None = None,
    url_small: str | None = None,
    url_tiny: str | None = None,
    readiness_timeout: float = 120.0,
) -> None:
    """Initialize all three llama-server clients and verify readiness.

    All three model servers (large, small, tiny) must be running. Exits if any
    is unreachable. Start all with: make llama-server
    """
    global _llama_client_large, _llama_client_small, _llama_client_tiny

    def _parallel_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    parallel_large = _parallel_env("LLAMA_PARALLEL_LARGE", 2)
    parallel_small = _parallel_env("LLAMA_PARALLEL_SMALL", 2)
    parallel_tiny = _parallel_env("LLAMA_PARALLEL_TINY", 8)

    resolved_large = url_large or os.environ.get("LLAMA_URL", "http://localhost:8080")
    resolved_small = url_small or os.environ.get("LLAMA_URL_SMALL", "http://localhost:8081")
    resolved_tiny = url_tiny or os.environ.get("LLAMA_URL_TINY", "http://localhost:8082")

    _llama_client_large = _make_client(
        resolved_large, parallel_large, readiness_timeout, "large"
    )
    _llama_client_small = _make_client(
        resolved_small, parallel_small, readiness_timeout, "small"
    )
    _llama_client_tiny = _make_client(
        resolved_tiny, parallel_tiny, readiness_timeout, "tiny"
    )


def _pick_tier(
    content_chars: int,
    file_path: str = "",
    language: str | None = None,
    size_bytes: int = 0,
) -> str:
    """Return the tier label for a file: 'tiny', 'small', or 'large'."""
    if file_path and _is_tiny_routable(file_path, language, size_bytes):
        return "tiny"
    if content_chars > SMALL_MODEL_CHAR_THRESHOLD:
        return "small"
    return "large"


def _model_for_backend(backend_label: str) -> str:
    """Return the model ID for a backend label like 'local-tiny'."""
    tier = backend_label.removeprefix("local-")
    return _tier_models.get(tier, "")


def _client_for_tier(tier: str) -> LlamaClient:
    """Return the LlamaClient for the given tier."""
    if tier == "tiny":
        return _llama_client_tiny
    if tier == "small":
        return _llama_client_small
    return _llama_client_large


def call_local(
    prompt: str,
    max_new_tokens: int = 512,
    content_chars: int = 0,
    file_path: str = "",
    language: str | None = None,
    size_bytes: int = 0,
) -> tuple[str | None, str]:
    """Generate a description using the appropriate llama-server tier.

    Returns (description, backend_label).
    """
    from bgkit.inference.client import ContextOverflowError

    tier = _pick_tier(content_chars, file_path, language, size_bytes)
    client = _client_for_tier(tier)
    try:
        return (client.generate_sync(prompt, max_tokens=max_new_tokens), f"local-{tier}")
    except ContextOverflowError:
        if tier != "tiny":
            return (None, f"local-{tier}")
        # Tiny has the smallest per-slot context; fall back to small.
        logger.info("context_overflow_fallback", from_tier="tiny", to_tier="small")
        try:
            result = _llama_client_small.generate_sync(prompt, max_tokens=max_new_tokens)
            return (result, "local-small")
        except ContextOverflowError:
            return (None, "local-tiny")


def call_local_batch(
    prompts: list[str],
    tiers: list[str] | None = None,
    max_new_tokens: int = 512,
) -> list[tuple[str | None, str]]:
    """Generate descriptions for multiple prompts with tier-based routing.

    Each prompt is routed to its tier's model. All three models' semaphores
    independently control their own concurrency.
    Returns list of (description, backend_label) tuples.
    """
    if tiers is None:
        tiers = ["large"] * len(prompts)

    import asyncio

    from bgkit.inference.client import ContextOverflowError

    async def _generate_routed() -> list[tuple[str | None, str]]:
        async def _one(prompt: str, tier: str) -> tuple[str | None, str]:
            client = _client_for_tier(tier)
            try:
                result = await client.generate(prompt, max_tokens=max_new_tokens)
            except ContextOverflowError:
                if tier != "tiny":
                    return (None, f"local-{tier}")
                # Tiny has the smallest per-slot context; fall back to small.
                logger.info(
                    "context_overflow_fallback", from_tier="tiny", to_tier="small"
                )
                try:
                    result = await _llama_client_small.generate(
                        prompt, max_tokens=max_new_tokens
                    )
                except ContextOverflowError:
                    return (None, "local-tiny")
                return (result, "local-small")
            return (result, f"local-{tier}")

        return await asyncio.gather(
            *[_one(p, t) for p, t in zip(prompts, tiers, strict=True)]
        )

    from bgkit.inference.client import _run_sync

    return _run_sync(_generate_routed())


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

_CYCLE_LOCK = Lock()


def generate_description(
    prompt: str,
    backend: str,
    backend_cycle: itertools.cycle | None = None,
    content_chars: int = 0,
    file_path: str = "",
    language: str | None = None,
    size_bytes: int = 0,
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
        return call_local(
            prompt, content_chars=content_chars, file_path=file_path,
            language=language, size_bytes=size_bytes,
        )
    else:
        return (None, chosen)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_file_prompt(file_path: str, content: str, language: str | None) -> str:
    """Build a prompt asking for a file-level description."""
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n... (truncated)"

    lang_note = f" ({language})" if language else ""
    return (
        f"File: {file_path}{lang_note}\n\n"
        f"```\n{content}\n```\n\n"
        f"Write a single dense paragraph describing this file. "
        f"Include: what it does, the names of key exports (classes, functions, constants), "
        f"and what it imports or depends on. "
        f"Omit any category that doesn't apply. "
        f"No headers, bullet points, or labels — just a compact paragraph "
        f"where every word carries information. Use actual identifier names from the code."
    )


def _truncate_desc(desc: str) -> str:
    """Truncate a description for embedding in module/repo prompts."""
    if len(desc) <= MAX_DESC_CHARS_IN_PROMPT:
        return desc
    return desc[:MAX_DESC_CHARS_IN_PROMPT] + "..."


def build_module_prompt(
    module_path: str,
    file_descriptions: list[dict],
    skeleton_text: str | None = None,
) -> str:
    """Build a prompt asking for a module-level description."""
    parts = [f"Module: {module_path}\n\nFiles:\n"]

    for fd in file_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
        parts.append(f"- {fd['file_path']}: {_truncate_desc(fd['description'])}")

    if skeleton_text:
        parts.append(f"\nSkeleton:\n{skeleton_text}")

    parts.append(
        "\nWrite a single dense paragraph describing this module. "
        "Include: what it provides, its public API (by name), "
        "how its files relate to each other, and what external packages it depends on. "
        "No headers or bullet points — just a compact paragraph."
    )
    return "\n".join(parts)


def build_repo_prompt(
    repo_path: str,
    module_descriptions: list[dict],
    file_descriptions: list[dict],
) -> str:
    """Build a prompt asking for a repo-level description."""
    parts = [f"Project: {repo_path}\n\n"]

    if module_descriptions:
        parts.append("Modules:")
        for md in module_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
            parts.append(f"- {md['module_path']}: {_truncate_desc(md['description'])}")
        parts.append("")

    if file_descriptions:
        parts.append("Key files:")
        for fd in file_descriptions[:MAX_FILE_DESCS_IN_PROMPT]:
            parts.append(f"- {fd['file_path']}: {_truncate_desc(fd['description'])}")
        parts.append("")

    parts.append(
        "Write a single dense paragraph describing this project. "
        "Include: what it does, its architecture (monorepo, client-server, library, CLI, etc.), "
        "key technologies and frameworks, and main entry points. "
        "No headers or bullet points — just a compact paragraph."
    )
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

        logger.info(
            "repo_snapshot", repo=rel_key, files=snapshot.total_files,
            skipped_binary=snapshot.skipped_binary,
            skipped_large=snapshot.skipped_large,
            skipped_pattern=snapshot.skipped_pattern,
        )

        if not snapshot.files:
            return {"repo": rel_key, "status": "empty"}

        # Cap file count per repo for deterministic, bounded processing
        if len(snapshot.files) > MAX_FILES_PER_REPO:
            logger.warning(
                "repo_truncated", repo=rel_key,
                total=len(snapshot.files), kept=MAX_FILES_PER_REPO,
            )
            snapshot.files = snapshot.files[:MAX_FILES_PER_REPO]

        commit_sha = snapshot.commit_sha
        records: list[dict] = []

        # Load structural skeletons if available
        module_skeletons: dict[str, str] = {}
        if structural_dir is not None:
            module_skeletons = _load_structural_skeletons(structural_dir, rel_key)

        # --- File-level descriptions ---
        file_descriptions: list[dict] = []

        if scope in ("file", "all"):
            # Build all prompts first, then generate in batch for local backend
            file_prompts = [
                (fr, build_file_prompt(fr.path, fr.content, fr.language))
                for fr in snapshot.files
            ]

            if backend == "local":
                # Batch: all prompts sent concurrently, routed by tier
                file_tiers = [
                    _pick_tier(len(fr.content), fr.path, fr.language, fr.size_bytes)
                    for fr, _ in file_prompts
                ]
                results = call_local_batch(
                    [p for _, p in file_prompts],
                    tiers=file_tiers,
                )
                for (fr, _prompt), (desc, backend_label) in zip(
                    file_prompts, results, strict=True
                ):
                    if not desc:
                        logger.warning(
                            "file_description_failed", file_path=fr.path,
                            content_chars=len(fr.content), repo=rel_key,
                        )
                    else:
                        rec = {
                            "scope": "file",
                            "file_path": fr.path,
                            "commit_sha": commit_sha,
                            "description": desc,
                            "language": fr.language or "",
                            "backend": backend_label,
                            "model": _model_for_backend(backend_label),
                            "prompt_version": PROMPT_VERSION,
                        }
                        records.append(rec)
                        file_descriptions.append(rec)
            else:
                # Sequential: haiku/mixed backends use per-request dispatch
                for fr, prompt in file_prompts:
                    desc, used_backend = generate_description(
                        prompt, backend, backend_cycle,
                        content_chars=len(fr.content),
                        file_path=fr.path,
                        language=fr.language,
                        size_bytes=fr.size_bytes,
                    )
                    if desc is None:
                        logger.warning(
                            "file_description_failed", file_path=fr.path,
                            content_chars=len(fr.content), repo=rel_key,
                        )
                    elif desc:
                        rec = {
                            "scope": "file",
                            "file_path": fr.path,
                            "commit_sha": commit_sha,
                            "description": desc,
                            "language": fr.language or "",
                            "backend": used_backend,
                            "model": _model_for_backend(used_backend),
                            "prompt_version": PROMPT_VERSION,
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
                        "model": _model_for_backend(used_backend),
                        "prompt_version": PROMPT_VERSION,
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
                    "model": _model_for_backend(used_backend),
                    "prompt_version": PROMPT_VERSION,
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
        description="Generate file/module/repo descriptions using Claude and/or llama-server."
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
        "--llama-url-large", type=str, default=None,
        help="URL for the large llama-server (default: LLAMA_URL env or http://localhost:8080)",
    )
    parser.add_argument(
        "--llama-url-small", type=str, default=None,
        help="URL for the small llama-server (default: LLAMA_URL_SMALL env or http://localhost:8081)",
    )
    parser.add_argument(
        "--llama-url-tiny", type=str, default=None,
        help="URL for the tiny llama-server (default: LLAMA_URL_TINY env or http://localhost:8082)",
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
        "--workers", type=int, default=12,
        help="Number of parallel workers (default: 12)",
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

    # Initialize llama-server clients if needed
    if args.backend in ("local", "mixed"):
        init_local_client(
            url_large=args.llama_url_large,
            url_small=args.llama_url_small,
            url_tiny=args.llama_url_tiny,
        )

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

    # Use ThreadPoolExecutor since the work is I/O-bound (HTTP calls to llama-server)
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
