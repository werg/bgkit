#!/usr/bin/env python3
"""Extract dependency tags from repository manifests and build a taxonomy.

Parses all planned manifest types: requirements.txt, setup.py, pyproject.toml,
package.json, Cargo.toml, go.mod, Gemfile, pom.xml, build.gradle, mix.exs.

Also extracts language tags from file extensions, Wikipedia categories from KILT
metadata, and MeSH terms from PubMedQA metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.taxonomy import TagTaxonomy

# ---------------------------------------------------------------------------
# Manifest parsers
# ---------------------------------------------------------------------------


def _parse_requirements_txt(text: str) -> list[str]:
    deps = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip version specifiers: ==, >=, <=, ~=, !=, <, >, [extras]
        name = re.split(r"[=<>!~\[;@]", line)[0].strip()
        if name:
            deps.append(name.lower())
    return deps


def _parse_setup_py(text: str) -> list[str]:
    deps = []
    # Extract install_requires list
    match = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if match:
        for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            name = re.split(r"[=<>!~\[;]", item)[0].strip()
            if name:
                deps.append(name.lower())
    return deps


def _parse_pyproject_toml(text: str) -> list[str]:
    deps = []
    # Extract [project] dependencies
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies") and "=" in stripped:
            in_deps = True
            # Handle inline list
            match = re.search(r"\[(.+)\]", stripped)
            if match:
                for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                    name = re.split(r"[=<>!~\[;]", item)[0].strip()
                    if name:
                        deps.append(name.lower())
                in_deps = False
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                continue
            match = re.search(r"['\"]([^'\"]+)['\"]", stripped)
            if match:
                name = re.split(r"[=<>!~\[;]", match.group(1))[0].strip()
                if name:
                    deps.append(name.lower())
    return deps


def _parse_package_json(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    deps = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        section = data.get(key, {})
        if isinstance(section, dict):
            deps.update(section.keys())
    return sorted(deps)


def _parse_cargo_toml(text: str) -> list[str]:
    return re.findall(r"^([A-Za-z0-9_-]+)\s*=", text, flags=re.MULTILINE)


def _parse_go_mod(text: str) -> list[str]:
    deps = []
    for match in re.finditer(r"^\s*([A-Za-z0-9_./-]+)\s+v", text, flags=re.MULTILINE):
        path = match.group(1)
        # Use last path segment as dep name
        name = path.rsplit("/", 1)[-1]
        deps.append(name)
    return deps


def _parse_gemfile(text: str) -> list[str]:
    deps = []
    for match in re.finditer(r"gem\s+['\"]([^'\"]+)['\"]", text):
        deps.append(match.group(1))
    return deps


def _parse_pom_xml(text: str) -> list[str]:
    deps = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return deps
    ns = ""
    # Detect Maven namespace
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    for dep in root.iter(f"{ns}dependency"):
        group = dep.find(f"{ns}groupId")
        artifact = dep.find(f"{ns}artifactId")
        if artifact is not None and artifact.text:
            name = artifact.text
            if group is not None and group.text:
                name = f"{group.text}:{name}"
            deps.append(name)
    return deps


def _parse_build_gradle(text: str) -> list[str]:
    deps = []
    for match in re.finditer(
        r"(?:implementation|compile|api|runtimeOnly|compileOnly|testImplementation)"
        r"\s+['\"]([^'\"]+)['\"]",
        text,
    ):
        dep = match.group(1)
        # Extract artifact from group:artifact:version format
        parts = dep.split(":")
        if len(parts) >= 2:
            deps.append(f"{parts[0]}:{parts[1]}")
        else:
            deps.append(dep)
    return deps


def _parse_mix_exs(text: str) -> list[str]:
    deps = []
    for match in re.finditer(r"\{:([a-zA-Z_][a-zA-Z0-9_]*)", text):
        deps.append(match.group(1))
    return deps


_MANIFEST_PARSERS = {
    "requirements.txt": ("Python", _parse_requirements_txt),
    "setup.py": ("Python", _parse_setup_py),
    "pyproject.toml": ("Python", _parse_pyproject_toml),
    "package.json": ("JavaScript", _parse_package_json),
    "Cargo.toml": ("Rust", _parse_cargo_toml),
    "go.mod": ("Go", _parse_go_mod),
    "Gemfile": ("Ruby", _parse_gemfile),
    "pom.xml": ("Java", _parse_pom_xml),
    "build.gradle": ("Java", _parse_build_gradle),
    "mix.exs": ("Elixir", _parse_mix_exs),
}

# Language detection by file extension
_EXTENSION_TO_LANGUAGE = {
    ".py": "Python", ".pyx": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".rs": "Rust",
    ".go": "Go",
    ".rb": "Ruby",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cxx": "C++", ".cc": "C++", ".hpp": "C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".scala": "Scala",
    ".ex": "Elixir", ".exs": "Elixir",
    ".php": "PHP",
    ".r": "R", ".R": "R",
    ".lua": "Lua",
    ".jl": "Julia",
    ".dart": "Dart",
    ".zig": "Zig",
    ".nim": "Nim",
    ".ml": "OCaml", ".mli": "OCaml",
    ".hs": "Haskell",
    ".erl": "Erlang",
    ".clj": "Clojure",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
}


def extract_repo_tags(
    root_dir: Path,
    output_dir: Path,
    min_dep_frequency: int = 50,
    min_language_frequency: int = 500,
) -> None:
    """Extract dependency and language tags from repos."""
    repo_tags: dict[str, list[str]] = defaultdict(list)
    dep_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    dep_details: list[dict] = []

    # Scan manifests
    for manifest_name, (language, parser) in _MANIFEST_PARSERS.items():
        for manifest_path in root_dir.rglob(manifest_name):
            try:
                text = manifest_path.read_text(errors="replace")
                dependencies = [dep for dep in parser(text) if dep]
            except Exception:
                continue
            repo_key = str(manifest_path.parent.relative_to(root_dir))
            tags = [f"dependency/{dep}" for dep in dependencies]
            repo_tags[repo_key].extend(tags)
            repo_tags[repo_key].append(f"language/{language}")
            dep_counts.update(set(tags))
            lang_counts[f"language/{language}"] += 1
            for dep in dependencies:
                dep_details.append({
                    "dep_name": dep,
                    "language": language,
                    "repo_path": repo_key,
                })

    # Scan file extensions for language tags
    for ext, language in _EXTENSION_TO_LANGUAGE.items():
        for file_path in root_dir.rglob(f"*{ext}"):
            try:
                repo_key = str(file_path.parent.relative_to(root_dir))
            except ValueError:
                continue
            tag = f"language/{language}"
            if tag not in repo_tags.get(repo_key, []):
                repo_tags[repo_key].append(tag)
            lang_counts[tag] += 1

    # Merge all counts
    all_counts: Counter[str] = Counter()
    for tag, count in dep_counts.items():
        if count >= min_dep_frequency:
            all_counts[tag] = count
    for tag, count in lang_counts.items():
        if count >= min_language_frequency:
            all_counts[tag] = count

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and save taxonomy
    taxonomy = TagTaxonomy.from_tag_counts(all_counts, min_frequency=1)
    taxonomy.save(output_dir / "taxonomy.json")

    # Save dependency counts
    dep_names = sorted(dep_counts)
    pq.write_table(
        pa.table({
            "dep_name": pa.array(dep_names, type=pa.string()),
            "repo_count": pa.array([dep_counts[name] for name in dep_names], type=pa.int64()),
        }),
        output_dir / "dependency_counts.parquet",
    )

    # Save per-repo tags
    repo_keys = sorted(repo_tags)
    pq.write_table(
        pa.table({
            "repo_path": pa.array(repo_keys, type=pa.string()),
            "tag_list_json": pa.array(
                [json.dumps(sorted(set(repo_tags[path]))) for path in repo_keys],
                type=pa.string(),
            ),
        }),
        output_dir / "repo_tags.parquet",
    )

    print(
        f"Extracted {len(dep_counts)} unique dependencies, "
        f"{len(lang_counts)} languages from {len(repo_tags)} repos",
    )
    print(f"Taxonomy has {len(taxonomy)} tags after pruning")


def extract_kilt_tags(
    kilt_metadata_path: Path,
    output_dir: Path,
    min_frequency: int = 100,
) -> None:
    """Extract Wikipedia categories from KILT metadata parquet."""
    if not kilt_metadata_path.exists():
        print(f"KILT metadata not found at {kilt_metadata_path}", file=sys.stderr)
        return

    table = pq.read_table(kilt_metadata_path)
    category_counts: Counter[str] = Counter()

    tag_col = None
    for col_name in ("tag_list_json", "categories"):
        if col_name in table.column_names:
            tag_col = col_name
            break

    if tag_col is None:
        print("No category column found in KILT metadata", file=sys.stderr)
        return

    for value in table.column(tag_col).to_pylist():
        if not value:
            continue
        try:
            cats = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            continue
        if isinstance(cats, list):
            for cat in cats:
                category_counts[f"wikipedia/{cat}"] += 1

    pruned = {tag: count for tag, count in category_counts.items() if count >= min_frequency}

    taxonomy_path = output_dir / "taxonomy.json"
    if taxonomy_path.exists():
        existing = TagTaxonomy.load(taxonomy_path)
        merged_counts = Counter({tag: existing.frequency(tag) for tag in existing.tags})
        merged_counts.update(pruned)
        taxonomy = TagTaxonomy.from_tag_counts(merged_counts, min_frequency=1)
    else:
        taxonomy = TagTaxonomy.from_tag_counts(pruned, min_frequency=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy.save(taxonomy_path)
    print(f"Added {len(pruned)} Wikipedia category tags to taxonomy")


def extract_pubmedqa_tags(
    pubmedqa_metadata_path: Path,
    output_dir: Path,
    min_frequency: int = 50,
) -> None:
    """Extract MeSH terms from PubMedQA metadata parquet."""
    if not pubmedqa_metadata_path.exists():
        print(f"PubMedQA metadata not found at {pubmedqa_metadata_path}", file=sys.stderr)
        return

    table = pq.read_table(pubmedqa_metadata_path)
    mesh_counts: Counter[str] = Counter()

    for col_name in ("tag_list_json", "mesh_terms"):
        if col_name not in table.column_names:
            continue
        for value in table.column(col_name).to_pylist():
            if not value:
                continue
            try:
                terms = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                continue
            if isinstance(terms, list):
                for term in terms:
                    mesh_counts[f"mesh/{term}"] += 1
        break

    pruned = {tag: count for tag, count in mesh_counts.items() if count >= min_frequency}

    taxonomy_path = output_dir / "taxonomy.json"
    if taxonomy_path.exists():
        existing = TagTaxonomy.load(taxonomy_path)
        merged_counts = Counter({tag: existing.frequency(tag) for tag in existing.tags})
        merged_counts.update(pruned)
        taxonomy = TagTaxonomy.from_tag_counts(merged_counts, min_frequency=1)
    else:
        taxonomy = TagTaxonomy.from_tag_counts(pruned, min_frequency=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy.save(taxonomy_path)
    print(f"Added {len(pruned)} MeSH term tags to taxonomy")


def extract_memory_tags(
    memory_metadata_path: Path,
    output_dir: Path,
    min_frequency: int = 100,
) -> None:
    """Extract memory types from memory dataset metadata."""
    if not memory_metadata_path.exists():
        print(f"Memory metadata not found at {memory_metadata_path}", file=sys.stderr)
        return

    table = pq.read_table(memory_metadata_path)
    type_counts: Counter[str] = Counter()

    for col_name in ("memory_type", "tag_list_json"):
        if col_name not in table.column_names:
            continue
        for value in table.column(col_name).to_pylist():
            if not value:
                continue
            if col_name == "memory_type":
                type_counts[f"memory/{value}"] += 1
            else:
                try:
                    types = json.loads(value) if isinstance(value, str) else value
                except json.JSONDecodeError:
                    continue
                if isinstance(types, list):
                    for t in types:
                        type_counts[f"memory/{t}"] += 1
        break

    pruned = {tag: count for tag, count in type_counts.items() if count >= min_frequency}

    taxonomy_path = output_dir / "taxonomy.json"
    if taxonomy_path.exists():
        existing = TagTaxonomy.load(taxonomy_path)
        merged_counts = Counter({tag: existing.frequency(tag) for tag in existing.tags})
        merged_counts.update(pruned)
        taxonomy = TagTaxonomy.from_tag_counts(merged_counts, min_frequency=1)
    else:
        taxonomy = TagTaxonomy.from_tag_counts(pruned, min_frequency=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy.save(taxonomy_path)
    print(f"Added {len(pruned)} memory type tags to taxonomy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    repos_cmd = sub.add_parser("repos", help="Extract from code repositories")
    repos_cmd.add_argument("--root-dir", type=Path, required=True)
    repos_cmd.add_argument("--output-dir", type=Path, required=True)
    repos_cmd.add_argument("--min-dep-frequency", type=int, default=50)
    repos_cmd.add_argument("--min-language-frequency", type=int, default=500)

    kilt_cmd = sub.add_parser("kilt", help="Extract Wikipedia categories from KILT")
    kilt_cmd.add_argument("--metadata", type=Path, required=True)
    kilt_cmd.add_argument("--output-dir", type=Path, required=True)
    kilt_cmd.add_argument("--min-frequency", type=int, default=100)

    pubmed_cmd = sub.add_parser("pubmedqa", help="Extract MeSH terms from PubMedQA")
    pubmed_cmd.add_argument("--metadata", type=Path, required=True)
    pubmed_cmd.add_argument("--output-dir", type=Path, required=True)
    pubmed_cmd.add_argument("--min-frequency", type=int, default=50)

    memory_cmd = sub.add_parser("memory", help="Extract memory types from memory datasets")
    memory_cmd.add_argument("--metadata", type=Path, required=True)
    memory_cmd.add_argument("--output-dir", type=Path, required=True)
    memory_cmd.add_argument("--min-frequency", type=int, default=100)

    args = parser.parse_args()
    if args.command == "repos":
        extract_repo_tags(
            args.root_dir, args.output_dir, args.min_dep_frequency, args.min_language_frequency,
        )
    elif args.command == "kilt":
        extract_kilt_tags(args.metadata, args.output_dir, args.min_frequency)
    elif args.command == "pubmedqa":
        extract_pubmedqa_tags(args.metadata, args.output_dir, args.min_frequency)
    elif args.command == "memory":
        extract_memory_tags(args.metadata, args.output_dir, args.min_frequency)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
