"""Tests for repo file extraction and processing."""

from __future__ import annotations

import os

import pytest

from bgkit.data.repo_processing import (
    FileRecord,
    RepoSnapshot,
    looks_minified,
    _should_skip_path,
    detect_language,
    extract_repo_snapshot,
    load_repo_files,
)
from bgkit.utils.git_utils import is_git_repo


# Use a real repo from the data collection for integration-style unit tests.
# Pick the first available repo in data/repos/.
def _find_test_repo() -> str | None:
    repos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "repos")
    repos_dir = os.path.normpath(repos_dir)
    if not os.path.isdir(repos_dir):
        return None
    for owner in sorted(os.listdir(repos_dir))[:10]:
        owner_path = os.path.join(repos_dir, owner)
        if not os.path.isdir(owner_path):
            continue
        for repo_name in os.listdir(owner_path):
            repo_path = os.path.join(owner_path, repo_name)
            if is_git_repo(repo_path):
                return repo_path
    return None


TEST_REPO = _find_test_repo()
needs_repo = pytest.mark.skipif(TEST_REPO is None, reason="No test repo available in data/repos/")


# --- Pure unit tests (no repo needed) ---


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == "Python"

    def test_typescript(self):
        assert detect_language("components/App.tsx") == "TypeScript"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "Dockerfile"
        assert detect_language("Dockerfile.prod") == "Dockerfile"

    def test_makefile(self):
        assert detect_language("Makefile") == "Makefile"

    def test_unknown(self):
        assert detect_language("README") is None

    def test_nested_path(self):
        assert detect_language("pkg/internal/handler.go") == "Go"


class TestShouldSkipPath:
    def test_skip_node_modules(self):
        assert _should_skip_path("node_modules/lodash/index.js") is True

    def test_skip_pycache(self):
        assert _should_skip_path("src/__pycache__/main.cpython-312.pyc") is True

    def test_skip_lockfile(self):
        assert _should_skip_path("package-lock.json") is True

    def test_skip_binary_ext(self):
        assert _should_skip_path("assets/logo.png") is True

    def test_keep_source(self):
        assert _should_skip_path("src/main.py") is False

    def test_keep_config(self):
        assert _should_skip_path(".gitignore") is False  # has extension

    def test_skip_vendor(self):
        assert _should_skip_path("vendor/github.com/pkg/errors/errors.go") is True

    def test_keep_github_dir(self):
        assert _should_skip_path(".github/workflows/ci.yml") is False

    def test_skip_bundle_js(self):
        assert _should_skip_path("src/main.bundle.js") is True

    def test_skip_chunk_js(self):
        assert _should_skip_path("static/0.chunk.js") is True

    def test_skip_source_map(self):
        assert _should_skip_path("dist/app.js.map") is True

    def test_skip_ts_declaration_map(self):
        assert _should_skip_path("out/index.d.ts.map") is True


class TestLooksMinified:
    def test_normal_python_is_not_minified(self):
        code = (
            "import os\n"
            "\n"
            "def greet(name):\n"
            "    print(f'Hello, {name}!')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    greet('world')\n"
        ) * 10  # pad so file is >200 bytes
        assert looks_minified(code) is False

    def test_normal_typescript_is_not_minified(self):
        code = (
            "import { Component } from 'react';\n"
            "\n"
            "export class App extends Component {\n"
            "    render() {\n"
            "        return <div>hello</div>;\n"
            "    }\n"
            "}\n"
        ) * 10
        assert looks_minified(code) is False

    def test_very_short_file_not_flagged(self):
        # Below _MINIFIED_MIN_BYTES — we don't have enough signal.
        assert looks_minified("x=1\n") is False
        assert looks_minified("export default 42;\n") is False

    def test_single_long_line_is_minified(self):
        code = "var a=1;" * 2000  # one very long line
        assert looks_minified(code) is True

    def test_few_lines_with_long_line_is_minified(self):
        # Two-line file where one line is 1000 chars of minified JS.
        code = "var x=1;\n" + ("var y=2;" * 200) + "\n"
        assert looks_minified(code) is True

    def test_dense_multi_line_is_minified(self):
        # 20 lines averaging ~300 chars each — mean > 200.
        long_line = "a(); b(); c(); " * 25
        code = "\n".join([long_line] * 20)
        assert looks_minified(code) is True

    def test_code_with_one_long_url_comment_not_flagged(self):
        # Valid code with a single long URL comment shouldn't be flagged.
        # The URL line is long but the overall mean stays low.
        url_line = "// " + ("https://example.com/" + "x" * 180)
        normal = "def foo():\n    return 1\n" * 10
        code = url_line + "\n" + normal
        # max_len is 200ish, well under 2000; mean is dominated by normal lines
        assert looks_minified(code) is False

    def test_empty_file_not_flagged(self):
        assert looks_minified("") is False

    def test_whitespace_only_file_not_flagged(self):
        assert looks_minified("\n\n\n") is False

    def test_invalid_content_type_raises(self):
        import pytest
        with pytest.raises(ValueError, match="content_type"):
            looks_minified("some text", content_type="bogus")

    def test_prose_mode_allows_single_paragraph_abstract(self):
        """A 1200-char single-line abstract (PubMedQA-style) is legitimate
        prose. In prose mode the short-file + mean-line rules are relaxed,
        so it must not be flagged."""
        abstract = "BACKGROUND: " + "word " * 150 + "CONCLUSION: " + "word " * 80
        assert len(abstract) > 1000
        assert looks_minified(abstract, content_type="code") is True
        assert looks_minified(abstract, content_type="prose") is False

    def test_prose_mode_allows_wikipedia_paragraph(self):
        """Single-paragraph Wikipedia-style text must not be flagged in prose mode."""
        wiki = (
            "The Battle of Waterloo was fought on Sunday 18 June 1815 near "
            "Waterloo, at the time in the United Kingdom of the Netherlands, "
            "now in Belgium. A French army under the command of Napoleon was "
            "defeated by two of the armies of the Seventh Coalition. One of "
            "these was a British-led coalition consisting of units from the "
            "United Kingdom, the Netherlands, Hanover, Brunswick and Nassau, "
            "under the command of the Duke of Wellington."
        )
        assert looks_minified(wiki, content_type="prose") is False

    def test_prose_mode_still_flags_pathological_long_line(self):
        """Long single lines (> _MINIFIED_MAX_LINE_LEN=2000) stay flagged
        even in prose mode — a 5000-char unbroken blob suggests base64 /
        CSV-in-text / HTML-in-field content, not natural writing."""
        blob = "a" * 5000
        assert looks_minified(blob, content_type="prose") is True

    def test_prose_mode_flags_html_stuffed_in_text(self):
        htmlish = "<html><body>" + "<div>x</div>" * 200 + "</body></html>"
        assert looks_minified(htmlish, content_type="prose") is True

    def test_minified_js_flagged_in_both_modes(self):
        """A classic minified JS file (long single line) is caught in both
        modes because max_line > 2000 is an unconditional rule."""
        minjs = "var a=1;" * 300 + ";" * 100
        assert looks_minified(minjs, content_type="code") is True
        assert looks_minified(minjs, content_type="prose") is True

    def test_conversational_dialogue_not_flagged_in_prose_mode(self):
        """Short human dialogue turns (used by memory datasets) must pass
        through — they have short lines and would never match any rule."""
        dialog = (
            "A: How was your day?\n"
            "B: Pretty good, thanks. I went hiking with Sarah and we saw "
            "a waterfall about two hours into the trail. You?\n"
            "A: Oh nice. I stayed home and read most of the day.\n"
        )
        assert looks_minified(dialog, content_type="prose") is False
        assert looks_minified(dialog, content_type="code") is False


# --- Tests against real repos ---


@needs_repo
class TestExtractRepoSnapshot:
    def test_returns_snapshot(self):
        snap = extract_repo_snapshot(TEST_REPO)
        assert isinstance(snap, RepoSnapshot)
        assert snap.total_files > 0
        assert snap.commit_sha  # non-empty

    def test_files_have_content(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files[:5]:
            assert isinstance(f, FileRecord)
            assert len(f.content) > 0
            assert f.size_bytes > 0
            assert f.path

    def test_no_binary_in_results(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files:
            # Content should be valid utf-8 (it's a string, so it is by construction)
            assert isinstance(f.content, str)

    def test_no_skipped_dirs_in_results(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files:
            parts = f.path.split("/")
            assert "node_modules" not in parts
            assert "__pycache__" not in parts
            assert ".git" not in parts

    def test_max_file_size(self):
        snap = extract_repo_snapshot(TEST_REPO, max_file_size=1024)
        for f in snap.files:
            assert f.size_bytes <= 1024

    def test_specific_commit(self):
        """Extracting at HEAD explicitly should match default."""
        snap_default = extract_repo_snapshot(TEST_REPO)
        snap_explicit = extract_repo_snapshot(TEST_REPO, commit_sha=snap_default.commit_sha)
        assert snap_default.total_files == snap_explicit.total_files


@needs_repo
class TestLoadRepoFiles:
    def test_returns_dict(self):
        files = load_repo_files(TEST_REPO)
        assert isinstance(files, dict)
        assert len(files) > 0

    def test_keys_are_paths(self):
        files = load_repo_files(TEST_REPO)
        for path in list(files.keys())[:5]:
            assert "/" in path or "." in path  # should look like a file path
            assert not path.startswith("/")  # relative paths

    def test_values_are_strings(self):
        files = load_repo_files(TEST_REPO)
        for content in list(files.values())[:5]:
            assert isinstance(content, str)
            assert len(content) > 0
