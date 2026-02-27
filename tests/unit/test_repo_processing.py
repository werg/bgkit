"""Tests for repo_processing filters and size caps."""

from bgkit.data.repo_processing import (
    _PUBLIC_KEEP_EXTS,
    EXTENSION_SIZE_CAPS,
    SKIP_DIRS,
    SKIP_EXTENSIONS,
    SKIP_FILENAMES,
    _should_skip_path,
)


class TestSkipExtensions:
    """Test that new skip extensions are present."""

    def test_snap_files_skipped(self):
        assert ".snap" in SKIP_EXTENSIONS

    def test_csv_files_skipped(self):
        assert ".csv" in SKIP_EXTENSIONS

    def test_tsv_files_skipped(self):
        assert ".tsv" in SKIP_EXTENSIONS

    def test_srt_files_skipped(self):
        assert ".srt" in SKIP_EXTENSIONS

    def test_stl_files_skipped(self):
        assert ".stl" in SKIP_EXTENSIONS

    def test_resx_files_skipped(self):
        assert ".resx" in SKIP_EXTENSIONS

    def test_pbxproj_files_skipped(self):
        assert ".pbxproj" in SKIP_EXTENSIONS

    def test_sln_files_skipped(self):
        assert ".sln" in SKIP_EXTENSIONS


class TestSkipFilenames:
    def test_shrinkwrap_skipped(self):
        assert "shrinkwrap.json" in SKIP_FILENAMES

    def test_gopkg_lock_skipped(self):
        assert "Gopkg.lock" in SKIP_FILENAMES


class TestSkipDirs:
    def test_snapshots_dir_skipped(self):
        assert "__snapshots__" in SKIP_DIRS

    def test_coverage_dir_skipped(self):
        assert "coverage" in SKIP_DIRS

    def test_cache_dir_skipped(self):
        assert ".cache" in SKIP_DIRS

    def test_target_dir_skipped(self):
        assert "target" in SKIP_DIRS

    def test_godeps_dir_skipped(self):
        assert "Godeps" in SKIP_DIRS

    def test_site_dir_skipped(self):
        assert "site" in SKIP_DIRS

    def test_external_dir_skipped(self):
        assert "external" in SKIP_DIRS


class TestShouldSkipPath:
    """Test _should_skip_path with new filter rules."""

    # --- Generated file detection ---

    def test_skip_pb_go(self):
        assert _should_skip_path("api/v1/service.pb.go") is True

    def test_skip_pb_rs(self):
        assert _should_skip_path("proto/msg.pb.rs") is True

    def test_skip_pb2_py(self):
        assert _should_skip_path("gen/model_pb2.py") is True

    def test_keep_regular_go(self):
        assert _should_skip_path("api/v1/service.go") is False

    def test_skip_generated_go(self):
        assert _should_skip_path("pkg/types_generated.go") is True

    def test_skip_generated_ts(self):
        assert _should_skip_path("src/api_generated.ts") is True

    def test_skip_generated_cs(self):
        assert _should_skip_path("Models/User.generated.cs") is True

    def test_skip_designer_cs(self):
        assert _should_skip_path("Forms/Main.designer.cs") is True

    def test_skip_swagger_generated(self):
        assert _should_skip_path("docs/swagger_doc_generated.json") is True

    # --- public/ directory conditional skip ---

    def test_skip_html_in_public(self):
        assert _should_skip_path("public/index.html") is True

    def test_skip_css_in_public(self):
        assert _should_skip_path("public/styles/main.css") is True

    def test_skip_txt_in_public(self):
        assert _should_skip_path("public/lyrics/song.txt") is True

    def test_keep_ts_in_public(self):
        assert _should_skip_path("public/src/app.ts") is False

    def test_keep_tsx_in_public(self):
        assert _should_skip_path("public/components/App.tsx") is False

    def test_keep_js_in_public(self):
        assert _should_skip_path("public/main.js") is False

    def test_keep_vue_in_public(self):
        assert _should_skip_path("public/App.vue") is False

    def test_keep_py_in_public(self):
        assert _should_skip_path("public/server.py") is False

    def test_html_outside_public_not_skipped(self):
        """HTML files outside public/ are not skipped by the public rule."""
        assert _should_skip_path("src/templates/index.html") is False

    # --- Existing filters still work ---

    def test_skip_node_modules(self):
        assert _should_skip_path("node_modules/foo/index.js") is True

    def test_skip_package_lock(self):
        assert _should_skip_path("package-lock.json") is True

    def test_skip_png(self):
        assert _should_skip_path("assets/logo.png") is True

    def test_keep_regular_py(self):
        assert _should_skip_path("src/main.py") is False

    def test_skip_snap_file(self):
        assert _should_skip_path("__tests__/Button.test.snap") is True

    def test_skip_csv_file(self):
        assert _should_skip_path("data/results.csv") is True

    def test_skip_target_dir(self):
        assert _should_skip_path("target/debug/main.rs") is True

    def test_skip_external_dir(self):
        assert _should_skip_path("external/glfw/src/init.c") is True


class TestExtensionSizeCaps:
    """Test that EXTENSION_SIZE_CAPS has expected entries and values."""

    def test_html_cap(self):
        assert EXTENSION_SIZE_CAPS[".html"] == 20 * 1024

    def test_json_cap(self):
        assert EXTENSION_SIZE_CAPS[".json"] == 10 * 1024

    def test_txt_cap(self):
        assert EXTENSION_SIZE_CAPS[".txt"] == 10 * 1024

    def test_css_cap(self):
        assert EXTENSION_SIZE_CAPS[".css"] == 20 * 1024

    def test_no_extension_cap(self):
        assert EXTENSION_SIZE_CAPS[""] == 5 * 1024

    def test_ipynb_cap(self):
        assert EXTENSION_SIZE_CAPS[".ipynb"] == 20 * 1024

    def test_py_not_capped(self):
        """Python files should not have a size cap."""
        assert ".py" not in EXTENSION_SIZE_CAPS

    def test_go_not_capped(self):
        assert ".go" not in EXTENSION_SIZE_CAPS


class TestPublicKeepExts:
    """Verify the _PUBLIC_KEEP_EXTS set."""

    def test_contains_ts(self):
        assert ".ts" in _PUBLIC_KEEP_EXTS

    def test_contains_tsx(self):
        assert ".tsx" in _PUBLIC_KEEP_EXTS

    def test_does_not_contain_html(self):
        assert ".html" not in _PUBLIC_KEEP_EXTS

    def test_does_not_contain_css(self):
        assert ".css" not in _PUBLIC_KEEP_EXTS
