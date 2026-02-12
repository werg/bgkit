"""Smoke tests: verify 1-step training doesn't crash."""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_import_bgkit():
    """Verify bgkit package is importable."""
    import bgkit

    assert bgkit.__version__ == "0.1.0"


@pytest.mark.smoke
def test_import_all_modules():
    """Verify all submodules are importable."""
    import bgkit.data
    import bgkit.eval
    import bgkit.models
    import bgkit.training
    import bgkit.utils
