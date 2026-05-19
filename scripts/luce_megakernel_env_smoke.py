#!/usr/bin/env python3
"""Smoke-test the optional Luce Qwen3.5 megakernel integration."""

from __future__ import annotations

import json
import os

from bgkit.inference.luce_megakernel import status, supports_spliced_embedding_prefill


def main() -> None:
    st = status(os.environ.get("BGKIT_LUCE_MEGAKERNEL_BACKEND", "auto"))
    payload = {
        "source_mounted": st.source_mounted,
        "cache_present": st.cache_present,
        "embedding_prefill_available": st.embedding_prefill_available,
        "hidden_prefill_available": st.hidden_prefill_available,
        "package_importable": st.package_importable,
        "extension_importable": st.extension_importable,
        "cuda_available": st.cuda_available,
        "capability": st.capability,
        "backend": st.backend,
        "usable": st.usable,
        "spliced_embedding_prefill": supports_spliced_embedding_prefill(),
        "error": st.error,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not st.usable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
