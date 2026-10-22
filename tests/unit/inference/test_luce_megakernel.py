from __future__ import annotations

import pytest

from bgkit.inference import luce_megakernel


def test_luce_megakernel_status_is_non_throwing() -> None:
    st = luce_megakernel.status()
    assert isinstance(st.usable, bool)
    assert st.backend in {"bf16", "nvfp4"}


def test_luce_megakernel_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        luce_megakernel.load_decoder(backend="bogus")


def test_luce_megakernel_spliced_embedding_prefill_supported() -> None:
    assert luce_megakernel.supports_spliced_embedding_prefill() is True


def test_luce_megakernel_spliced_hidden_prefill_supported() -> None:
    assert luce_megakernel.supports_spliced_hidden_prefill() is True


def test_luce_splice_generator_rejects_sampling() -> None:
    gen = luce_megakernel.LuceSingleSpliceGenerator(decoder=object(), tokenizer=object())
    with pytest.raises(ValueError, match="greedy"):
        gen.generate_with_single_splice(
            survivor_embeddings=None,
            survivor_cu_seqlens=None,
            prefix_ids=None,
            suffix_ids=None,
            temperature=0.7,
        )
