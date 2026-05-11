"""Helpers for reasoning about encoder/decoder tokenizer compatibility.

Phase 1 description / structural / commit-repro / QA subsets concatenate
encoder-tokenized content into a decoder-tokenized target sequence. When
encoder and decoder share a vocabulary (Qwen→Qwen) this is correct; when
they don't (Qwen→Falcon) the target ends up vocab-mixed and the decoder
sees garbage. The subsets call ``tokenizers_share_vocab`` at construction
time to fail fast in that case.
"""

from __future__ import annotations


def tokenizers_share_vocab(a, b) -> bool:
    """Best-effort equality check between two HF tokenizers' vocabularies.

    Same identity → True. Different identity but same ``name_or_path`` →
    True (cheap shortcut). Otherwise compare ``vocab_size`` and a handful
    of fixed probes; perfect equality not guaranteed but enough to catch
    Qwen × Falcon swaps without exhaustive vocab enumeration.
    """

    if a is b:
        return True
    a_name = getattr(a, "name_or_path", None)
    b_name = getattr(b, "name_or_path", None)
    if a_name is not None and a_name == b_name:
        return True
    try:
        if getattr(a, "vocab_size", None) != getattr(b, "vocab_size", None):
            return False
        for probe in ("hello", "def", "</s>"):
            if a.encode(probe, add_special_tokens=False) != b.encode(
                probe, add_special_tokens=False,
            ):
                return False
        return True
    except Exception:
        return False


def assert_subset_tokenizers_compatible(
    *,
    subset_name: str,
    encoder_tokenizer,
    decoder_tokenizer,
    objective: str,
) -> None:
    """Raise if a subset would feed encoder-tokenized content into a
    decoder-tokenized target while ``encoder_tokenizer`` and
    ``decoder_tokenizer`` use different vocabularies.

    Use in the constructor of any subset that builds ``target_token_ids``
    via ``tokenize_with_sentinel(decoder_tokenizer, … , encoder_ids)``.
    DataReconstructionSubset is the exception: it uses
    ``inner["decoder_content_token_ids"]`` from the Falcon companion stream
    instead of encoder-tokenized content, so it does not need this guard.
    """

    if encoder_tokenizer is decoder_tokenizer:
        return
    if tokenizers_share_vocab(encoder_tokenizer, decoder_tokenizer):
        return
    raise ValueError(
        f"{subset_name} was constructed with encoder and decoder tokenizers "
        f"with different vocabularies. Building the {objective!r} target "
        "would mix vocab IDs because the source/answer/description mmap was "
        "tokenized with the encoder tokenizer. Build a decoder-side mmap "
        "for this objective or restrict the run to a single tokenizer family."
    )
