"""The missing control: what happens when the decoder gets the DOCUMENT.

Every ablation arm in this system degrades the reps -- zeroed, noise, rescaled,
oracle_span. None establishes the CEILING. That made a low rep_gain
uninterpretable: it can mean compression destroys the information, or that the
decoder cannot exploit document context for this task, or that the task is
unanswerable as posed. Those demand opposite fixes.

Measured 2026-08-30 on v8 (128 samples), which is why this matters:

    family      n    EM zeroed   EM reps    dEM
    lognav     31      0.258      0.387    +0.129
    fileneedle 62      0.3065     0.3065    0.000
    grepset    35      0.0857     0.0857    0.000

On 97 of 128 samples the reps are inert to full float precision. grepset has 91
points of headroom and gains nothing, and is not guessable (zeroed EM 0.086), so
"no headroom" does not explain it. Without a full-text ceiling there is no way
to say whether compression is the bottleneck.
"""

from __future__ import annotations

import inspect

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_the_arm_exists_and_is_not_degenerate() -> None:
    assert KRKBTrainer.ABLATION_FULL_TEXT == "full_text"
    assert KRKBTrainer.ABLATION_FULL_TEXT not in KRKBTrainer._DEGENERATE_REP_ABLATIONS, (
        "full text is the UPPER BOUND, not a degraded arm; marking it degenerate "
        "would suppress the splice guard on the one arm whose liveness matters most"
    )


def test_guard_expects_embedding_norm_for_this_arm() -> None:
    """embed_tokens output IS the embedding distribution, so ratio 1.0. Without
    a declared expectation the guard would warn on every sample of a healthy
    run -- the failure already seen with the rescaled arm."""
    assert KRKBTrainer._REP_ABLATION_EXPECTED_RATIO[KRKBTrainer.ABLATION_FULL_TEXT] == 1.0


def test_tokenizer_round_trip_is_mandatory() -> None:
    """_token_store holds ENCODER-vocab ids. Feeding them to the decoder's
    embedding table indexes a different vocabulary and yields a shape-correct,
    meaning-wrong tensor -- the exact silent-corruption class this codebase
    keeps hitting. Decode with the encoder tokenizer, re-encode with the
    decoder's."""
    src = inspect.getsource(KRKBTrainer._full_text_payload)
    assert "self.encoder_tokenizer.decode(" in src
    assert "self.tokenizer.encode(" in src
    di, ei = src.index("encoder_tokenizer.decode("), src.index("self.tokenizer.encode(")
    assert di < ei, "must decode from encoder vocab BEFORE re-encoding for the decoder"


def test_it_raises_rather_than_degrading_silently() -> None:
    """A full-text arm that quietly falls back to an empty or partial context
    would be reported as a valid ceiling and would make compression look
    blameless. Both failure paths must raise."""
    src = inspect.getsource(KRKBTrainer._full_text_payload)
    assert src.count("raise RuntimeError") >= 3
    assert "live_l0" in src, "the cached-L0 path never reads tokens; say so"


def test_payload_is_embeddings_so_span_remapping_is_shared() -> None:
    """Returning embeddings rather than a TokenSegment keeps ONE splice path:
    running_delta += n_emb - sentinel_tok_len already generalises over payload
    length, so answer/tool-call span remapping needs no second implementation
    to drift out of sync."""
    src = inspect.getsource(KRKBTrainer._assemble_sample_segments)
    assert "_full_text_payload(" in src
    assert "running_delta += n_emb - sentinel_tok_len" in src
    # the arm must bypass _apply_context_ablation, which only degrades
    i = src.index("_full_text_payload(")
    assert "else:" in src[i : i + 400]


def test_length_is_capped_configurably() -> None:
    """Full documents are far longer than the rep budget; an uncapped splice
    OOMs on the long tail and the ceiling never gets measured at all."""
    src = inspect.getsource(KRKBTrainer._full_text_payload)
    assert "eval_full_text_max_tokens" in src
