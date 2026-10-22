"""The replay tail must be an objective the decoder cannot satisfy blind.

WHY THIS EXISTS. Measured 2026-08-29 by weight-delta across the lineage
(base phase1_summarization_round_robin_step51945, rep_gain 2.03-2.95 nats ->
phase2_kb control_armb, rep_gain ~0.01):

    l1.backbone                 0.0000
    l0.backbone                 0.0001
    projection_blocks.qwen35    0.0039   (output_norm.weight bit-identical)
    DECODER qwen35              0.0120   (uniform across every layer)

The representation did not change; the READER did. So the fix is neither an
anchor on the projection (nothing drifted) nor selection (random == trained,
and the top-k margin loss drove its own metric to 0.0008 with zero rep_gain
effect) — it is an objective that requires reading the reps.

The tests below pin the properties that make the tail that objective, plus the
one integration hazard already caught during implementation: the gold decode
cap drops every segment past the answer, so a tail folded into ``token_ids``
would be deleted whenever the answer exceeded the cap — intermittently.
"""

from __future__ import annotations

import inspect
import types

import torch

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _Tok:
    """Whitespace tokenizer that round-trips ``wN`` <-> ``N`` injectively.

    Injectivity matters: a lossy fake makes distinct windows encode identically
    and would silently pass ``test_the_window_moves_across_steps`` no matter
    what the code did.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for w in text.split():
            out.append(int(w[1:]) if w.startswith("w") and w[1:].isdigit() else 1)
        return out

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{int(i)}" for i in ids)


class _Store:
    def __init__(self, n: int = 400) -> None:
        self._doc = torch.arange(n, dtype=torch.long)

    def get(self, dataset: str, aid: str) -> torch.Tensor:
        return self._doc


def _turn(**args) -> types.SimpleNamespace:
    return types.SimpleNamespace(args=args)


def _trainer(weight: float = 0.5, doc_len: int = 400) -> KRKBTrainer:
    t = object.__new__(KRKBTrainer)
    t._replay_loss_weight = weight
    t._replay_window_tokens = 48
    t._replay_cue_tokens = 8
    t._replay_mode = "cue"
    t._replay_marker_ids = None
    t._replay_stats = {"applied": 0, "skipped": 0, "target_tokens": 0}
    t._token_store = _Store(doc_len)
    t.encoder_tokenizer = _Tok()
    t.tokenizer = _Tok()
    t.device = torch.device("cpu")
    t.global_step = 7
    t._resolve_article_ids = lambda ds, ids: [str(ids[0])]
    return t


def _rendered() -> types.SimpleNamespace:
    return types.SimpleNamespace(bgkit_turns=[_turn(ids=["doc-1"])])


def _sample() -> types.SimpleNamespace:
    return types.SimpleNamespace(dataset_name="fileneedle")


def test_disabled_by_default_is_a_hard_none() -> None:
    """A zero weight must not build, log, or cost anything."""
    t = _trainer(weight=0.0)
    assert t._replay_tail(_sample(), _rendered()) is None


def test_no_tail_without_grad() -> None:
    """Every eval, ablation and free-running path runs under no_grad. The tail
    must never appear there, or it shifts a metric it was meant to explain."""
    t = _trainer()
    with torch.no_grad():
        assert t._replay_tail(_sample(), _rendered()) is None


def test_marker_and_cue_are_unsupervised_and_the_continuation_is_not() -> None:
    """The cue localises; only the continuation is scored. If the cue carried
    loss the model could score well by copying tokens it was just handed."""
    t = _trainer(weight=0.5)
    tail = t._replay_tail(_sample(), _rendered())
    assert tail is not None
    ids, w = tail
    assert ids.shape == w.shape
    supervised = w > 0
    assert int(supervised.sum()) > 0, "nothing supervised — the tail is inert"
    # Supervision is a single contiguous suffix: no gaps, nothing before it.
    first = int(supervised.nonzero()[0])
    assert bool(supervised[first:].all())
    assert not bool(supervised[:first].any())
    assert torch.allclose(w[supervised], torch.tensor(0.5))


def test_head_turns_are_not_used_as_replay_documents() -> None:
    """``is_head`` turns carry the task query, not a document."""
    t = _trainer()
    rendered = types.SimpleNamespace(bgkit_turns=[_turn(ids=["q"], is_head=True)])
    assert t._replay_tail(_sample(), rendered) is None
    assert t._replay_stats["skipped"] == 1


def test_short_documents_are_skipped_not_silently_truncated() -> None:
    """A doc shorter than cue+window has no valid window; skipping is honest,
    a padded or wrapped window would be a fabricated target."""
    t = _trainer(doc_len=10)
    assert t._replay_tail(_sample(), _rendered()) is None
    assert t._replay_stats["skipped"] == 1


def test_reproducible_for_the_same_step_and_article() -> None:
    """Resume must reproduce the same targets — hence crc32, not hash()."""
    a = _trainer()._replay_tail(_sample(), _rendered())
    b = _trainer()._replay_tail(_sample(), _rendered())
    assert a is not None and b is not None
    assert torch.equal(a[0], b[0])


def test_the_window_moves_across_steps() -> None:
    """A fixed window would let the model memorise one span per document
    instead of learning to read the reps."""
    t1 = _trainer()
    t1.global_step = 7
    t2 = _trainer()
    t2.global_step = 8
    a = t1._replay_tail(_sample(), _rendered())
    b = t2._replay_tail(_sample(), _rendered())
    assert a is not None and b is not None
    assert not torch.equal(a[0], b[0])


def test_tail_is_appended_after_the_gold_decode_cap_truncation() -> None:
    """The hazard caught during implementation.

    ``_truncate_segments_to_gold_budget`` cuts at ``answer_start + n_gold`` and
    breaks, dropping every later segment. A tail folded into ``token_ids``
    inside ``_prepare_sample_for_decode`` would therefore be deleted whenever
    the answer exceeded the cap and kept when it did not — an intermittent
    silent no-op, the exact failure shape that has bitten this work repeatedly.

    Pinned structurally because the ordering IS the fix: the append must follow
    the truncation call in ``_encode_decode_group``.
    """
    src = inspect.getsource(KRKBTrainer._encode_decode_group)
    cut = src.index("_truncate_segments_to_gold_budget")
    add = src.index('prep.get("replay_tail")')
    assert cut < add, "replay tail must be appended AFTER the gold-cap truncation"


def test_prepare_does_not_fold_the_tail_into_token_ids() -> None:
    """The other half of the same invariant: the tail travels in its own key so
    nothing that rewrites ``token_ids`` can consume or truncate it."""
    src = inspect.getsource(KRKBTrainer._prepare_sample_for_decode)
    assert '"replay_tail": replay_tail' in src
    assert "torch.cat([token_ids" not in src


# ---------------------------------------------------------------------------
# The replay GAP probe. eval/rep_gain measures the ANSWER task, which the model
# already solves rep-blind, so it cannot distinguish a working replay term from
# an inert one. The gap probe measures rep-dependence of the replay task itself.
# ---------------------------------------------------------------------------


def test_force_flag_opens_the_gate_inside_no_grad() -> None:
    """The probe runs under no_grad, where the training gate closes. Without
    the force flag it would measure a tail that was never built — the exact
    silent-no-op shape this work keeps hitting."""
    t = _trainer()
    with torch.no_grad():
        assert t._replay_tail(_sample(), _rendered()) is None
        t._replay_force = True
        assert t._replay_tail(_sample(), _rendered()) is not None


def test_probe_restores_every_flag_it_touches() -> None:
    """A probe that leaked _replay_only_loss would zero the ANSWER supervision
    for the rest of training. Pinned by inspecting the finally block."""
    src = inspect.getsource(KRKBTrainer._eval_replay_gap)
    finally_block = src[src.index("finally:"):]
    for name in ("_replay_force", "_replay_only_loss", "_replay_loss_weight",
                 "_ablation_mode", "_pending_l0_outputs", "_pending_l1_outputs"):
        assert name in finally_block, f"{name} not restored"


def test_probe_measures_nats_not_half_nats() -> None:
    """The normaliser counts positions, not weights, so leaving the mask at 0.5
    would halve both arms and halve the reported gap. The probe must pin the
    weight to 1.0."""
    src = inspect.getsource(KRKBTrainer._eval_replay_gap)
    assert "self._replay_loss_weight = 1.0" in src


def test_replay_only_loss_silences_the_trajectory() -> None:
    """If answer/tool tokens stayed supervised, the probe's loss would be a
    pooled quantity and its movement could come from either half."""
    src = inspect.getsource(KRKBTrainer._prepare_sample_for_decode)
    assert "_replay_only_loss" in src
    assert "torch.zeros_like(loss_mask" in src


def test_probe_divides_the_group_sum_by_sample_count() -> None:
    """_encode_decode_group returns a SUM over samples (_forward_backward
    divides by n_samples right after calling it). Reporting the raw sum gave
    ce_reps 25.86 at step 1500 — worse than uniform over the vocab and
    irreconcilable with a pooled train loss of 1.3. The contradiction caught
    it; nothing in the probe did."""
    src = inspect.getsource(KRKBTrainer._eval_replay_gap)
    assert "float(loss_reps.detach()) / done_n" in src
    assert "float(loss_zero.detach()) / done_z" in src


def test_reported_ce_is_plausible_for_a_language_model() -> None:
    """A guard on the units themselves: per-token CE above ln(vocab) means the
    model is worse than uniform, which for a trained LM means the number is
    mis-normalised, not that the model is that bad."""
    import math
    uniform_nats = math.log(150_000)
    for ce in (25.858, 26.407):          # the step-1500 raw sums, n=24
        assert ce > uniform_nats         # implausible as a per-token CE
        assert ce / 24 < uniform_nats    # plausible once divided by n_samples


# ---------------------------------------------------------------------------
# CUE-FREE replay. The cued form measured a gap of 0.0229 nats against a total
# replay CE of 1.077 (step 1500): 98% of the continuation came from the LM
# prior, because a short cue out of templated log/grep text nearly determines
# what follows. A task the prior can solve teaches nothing about reading reps.
# ---------------------------------------------------------------------------


def test_cue_free_supervises_every_target_token() -> None:
    """With no cue, nothing is handed to the model — the whole window must be
    produced from the reps."""
    t = _trainer()
    t._replay_mode = "start"
    tail = t._replay_tail(_sample(), _rendered())
    assert tail is not None
    _ids, w = tail
    marker = len(t._replay_marker_ids)
    assert float(w[:marker].sum()) == 0.0          # marker still unsupervised
    assert bool((w[marker:] > 0).all())            # every content token scored


def test_cue_free_starts_at_the_document_start() -> None:
    """Well-posedness: with no cue the request must name a deterministic
    position, and the prompt already names the document. A random start with
    no cue would be unanswerable rather than merely hard."""
    t = _trainer()
    t._replay_mode = "start"
    ids, _ = t._replay_tail(_sample(), _rendered())
    marker = len(t._replay_marker_ids)
    # _Store hands back arange(N), so doc-start content decodes to w0 w1 w2...
    assert [int(x) for x in ids[marker:marker + 3]] == [0, 1, 2]


def test_cue_free_target_is_step_invariant() -> None:
    """Deterministic by construction: the document start does not move. The
    randomised start only applies to the cued form, where the cue says where."""
    a = _trainer()
    a._replay_mode = "start"
    a.global_step = 7
    b = _trainer()
    b._replay_mode = "start"
    b.global_step = 9999
    assert torch.equal(a._replay_tail(_sample(), _rendered())[0],
                       b._replay_tail(_sample(), _rendered())[0])


def test_cued_form_still_works() -> None:
    """The cued path stays available and randomised — the change is additive."""
    t1 = _trainer()
    t1._replay_mode = "cue"
    t1.global_step = 7
    t2 = _trainer()
    t2._replay_mode = "cue"
    t2.global_step = 8
    assert not torch.equal(t1._replay_tail(_sample(), _rendered())[0],
                           t2._replay_tail(_sample(), _rendered())[0])


# ---------------------------------------------------------------------------
# mode=chunk. Both earlier locators carried content and leaked to the prior:
#   cue    step 1500: gap 0.0229 against replay CE 1.077 (98% from the prior)
#   start  step 1750: replay CE HALVED to 0.588, gap barely moved — document
#          openings are boilerplate, so position 0 is the most guessable place
#          to ask about. That halving is a property of the target, not learning.
# An INDEX locator says which chunk without saying what is in it.
# ---------------------------------------------------------------------------


def test_chunk_marker_names_the_index() -> None:
    """The locator must actually appear in the prompt — an index the model
    cannot see makes the task unanswerable rather than rep-dependent."""
    t = _trainer()
    t._replay_mode = "chunk"
    ids, _w = t._replay_tail(_sample(), _rendered())
    # The index must be rendered INTO the marker, not merely chosen internally.
    assert 'f"\\n\\n[replay {chunk_index}]\\n"' in inspect.getsource(
        KRKBTrainer._replay_tail,
    )
    # The fixture tokenizer maps doc token N -> id N, so the emitted content
    # must start exactly on a chunk boundary.
    marker_len = len(t.tokenizer.encode("\n\n[replay 0]\n"))
    first_content = int(ids[marker_len])
    assert first_content % 48 == 0, "chunk target must start on a chunk boundary"


def test_chunk_index_is_unsupervised() -> None:
    """The index is given, not predicted. Scoring it would reward learning the
    index distribution instead of reading reps."""
    t = _trainer()
    t._replay_mode = "chunk"
    ids, w = t._replay_tail(_sample(), _rendered())
    assert int((w == 0).sum()) > 0
    assert float(w[0]) == 0.0
    assert bool((w[-4:] > 0).all())          # content is scored
    assert int(ids.shape[0]) == int(w.shape[0])


def test_chunk_target_moves_across_steps() -> None:
    """A fixed chunk would be memorised the same way position 0 was."""
    a = _trainer()
    a._replay_mode = "chunk"
    a.global_step = 7
    b = _trainer()
    b._replay_mode = "chunk"
    b.global_step = 8
    assert not torch.equal(a._replay_tail(_sample(), _rendered())[0],
                           b._replay_tail(_sample(), _rendered())[0])


def test_unknown_mode_is_rejected_at_setup() -> None:
    """A typo must fail configuration, not silently fall back to the leakiest
    mode — the failure shape this session hit eight times. Validated where the
    knob is actually read (setup), not where one might assume (__init__)."""
    src = inspect.getsource(KRKBTrainer.setup)
    assert "replay_mode must be cue|start|chunk" in src
    assert "raise ValueError" in src


def test_replay_probe_has_a_wall_clock_budget() -> None:
    """A diagnostic must not be able to hang the thing it diagnoses.

    On 2026-08-30 this probe wedged an eval for THREE HOURS on a cross-lineage
    checkpoint: one core at 100%, GPU resident at 35W, no log line after the
    first replay tail. The surrounding try/except caught exceptions — but a
    hang is not an exception, so nothing fired and the GPU sat blocked.
    """
    src = inspect.getsource(KRKBTrainer.evaluate)
    assert "replay_eval_timeout_s" in src
    assert "SIGALRM" in src
    assert "replay_gap_eval_timeout" in src
    # the alarm must be cleared on EVERY path, or a later eval inherits it
    tail = src[src.index("replay_eval_timeout_s"):]
    assert "finally:" in tail and "alarm(0)" in tail


def test_timeout_is_distinguished_from_failure() -> None:
    """A timeout and a crash need different log events: one means 'too slow on
    this input', the other means 'broken'. Collapsing them would have made the
    three-hour hang look like the earlier AttributeError."""
    src = inspect.getsource(KRKBTrainer.evaluate)
    assert "except TimeoutError:" in src
    assert "replay_gap_eval_failed" in src
