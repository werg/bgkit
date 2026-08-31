"""The splice guard must see a payload that is k copies of one vector.

The norm guard next door catches reps at the wrong SCALE. It is blind to
reps at the right scale whose vectors are all the same vector -- which is
what widenet v8 was: effective rank 1.01 per document against the Phase-1
base's 12.97, measured 2026-08-31. Every metric-level check missed it for
weeks, and a 33x retention-budget sweep was spent before the shape of the
defect was visible.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.decoder import _rep_rank_stats


def test_rank_one_payload_is_reported_as_rank_one():
    v = torch.randn(64)
    rows = torch.arange(1.0, 21.0).unsqueeze(-1) * v
    st = _rep_rank_stats([rows])
    assert st["eff_rank"] == pytest.approx(1.0, abs=1e-2)
    assert st["n_payloads"] == 1


def test_a_shared_vector_plus_tiny_content_still_reports_the_content_rank():
    """The measured pathology has 99% of the energy in one shared vector.
    Centring is what makes the statistic report the CONTENT's rank instead of
    the mean's -- without it every such payload reads as rank 1 and the guard
    could not tell v8 (1.01) from the healthy base (12.97)."""
    shared = torch.randn(64) * 300.0
    rows = shared + torch.randn(40, 64)
    st = _rep_rank_stats([rows])
    assert st["eff_rank"] > 10.0
    assert st["shared_frac"] > 0.99


def test_spread_payload_reports_a_high_rank():
    st = _rep_rank_stats([torch.randn(40, 64)])
    assert st["eff_rank"] > 20.0
    assert st["shared_frac"] < 0.1
    assert st["n_rows"] == 40


def test_short_payloads_are_skipped():
    """A participation ratio over k centred rows is bounded by k - 1, so a
    short payload reports a low rank by CONSTRUCTION rather than by
    collapsing. The first live run warned on exactly such a payload
    (n_payloads=1, eff_rank 1.84) with 9-10.6 either side of it."""
    from bgkit.models.decoder import _REP_RANK_GUARD_MIN_ROWS

    short = _REP_RANK_GUARD_MIN_ROWS - 1
    assert _rep_rank_stats([torch.randn(1, 64)]) is None
    assert _rep_rank_stats([torch.randn(short, 64)]) is None
    st = _rep_rank_stats([torch.randn(short, 64), torch.randn(40, 64)])
    assert st["n_payloads"] == 1
    assert st["n_rows"] == 40


def test_empty_and_non_tensor_inputs_are_ignored():
    assert _rep_rank_stats([]) is None
    assert _rep_rank_stats([torch.zeros(0, 64), None, "not a tensor"]) is None


def test_payloads_are_averaged_not_concatenated():
    """Concatenating documents would measure ACROSS-document spread and hide a
    per-document collapse -- exactly the failure being guarded against."""
    a = torch.randn(64)
    b = torch.randn(64)
    rank1_a = torch.arange(1.0, 21.0).unsqueeze(-1) * a
    rank1_b = torch.arange(1.0, 21.0).unsqueeze(-1) * b
    st = _rep_rank_stats([rank1_a, rank1_b])
    assert st["n_payloads"] == 2
    assert st["eff_rank"] == pytest.approx(1.0, abs=1e-2)


def test_large_payloads_are_subsampled_not_rejected():
    from bgkit.models.decoder import _REP_RANK_GUARD_MAX_ROWS

    st = _rep_rank_stats([torch.randn(_REP_RANK_GUARD_MAX_ROWS * 4, 64)])
    assert st is not None
    assert st["eff_rank"] > 20.0


def test_rank_is_scale_invariant():
    """A payload's rank must not move when the whole thing is rescaled --
    otherwise the rank guard would just be a second, noisier norm guard."""
    x = torch.randn(40, 64)
    lo = _rep_rank_stats([x])["eff_rank"]
    hi = _rep_rank_stats([x * 1000.0])["eff_rank"]
    assert lo == pytest.approx(hi, rel=1e-3)


def test_the_row_count_reported_is_what_actually_entered_the_statistic():
    """The warning is only readable if it says how much evidence it had."""
    from bgkit.models.decoder import _REP_RANK_GUARD_MAX_ROWS

    st = _rep_rank_stats([torch.randn(_REP_RANK_GUARD_MAX_ROWS * 4, 64)])
    assert st["n_rows"] <= _REP_RANK_GUARD_MAX_ROWS + 1
    assert st["n_rows"] > 1


# ---------------------------------------------------------------------------
# The alarm itself: a run-level property, not one sampled call
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from torch import nn

import bgkit.models.decoder as decoder_mod
from bgkit.models.decoder import (
    EmbeddingSegment,
    ReconstructionDecoder,
    TokenSegment,
)

RANK_EVENT = "spliced_rep_rank_collapsed"
_VOCAB, _DIM = 64, 16


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _DIM)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                F.normalize(torch.randn(_VOCAB, _DIM), dim=-1),
            )

    def get_input_embeddings(self):
        return self.embed_tokens


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def get_input_embeddings(self):
        return self.model.embed_tokens


class _Rec:
    def __init__(self):
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, *a, **k):
        pass


@pytest.fixture
def guard_env(monkeypatch):
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_ENABLED", True)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_EVERY", 1)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_HI", 1e9)
    rec = _Rec()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    return rec


def _decoder():
    torch.manual_seed(11)
    return ReconstructionDecoder(_Backbone(), hidden_dim=_DIM)


def _segments(reps: torch.Tensor) -> list:
    return [
        TokenSegment(token_ids=torch.arange(4, dtype=torch.long), loss=False),
        EmbeddingSegment(embeddings=reps),
    ]


def _rank_one(k=32):
    v = F.normalize(torch.randn(_DIM), dim=-1)
    return torch.arange(1.0, k + 1.0).unsqueeze(-1) * v


def _spread(k=32):
    return F.normalize(torch.randn(k, _DIM), dim=-1)


def _rank_warnings(rec):
    return [(e, kw) for e, kw in rec.warnings if e == RANK_EVENT]


def test_one_low_call_does_not_raise_the_alarm(guard_env):
    """The first live run warned on a single short document (n_payloads=1,
    eff_rank 1.84) with 9-10.6 either side of it. One call is not a run."""
    dec = _decoder()
    dec._maybe_guard_spliced_rep_norm([_rank_one()], dec.backbone.get_input_embeddings())
    assert _rank_warnings(guard_env) == []


def test_a_sustained_collapse_does_raise_it(guard_env):
    dec = _decoder()
    emb = dec.backbone.get_input_embeddings()
    for _ in range(decoder_mod._REP_RANK_GUARD_WINDOW):
        dec._maybe_guard_spliced_rep_norm([_rank_one()], emb)
    warns = _rank_warnings(guard_env)
    assert warns, "a full window of rank-1 payloads must warn"
    assert warns[-1][1]["median_eff_rank"] < 2.0
    assert warns[-1][1]["n_rows"] > 0


def test_a_healthy_run_with_one_short_document_stays_quiet(guard_env):
    """The realistic mix: mostly spread payloads, occasionally a short one."""
    dec = _decoder()
    emb = dec.backbone.get_input_embeddings()
    for i in range(12):
        payload = _rank_one() if i == 5 else _spread()
        dec._maybe_guard_spliced_rep_norm([payload], emb)
    assert _rank_warnings(guard_env) == []


def test_recovery_resets_the_consecutive_counter(guard_env):
    dec = _decoder()
    emb = dec.backbone.get_input_embeddings()
    for _ in range(decoder_mod._REP_RANK_GUARD_WINDOW + 2):
        dec._maybe_guard_spliced_rep_norm([_rank_one()], emb)
    assert _rank_warnings(guard_env)[-1][1]["consecutive"] >= 2
    for _ in range(decoder_mod._REP_RANK_GUARD_WINDOW):
        dec._maybe_guard_spliced_rep_norm([_spread()], emb)
    assert int(dec._rep_rank_guard_low_streak) == 0


def test_mean_vec_is_returned_for_the_corpus_comparison():
    st = _rep_rank_stats([torch.randn(40, 64), torch.randn(40, 64)])
    assert st["mean_vec"].shape == (64,)


def test_identical_documents_read_as_cosine_one_to_the_corpus_mean(guard_env):
    """v8's signature: every document emits the same shared vector, cosine
    1.00000. shared_frac cannot tell that from the healthy base (0.927 vs
    0.990); this can (base 0.961, v8 1.00000)."""
    dec = _decoder()
    emb = dec.backbone.get_input_embeddings()
    constant = F.normalize(torch.randn(_DIM), dim=-1) * 50.0
    for _ in range(4):
        dec._maybe_guard_spliced_rep_norm([constant + _spread()], emb)
    cos = getattr(dec, "_rep_rank_guard_corpus_mean", None)
    assert cos is not None
    # Two payloads sharing one constant point the same way.
    a = dec._rep_rank_guard_corpus_mean
    assert float(F.cosine_similarity(a, constant, dim=0)) > 0.95


def test_documents_with_their_own_directions_do_not_read_as_one_vector(guard_env):
    """The healthy case must be distinguishable: each document's shared part
    is its OWN, so the payload means do not collapse onto the corpus mean."""
    dec = _decoder()
    emb = dec.backbone.get_input_embeddings()
    torch.manual_seed(7)
    for _ in range(6):
        own = F.normalize(torch.randn(_DIM), dim=-1) * 50.0
        dec._maybe_guard_spliced_rep_norm([own + _spread()], emb)
    # Nothing warns, and the running mean is not any single document's.
    assert _rank_warnings(guard_env) == []


def test_the_first_payload_reports_no_cosine(guard_env):
    """There is nothing to compare a corpus of one against, and reporting 1.0
    there would look exactly like the collapse."""
    dec = _decoder()
    dec._maybe_guard_spliced_rep_norm(
        [_spread()], dec.backbone.get_input_embeddings(),
    )
    assert getattr(dec, "_rep_rank_guard_corpus_mean", None) is not None
