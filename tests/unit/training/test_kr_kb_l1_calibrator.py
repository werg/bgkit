"""Tests that the KRKBTrainer's L1 path drives the ThresholdCalibrator
instead of using raw torch.topk."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _StubICE(torch.nn.Module):
    """Returns scores = position index so torch.topk and quantile both
    have a deterministic monotone signal we can verify against."""

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        # content: (B, seq_len, D); return (B, seq_len)
        b, seq_len, _ = content.shape
        # Linearly increasing scores by position (0, 1, ..., seq_len-1)
        return (
            torch.arange(seq_len, dtype=torch.float32, device=content.device)
            .unsqueeze(0)
            .expand(b, -1)
            .clone()
        )


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Two tokens per ID, deterministic so the test is reproducible.
        return [hash(text) % 31 + 1, (hash(text) // 7) % 31 + 1]


class _StubBackbone:
    def __init__(self, hidden_dim: int = 8) -> None:
        self._embed = torch.nn.Embedding(64, hidden_dim)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._embed


class _StubCompressor:
    def __init__(self, hidden_dim: int = 8) -> None:
        self.backbone = _StubBackbone(hidden_dim)
        self.hidden_dim = hidden_dim


class _StubEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.compressor = _StubCompressor(hidden_dim)


def _make_trainer_with_l0_batch(
    *,
    n_articles: int = 4,
    surv_per_article: int = 6,
    hidden_dim: int = 8,
    l1_retention: float = 0.30,
):
    """Build a bare-minimum KRKBTrainer instance with L1 dependencies stubbed."""
    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer.encoder = _StubEncoder(hidden_dim)
    trainer.encoder_tokenizer = _StubTokenizer()
    trainer.ice = _StubICE()
    trainer._l1_retention = l1_retention
    trainer._l1_calibrator = ThresholdCalibrator(
        quantile_points=21, ema_decay=0.9, warmup_batches=2,
    )
    # Pre-warm the calibrator with the actual non-pinned score
    # distribution we'll see at L1 time. The stub ICE returns
    # ``arange(L)`` for an input of length L, where L for the prepared
    # turn is ``n_articles * (id_tokens + surv_per_article)`` and the
    # non-pinned subset is the surv_per_article positions per article.
    # We approximate the distribution by sampling those positions
    # directly from the expected layout.
    id_tokens_per_article = 2  # _StubTokenizer always returns 2 tokens
    L = n_articles * (id_tokens_per_article + surv_per_article)
    full_scores = torch.arange(L, dtype=torch.float32)
    pinned_positions = []
    cursor = 0
    for _ in range(n_articles):
        pinned_positions.extend(range(cursor, cursor + id_tokens_per_article))
        cursor += id_tokens_per_article + surv_per_article
    pinned_set = set(pinned_positions)
    non_pinned_scores = torch.tensor(
        [s for i, s in enumerate(full_scores.tolist()) if i not in pinned_set],
        dtype=torch.float32,
    )
    trainer._l1_calibrator.update_from_flat(non_pinned_scores)
    trainer._l1_calibrator.update_from_flat(non_pinned_scores)
    # Stubbed _resolve_article_ids: identity passthrough.
    trainer._resolve_article_ids = lambda dataset, ids: list(ids)
    # Stubbed _l0_for_articles: produce a deterministic L0 batch.
    l0_batch = torch.zeros(n_articles, surv_per_article, hidden_dim)
    l0_mask = torch.ones(n_articles, surv_per_article, dtype=torch.bool)
    trainer._l0_for_articles = lambda dataset, ids: (l0_batch, l0_mask)
    return trainer, n_articles, surv_per_article


def test_l1_path_uses_calibrator_threshold():
    """When the calibrator is warmed up, the L1 mask should reflect a
    quantile threshold rather than raw top-K. With monotone-by-position
    ICE scores and 30% retention, the calibrator picks the top ~70%
    threshold and the surviving non-pinned positions are the ones with
    the highest position indices."""
    trainer, _n, surv_per = _make_trainer_with_l0_batch(
        n_articles=3, surv_per_article=10, l1_retention=0.30,
    )
    out = trainer._prepare_l1_turn(
        "toy", ["a1", "a2", "a3"], "what?",
    )
    assert out is not None
    pinned = out["pinned"]
    survivor_mask = out["survivor_mask"]
    # All pinned positions survive.
    assert bool((pinned & survivor_mask)[pinned].all().item())
    # Total non-pinned survivor count is approximately retention * non-pinned.
    n_non_pinned = int((~pinned).sum().item())
    n_non_pinned_survivors = int((~pinned & survivor_mask).sum().item())
    # The calibrator-driven threshold may differ slightly from the exact
    # ratio because it operates over EMA quantiles, but it should still
    # land within a reasonable bound around the configured retention.
    expected = max(1, round(n_non_pinned * 0.30))
    assert abs(n_non_pinned_survivors - expected) <= max(2, expected // 2), (
        f"got {n_non_pinned_survivors} survivors, expected ~{expected}"
    )


def test_l1_path_falls_back_to_topk_before_warmup():
    """A cold calibrator (warmup not yet reached) must use top-K fallback
    so the trainer doesn't crash on the first batch with all-zero
    survivor masks."""
    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer.encoder = _StubEncoder(8)
    trainer.encoder_tokenizer = _StubTokenizer()
    trainer.ice = _StubICE()
    trainer._l1_retention = 0.20
    trainer._l1_calibrator = ThresholdCalibrator(
        quantile_points=21, ema_decay=0.9, warmup_batches=100,  # high warmup
    )
    trainer._resolve_article_ids = lambda dataset, ids: list(ids)
    l0_batch = torch.zeros(2, 8, 8)
    l0_mask = torch.ones(2, 8, dtype=torch.bool)
    trainer._l0_for_articles = lambda dataset, ids: (l0_batch, l0_mask)
    out = trainer._prepare_l1_turn("toy", ["x", "y"], "?")
    assert out is not None
    pinned = out["pinned"]
    survivor_mask = out["survivor_mask"]
    n_non_pinned_survivors = int((~pinned & survivor_mask).sum().item())
    # 16 non-pinned positions × 0.20 → 4 (rounded up). Top-K fallback
    # should pick exactly 4.
    assert n_non_pinned_survivors == 4
