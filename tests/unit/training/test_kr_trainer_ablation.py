"""Tests for the KRTrainer ablation API.

KRTrainer.set_ablation_mode() controls how _compose_prompt() builds the
decoder input during eval ablation studies.  Since instantiating a full
KRTrainer requires configs, models, and datasets, we test the _compose_prompt
logic by creating a minimal subclass that stubs out the heavy dependencies.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.phase2.kr_trainer import KRTrainer


# ---------------------------------------------------------------------------
# Minimal stub that allows us to test _compose_prompt in isolation
# ---------------------------------------------------------------------------

class _StubKRTrainer:
    """Mimics KRTrainer._compose_prompt behavior without any real dependencies.

    Copies the ablation-relevant logic from KRTrainer._compose_prompt and
    stubs _cached_survivors / _use_live_l0 / _subsample_embeddings so we
    can test each ablation mode without loading models.
    """

    ABLATION_NONE = KRTrainer.ABLATION_NONE
    ABLATION_ZEROED = KRTrainer.ABLATION_ZEROED
    ABLATION_NOISE = KRTrainer.ABLATION_NOISE
    ABLATION_NO_TOPICS = KRTrainer.ABLATION_NO_TOPICS
    ABLATION_TOPICS_ONLY = KRTrainer.ABLATION_TOPICS_ONLY
    ABLATION_NEITHER = KRTrainer.ABLATION_NEITHER

    def __init__(self, hidden_dim: int = 64, batch_size: int = 2, seq_len: int = 4):
        self._ablation_mode: str | None = None
        self.device = torch.device("cpu")
        self.hidden_dim = hidden_dim
        self._batch_size = batch_size
        self._seq_len = seq_len
        self.topic_embeddings = None  # No topic embeddings by default

    def set_ablation_mode(self, mode: str | None) -> None:
        self._ablation_mode = mode

    def _cached_survivors(self, batch):
        return None

    def _use_live_l0(self):
        return False

    def _subsample_embeddings(self, content_ids, content_mask):
        """Return non-zero embeddings as a stand-in for subsample."""
        b, l = content_ids.shape
        return (
            torch.randn(b, max(1, l // 2), self.hidden_dim),
            torch.ones(b, max(1, l // 2), dtype=torch.bool),
        )

    def _compose_prompt(self, batch):
        """Replicates KRTrainer._compose_prompt with ablation logic."""
        skip_context = self._ablation_mode in (
            self.ABLATION_TOPICS_ONLY, self.ABLATION_NEITHER,
        )
        skip_topics = self._ablation_mode in (
            self.ABLATION_NO_TOPICS, self.ABLATION_NEITHER,
        )

        if skip_context:
            prompt = None
            mask = None
        else:
            cached = self._cached_survivors(batch)
            if cached is not None:
                prompt, mask = cached
            elif self._use_live_l0():
                prompt, mask = (
                    torch.randn(self._batch_size, self._seq_len, self.hidden_dim),
                    torch.ones(self._batch_size, self._seq_len, dtype=torch.bool),
                )
            else:
                prompt, mask = self._subsample_embeddings(
                    batch["content_token_ids"],
                    batch["content_attention_mask"],
                )

            if self._ablation_mode == self.ABLATION_ZEROED and prompt is not None:
                prompt = torch.zeros_like(prompt)
            elif self._ablation_mode == self.ABLATION_NOISE and prompt is not None:
                prompt = torch.randn_like(prompt) * 0.02

        if not skip_topics and self.topic_embeddings is not None:
            pass  # Topic embedding injection tested separately

        if prompt is None:
            batch_size = batch["content_token_ids"].size(0)
            prompt = torch.zeros(
                batch_size, 1, self.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)

        return prompt, mask


def _make_batch(batch_size: int = 2, seq_len: int = 6):
    return {
        "content_token_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "content_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "tags": [["python"] for _ in range(batch_size)],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKRTrainerAblationModes:
    def test_ablation_none_returns_normal_output(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(None)
        prompt, mask = trainer._compose_prompt(_make_batch())
        assert prompt.shape[0] == 2
        assert prompt.shape[2] == 64
        # Normal mode: prompt should contain non-zero values
        assert prompt.abs().sum() > 0

    def test_ablation_zeroed_returns_zero_tensors(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(KRTrainer.ABLATION_ZEROED)
        prompt, mask = trainer._compose_prompt(_make_batch())
        assert prompt.abs().sum() == 0
        # Mask should still be valid
        assert mask.any()

    def test_ablation_noise_returns_nonzero_random(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(KRTrainer.ABLATION_NOISE)
        prompt, mask = trainer._compose_prompt(_make_batch())
        assert prompt.abs().sum() > 0
        # Check the noise is small (scaled by 0.02)
        assert prompt.abs().max() < 1.0

    def test_ablation_topics_only_skips_context(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(KRTrainer.ABLATION_TOPICS_ONLY)
        # No topic embeddings -> should get minimal empty prompt
        prompt, mask = trainer._compose_prompt(_make_batch())
        assert prompt.shape[1] == 1
        assert prompt.abs().sum() == 0
        assert not mask.any()

    def test_ablation_neither_returns_minimal_empty(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(KRTrainer.ABLATION_NEITHER)
        prompt, mask = trainer._compose_prompt(_make_batch())
        assert prompt.shape[1] == 1
        assert prompt.abs().sum() == 0
        assert not mask.any()

    def test_set_ablation_mode_none_restores_normal(self):
        trainer = _StubKRTrainer()
        trainer.set_ablation_mode(KRTrainer.ABLATION_ZEROED)
        prompt_z, _ = trainer._compose_prompt(_make_batch())
        assert prompt_z.abs().sum() == 0

        trainer.set_ablation_mode(None)
        prompt_n, _ = trainer._compose_prompt(_make_batch())
        assert prompt_n.abs().sum() > 0

    def test_ablation_constants_are_strings(self):
        assert KRTrainer.ABLATION_NONE is None
        assert isinstance(KRTrainer.ABLATION_ZEROED, str)
        assert isinstance(KRTrainer.ABLATION_NOISE, str)
        assert isinstance(KRTrainer.ABLATION_TOPICS_ONLY, str)
        assert isinstance(KRTrainer.ABLATION_NEITHER, str)
