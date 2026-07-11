"""Tests for Phase 3 DistillationTrainer — packed rewrite.

Covers:
- _collate: packed output shape and field correctness
- _get_bgkit_context: packed (K_total, D) + cu_seqlens output
- _build_decoder_inputs: prefix_ids / suffix_ids / loss_mask_parts correctness
- _make_flat_loss_mask: shape and content
- _forward_backward: runs one step with mock decoder, produces non-NaN scalar loss
- evaluate: runs without error with mock decoder
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from bgkit.training.phase3.distillation_trainer import (
    DistillationTrainer,
    _ContextSourceCache,
    _make_cu_seqlens,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
VOCAB_SIZE = 512
B = 3  # batch size


# ---------------------------------------------------------------------------
# Minimal mock decoder (stands in for ReconstructionDecoder)
# ---------------------------------------------------------------------------


class _MockDecoderBackbone(nn.Module):
    """Minimal backbone that forward_with_single_splice can call."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _MockDecoder(nn.Module):
    """Mock ReconstructionDecoder that records calls and returns a fixed loss."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = _MockDecoderBackbone(hidden_dim=hidden_dim)
        self._call_kwargs: list[dict] = []

    def forward_with_single_splice(
        self,
        *,
        survivor_embeddings,
        survivor_cu_seqlens,
        prefix_ids,
        suffix_ids,
        loss_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        self._call_kwargs.append(
            {
                "survivor_embeddings": survivor_embeddings,
                "survivor_cu_seqlens": survivor_cu_seqlens,
                "prefix_ids": prefix_ids,
                "suffix_ids": suffix_ids,
                "loss_mask": loss_mask,
            }
        )
        # Return a differentiable scalar so backward() can be called.
        return survivor_embeddings.sum() * 0.0 + torch.tensor(
            1.5, dtype=torch.float32, requires_grad=True
        )

    def train(self, mode=True):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# Trainer stub builder
# ---------------------------------------------------------------------------


def _make_trainer_stub(
    batch_size: int = B,
    hidden_dim: int = HIDDEN_DIM,
    no_injection_fraction: float = 0.0,
) -> DistillationTrainer:
    """Build a DistillationTrainer with mocked internals (no real model/tokenizer)."""
    trainer = DistillationTrainer.__new__(DistillationTrainer)
    trainer.device = torch.device("cpu")
    trainer._no_injection_fraction = no_injection_fraction
    trainer._bgkit_cache_dir = None
    trainer._fs_cache = None
    trainer._git_cache = None
    trainer._session_cache = None
    trainer._accum_steps = 1

    # Fake tokenizer prefix/suffix ids
    trainer._distill_bgkit_prefix_ids = torch.tensor([100, 101], dtype=torch.long)
    trainer._distill_bgkit_suffix_ids = torch.tensor([200, 201], dtype=torch.long)

    # Mock decoder
    trainer.decoder = _MockDecoder(hidden_dim=hidden_dim)
    trainer.encoder = None

    return trainer


# ---------------------------------------------------------------------------
# Helper to build a packed batch dict
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = B,
    traj_lens: list[int] | None = None,
    issue_lens: list[int] | None = None,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    """Build a minimal packed batch as _collate would produce."""
    if traj_lens is None:
        traj_lens = [5, 8, 3]
    if issue_lens is None:
        issue_lens = [4, 2, 6]

    repos = [f"org/repo{i}" for i in range(batch_size)]
    base_commits = [f"abc{i}" * 8 for i in range(batch_size)]

    traj_seqs = [torch.randint(0, vocab_size, (ln,)) for ln in traj_lens]
    traj_cu = _make_cu_seqlens(traj_lens)
    traj_total = int(traj_cu[-1])
    from bgkit.utils.packing import position_ids_from_cu
    traj_pos = position_ids_from_cu(traj_cu, traj_total)

    issue_seqs = [torch.randint(0, vocab_size, (ln,)) for ln in issue_lens]
    issue_cu = _make_cu_seqlens(issue_lens)
    issue_total = int(issue_cu[-1])
    issue_pos = position_ids_from_cu(issue_cu, issue_total)

    return {
        "instance_ids": [f"inst_{i}" for i in range(batch_size)],
        "repos": repos,
        "base_commits": base_commits,
        "issue_texts": [f"issue text {i}" for i in range(batch_size)],
        "trajectory_token_ids": torch.cat(traj_seqs),
        "trajectory_cu_seqlens": traj_cu,
        "trajectory_position_ids": traj_pos,
        "trajectory_max_seqlen": max(traj_lens),
        "issue_token_ids": torch.cat(issue_seqs),
        "issue_cu_seqlens": issue_cu,
        "issue_position_ids": issue_pos,
        "issue_max_seqlen": max(issue_lens),
    }


# ---------------------------------------------------------------------------
# Tests: _collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_output_has_packed_trajectory(self):
        trainer = _make_trainer_stub()
        traj_lens = [5, 8, 3]
        issue_lens = [4, 2, 6]
        samples = [
            {
                "instance_id": f"id_{i}",
                "repo": f"repo{i}",
                "base_commit": f"sha{i}",
                "issue_text": f"issue {i}",
                "trajectory_token_ids": torch.randint(0, 100, (traj_lens[i],)),
                "issue_token_ids": torch.randint(0, 100, (issue_lens[i],)),
            }
            for i in range(len(traj_lens))
        ]
        result = trainer._collate(samples)

        assert "trajectory_token_ids" in result
        assert "trajectory_cu_seqlens" in result
        assert "trajectory_position_ids" in result
        assert "trajectory_max_seqlen" in result

        assert result["trajectory_token_ids"].shape == (sum(traj_lens),)
        assert result["trajectory_cu_seqlens"].shape == (len(traj_lens) + 1,)
        assert int(result["trajectory_cu_seqlens"][-1].item()) == sum(traj_lens)

    def test_output_has_packed_issue(self):
        trainer = _make_trainer_stub()
        issue_lens = [3, 7, 2]
        samples = [
            {
                "instance_id": f"id_{i}",
                "repo": f"repo{i}",
                "base_commit": f"sha{i}",
                "issue_text": f"issue {i}",
                "issue_token_ids": torch.randint(0, 100, (issue_lens[i],)),
            }
            for i in range(len(issue_lens))
        ]
        result = trainer._collate(samples)

        assert "issue_token_ids" in result
        assert "issue_cu_seqlens" in result
        assert result["issue_token_ids"].shape == (sum(issue_lens),)

    def test_no_padding_tokens(self):
        """Flat buffer should have exactly sum(lengths) tokens — no padding."""
        trainer = _make_trainer_stub()
        traj_lens = [10, 3, 7]
        samples = [
            {
                "instance_id": f"id_{i}",
                "repo": f"repo{i}",
                "base_commit": "sha",
                "issue_text": "x",
                "trajectory_token_ids": torch.randint(0, 50, (traj_lens[i],)),
            }
            for i in range(len(traj_lens))
        ]
        result = trainer._collate(samples)
        assert result["trajectory_token_ids"].shape[0] == sum(traj_lens)

    def test_position_ids_restart_per_sample(self):
        trainer = _make_trainer_stub()
        traj_lens = [4, 3]
        samples = [
            {
                "instance_id": f"id_{i}",
                "repo": f"r{i}",
                "base_commit": "sha",
                "issue_text": "x",
                "trajectory_token_ids": torch.arange(traj_lens[i]),
            }
            for i in range(len(traj_lens))
        ]
        result = trainer._collate(samples)
        pos = result["trajectory_position_ids"]
        # Sample 0: [0,1,2,3], sample 1: [0,1,2]
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)
        assert torch.equal(pos, expected)

    def test_cu_seqlens_dtype_is_int32(self):
        trainer = _make_trainer_stub()
        samples = [
            {
                "instance_id": "id_0",
                "repo": "r",
                "base_commit": "sha",
                "issue_text": "x",
                "trajectory_token_ids": torch.arange(5),
            }
        ]
        result = trainer._collate(samples)
        assert result["trajectory_cu_seqlens"].dtype == torch.int32

    def test_no_attention_mask_produced(self):
        """Packed collator must not produce attention_mask."""
        trainer = _make_trainer_stub()
        samples = [
            {
                "instance_id": "id_0",
                "repo": "r",
                "base_commit": "sha",
                "issue_text": "x",
                "trajectory_token_ids": torch.arange(5),
                "issue_token_ids": torch.arange(3),
            }
        ]
        result = trainer._collate(samples)
        assert "trajectory_attention_mask" not in result
        assert "issue_attention_mask" not in result


# ---------------------------------------------------------------------------
# Tests: _get_bgkit_context
# ---------------------------------------------------------------------------


class TestGetBgkitContext:
    def test_returns_none_when_no_caches(self):
        trainer = _make_trainer_stub()
        batch = _make_batch()
        result = trainer._get_bgkit_context(batch)
        assert result is None

    def test_returns_none_when_caches_empty(self):
        trainer = _make_trainer_stub()
        # Create a cache that returns no rows
        cache = _ContextSourceCache.__new__(_ContextSourceCache)
        cache._key_columns = ["repo"]
        cache._index = {}
        trainer._git_cache = cache

        batch = _make_batch(batch_size=2)
        result = trainer._get_bgkit_context(batch)
        assert result is None

    def test_returns_flat_and_cu_seqlens_when_data_available(self, tmp_path):
        """When cache has survivors, returns (K_total, D) flat + (B+1,) cu_seqlens."""
        trainer = _make_trainer_stub(batch_size=2)

        # Build a fake git cache with npy data
        npy_dir = tmp_path / "git_history"
        npy_dir.mkdir()
        surv_data = torch.randn(5, HIDDEN_DIM).numpy()
        import numpy as np

        npy_path = npy_dir / "repo0.npy"
        np.save(str(npy_path), surv_data)

        cache = _ContextSourceCache.__new__(_ContextSourceCache)
        cache._key_columns = ["repo"]
        cache._index = {
            ("org/repo0",): [{"path": str(npy_path)}],
        }
        trainer._git_cache = cache

        batch = _make_batch(batch_size=2, traj_lens=[3, 4], issue_lens=[2, 2])
        batch["repos"] = ["org/repo0", "org/repo1"]

        result = trainer._get_bgkit_context(batch)
        # repo0 has 5 survivors, repo1 has 0
        assert result is not None
        flat, surv_cu = result
        assert flat.shape == (5, HIDDEN_DIM)
        assert surv_cu.shape == (3,)  # (B+1,) = (2+1,)
        assert int(surv_cu[0].item()) == 0
        assert int(surv_cu[1].item()) == 5  # repo0
        assert int(surv_cu[2].item()) == 5  # repo1 contributes 0

    def test_output_flat_is_2d(self, tmp_path):
        """Flat embeddings must be (K, D) with D > 1."""
        trainer = _make_trainer_stub(batch_size=1)
        npy_dir = tmp_path / "git"
        npy_dir.mkdir()
        import numpy as np

        data = torch.randn(3, HIDDEN_DIM).numpy()
        npy_path = npy_dir / "r.npy"
        np.save(str(npy_path), data)

        cache = _ContextSourceCache.__new__(_ContextSourceCache)
        cache._key_columns = ["repo"]
        cache._index = {("repo0",): [{"path": str(npy_path)}]}
        trainer._git_cache = cache

        batch = _make_batch(batch_size=1, traj_lens=[4], issue_lens=[2])
        batch["repos"] = ["repo0"]

        result = trainer._get_bgkit_context(batch)
        assert result is not None
        flat, _cu = result
        assert flat.ndim == 2
        assert flat.shape[1] == HIDDEN_DIM

    def test_prior_sessions_use_timestamps_not_commit_sha_order(self, tmp_path):
        trainer = _make_trainer_stub(batch_size=1)
        import numpy as np

        old_path = tmp_path / "old.npy"
        future_path = tmp_path / "future.npy"
        np.save(old_path, torch.ones(2, HIDDEN_DIM).numpy())
        np.save(future_path, torch.ones(3, HIDDEN_DIM).numpy())
        cache = _ContextSourceCache.__new__(_ContextSourceCache)
        cache._key_columns = ["repo"]
        cache._index = {
            ("repo",): [
                {"path": str(old_path), "timestamp": 100},
                # Lexically smaller SHA-like values must not affect ordering.
                {"path": str(future_path), "timestamp": 300},
            ],
        }

        survivors = trainer._load_source_survivors(
            cache, "repo", before_timestamp=200,
        )
        assert survivors is not None
        assert survivors.shape == (2, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Tests: _build_decoder_inputs
# ---------------------------------------------------------------------------


class TestBuildDecoderInputs:
    def test_returns_b_prefix_and_suffix_lists(self):
        trainer = _make_trainer_stub()
        batch = _make_batch()
        prefix_ids, suffix_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)

        assert len(prefix_ids) == B
        assert len(suffix_ids) == B
        assert len(loss_mask_parts) == B

    def test_prefix_contains_issue_and_bgkit_prefix(self):
        """prefix_ids[i] = [issue_i | bgkit_prefix_ids]"""
        trainer = _make_trainer_stub()
        issue_lens = [3, 5, 2]
        batch = _make_batch(batch_size=3, issue_lens=issue_lens)

        prefix_ids, _suf, _lm = trainer._build_decoder_inputs(batch)

        prefix_extra = trainer._distill_bgkit_prefix_ids.size(0)
        for i, pre in enumerate(prefix_ids):
            expected_len = issue_lens[i] + prefix_extra
            assert pre.size(0) == expected_len, (
                f"sample {i}: expected {expected_len}, got {pre.size(0)}"
            )

    def test_suffix_contains_bgkit_suffix_and_trajectory(self):
        """suffix_ids[i] = [bgkit_suffix | traj_i]"""
        trainer = _make_trainer_stub()
        traj_lens = [5, 8, 3]
        batch = _make_batch(batch_size=3, traj_lens=traj_lens)

        _pre, suffix_ids, _lm = trainer._build_decoder_inputs(batch)

        suffix_extra = trainer._distill_bgkit_suffix_ids.size(0)
        for i, suf in enumerate(suffix_ids):
            expected_len = traj_lens[i] + suffix_extra
            assert suf.size(0) == expected_len, (
                f"sample {i}: expected {expected_len}, got {suf.size(0)}"
            )

    def test_no_sentinel_tokens_in_prefix(self):
        """The sentinel tokens must not appear in prefix_ids (dropped in packed path)."""
        trainer = _make_trainer_stub()
        issue_lens = [3, 5, 2]
        batch_check = _make_batch(batch_size=3, issue_lens=issue_lens)
        prefix_ids_check, _, _ = trainer._build_decoder_inputs(batch_check)
        prefix_extra = trainer._distill_bgkit_prefix_ids.size(0)
        # Each prefix must be exactly [issue_i | bgkit_prefix]; no extra sentinel tokens.
        for i, pre in enumerate(prefix_ids_check):
            assert pre.size(0) == issue_lens[i] + prefix_extra

    def test_loss_mask_covers_only_trajectory_tokens(self):
        """In lm_suf: first bgkit_suffix_len tokens are False, rest are True."""
        trainer = _make_trainer_stub()
        batch = _make_batch(batch_size=1, traj_lens=[6], issue_lens=[3])

        _pre, _suffix_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)

        suf_header_len = trainer._distill_bgkit_suffix_ids.size(0)
        _pre_len, _suf_len, lm_suf = loss_mask_parts[0]

        # bgkit suffix wrapper: False
        assert lm_suf[:suf_header_len].all() == False  # noqa: E712
        # trajectory tokens: True
        assert lm_suf[suf_header_len:].all() == True  # noqa: E712

    def test_handles_missing_issue(self):
        """Works when batch has no issue_token_ids."""
        trainer = _make_trainer_stub()
        batch = {
            "repos": ["r0"],
            "base_commits": ["abc"],
            "trajectory_token_ids": torch.tensor([1, 2, 3]),
            "trajectory_cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        }
        prefix_ids, _suf_ids, _lm_parts = trainer._build_decoder_inputs(batch)
        assert len(prefix_ids) == 1
        # prefix = only bgkit_prefix (no issue)
        assert prefix_ids[0].size(0) == trainer._distill_bgkit_prefix_ids.size(0)

    def test_handles_missing_trajectory(self):
        """Works when batch has no trajectory_token_ids."""
        trainer = _make_trainer_stub()
        batch = {
            "repos": ["r0"],
            "base_commits": ["abc"],
            "issue_token_ids": torch.tensor([10, 20]),
            "issue_cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
        }
        _pre_ids, suffix_ids, _lm_parts = trainer._build_decoder_inputs(batch)
        # suffix = only bgkit_suffix (no trajectory)
        assert suffix_ids[0].size(0) == trainer._distill_bgkit_suffix_ids.size(0)


# ---------------------------------------------------------------------------
# Tests: _make_flat_loss_mask
# ---------------------------------------------------------------------------


class TestMakeFlatLossMask:
    def test_shape_matches_total_tokens(self):
        trainer = _make_trainer_stub()
        traj_lens = [5, 3, 7]
        issue_lens = [2, 4, 1]
        batch = _make_batch(batch_size=3, traj_lens=traj_lens, issue_lens=issue_lens)

        prefix_ids, suffix_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)

        # No survivors
        survivor_cu = _make_cu_seqlens([0, 0, 0])
        flat_lm = trainer._make_flat_loss_mask(loss_mask_parts, survivor_cu)

        # Total = sum(pre_i + 0 + suf_i)
        expected = sum(
            pre.size(0) + suf.size(0)
            for pre, suf in zip(prefix_ids, suffix_ids, strict=True)
        )
        assert flat_lm.shape == (expected,)

    def test_shape_matches_with_survivors(self):
        trainer = _make_trainer_stub()
        traj_lens = [5, 3]
        issue_lens = [2, 4]
        batch = _make_batch(batch_size=2, traj_lens=traj_lens, issue_lens=issue_lens)

        prefix_ids, suffix_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)

        surv_counts = [7, 4]
        survivor_cu = _make_cu_seqlens(surv_counts)
        flat_lm = trainer._make_flat_loss_mask(loss_mask_parts, survivor_cu)

        expected = sum(
            pre.size(0) + k + suf.size(0)
            for pre, suf, k in zip(prefix_ids, suffix_ids, surv_counts, strict=True)
        )
        assert flat_lm.shape == (expected,)

    def test_only_trajectory_positions_are_true(self):
        """Only trajectory tokens inside the suffix should be True."""
        trainer = _make_trainer_stub()
        traj_len = 5
        issue_len = 3
        batch = _make_batch(batch_size=1, traj_lens=[traj_len], issue_lens=[issue_len])

        prefix_ids, _suf_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)
        pre_len = prefix_ids[0].size(0)
        suf_header_len = trainer._distill_bgkit_suffix_ids.size(0)

        # 2 survivors
        survivor_cu = _make_cu_seqlens([2])
        flat_lm = trainer._make_flat_loss_mask(loss_mask_parts, survivor_cu)

        # Layout: [pre | surv | suf_header | traj]
        # False zones: pre, surv, suf_header
        assert not flat_lm[:pre_len].any()
        assert not flat_lm[pre_len : pre_len + 2].any()  # survivors
        assert not flat_lm[pre_len + 2 : pre_len + 2 + suf_header_len].any()
        # True zone: trajectory tokens
        traj_start = pre_len + 2 + suf_header_len
        assert flat_lm[traj_start : traj_start + traj_len].all()

    def test_total_true_count_equals_sum_traj_lens(self):
        trainer = _make_trainer_stub()
        traj_lens = [4, 7, 3]
        batch = _make_batch(batch_size=3, traj_lens=traj_lens, issue_lens=[2, 2, 2])

        _pre_ids, _suf_ids, loss_mask_parts = trainer._build_decoder_inputs(batch)
        survivor_cu = _make_cu_seqlens([0, 0, 0])
        flat_lm = trainer._make_flat_loss_mask(loss_mask_parts, survivor_cu)

        assert int(flat_lm.sum().item()) == sum(traj_lens)


# ---------------------------------------------------------------------------
# Tests: _forward_backward (one step, non-NaN loss)
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def _make_trainer_with_mock_decoder(self, no_injection: float = 0.0) -> DistillationTrainer:
        trainer = _make_trainer_stub(no_injection_fraction=no_injection)
        return trainer

    def test_returns_non_nan_loss(self):
        trainer = self._make_trainer_with_mock_decoder()
        batch = _make_batch()
        metrics = trainer._forward_backward(batch)

        assert "loss" in metrics
        loss_val = float(metrics["loss"])
        assert loss_val == loss_val, f"loss is NaN: {loss_val}"

    def test_forward_with_injection_passes_flat_embeddings(self, tmp_path):
        """When context is available, decoder receives (K_total, D) flat embeddings."""
        trainer = self._make_trainer_with_mock_decoder(no_injection=0.0)

        # Add a fake git cache with survivors
        import numpy as np

        npy_dir = tmp_path / "git"
        npy_dir.mkdir()
        data = torch.randn(4, HIDDEN_DIM).numpy()
        npy_path = npy_dir / "r.npy"
        np.save(str(npy_path), data)

        cache = _ContextSourceCache.__new__(_ContextSourceCache)
        cache._key_columns = ["repo"]
        cache._index = {("org/repo0",): [{"path": str(npy_path)}]}
        trainer._git_cache = cache

        batch = _make_batch(batch_size=2, traj_lens=[3, 3], issue_lens=[2, 2])
        batch["repos"] = ["org/repo0", "org/repo1"]

        # Force inject=True by patching random.random to return > no_injection_fraction
        with patch("bgkit.training.phase3.distillation_trainer.random.random", return_value=1.0):
            metrics = trainer._forward_backward(batch)

        assert "loss" in metrics
        # Decoder should have been called with some survivors from repo0
        kwargs = trainer.decoder._call_kwargs[-1]
        assert kwargs["survivor_embeddings"].shape[0] == 4  # 4 survivors from repo0

    def test_forward_without_injection_passes_empty_survivors(self):
        """No injection: survivor_embeddings is shape (0, D) and cu_seqlens all zeros."""
        trainer = self._make_trainer_with_mock_decoder()

        batch = _make_batch()

        # Force inject=False
        with patch("bgkit.training.phase3.distillation_trainer.random.random", return_value=0.0):
            trainer._forward_backward(batch)

        # Check the mock decoder was called
        kwargs = trainer.decoder._call_kwargs[-1]
        surv = kwargs["survivor_embeddings"]
        cu = kwargs["survivor_cu_seqlens"]
        assert surv.shape[0] == 0  # no survivors
        assert int(cu[-1].item()) == 0  # all zero
        # The baseline arm omits the synthetic BgKIT tool wrapper too.
        issue_lengths = (
            batch["issue_cu_seqlens"][1:] - batch["issue_cu_seqlens"][:-1]
        ).tolist()
        trajectory_lengths = (
            batch["trajectory_cu_seqlens"][1:]
            - batch["trajectory_cu_seqlens"][:-1]
        ).tolist()
        assert [len(x) for x in kwargs["prefix_ids"]] == issue_lengths
        assert [len(x) for x in kwargs["suffix_ids"]] == trajectory_lengths

    def test_decoder_called_with_lists_for_prefix_and_suffix(self):
        """prefix_ids and suffix_ids must be Python lists (not tensors)."""
        trainer = self._make_trainer_with_mock_decoder()
        batch = _make_batch(batch_size=2)

        trainer._forward_backward(batch)

        kwargs = trainer.decoder._call_kwargs[-1]
        assert isinstance(kwargs["prefix_ids"], list)
        assert isinstance(kwargs["suffix_ids"], list)
        assert len(kwargs["prefix_ids"]) == 2
        assert len(kwargs["suffix_ids"]) == 2

    def test_injected_flag_reported(self):
        trainer = self._make_trainer_with_mock_decoder()
        batch = _make_batch()

        with patch("bgkit.training.phase3.distillation_trainer.random.random", return_value=1.0):
            metrics_injected = trainer._forward_backward(batch)

        # Clear call log
        trainer.decoder._call_kwargs.clear()

        with patch("bgkit.training.phase3.distillation_trainer.random.random", return_value=0.0):
            metrics_not_injected = trainer._forward_backward(batch)

        assert metrics_injected["injected"] == 1.0
        assert metrics_not_injected["injected"] == 0.0


# ---------------------------------------------------------------------------
# Tests: evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _make_eval_trainer(self) -> tuple[DistillationTrainer, list[dict]]:
        trainer = _make_trainer_stub()
        trainer.model = nn.ModuleDict({"decoder": trainer.decoder})
        # Two tiny batches for the eval dataloader
        batches = [
            _make_batch(batch_size=2, traj_lens=[4, 3], issue_lens=[2, 3]),
            _make_batch(batch_size=1, traj_lens=[6], issue_lens=[2]),
        ]
        trainer.eval_dataloader = batches  # DataLoader duck-typing
        return trainer, batches

    def test_returns_eval_loss_key(self):
        trainer, _batches = self._make_eval_trainer()
        metrics = trainer.evaluate()
        assert "eval/loss" in metrics

    def test_eval_loss_is_finite(self):
        trainer, _batches = self._make_eval_trainer()
        metrics = trainer.evaluate()
        val = metrics["eval/loss"]
        import math
        assert val == val, f"eval/loss is NaN: {val}"
        assert math.isfinite(val)

    def test_context_coverage_key_present(self):
        trainer, _batches = self._make_eval_trainer()
        metrics = trainer.evaluate()
        assert "eval/context_coverage" in metrics

    def test_evaluation_is_paired_on_every_batch(self):
        trainer, batches = self._make_eval_trainer()
        metrics = trainer.evaluate()
        assert len(trainer.decoder._call_kwargs) == 2 * len(batches)
        assert "eval/loss_with_context" in metrics
        assert "eval/loss_no_context" in metrics
        assert "eval/context_delta" in metrics

    def test_evaluate_sets_model_to_train_after(self):
        trainer, _batches = self._make_eval_trainer()
        trainer.evaluate()
        # Model should be back in train mode (evaluate() calls self.model.train())
        assert trainer.decoder.training


# ---------------------------------------------------------------------------
# Tests: _make_cu_seqlens helper
# ---------------------------------------------------------------------------


class TestMakeCuSeqlens:
    def test_shape(self):
        cu = _make_cu_seqlens([3, 5, 2])
        assert cu.shape == (4,)

    def test_starts_at_zero(self):
        cu = _make_cu_seqlens([3, 5, 2])
        assert int(cu[0].item()) == 0

    def test_ends_at_total(self):
        cu = _make_cu_seqlens([3, 5, 2])
        assert int(cu[-1].item()) == 10

    def test_dtype_is_int32(self):
        cu = _make_cu_seqlens([3, 5, 2])
        assert cu.dtype == torch.int32

    def test_empty_lengths(self):
        cu = _make_cu_seqlens([])
        assert cu.shape == (1,)
        assert int(cu[0].item()) == 0
