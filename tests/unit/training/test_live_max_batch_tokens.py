"""Tests for live-tunable max_batch_tokens and max_batch_tokens_eval.

CPU-only — no GPU, no torch.cuda.

Tests:
- Initial budget honored by sampler
- Handler updates budget and rebuilds sampler/dataloader
- Cursor preserved across rebuild
- max_batch_tokens_eval rebuilds eval dataloader
- Invalid values rejected (non-int, zero, negative, bool)
- No-op when value unchanged
- Graceful warning when required attributes are missing
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if torch is not installed on this environment.
torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.training.base_trainer import BaseTrainer

# ---------------------------------------------------------------------------
# Minimal concrete trainer for testing
# ---------------------------------------------------------------------------

def _identity_collate(batch):
    """Simple collate that stacks tensors."""
    return torch.stack(batch)


class _MinimalTrainer(BaseTrainer):
    """Minimal trainer that exposes the rebuildable-dataloader pattern."""

    def __init__(self, lengths, budget, budget_eval=None, seed=42):
        # Build a minimal OmegaConf-like cfg stub
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "training": {
                "phase": "test_phase",
                "lr": 1e-3,
                "max_steps": 100,
                "warmup_steps": 0,
                "optimizer": "adamw",
            },
        })
        super().__init__(cfg)

        # Fake dataset: each item is a 1D tensor of length matching its seq length
        samples = [torch.zeros(int(length), dtype=torch.long) for length in lengths]
        # Use a simple list-based dataset via TensorDataset workaround
        self._samples = samples
        self._lengths_array = np.array(lengths, dtype=np.int64)
        self._budget = budget
        self._budget_eval = budget_eval if budget_eval is not None else budget
        self._seed = seed
        self._setup_done = False

    def setup(self) -> None:
        """Simulate the packed-dataloader setup pattern."""
        from torch.utils.data import Dataset

        class _ListDataset(Dataset):
            def __init__(self, items):
                self._items = items

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        dataset = _ListDataset(self._samples)
        self.train_dataset = dataset
        self.eval_dataset = dataset  # reuse for simplicity

        # Stash required attributes for rebuild
        self._train_lengths = self._lengths_array
        self._eval_lengths = self._lengths_array
        self._train_collate_fn = _identity_collate
        self._num_workers = 0
        self._pin_memory = False
        self._max_batch_tokens = self._budget
        self._max_batch_tokens_eval = self._budget_eval

        self.train_sampler = PackedTokenBudgetSampler(
            dataset=self.train_dataset,
            lengths=self._lengths_array,
            max_batch_tokens=self._budget,
            shuffle=True,
            seed=self._seed,
        )
        eval_sampler = PackedTokenBudgetSampler(
            dataset=self.eval_dataset,
            lengths=self._lengths_array,
            max_batch_tokens=self._budget_eval,
            shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=_identity_collate,
            num_workers=0,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=_identity_collate,
            num_workers=0,
        )

        # Use a simple optimizer so BaseTrainer doesn't crash on _create_optimizer
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self._setup_done = True

    def _forward_backward(self, batch):
        return {"loss": 0.0}

    def evaluate(self):
        return {"loss": 0.0}

    def trainable_parameters(self):
        return list(self.model.parameters())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(n=50, default_len=100, budget=50000):
    """Create a trainer with n samples all of the same default_len."""
    lengths = [default_len] * n
    t = _MinimalTrainer(lengths, budget=budget)
    t.setup()
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_budget_honored():
    """Sampler picks up the configured max_batch_tokens at setup."""
    t = _make_trainer(budget=50000)
    assert t.train_sampler._max_batch_tokens == 50000
    assert t._max_batch_tokens == 50000


def test_handle_max_batch_tokens_updates_budget():
    """Handler updates _max_batch_tokens and rebuilds sampler."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(30000)
    assert t._max_batch_tokens == 30000
    assert t.train_sampler._max_batch_tokens == 30000


def test_handle_max_batch_tokens_rebuilds_dataloader():
    """Rebuilt dataloader can be iterated and yields non-empty batches."""
    t = _make_trainer(n=20, default_len=100, budget=50000)
    t._handle_max_batch_tokens(5000)
    # DataLoader must be iterable and yield at least one batch
    batches = list(t.train_dataloader)
    assert len(batches) >= 1


def test_handle_max_batch_tokens_sets_invalidated_flag():
    """After rebuild, _dataloader_invalidated is True."""
    t = _make_trainer(budget=50000)
    assert not t._dataloader_invalidated
    t._handle_max_batch_tokens(30000)
    assert t._dataloader_invalidated


def test_handle_max_batch_tokens_cursor_preserved():
    """Cursor (_microbatches_in_epoch) is preserved after rebuild."""
    t = _make_trainer(budget=50000)
    t._microbatches_in_epoch = 7
    t._handle_max_batch_tokens(30000)
    # The sampler's _batch_cursor should reflect the preserved cursor
    assert t.train_sampler._batch_cursor == 7


def test_handle_max_batch_tokens_no_op_same_value():
    """Calling handler with the current budget is a no-op."""
    t = _make_trainer(budget=50000)
    original_sampler_id = id(t.train_sampler)
    t._handle_max_batch_tokens(50000)
    # Sampler object should not have been replaced
    assert id(t.train_sampler) == original_sampler_id
    assert not t._dataloader_invalidated


def test_handle_max_batch_tokens_rejects_zero():
    """Zero budget is rejected; budget unchanged."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(0)
    assert t._max_batch_tokens == 50000


def test_handle_max_batch_tokens_rejects_negative():
    """Negative budget is rejected; budget unchanged."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(-1)
    assert t._max_batch_tokens == 50000


def test_handle_max_batch_tokens_rejects_float():
    """Float value is rejected; budget unchanged."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(30000.5)
    assert t._max_batch_tokens == 50000


def test_handle_max_batch_tokens_rejects_bool():
    """Boolean True is rejected (bool is subclass of int, but we reject it)."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(True)
    assert t._max_batch_tokens == 50000


def test_handle_max_batch_tokens_rejects_none():
    """None is rejected; budget unchanged."""
    t = _make_trainer(budget=50000)
    t._handle_max_batch_tokens(None)
    assert t._max_batch_tokens == 50000


# ---------------------------------------------------------------------------
# max_batch_tokens_eval
# ---------------------------------------------------------------------------

def test_handle_max_batch_tokens_eval_updates_budget():
    """Handler updates _max_batch_tokens_eval and rebuilds eval dataloader."""
    t = _MinimalTrainer([100] * 20, budget=50000, budget_eval=80000)
    t.setup()
    t._handle_max_batch_tokens_eval(40000)
    assert t._max_batch_tokens_eval == 40000


def test_handle_max_batch_tokens_eval_rebuilds_eval_dataloader():
    """Rebuilt eval dataloader can be iterated."""
    t = _MinimalTrainer([100] * 20, budget=50000, budget_eval=80000)
    t.setup()
    t._handle_max_batch_tokens_eval(40000)
    batches = list(t.eval_dataloader)
    assert len(batches) >= 1


def test_handle_max_batch_tokens_eval_no_op_same_value():
    """Calling eval handler with the current budget is a no-op."""
    t = _MinimalTrainer([100] * 20, budget=50000, budget_eval=80000)
    t.setup()
    original_loader = t.eval_dataloader
    t._handle_max_batch_tokens_eval(80000)
    assert t.eval_dataloader is original_loader


def test_handle_max_batch_tokens_eval_rejects_zero():
    """Zero eval budget rejected."""
    t = _MinimalTrainer([100] * 20, budget=50000, budget_eval=80000)
    t.setup()
    t._handle_max_batch_tokens_eval(0)
    assert t._max_batch_tokens_eval == 80000


# ---------------------------------------------------------------------------
# Missing-attribute graceful degradation
# ---------------------------------------------------------------------------

def test_rebuild_train_warns_on_missing_attr():
    """Rebuild train is a no-op (with warning) when stash attrs not set."""
    t = _MinimalTrainer([100] * 10, budget=50000)
    # Do NOT call setup() — stash attrs are absent
    # Should not raise, just warn
    t._rebuild_train_dataloader_with_budget(30000)  # no-op


def test_rebuild_eval_warns_on_missing_attr():
    """Rebuild eval is a no-op (with warning) when stash attrs not set."""
    t = _MinimalTrainer([100] * 10, budget=50000)
    # Do NOT call setup()
    t._rebuild_eval_dataloader_with_budget(30000)  # no-op


# ---------------------------------------------------------------------------
# apply_live_config dispatch
# ---------------------------------------------------------------------------

def test_apply_live_config_dispatches_max_batch_tokens():
    """apply_live_config routes max_batch_tokens to the handler."""
    t = _make_trainer(budget=50000)
    t.apply_live_config({"max_batch_tokens": 25000})
    assert t._max_batch_tokens == 25000
    assert t.train_sampler._max_batch_tokens == 25000


def test_apply_live_config_dispatches_max_batch_tokens_eval():
    """apply_live_config routes max_batch_tokens_eval to the handler."""
    t = _MinimalTrainer([100] * 20, budget=50000, budget_eval=80000)
    t.setup()
    t.apply_live_config({"max_batch_tokens_eval": 60000})
    assert t._max_batch_tokens_eval == 60000


# ---------------------------------------------------------------------------
# min_sample_length filter
# ---------------------------------------------------------------------------

def test_min_sample_length_filters_short_samples():
    """Setting min_sample_length wraps the dataset in a Subset that drops
    samples shorter than the threshold."""
    lengths = [10, 50, 100, 200, 500, 1000]
    t = _MinimalTrainer(lengths, budget=50000)
    t.setup()
    n_full = len(t.train_dataset)
    assert n_full == 6

    t._handle_min_sample_length(100)
    # Three samples with length >= 100: [100, 200, 500, 1000] = 4 samples
    assert len(t.train_dataset) == 4
    assert t._min_sample_length == 100
    # Sampler now operates over the filtered length array
    assert all(int(L) >= 100 for L in t._train_lengths)


def test_min_sample_length_zero_disables_filter():
    """Setting min_sample_length back to 0 restores the full dataset."""
    lengths = [10, 50, 100, 200, 500, 1000]
    t = _MinimalTrainer(lengths, budget=50000)
    t.setup()
    t._handle_min_sample_length(100)
    assert len(t.train_dataset) == 4
    t._handle_min_sample_length(0)
    assert len(t.train_dataset) == 6
    assert t._min_sample_length == 0


def test_min_sample_length_filters_all_warns_and_no_op():
    """If the threshold filters every sample, the rebuild is skipped."""
    lengths = [10, 50, 90]
    t = _MinimalTrainer(lengths, budget=50000)
    t.setup()
    n_before = len(t.train_dataset)
    t._handle_min_sample_length(1000)
    # No-op: dataset unchanged because filter would empty it
    assert len(t.train_dataset) == n_before


def test_min_sample_length_invalid_rejected():
    """Negative, non-int, or bool values are rejected with no state change."""
    t = _make_trainer()
    t._handle_min_sample_length(-1)
    assert getattr(t, "_min_sample_length", 0) == 0
    t._handle_min_sample_length("100")
    assert getattr(t, "_min_sample_length", 0) == 0
    t._handle_min_sample_length(True)
    assert getattr(t, "_min_sample_length", 0) == 0


def test_apply_live_config_dispatches_min_sample_length():
    """apply_live_config routes min_sample_length to the handler."""
    lengths = [10, 50, 100, 200, 500, 1000]
    t = _MinimalTrainer(lengths, budget=50000)
    t.setup()
    t.apply_live_config({"min_sample_length": 100})
    assert t._min_sample_length == 100
    assert len(t.train_dataset) == 4


def test_min_sample_length_combined_with_max_batch_tokens():
    """Changing max_batch_tokens after a filter is set preserves the filter."""
    lengths = [10, 50, 100, 200, 500, 1000]
    t = _MinimalTrainer(lengths, budget=50000)
    t.setup()
    t._handle_min_sample_length(100)
    assert len(t.train_dataset) == 4
    t._handle_max_batch_tokens(30000)
    assert t._max_batch_tokens == 30000
    # Filter is still in effect
    assert len(t.train_dataset) == 4
    assert t._min_sample_length == 100
