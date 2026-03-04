"""Tests for ChatReproDataset: chat template, loss masking, variant selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import (
    CONTENT_SENTINEL,
    ChatReproDataset,
    _build_messages,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the"
        " bgkit_read_file tool for reading file contents."
    ),
    "user_prompt": "Read the file `{file_path}`",
    "compression_prompt": "Return the file contents verbatim",
    "response_prefix": "Here are the contents of `{file_path}`:",
}

ALT_VARIANT = {
    "system_prompt": "AI code helper. Use bgkit_read_file.",
    "user_prompt": "Show `{file_path}`",
    "compression_prompt": "Return file verbatim",
    "response_prefix": "Contents of `{file_path}`:",
}


class MockTokenizer:
    """Minimal tokenizer mock for testing template construction.

    Uses a simple word-level tokenization (split on characters) to make
    token boundaries predictable.
    """

    def __init__(self):
        # Simple char-level "tokenizer" for predictable behavior
        self._vocab = {}
        self._next_id = 100

    def _get_id(self, char: str) -> int:
        if char not in self._vocab:
            self._vocab[char] = self._next_id
            self._next_id += 1
        return self._vocab[char]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._get_id(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        reverse = {v: k for k, v in self._vocab.items()}
        return "".join(reverse.get(i, "?") for i in ids)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ) -> str | list[int]:
        """Simple template: join messages with role markers."""
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        result = "".join(parts)
        if tokenize:
            return self.encode(result, add_special_tokens=False)
        return result


class MockInnerDataset:
    """Mock MmapTokenDataset returning fixed samples."""

    def __init__(self, samples: list[dict]):
        self._samples = samples
        self._chunk_lengths = np.array([len(s["token_ids"]) for s in samples], dtype=np.int32)

    @property
    def lengths(self) -> np.ndarray:
        return self._chunk_lengths

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "token_ids": torch.tensor(s["token_ids"], dtype=torch.long),
            "file_path": s["file_path"],
            "language": s["language"],
        }


@pytest.fixture()
def variant_bank(tmp_path):
    """Write a variant bank JSON and return the path."""
    bank = [SEED_VARIANT, ALT_VARIANT]
    path = tmp_path / "variants.json"
    with open(path, "w") as f:
        json.dump(bank, f)
    return path


@pytest.fixture()
def inner_dataset():
    """Create a mock inner dataset with 3 samples."""
    return MockInnerDataset([
        {
            "token_ids": [1, 2, 3, 4, 5],
            "file_path": "src/main.py",
            "language": "python",
        },
        {
            "token_ids": [10, 20, 30],
            "file_path": "lib/utils.js",
            "language": "javascript",
        },
        {
            "token_ids": [100, 200, 300, 400],
            "file_path": "README.md",
            "language": "",
        },
    ])


@pytest.fixture()
def dataset(inner_dataset, variant_bank):
    """Create a ChatReproDataset with mock components."""
    tokenizer = MockTokenizer()
    return ChatReproDataset(
        inner_dataset,
        tokenizer=tokenizer,
        variant_bank_path=variant_bank,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Template construction tests
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_messages_have_correct_roles(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content here")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

    def test_file_path_in_user_message(self):
        messages = _build_messages(SEED_VARIANT, "src/foo.py", "python", "content")
        assert "src/foo.py" in messages[1]["content"]

    def test_file_path_in_response_prefix(self):
        messages = _build_messages(SEED_VARIANT, "src/foo.py", "python", "content")
        assert "src/foo.py" in messages[4]["content"]

    def test_tool_call_in_assistant_message(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content")
        assert "<tool_call>" in messages[2]["content"]
        assert "bgkit_read_file" in messages[2]["content"]

    def test_tool_response_in_user_message(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content")
        assert "<tool_response>" in messages[3]["content"]

    def test_think_block_in_final_response(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content")
        assert "<think>" in messages[4]["content"]
        assert "</think>" in messages[4]["content"]

    def test_code_fence_with_language(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "CONTENT")
        assert "```python" in messages[4]["content"]
        assert "CONTENT" in messages[4]["content"]

    def test_compression_prompt_in_tool_call(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content")
        assert "Return the file contents verbatim" in messages[2]["content"]

    def test_sentinel_preserved(self):
        messages = _build_messages(SEED_VARIANT, "test.py", "python", CONTENT_SENTINEL)
        assert CONTENT_SENTINEL in messages[4]["content"]


# ---------------------------------------------------------------------------
# ChatReproDataset tests
# ---------------------------------------------------------------------------


class TestChatReproDataset:
    def test_len_matches_inner(self, dataset, inner_dataset):
        assert len(dataset) == len(inner_dataset)

    def test_getitem_returns_expected_keys(self, dataset):
        sample = dataset[0]
        expected_keys = {
            "token_ids", "loss_mask", "content_token_ids",
            "compression_prompt_ids", "prefix_ids", "language",
        }
        assert set(sample.keys()) == expected_keys

    def test_token_ids_is_long_tensor(self, dataset):
        sample = dataset[0]
        assert sample["token_ids"].dtype == torch.long

    def test_loss_mask_shape_matches_token_ids(self, dataset):
        sample = dataset[0]
        assert sample["loss_mask"].shape == sample["token_ids"].shape

    def test_loss_mask_is_binary(self, dataset):
        sample = dataset[0]
        assert set(sample["loss_mask"].unique().tolist()).issubset({0, 1})

    def test_loss_mask_has_ones(self, dataset):
        """Loss mask should have 1s for content tokens."""
        sample = dataset[0]
        assert sample["loss_mask"].sum().item() > 0

    def test_loss_mask_content_count_matches(self, dataset, inner_dataset):
        """Number of 1s in loss_mask should equal content token count."""
        sample = dataset[0]
        inner_sample = inner_dataset[0]
        expected_content_len = len(inner_sample["token_ids"])
        actual_ones = sample["loss_mask"].sum().item()
        assert actual_ones == expected_content_len

    def test_content_token_ids_matches_inner(self, dataset, inner_dataset):
        """content_token_ids should match the inner dataset's token_ids."""
        for idx in range(len(inner_dataset)):
            chat_sample = dataset[idx]
            inner_sample = inner_dataset[idx]
            assert torch.equal(
                chat_sample["content_token_ids"],
                torch.tensor(inner_sample["token_ids"], dtype=torch.long),
            )

    def test_compression_prompt_ids_nonempty(self, dataset):
        sample = dataset[0]
        assert len(sample["compression_prompt_ids"]) > 0

    def test_prefix_ids_nonempty(self, dataset):
        sample = dataset[0]
        assert len(sample["prefix_ids"]) > 0

    def test_prefix_ids_shorter_than_token_ids(self, dataset):
        """Prefix should be shorter than full sequence (content + suffix omitted)."""
        sample = dataset[0]
        assert len(sample["prefix_ids"]) < len(sample["token_ids"])

    def test_suffix_ids_constant_1d_tensor(self, dataset):
        """suffix_ids should be a constant 1D tensor."""
        suffix = dataset.suffix_ids
        assert suffix.dim() == 1
        assert len(suffix) > 0

    def test_suffix_ids_same_across_access(self, dataset):
        """suffix_ids should return the same tensor each time."""
        assert torch.equal(dataset.suffix_ids, dataset.suffix_ids)

    def test_token_ids_longer_than_content(self, dataset, inner_dataset):
        """Chat-formatted tokens should be longer than raw content (template overhead)."""
        sample = dataset[0]
        inner_sample = inner_dataset[0]
        assert len(sample["token_ids"]) > len(inner_sample["token_ids"])


class TestVariantSelection:
    def test_deterministic_per_idx(self, dataset):
        """Same idx should always select the same variant."""
        s1 = dataset[0]
        s2 = dataset[0]
        assert torch.equal(s1["token_ids"], s2["token_ids"])

    def test_different_epochs_different_variants(self, inner_dataset, variant_bank):
        """Different epoch seeds should produce different variant selections for some samples."""
        tokenizer = MockTokenizer()
        ds = ChatReproDataset(inner_dataset, tokenizer, variant_bank, seed=0)
        s_epoch0 = ds[0]["token_ids"].clone()

        ds.set_epoch(1)
        s_epoch1 = ds[0]["token_ids"].clone()

        ds.set_epoch(2)
        s_epoch2 = ds[0]["token_ids"].clone()

        # At least one epoch should differ (with 2 variants, very likely)
        differs = (
            not torch.equal(s_epoch0, s_epoch1)
            or not torch.equal(s_epoch1, s_epoch2)
        )
        assert differs, "Variant selection should change across epochs"

    def test_variant_selection_stable_hash(self, inner_dataset, variant_bank):
        """Selection should be deterministic (not dependent on Python hash randomization)."""
        tokenizer = MockTokenizer()
        ds1 = ChatReproDataset(inner_dataset, tokenizer, variant_bank, seed=42)
        ds2 = ChatReproDataset(inner_dataset, tokenizer, variant_bank, seed=42)
        assert torch.equal(ds1[1]["token_ids"], ds2[1]["token_ids"])

    def test_different_seeds_different_schedules(self, inner_dataset, variant_bank):
        """Different base seeds should produce different variant schedules."""
        tokenizer = MockTokenizer()
        ds_a = ChatReproDataset(inner_dataset, tokenizer, variant_bank, seed=0)
        ds_b = ChatReproDataset(inner_dataset, tokenizer, variant_bank, seed=999)

        # Check across multiple samples — at least one should differ
        differs = any(
            not torch.equal(ds_a[i]["token_ids"], ds_b[i]["token_ids"])
            for i in range(len(inner_dataset))
        )
        assert differs, "Different seeds should produce different variant selections"


class TestVariantValidation:
    def test_file_path_in_all_user_prompts(self, variant_bank):
        """All variants should have {file_path} in user_prompt."""
        with open(variant_bank) as f:
            variants = json.load(f)
        for i, v in enumerate(variants):
            assert "{file_path}" in v["user_prompt"], (
                f"Variant {i} missing {{file_path}} in user_prompt"
            )

    def test_file_path_in_all_response_prefixes(self, variant_bank):
        """All variants should have {file_path} in response_prefix."""
        with open(variant_bank) as f:
            variants = json.load(f)
        for i, v in enumerate(variants):
            assert "{file_path}" in v["response_prefix"], (
                f"Variant {i} missing {{file_path}} in response_prefix"
            )


class TestLengthProperties:
    def test_lengths_includes_overhead(self, dataset, inner_dataset):
        """Lengths should be content_length + overhead (> raw content length)."""
        for i in range(len(inner_dataset)):
            assert dataset.lengths[i] > inner_dataset.lengths[i]

    def test_content_lengths_match_inner(self, dataset, inner_dataset):
        """content_lengths should exactly match inner dataset lengths."""
        np.testing.assert_array_equal(dataset.content_lengths, inner_dataset.lengths)


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestCollateChatRepro:
    def test_pads_to_max_length(self, dataset):
        """Batch should be padded to the longest sequence."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        max_len = max(len(s["token_ids"]) for s in samples)
        assert batch["token_ids"].shape == (2, max_len)
        assert batch["loss_mask"].shape == (2, max_len)
        assert batch["attention_mask"].shape == (2, max_len)

    def test_attention_mask_correct(self, dataset):
        """Attention mask should be True for real tokens, False for padding."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        for i, s in enumerate(samples):
            real_len = len(s["token_ids"])
            assert batch["attention_mask"][i, :real_len].all()
            if real_len < batch["attention_mask"].shape[1]:
                assert not batch["attention_mask"][i, real_len:].any()

    def test_loss_mask_padded_with_zeros(self, dataset):
        """Loss mask padding positions should be 0."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        for i, s in enumerate(samples):
            real_len = len(s["token_ids"])
            if real_len < batch["loss_mask"].shape[1]:
                assert (batch["loss_mask"][i, real_len:] == 0).all()

    def test_all_expected_keys(self, dataset):
        """Batch should contain all expected keys."""
        samples = [dataset[0]]
        batch = collate_chat_repro(samples)
        expected = {
            "token_ids", "attention_mask", "loss_mask",
            "content_token_ids", "content_attention_mask",
            "compression_prompt_ids", "compression_prompt_mask",
            "prefix_ids", "prefix_attention_mask",
            "languages",
        }
        assert set(batch.keys()) == expected

    def test_content_token_ids_padded(self, dataset):
        """Content token IDs should be padded to max content length."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        max_content_len = max(len(s["content_token_ids"]) for s in samples)
        assert batch["content_token_ids"].shape == (2, max_content_len)
        assert batch["content_attention_mask"].shape == (2, max_content_len)

    def test_prefix_ids_padded(self, dataset):
        """Prefix IDs should be padded to max prefix length."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        max_prefix_len = max(len(s["prefix_ids"]) for s in samples)
        assert batch["prefix_ids"].shape == (2, max_prefix_len)
        assert batch["prefix_attention_mask"].shape == (2, max_prefix_len)

    def test_prefix_attention_mask_correct(self, dataset):
        """Prefix attention mask should be True for real tokens, False for padding."""
        samples = [dataset[0], dataset[1]]
        batch = collate_chat_repro(samples)

        for i, s in enumerate(samples):
            real_len = len(s["prefix_ids"])
            assert batch["prefix_attention_mask"][i, :real_len].all()
            if real_len < batch["prefix_attention_mask"].shape[1]:
                assert not batch["prefix_attention_mask"][i, real_len:].any()


# ---------------------------------------------------------------------------
# Full variant bank validation
# ---------------------------------------------------------------------------


class TestFullVariantBank:
    """Validate the checked-in variant bank."""

    @pytest.fixture()
    def full_bank(self):
        bank_path = Path("configs/prompt_variants/file_read_repro.json")
        if not bank_path.exists():
            pytest.skip("Full variant bank not present")
        with open(bank_path) as f:
            return json.load(f)

    def test_bank_not_empty(self, full_bank):
        assert len(full_bank) > 0

    def test_all_variants_have_required_fields(self, full_bank):
        required = {"system_prompt", "user_prompt", "compression_prompt", "response_prefix"}
        for i, v in enumerate(full_bank):
            assert set(v.keys()) == required, f"Variant {i} has wrong fields: {set(v.keys())}"

    def test_all_variants_have_file_path_placeholder(self, full_bank):
        for i, v in enumerate(full_bank):
            assert "{file_path}" in v["user_prompt"], f"Variant {i}: missing in user_prompt"
            assert "{file_path}" in v["response_prefix"], f"Variant {i}: missing in response_prefix"

    def test_no_structural_tokens_leaked(self, full_bank):
        structural = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "```"]
        for i, v in enumerate(full_bank):
            for field, text in v.items():
                for token in structural:
                    assert token not in text, (
                        f"Variant {i}: structural token '{token}' in field '{field}'"
                    )
