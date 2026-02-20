"""Tests for shared chat template module."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.chat_template import (
    CONTENT_SENTINEL,
    TOOL_CONFIGS,
    build_messages,
    compute_suffix_ids,
    make_tool_definition,
    select_variant,
    tokenize_with_sentinel,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SEED_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the bgkit_read_file"
        " tool for reading file contents."
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

COMMIT_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the bgkit_reproduce_commit"
        " tool for reproducing commits."
    ),
    "user_prompt": "Reproduce the commit from `{file_path}`",
    "compression_prompt": "Reproduce the commit faithfully",
    "response_prefix": "Here is the reproduced commit from `{file_path}`:",
}

DESC_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the bgkit_describe"
        " tool for generating descriptions."
    ),
    "user_prompt": "Describe `{file_path}`",
    "compression_prompt": "Generate a clear description",
    "response_prefix": "Here is a description of `{file_path}`:",
}

STRUCTURAL_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the bgkit_extract_structure"
        " tool for extracting structure."
    ),
    "user_prompt": "Extract the structure of `{file_path}`",
    "compression_prompt": "Extract the structural skeleton faithfully",
    "response_prefix": "Here is the structural extraction of `{file_path}`:",
}


class MockTokenizer:
    """Minimal tokenizer mock for testing template construction."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
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
        self, messages, tokenize=True, add_generation_prompt=False,
    ) -> str | list[int]:
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        result = "".join(parts)
        if tokenize:
            return self.encode(result, add_special_tokens=False)
        return result


# ---------------------------------------------------------------------------
# build_messages tests
# ---------------------------------------------------------------------------


class TestBuildMessagesFileReadRepro:
    """Verify backward compat with old _build_messages for file_read_repro."""

    config = TOOL_CONFIGS["file_read_repro"]

    def test_messages_have_correct_roles(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content here",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

    def test_file_path_in_user_message(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "src/foo.py", "python", "content",
        )
        assert "src/foo.py" in messages[1]["content"]

    def test_file_path_in_response_prefix(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "src/foo.py", "python", "content",
        )
        assert "src/foo.py" in messages[4]["content"]

    def test_tool_call_uses_config_tool_name(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "<tool_call>" in messages[2]["content"]
        assert "bgkit_read_file" in messages[2]["content"]

    def test_tool_response_in_user_message(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "<tool_response>" in messages[3]["content"]

    def test_think_block_in_final_response(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "<think>" in messages[4]["content"]
        assert "</think>" in messages[4]["content"]

    def test_code_fence_with_language(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "CONTENT",
        )
        assert "```python" in messages[4]["content"]
        assert "CONTENT" in messages[4]["content"]

    def test_compression_prompt_in_tool_call(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "Return the file contents verbatim" in messages[2]["content"]


class TestBuildMessagesCommitRepro:
    """Verify commit repro template."""

    config = TOOL_CONFIGS["commit_repro"]

    def test_correct_roles(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff content",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

    def test_tool_name_is_commit_repro(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff",
        )
        assert "bgkit_reproduce_commit" in messages[2]["content"]

    def test_no_code_fence(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff content",
        )
        assert "```" not in messages[4]["content"]

    def test_repo_in_tool_arguments(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff",
        )
        tool_call = json.loads(
            messages[2]["content"]
            .replace("<tool_call>\n", "")
            .replace("\n</tool_call>", "")
        )
        assert tool_call["arguments"]["repo"] == "owner/repo"


class TestBuildMessagesDescriptionGen:
    """Verify description generation template."""

    config = TOOL_CONFIGS["description_gen"]

    def test_tool_name(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module", "python", "description text",
        )
        assert "bgkit_describe" in messages[2]["content"]

    def test_no_code_fence(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module", "python", "description text",
        )
        assert "```" not in messages[4]["content"]

    def test_target_in_tool_arguments(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module.py", "python", "desc",
        )
        tool_call = json.loads(
            messages[2]["content"]
            .replace("<tool_call>\n", "")
            .replace("\n</tool_call>", "")
        )
        assert tool_call["arguments"]["target"] == "src/module.py"


class TestBuildMessagesStructuralRepro:
    """Verify structural reconstruction template."""

    config = TOOL_CONFIGS["structural_repro"]

    def test_tool_name(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skeleton",
        )
        assert "bgkit_extract_structure" in messages[2]["content"]

    def test_no_code_fence(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skeleton",
        )
        assert "```" not in messages[4]["content"]

    def test_file_path_in_tool_arguments(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skel",
        )
        tool_call = json.loads(
            messages[2]["content"]
            .replace("<tool_call>\n", "")
            .replace("\n</tool_call>", "")
        )
        assert tool_call["arguments"]["file_path"] == "src/foo.py"


# ---------------------------------------------------------------------------
# Content sentinel tests
# ---------------------------------------------------------------------------


class TestContentSentinel:
    def test_sentinel_appears_exactly_once_file_read(self):
        config = TOOL_CONFIGS["file_read_repro"]
        messages = build_messages(
            SEED_VARIANT, config, "test.py", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_commit_repro(self):
        config = TOOL_CONFIGS["commit_repro"]
        messages = build_messages(
            COMMIT_VARIANT, config, "repo", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_description_gen(self):
        config = TOOL_CONFIGS["description_gen"]
        messages = build_messages(
            DESC_VARIANT, config, "target", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_structural_repro(self):
        config = TOOL_CONFIGS["structural_repro"]
        messages = build_messages(
            STRUCTURAL_VARIANT, config, "file.py", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m["content"] for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1


# ---------------------------------------------------------------------------
# Code fence wrapping tests
# ---------------------------------------------------------------------------


class TestCodeFenceWrapping:
    def test_code_fence_present_for_file_read(self):
        config = TOOL_CONFIGS["file_read_repro"]
        messages = build_messages(
            SEED_VARIANT, config, "test.py", "python", "content",
        )
        assert "```python" in messages[4]["content"]
        # Closing fence
        assert messages[4]["content"].endswith("```")

    def test_code_fence_absent_for_commit_repro(self):
        config = TOOL_CONFIGS["commit_repro"]
        messages = build_messages(
            COMMIT_VARIANT, config, "repo", "python", "content",
        )
        assert "```" not in messages[4]["content"]

    def test_code_fence_absent_for_description_gen(self):
        config = TOOL_CONFIGS["description_gen"]
        messages = build_messages(
            DESC_VARIANT, config, "target", "python", "content",
        )
        assert "```" not in messages[4]["content"]

    def test_code_fence_absent_for_structural_repro(self):
        config = TOOL_CONFIGS["structural_repro"]
        messages = build_messages(
            STRUCTURAL_VARIANT, config, "file.py", "python", "content",
        )
        assert "```" not in messages[4]["content"]


# ---------------------------------------------------------------------------
# Variant selection tests
# ---------------------------------------------------------------------------


class TestSelectVariant:
    def test_deterministic(self):
        variants = [SEED_VARIANT, ALT_VARIANT]
        v1 = select_variant(variants, idx=5, epoch_seed=42)
        v2 = select_variant(variants, idx=5, epoch_seed=42)
        assert v1 == v2

    def test_varies_by_epoch(self):
        variants = [SEED_VARIANT, ALT_VARIANT]
        selections = set()
        for epoch_seed in range(100):
            v = select_variant(variants, idx=0, epoch_seed=epoch_seed)
            selections.add(v["system_prompt"])
        # With 2 variants and 100 epoch seeds, both should appear
        assert len(selections) == 2

    def test_varies_by_idx(self):
        variants = [SEED_VARIANT, ALT_VARIANT]
        selections = set()
        for idx in range(100):
            v = select_variant(variants, idx=idx, epoch_seed=42)
            selections.add(v["system_prompt"])
        assert len(selections) == 2


# ---------------------------------------------------------------------------
# tokenize_with_sentinel tests
# ---------------------------------------------------------------------------


class TestTokenizeWithSentinel:
    def test_returns_expected_keys(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        expected_keys = {
            "token_ids", "loss_mask", "content_token_ids",
            "compression_prompt_ids", "prefix_ids",
        }
        assert set(result.keys()) == expected_keys

    def test_loss_mask_shape_matches(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        assert result["loss_mask"].shape == result["token_ids"].shape

    def test_loss_mask_ones_count(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        assert result["loss_mask"].sum().item() == 5

    def test_content_token_ids_preserved(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([10, 20, 30], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        assert torch.equal(result["content_token_ids"], content_ids)

    def test_prefix_ids_shorter_than_full(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        assert len(result["prefix_ids"]) < len(result["token_ids"])

    def test_compression_prompt_ids_nonempty(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        assert len(result["compression_prompt_ids"]) > 0

    def test_works_for_all_configs(self):
        """tokenize_with_sentinel should work for all task configs."""
        tokenizer = MockTokenizer()
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        variants_by_config = {
            "file_read_repro": SEED_VARIANT,
            "commit_repro": COMMIT_VARIANT,
            "description_gen": DESC_VARIANT,
            "structural_repro": STRUCTURAL_VARIANT,
        }
        for config_name, variant in variants_by_config.items():
            config = TOOL_CONFIGS[config_name]
            result = tokenize_with_sentinel(
                tokenizer, variant, config, "test.py", "python", content_ids,
            )
            assert result["loss_mask"].sum().item() == 3
            assert len(result["token_ids"]) > 3


# ---------------------------------------------------------------------------
# compute_suffix_ids tests
# ---------------------------------------------------------------------------


class TestComputeSuffixIds:
    def test_returns_1d_tensor(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        suffix = compute_suffix_ids(tokenizer, [SEED_VARIANT, ALT_VARIANT], config)
        assert suffix.dim() == 1
        assert len(suffix) > 0

    def test_consistent_across_variants(self):
        """Should not raise even with different variants (suffix is constant)."""
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        suffix = compute_suffix_ids(tokenizer, [SEED_VARIANT, ALT_VARIANT], config)
        assert suffix.dtype == torch.long


# ---------------------------------------------------------------------------
# Backward compat: refactored ChatReproDataset produces identical output
# ---------------------------------------------------------------------------


class MockInnerDataset:
    """Mock MmapTokenDataset returning fixed samples."""

    def __init__(self, samples: list[dict]):
        self._samples = samples
        self._chunk_lengths = __import__("numpy").array(
            [len(s["token_ids"]) for s in samples], dtype=__import__("numpy").int32,
        )

    @property
    def lengths(self):
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


class TestBackwardCompatWithChatReproDataset:
    """Verify the refactored ChatReproDataset produces identical output."""

    @pytest.fixture()
    def inner_dataset(self):
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
    def variant_bank(self, tmp_path):
        bank = [SEED_VARIANT, ALT_VARIANT]
        path = tmp_path / "variants.json"
        with open(path, "w") as f:
            __import__("json").dump(bank, f)
        return path

    @pytest.fixture()
    def dataset(self, inner_dataset, variant_bank):
        from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset

        tokenizer = MockTokenizer()
        return ChatReproDataset(
            inner_dataset, tokenizer=tokenizer, variant_bank_path=variant_bank, seed=42,
        )

    def test_getitem_returns_expected_keys(self, dataset):
        sample = dataset[0]
        expected_keys = {
            "token_ids", "loss_mask", "content_token_ids",
            "compression_prompt_ids", "prefix_ids", "language",
        }
        assert set(sample.keys()) == expected_keys

    def test_token_ids_dtype(self, dataset):
        assert dataset[0]["token_ids"].dtype == torch.long

    def test_loss_mask_shape(self, dataset):
        sample = dataset[0]
        assert sample["loss_mask"].shape == sample["token_ids"].shape

    def test_loss_mask_content_count(self, dataset, inner_dataset):
        for idx in range(len(inner_dataset)):
            sample = dataset[idx]
            inner_sample = inner_dataset[idx]
            expected = len(inner_sample["token_ids"])
            assert sample["loss_mask"].sum().item() == expected

    def test_content_token_ids_preserved(self, dataset, inner_dataset):
        for idx in range(len(inner_dataset)):
            chat_sample = dataset[idx]
            inner_sample = inner_dataset[idx]
            assert torch.equal(
                chat_sample["content_token_ids"],
                torch.tensor(inner_sample["token_ids"], dtype=torch.long),
            )

    def test_suffix_ids_constant(self, dataset):
        suffix = dataset.suffix_ids
        assert suffix.dim() == 1
        assert len(suffix) > 0

    def test_deterministic_variant_selection(self, dataset):
        s1 = dataset[0]
        s2 = dataset[0]
        assert torch.equal(s1["token_ids"], s2["token_ids"])

    def test_len_matches_inner(self, dataset, inner_dataset):
        assert len(dataset) == len(inner_dataset)

    def test_lengths_includes_overhead(self, dataset, inner_dataset):
        for i in range(len(inner_dataset)):
            assert dataset.lengths[i] > inner_dataset.lengths[i]

    def test_build_messages_backward_compat(self):
        """_build_messages imported from chat_repro_dataset still works."""
        from bgkit.data.datasets.chat_repro_dataset import _build_messages

        messages = _build_messages(SEED_VARIANT, "test.py", "python", "content")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]
        assert "bgkit_read_file" in messages[2]["content"]
        assert "```python" in messages[4]["content"]

    def test_content_sentinel_importable(self):
        """CONTENT_SENTINEL should be importable from chat_repro_dataset."""
        from bgkit.data.datasets.chat_repro_dataset import CONTENT_SENTINEL as COMPAT_SENTINEL
        assert COMPAT_SENTINEL == CONTENT_SENTINEL
