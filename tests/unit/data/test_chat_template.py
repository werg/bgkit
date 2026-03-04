"""Tests for shared chat template module."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.chat_template import (
    CONTENT_SENTINEL,
    TOOL_CONFIGS,
    build_messages,
    build_tools,
    compute_suffix_ids,
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
    """Minimal tokenizer mock for testing template construction.

    Handles tool_calls on assistant messages and role="tool" messages,
    mimicking Qwen3.5's apply_chat_template behavior.
    """

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
        self, messages, tokenize=True, add_generation_prompt=False, **kwargs,
    ) -> str | list[int]:
        parts = []
        prev_role = None
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            # Render tool_calls on assistant messages
            if role == "assistant" and "tool_calls" in msg:
                tc_parts = []
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    args = fn["arguments"]
                    param_strs = "".join(
                        f"<parameter={k}>\n{v}\n</parameter>"
                        for k, v in args.items()
                    )
                    tc_parts.append(
                        f"<tool_call>\n<function={fn['name']}>"
                        f"{param_strs}</function>\n</tool_call>"
                    )
                content = "".join(tc_parts)
            # Render tool role as user with <tool_response> wrapper
            if role == "tool":
                role = "user"
                content = f"<tool_response>\n{content}\n</tool_response>"
            # Inject think blocks on assistant turns after user/tool
            if msg["role"] == "assistant" and prev_role in ("user", "tool"):
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            prev_role = msg["role"]
        result = "".join(parts)
        if tokenize:
            return self.encode(result, add_special_tokens=False)
        return result


# ---------------------------------------------------------------------------
# build_messages tests
# ---------------------------------------------------------------------------


class TestBuildMessagesFileReadRepro:
    """Verify official Qwen3.5 tool-call format for file_read_repro."""

    config = TOOL_CONFIGS["file_read_repro"]

    def test_messages_have_correct_roles(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content here",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

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

    def test_tool_calls_attribute_on_assistant(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "tool_calls" in messages[2]
        tc = messages[2]["tool_calls"][0]
        assert tc["function"]["name"] == "bgkit_read_file"

    def test_assistant_tool_call_content_empty(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert messages[2]["content"] == ""

    def test_tool_response_role(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert messages[3]["role"] == "tool"

    def test_no_think_block_in_final_response(self):
        """Template auto-injects think blocks; they should not be in content."""
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        assert "<think>" not in messages[4]["content"]

    def test_code_fence_with_language(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "CONTENT",
        )
        assert "```python" in messages[4]["content"]
        assert "CONTENT" in messages[4]["content"]

    def test_compression_prompt_in_tool_call_args(self):
        messages = build_messages(
            SEED_VARIANT, self.config, "test.py", "python", "content",
        )
        args = messages[2]["tool_calls"][0]["function"]["arguments"]
        assert args["prompt"] == "Return the file contents verbatim"


class TestBuildMessagesCommitRepro:
    """Verify commit repro template."""

    config = TOOL_CONFIGS["commit_repro"]

    def test_correct_roles(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff content",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_tool_name_is_commit_repro(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff",
        )
        assert messages[2]["tool_calls"][0]["function"]["name"] == "bgkit_reproduce_commit"

    def test_no_code_fence(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff content",
        )
        assert "```" not in messages[4]["content"]

    def test_repo_in_tool_arguments(self):
        messages = build_messages(
            COMMIT_VARIANT, self.config, "owner/repo", "python", "diff",
        )
        args = messages[2]["tool_calls"][0]["function"]["arguments"]
        assert args["repo"] == "owner/repo"


class TestBuildMessagesDescriptionGen:
    """Verify description generation template."""

    config = TOOL_CONFIGS["description_gen"]

    def test_tool_name(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module", "python", "description text",
        )
        assert messages[2]["tool_calls"][0]["function"]["name"] == "bgkit_describe"

    def test_no_code_fence(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module", "python", "description text",
        )
        assert "```" not in messages[4]["content"]

    def test_target_in_tool_arguments(self):
        messages = build_messages(
            DESC_VARIANT, self.config, "src/module.py", "python", "desc",
        )
        args = messages[2]["tool_calls"][0]["function"]["arguments"]
        assert args["target"] == "src/module.py"


class TestBuildMessagesStructuralRepro:
    """Verify structural reconstruction template."""

    config = TOOL_CONFIGS["structural_repro"]

    def test_tool_name(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skeleton",
        )
        assert messages[2]["tool_calls"][0]["function"]["name"] == "bgkit_extract_structure"

    def test_no_code_fence(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skeleton",
        )
        assert "```" not in messages[4]["content"]

    def test_file_path_in_tool_arguments(self):
        messages = build_messages(
            STRUCTURAL_VARIANT, self.config, "src/foo.py", "python", "skel",
        )
        args = messages[2]["tool_calls"][0]["function"]["arguments"]
        assert args["file_path"] == "src/foo.py"


# ---------------------------------------------------------------------------
# Content sentinel tests
# ---------------------------------------------------------------------------


class TestContentSentinel:
    def test_sentinel_appears_exactly_once_file_read(self):
        config = TOOL_CONFIGS["file_read_repro"]
        messages = build_messages(
            SEED_VARIANT, config, "test.py", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m.get("content", "") for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_commit_repro(self):
        config = TOOL_CONFIGS["commit_repro"]
        messages = build_messages(
            COMMIT_VARIANT, config, "repo", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m.get("content", "") for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_description_gen(self):
        config = TOOL_CONFIGS["description_gen"]
        messages = build_messages(
            DESC_VARIANT, config, "target", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m.get("content", "") for m in messages)
        assert full_text.count(CONTENT_SENTINEL) == 1

    def test_sentinel_appears_exactly_once_structural_repro(self):
        config = TOOL_CONFIGS["structural_repro"]
        messages = build_messages(
            STRUCTURAL_VARIANT, config, "file.py", "python", CONTENT_SENTINEL,
        )
        full_text = " ".join(m.get("content", "") for m in messages)
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


class TestChatReproDatasetIntegration:
    """Verify ChatReproDataset works with new tool-call format."""

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

    def test_content_sentinel_importable(self):
        """CONTENT_SENTINEL should be importable from chat_repro_dataset."""
        from bgkit.data.datasets.chat_repro_dataset import CONTENT_SENTINEL as COMPAT_SENTINEL
        assert COMPAT_SENTINEL == CONTENT_SENTINEL


# ---------------------------------------------------------------------------
# Real Qwen3.5 tokenizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBuildMessagesWithQwen35Tokenizer:
    """Verify build_messages + apply_chat_template(tools=...) matches
    the official Qwen3.5 tool-call format."""

    @pytest.fixture(autouse=True)
    def _load_tokenizer(self):
        transformers = pytest.importorskip("transformers")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B-Base", trust_remote_code=True,
            )
        except OSError:
            pytest.skip("Qwen3.5-0.8B-Base tokenizer not cached locally")

    def _render(self, variant, config, file_path="test.py", language="python",
                content="PLACEHOLDER"):
        messages = build_messages(variant, config, file_path, language, content)
        tools = build_tools(config)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
        )

    def test_sentinel_appears_exactly_once(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config, content=CONTENT_SENTINEL)
        assert out.count(CONTENT_SENTINEL) == 1

    def test_sentinel_split_produces_valid_prefix_suffix(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config, content=CONTENT_SENTINEL)
        prefix, suffix = out.split(CONTENT_SENTINEL)
        # Prefix should contain the system prompt, user message, tool call, etc.
        assert "<|im_start|>system" in prefix
        assert "<|im_start|>user" in prefix
        assert "<|im_start|>assistant" in prefix
        # Suffix should end with the end-of-turn token
        assert suffix.rstrip().endswith("<|im_end|>")

    def test_xml_parameter_tool_call_format(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config)
        assert "<function=bgkit_read_file>" in out
        assert "<parameter=file_path>" in out
        assert "<parameter=prompt>" in out

    def test_think_blocks_injected(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config)
        # Template injects think blocks on assistant turns
        assert out.count("<think>") >= 2  # tool-call turn + final response
        assert out.count("</think>") >= 2

    def test_tool_response_in_user_role(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config)
        assert "<tool_response>" in out

    def test_code_fence_in_response(self):
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config, language="python",
                           content="print('hello')")
        assert "```python" in out

    def test_no_code_fence_for_non_fenced_config(self):
        config = TOOL_CONFIGS["commit_repro"]
        out = self._render(COMMIT_VARIANT, config, file_path="owner/repo",
                           content="diff content")
        # Only tool_call tags should appear, not code fences in response
        response_after_tool = out.split("<tool_response>")[1]
        # After </tool_response> and the next assistant turn
        final_assistant = response_after_tool.split("<|im_start|>assistant")[-1]
        assert "```" not in final_assistant

    def test_suffix_constant_across_variants(self):
        config = TOOL_CONFIGS["file_read_repro"]
        tools = build_tools(config)
        suffixes = []
        for variant in [SEED_VARIANT, ALT_VARIANT]:
            messages = build_messages(
                variant, config, "test.py", "python", CONTENT_SENTINEL,
            )
            out = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, tools=tools,
            )
            _, suffix = out.split(CONTENT_SENTINEL)
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            suffixes.append(suffix_ids)
        assert suffixes[0] == suffixes[1], (
            f"Suffix IDs differ across variants:\n{suffixes[0]}\nvs\n{suffixes[1]}"
        )

    def test_all_configs_sentinel_unique(self):
        """Sentinel appears exactly once for all tool configs."""
        config_variants = {
            "file_read_repro": SEED_VARIANT,
            "commit_repro": COMMIT_VARIANT,
            "description_gen": DESC_VARIANT,
            "structural_repro": STRUCTURAL_VARIANT,
        }
        for config_name, variant in config_variants.items():
            config = TOOL_CONFIGS[config_name]
            out = self._render(variant, config, content=CONTENT_SENTINEL)
            count = out.count(CONTENT_SENTINEL)
            assert count == 1, (
                f"{config_name}: sentinel appeared {count} times, expected 1"
            )

    def test_system_prompt_at_end_of_system_message(self):
        """User's system prompt should appear at the end of the system message,
        after the auto-generated tool format instructions."""
        config = TOOL_CONFIGS["file_read_repro"]
        out = self._render(SEED_VARIANT, config)
        # Extract system message content
        system_start = out.index("<|im_start|>system\n") + len("<|im_start|>system\n")
        system_end = out.index("<|im_end|>", system_start)
        system_content = out[system_start:system_end]
        # User's system prompt should be at the end
        assert system_content.rstrip().endswith(SEED_VARIANT["system_prompt"])
