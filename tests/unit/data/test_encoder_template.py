"""Tests for encoder ChatML prefix template helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.chat_template import (
    CONTENT_SENTINEL,
    TOOL_CONFIGS,
    build_encoder_prefix_ids,
    build_encoder_user_only_prefix_ids,
    tokenize_with_sentinel,
)


# ---------------------------------------------------------------------------
# Mock tokenizer (reuse pattern from test_chat_template.py)
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock with ChatML apply_chat_template."""

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
            if role == "tool":
                role = "user"
                content = f"<tool_response>\n{content}\n</tool_response>"
            if msg["role"] == "assistant" and prev_role in ("user", "tool"):
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            prev_role = msg["role"]
        result = "".join(parts)
        if tokenize:
            return self.encode(result, add_special_tokens=False)
        return result


# ---------------------------------------------------------------------------
# build_encoder_prefix_ids tests
# ---------------------------------------------------------------------------


class TestBuildEncoderPrefixIds:
    def test_starts_with_im_start_system(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_prefix_ids(tokenizer, "Compress this code")
        decoded = tokenizer.decode(prefix.tolist())
        assert decoded.startswith("<|im_start|>system\n")

    def test_ends_with_im_start_user_newline(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_prefix_ids(tokenizer, "Compress this code")
        decoded = tokenizer.decode(prefix.tolist())
        assert decoded.endswith("<|im_start|>user\n")

    def test_contains_compression_prompt(self):
        tokenizer = MockTokenizer()
        prompt = "Return the file contents verbatim"
        prefix = build_encoder_prefix_ids(tokenizer, prompt)
        decoded = tokenizer.decode(prefix.tolist())
        assert prompt in decoded

    def test_returns_1d_long_tensor(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_prefix_ids(tokenizer, "test")
        assert prefix.dim() == 1
        assert prefix.dtype == torch.long

    def test_does_not_contain_sentinel(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_prefix_ids(tokenizer, "test")
        decoded = tokenizer.decode(prefix.tolist())
        assert CONTENT_SENTINEL not in decoded


# ---------------------------------------------------------------------------
# build_encoder_user_only_prefix_ids tests
# ---------------------------------------------------------------------------


class TestBuildEncoderUserOnlyPrefixIds:
    def test_produces_im_start_user_newline(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_user_only_prefix_ids(tokenizer)
        decoded = tokenizer.decode(prefix.tolist())
        assert "<|im_start|>user\n" in decoded

    def test_no_system_message(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_user_only_prefix_ids(tokenizer)
        decoded = tokenizer.decode(prefix.tolist())
        assert "system" not in decoded

    def test_returns_1d_long_tensor(self):
        tokenizer = MockTokenizer()
        prefix = build_encoder_user_only_prefix_ids(tokenizer)
        assert prefix.dim() == 1
        assert prefix.dtype == torch.long

    def test_shorter_than_full_prefix(self):
        tokenizer = MockTokenizer()
        user_only = build_encoder_user_only_prefix_ids(tokenizer)
        full = build_encoder_prefix_ids(tokenizer, "Some compression prompt")
        assert len(user_only) < len(full)


# ---------------------------------------------------------------------------
# tokenize_with_sentinel produces ChatML prefix in compression_prompt_ids
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


class TestTokenizeWithSentinelChatMLPrefix:
    def test_compression_prompt_ids_contain_im_start_system(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        decoded = tokenizer.decode(result["compression_prompt_ids"].tolist())
        assert "<|im_start|>system\n" in decoded

    def test_compression_prompt_ids_end_with_user_turn(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        decoded = tokenizer.decode(result["compression_prompt_ids"].tolist())
        assert decoded.endswith("<|im_start|>user\n")

    def test_compression_prompt_ids_contain_prompt_text(self):
        tokenizer = MockTokenizer()
        config = TOOL_CONFIGS["file_read_repro"]
        content_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        result = tokenize_with_sentinel(
            tokenizer, SEED_VARIANT, config, "test.py", "python", content_ids,
        )
        decoded = tokenizer.decode(result["compression_prompt_ids"].tolist())
        assert "Return the file contents verbatim" in decoded


# ---------------------------------------------------------------------------
# Real Qwen3.5 tokenizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEncoderPrefixWithQwen35Tokenizer:
    """Verify exact ChatML token IDs against the real Qwen3.5 tokenizer."""

    @pytest.fixture(autouse=True)
    def _load_tokenizer(self):
        transformers = pytest.importorskip("transformers")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B-Base", trust_remote_code=True,
            )
        except OSError:
            pytest.skip("Qwen3.5-0.8B-Base tokenizer not cached locally")

    # -- Known token IDs for Qwen3.5 ChatML special tokens --
    IM_START = 248045
    IM_END = 248046
    # "system" = 8678, "user" = 846, "\n" = 198
    SYSTEM_ID = 8678
    USER_ID = 846
    NEWLINE_ID = 198

    def test_full_prefix_starts_with_im_start_system(self):
        ids = build_encoder_prefix_ids(self.tokenizer, "Compress this code")
        assert ids[0].item() == self.IM_START
        assert ids[1].item() == self.SYSTEM_ID
        assert ids[2].item() == self.NEWLINE_ID

    def test_full_prefix_ends_with_im_start_user_newline(self):
        ids = build_encoder_prefix_ids(self.tokenizer, "Compress this code")
        assert ids[-3].item() == self.IM_START
        assert ids[-2].item() == self.USER_ID
        assert ids[-1].item() == self.NEWLINE_ID

    def test_full_prefix_contains_im_end(self):
        ids = build_encoder_prefix_ids(self.tokenizer, "Compress this code")
        assert self.IM_END in ids.tolist()

    def test_full_prefix_exact_ids(self):
        ids = build_encoder_prefix_ids(self.tokenizer, "Compress this code")
        expected = [
            self.IM_START, self.SYSTEM_ID, self.NEWLINE_ID,  # <|im_start|>system\n
            # "Compress this code" tokens
            1057, 1808, 411, 1970,
            self.IM_END, self.NEWLINE_ID,  # <|im_end|>\n
            self.IM_START, self.USER_ID, self.NEWLINE_ID,  # <|im_start|>user\n
        ]
        assert ids.tolist() == expected

    def test_user_only_prefix_exact_ids(self):
        ids = build_encoder_user_only_prefix_ids(self.tokenizer)
        expected = [self.IM_START, self.USER_ID, self.NEWLINE_ID]
        assert ids.tolist() == expected

    def test_user_only_is_suffix_of_full(self):
        full = build_encoder_prefix_ids(self.tokenizer, "test prompt")
        user_only = build_encoder_user_only_prefix_ids(self.tokenizer)
        assert full[-len(user_only):].tolist() == user_only.tolist()
