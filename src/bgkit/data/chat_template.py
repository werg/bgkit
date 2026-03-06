"""Shared chat template construction for all objectives.

Parameterized by ChatTemplateConfig to support file reproduction,
commit reproduction, description generation, and structural reconstruction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

# Sentinel used to locate exact content boundaries within the template.
# Long random suffix makes accidental collision near-impossible.
CONTENT_SENTINEL = "<<<BGKIT_CONTENT_a7f3b2e1>>>"


@dataclass
class ChatTemplateConfig:
    """Configuration for a specific task's chat template."""

    tool_name: str  # e.g., "bgkit_read_file", "bgkit_reproduce_commit"
    tool_description: str
    tool_parameters: dict  # JSON schema for tool params
    content_in_code_fence: bool = True  # Whether response wraps content in ```
    code_fence_language: str | None = None  # None = use file's language
    # Mapping from tool parameter names to variant fields that supply their values.
    # e.g. {"file_path": "file_path_value", "prompt": "compression_prompt"}
    # If not provided, defaults are inferred from the parameter schema.
    tool_arg_fields: dict[str, str] | None = None


TOOL_CONFIGS: dict[str, ChatTemplateConfig] = {
    "file_read_repro": ChatTemplateConfig(
        tool_name="bgkit_read_file",
        tool_description="Read the contents of a file from BgKIT compressed context.",
        tool_parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Instructions for how to process/return the file contents"
                    ),
                },
            },
            "required": ["file_path", "prompt"],
        },
    ),
    "commit_repro": ChatTemplateConfig(
        tool_name="bgkit_reproduce_commit",
        tool_description="Reproduce a commit from BgKIT compressed repository context.",
        tool_parameters={
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository identifier",
                },
                "prompt": {
                    "type": "string",
                    "description": "Instructions for reproducing the commit",
                },
            },
            "required": ["repo", "prompt"],
        },
        content_in_code_fence=False,
    ),
    "description_gen": ChatTemplateConfig(
        tool_name="bgkit_describe",
        tool_description="Generate a description of code from BgKIT compressed context.",
        tool_parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "What to describe (file, module, or repo)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Instructions for the description",
                },
            },
            "required": ["target", "prompt"],
        },
        content_in_code_fence=False,
    ),
    "structural_repro": ChatTemplateConfig(
        tool_name="bgkit_extract_structure",
        tool_description=(
            "Extract structural information from BgKIT compressed context."
        ),
        tool_parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "Instructions for structural extraction",
                },
            },
            "required": ["file_path", "prompt"],
        },
        content_in_code_fence=False,
    ),
    "file_read_query": ChatTemplateConfig(
        tool_name="bgkit_read_file",
        tool_description="Read and analyze file contents from BgKIT compressed context.",
        tool_parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Question or instructions for analyzing the file contents"
                    ),
                },
            },
            "required": ["file_path", "prompt"],
        },
        content_in_code_fence=False,
    ),
}


def build_tools(config: ChatTemplateConfig) -> list[dict]:
    """Build tools list for apply_chat_template(tools=...).

    Returns the format expected by Qwen3.5's chat template: a list of
    tool dicts with type/function/name/description/parameters.
    """
    return [{"type": "function", "function": {
        "name": config.tool_name,
        "description": config.tool_description,
        "parameters": config.tool_parameters,
    }}]


def _build_tool_call_arguments(
    config: ChatTemplateConfig,
    file_path: str,
    compression_prompt: str,
) -> dict[str, str]:
    """Build tool call argument dict from config and sample metadata.

    Maps each parameter in the tool schema to an appropriate value:
    - 'file_path' / 'repo' / 'target' params get the file_path value
    - 'prompt' param gets the compression_prompt value
    """
    if config.tool_arg_fields is not None:
        # Explicit mapping provided
        field_map = config.tool_arg_fields
        available = {"file_path": file_path, "compression_prompt": compression_prompt}
        return {k: available[v] for k, v in field_map.items()}

    # Default inference from parameter names
    args: dict[str, str] = {}
    properties = config.tool_parameters.get("properties", {})
    for param_name in properties:
        if param_name in ("file_path", "repo", "target"):
            args[param_name] = file_path
        elif param_name == "prompt":
            args[param_name] = compression_prompt
    return args


def build_messages(
    variant: dict[str, str],
    config: ChatTemplateConfig,
    file_path: str,
    language: str,
    content_placeholder: str,
) -> list[dict]:
    """Build chat messages in Qwen3.5's official tool-call format.

    Uses tool_calls attribute on assistant messages and role="tool" for
    tool responses, matching the format that apply_chat_template(tools=...)
    renders natively. The template auto-injects tool format instructions
    into the system prompt and <think> blocks into assistant turns.
    """
    system_prompt = variant["system_prompt"]
    user_prompt = variant["user_prompt"].replace("{file_path}", file_path)
    compression_prompt = variant["compression_prompt"]
    response_prefix = variant["response_prefix"].replace("{file_path}", file_path)

    # Build tool call arguments as a dict (not JSON string)
    tool_args = _build_tool_call_arguments(config, file_path, compression_prompt)

    # Build final assistant response content (no <think> — template injects it)
    if config.content_in_code_fence:
        fence_lang = config.code_fence_language if config.code_fence_language else language
        response_content = (
            f"{response_prefix}\n\n"
            f"```{fence_lang}\n"
            f"{content_placeholder}\n"
            f"```"
        )
    else:
        response_content = (
            f"{response_prefix}\n\n"
            f"{content_placeholder}"
        )

    # Assistant tool-call message with tool_calls attribute
    tool_call_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": config.tool_name,
                    "arguments": tool_args,
                },
            }
        ],
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        tool_call_msg,
        {
            "role": "tool",
            "content": "File contents provided as BgKIT compressed context.",
        },
        {"role": "assistant", "content": response_content},
    ]
    return messages


def compute_suffix_ids(
    tokenizer,
    variants: list[dict[str, str]],
    config: ChatTemplateConfig,
) -> torch.Tensor:
    """Compute constant suffix token IDs across variants.

    The suffix is structurally identical across all variants - only the
    text *before* the sentinel differs per variant. Verify across a few
    variants as a sanity check.
    """
    tools = build_tools(config)
    suffix_ids = None
    for variant in variants:
        messages = build_messages(
            variant, config, "test.py", "python", CONTENT_SENTINEL,
        )
        template_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            tools=tools,
        )
        _, suffix_str = template_str.split(CONTENT_SENTINEL)
        ids = tokenizer.encode(suffix_str, add_special_tokens=False)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        if suffix_ids is None:
            suffix_ids = ids_tensor
        elif not suffix_ids.equal(ids_tensor):
            raise ValueError(
                "Suffix token IDs differ across variants — expected constant suffix. "
                f"Got {suffix_ids.tolist()} vs {ids_tensor.tolist()}"
            )
    return suffix_ids  # type: ignore[return-value]


def select_variant(
    variants: list[dict[str, str]],
    idx: int,
    epoch_seed: int,
) -> dict[str, str]:
    """Select a variant deterministically per (epoch, idx).

    Uses hashlib.md5 for stability across DataLoader workers
    (Python's hash() is randomized by PYTHONHASHSEED per process).
    """
    key = f"{epoch_seed}:{idx}".encode()
    h = int.from_bytes(hashlib.md5(key).digest()[:8], "little")
    return variants[h % len(variants)]


def load_all_variant_banks(variant_dir: str | Path) -> list[dict[str, str]]:
    """Load and deduplicate compression prompts from all variant bank JSON files.

    Concatenates all *.json files in variant_dir, extracts the compression_prompt
    field from each variant, deduplicates, and returns a list of minimal variant
    dicts suitable for encoder prefix construction.
    """
    import json

    variant_dir = Path(variant_dir)
    all_prompts: set[str] = set()
    variants: list[dict[str, str]] = []

    for json_path in sorted(variant_dir.glob("*.json")):
        with open(json_path) as f:
            bank = json.load(f)
        for v in bank:
            prompt = v.get("compression_prompt", "")
            if prompt and prompt not in all_prompts:
                all_prompts.add(prompt)
                variants.append(v)

    return variants


def build_encoder_prefix_ids(tokenizer, compression_prompt: str) -> torch.Tensor:
    """Build ChatML-wrapped encoder prefix token IDs.

    Produces: <|im_start|>system\\n{compression_prompt}<|im_end|>\\n<|im_start|>user\\n

    Uses apply_chat_template with a sentinel to locate the exact boundary
    between the template prefix and the user content.
    """
    messages = [
        {"role": "system", "content": compression_prompt},
        {"role": "user", "content": CONTENT_SENTINEL},
    ]
    template_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_str, _ = template_str.split(CONTENT_SENTINEL)
    ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


def build_encoder_user_only_prefix_ids(tokenizer) -> torch.Tensor:
    """Build a user-only ChatML prefix: <|im_start|>user\\n

    For joint block pretrain where no compression prompt is needed.
    Uses apply_chat_template with a sentinel to extract just the user turn opener.
    """
    messages = [
        {"role": "user", "content": CONTENT_SENTINEL},
    ]
    template_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_str, _ = template_str.split(CONTENT_SENTINEL)
    ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


def tokenize_with_sentinel(
    tokenizer,
    variant: dict[str, str],
    config: ChatTemplateConfig,
    file_path: str,
    language: str,
    content_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Tokenize a sample using sentinel-based boundary detection.

    Returns dict with: token_ids, loss_mask, content_token_ids,
    compression_prompt_ids, prefix_ids
    """
    # Build template with sentinel for boundary detection
    tools = build_tools(config)
    messages_with_sentinel = build_messages(
        variant, config, file_path, language, CONTENT_SENTINEL,
    )
    template_str = tokenizer.apply_chat_template(
        messages_with_sentinel, tokenize=False, add_generation_prompt=False,
        tools=tools,
    )

    # Validate sentinel uniqueness
    sentinel_count = template_str.count(CONTENT_SENTINEL)
    if sentinel_count != 1:
        raise ValueError(
            f"Expected exactly 1 sentinel in template, found {sentinel_count}. "
            f"Variant text may accidentally contain the sentinel string."
        )

    # Split on sentinel to get prefix and suffix strings
    prefix_str, suffix_str = template_str.split(CONTENT_SENTINEL)

    # Tokenize each piece separately (no special tokens per piece)
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)

    # Content token IDs from the inner dataset (already tokenized)
    content_ids = content_token_ids.tolist()

    # Concatenate: prefix + content + suffix
    full_ids = prefix_ids + content_ids + suffix_ids
    token_ids = torch.tensor(full_ids, dtype=torch.long)

    # Build loss mask: 1 only for content tokens
    loss_mask = torch.zeros(len(full_ids), dtype=torch.long)
    content_start = len(prefix_ids)
    content_end = content_start + len(content_ids)
    loss_mask[content_start:content_end] = 1

    # Tokenize compression prompt as ChatML prefix for BgKIT conditioning
    compression_prompt = variant["compression_prompt"]
    compression_prompt_ids = build_encoder_prefix_ids(tokenizer, compression_prompt)

    return {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "content_token_ids": content_token_ids,
        "compression_prompt_ids": compression_prompt_ids,
        "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
    }
