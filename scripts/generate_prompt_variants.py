#!/usr/bin/env python3
"""Generate prompt variant banks for chat-template training.

Reads a template definition YAML and produces holistic reformulations using
``claude -p --model haiku --output-format json`` as an offline preprocessing step.
The output JSON file is checked into the repo and loaded at training time.

Usage:
    python scripts/generate_prompt_variants.py \
        --template configs/templates/file_read_repro.yaml \
        --num-variants 40 --languages tier1,tier2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Language tiers (Qwen3's supported languages)
# ---------------------------------------------------------------------------

TIER1_LANGUAGES = [
    "Chinese (zh)", "Spanish", "French", "German", "Japanese",
    "Korean", "Russian", "Portuguese", "Arabic",
]

TIER2_LANGUAGES = [
    "Italian", "Dutch", "Polish", "Turkish", "Vietnamese",
    "Thai", "Indonesian", "Hindi", "Czech", "Swedish", "Ukrainian",
]

VARIATION_AXES = [
    {"axis": "verbosity", "value": "terse", "description": "Very brief and concise"},
    {"axis": "verbosity", "value": "normal", "description": "Standard length"},
    {"axis": "verbosity", "value": "verbose", "description": "Detailed and elaborate"},
    {"axis": "style", "value": "formal", "description": "Professional and formal tone"},
    {"axis": "style", "value": "casual", "description": "Informal, conversational tone"},
    {"axis": "style", "value": "technical", "description": "Technical jargon-heavy"},
    {
        "axis": "distraction",
        "value": "mild_cruft",
        "description": "Some extra preamble or filler",
    },
    {
        "axis": "distraction",
        "value": "noisy",
        "description": "Irrelevant instructions and filler the model should ignore",
    },
]


def load_template(path: Path) -> dict:
    """Load and validate a template YAML file."""
    with open(path) as f:
        template = yaml.safe_load(f)

    required_keys = {"name", "description", "injections", "fields"}
    missing = required_keys - set(template.keys())
    if missing:
        raise ValueError(f"Template missing keys: {missing}")

    required_fields = {"system_prompt", "user_prompt", "compression_prompt", "response_prefix"}
    missing_fields = required_fields - set(template["fields"].keys())
    if missing_fields:
        raise ValueError(f"Template missing fields: {missing_fields}")

    return template


def validate_variant(variant: dict, template: dict) -> list[str]:
    """Validate a reformulated variant against template constraints.

    Returns list of error messages (empty if valid).
    """
    errors = []
    expected_fields = set(template["fields"].keys())
    variant_fields = set(variant.keys())

    if variant_fields != expected_fields:
        extra = variant_fields - expected_fields
        missing = expected_fields - variant_fields
        if extra:
            errors.append(f"Extra fields: {extra}")
        if missing:
            errors.append(f"Missing fields: {missing}")
            return errors  # can't validate content of missing fields

    # Validate injection sites
    for injection_name, injection_def in template["injections"].items():
        placeholder = "{" + injection_name + "}"
        for field_name in injection_def.get("in_fields", []):
            if field_name in variant and placeholder not in variant[field_name]:
                errors.append(
                    f"Injection '{placeholder}' missing from field '{field_name}': "
                    f"{variant[field_name]!r}"
                )

    # Check for leaked structural tokens
    structural_tokens = [
        "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>", "<think>", "</think>",
        "```",
    ]
    for field_name, field_value in variant.items():
        for token in structural_tokens:
            if token in field_value:
                errors.append(
                    f"Structural token '{token}' leaked into field '{field_name}'"
                )

    return errors


def build_meta_prompt(
    template: dict,
    axis: str | None = None,
    axis_value: str | None = None,
    axis_description: str | None = None,
    language: str | None = None,
) -> str:
    """Build the meta-prompt for claude to generate a reformulation."""
    seed_fields = template["fields"]
    injections_desc = []
    for name, defn in template["injections"].items():
        in_fields = defn.get("in_fields", [])
        injections_desc.append(
            f"  - {{{name}}}: {defn['description']}. "
            f"Must appear literally in fields: {in_fields}"
        )

    injections_text = chr(10).join(injections_desc)
    prompt = (
        "You are generating a reformulated version of a conversational "
        "template for AI training data augmentation.\n\n"
        "## Original template\n\n"
        f"Template name: {template['name']}\n"
        f"Description: {template['description']}\n\n"
        "### Seed fields:\n"
        f"- system_prompt: {seed_fields['system_prompt'].strip()}\n"
        f"- user_prompt: {seed_fields['user_prompt'].strip()}\n"
        f"- compression_prompt: {seed_fields['compression_prompt'].strip()}\n"
        f"- response_prefix: {seed_fields['response_prefix'].strip()}\n\n"
        "### Injection site constraints:\n"
        f"{injections_text}\n\n"
        "## Your task\n\n"
        "Produce a complete reformulation of ALL four fields. "
        "The reformulation must:\n"
        "1. Preserve all injection placeholders (like {file_path}) EXACTLY "
        "as written, including the curly braces, in the listed fields\n"
        "2. Keep the same semantic meaning for each field\n"
        "3. Use ONLY the four field names: system_prompt, user_prompt, "
        "compression_prompt, response_prefix\n"
        "4. NOT include any structural tokens like <|im_start|>, "
        "<|im_end|>, <tool_call>, </tool_call>, ```, <think>, etc.\n"
    )

    if language:
        prompt += (
            "\n## Language\n"
            f"Translate all fields into {language}. The injection "
            "placeholders ({file_path}) must remain exactly as written.\n"
        )

    if axis:
        prompt += (
            f"\n## Variation axis: {axis} = {axis_value}\n"
            f"{axis_description}\n"
            "Apply this variation consistently across ALL four fields.\n"
        )

    prompt += (
        "\n## Output format\n"
        "Return a JSON object with exactly these four keys: "
        "system_prompt, user_prompt, compression_prompt, response_prefix.\n"
        "Each value should be a string.\n"
    )

    return prompt


def call_claude(prompt: str) -> dict | None:
    """Call claude CLI and parse JSON response."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  claude CLI error: {result.stderr[:200]}", file=sys.stderr)
            return None

        response = json.loads(result.stdout)
        # Extract text from claude JSON output format
        if isinstance(response, dict) and "result" in response:
            text = response["result"]
        elif isinstance(response, dict) and "content" in response:
            # Handle different output formats
            content = response["content"]
            text = (
                content[0].get("text", "") if isinstance(content, list)
                else str(content)
            )
        else:
            text = result.stdout

        # Try to parse the JSON from the text response
        # The model might wrap JSON in markdown code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    except subprocess.TimeoutExpired:
        print("  claude CLI timeout", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Failed to parse response: {e}", file=sys.stderr)
        return None


def generate_variants(
    template: dict,
    num_english_variants: int = 20,
    languages: list[str] | None = None,
    variants_per_language: dict[str, int] | None = None,
) -> list[dict]:
    """Generate prompt variants for a template.

    Args:
        template: Loaded template dict.
        num_english_variants: Number of English variation-axis variants.
        languages: List of languages to translate into.
        variants_per_language: Override per-language variant count.

    Returns:
        List of validated variant dicts.
    """
    variants = []

    # Always include the seed as variant 0
    seed = dict(template["fields"])
    # Strip whitespace from YAML block scalars
    seed = {k: v.strip() for k, v in seed.items()}
    variants.append(seed)

    # English variation-axis variants
    print(f"Generating {num_english_variants} English variants...")
    generated = 0
    attempts = 0
    max_attempts = num_english_variants * 3

    while generated < num_english_variants and attempts < max_attempts:
        # Cycle through variation axes
        axis_info = VARIATION_AXES[generated % len(VARIATION_AXES)]
        attempts += 1

        prompt = build_meta_prompt(
            template,
            axis=axis_info["axis"],
            axis_value=axis_info["value"],
            axis_description=axis_info["description"],
        )

        result = call_claude(prompt)
        if result is None:
            continue

        errors = validate_variant(result, template)
        if errors:
            print(f"  Variant rejected: {errors}", file=sys.stderr)
            continue

        variants.append(result)
        generated += 1
        print(f"  [{generated}/{num_english_variants}] {axis_info['axis']}={axis_info['value']}")

    # Language variants
    if languages:
        default_counts = variants_per_language or {}
        for lang in languages:
            n = default_counts.get(lang, 5 if lang in TIER1_LANGUAGES else 2)
            print(f"Generating {n} variants for {lang}...")
            lang_generated = 0
            lang_attempts = 0

            while lang_generated < n and lang_attempts < n * 3:
                lang_attempts += 1
                # Use a variation axis for diversity
                axis_info = VARIATION_AXES[lang_generated % len(VARIATION_AXES)]

                prompt = build_meta_prompt(
                    template,
                    axis=axis_info["axis"],
                    axis_value=axis_info["value"],
                    axis_description=axis_info["description"],
                    language=lang,
                )

                result = call_claude(prompt)
                if result is None:
                    continue

                errors = validate_variant(result, template)
                if errors:
                    print(f"  Variant rejected ({lang}): {errors}", file=sys.stderr)
                    continue

                variants.append(result)
                lang_generated += 1
                print(f"  [{lang_generated}/{n}] {lang}")

    return variants


def parse_language_tiers(tier_str: str) -> list[str]:
    """Parse comma-separated tier names into language list."""
    languages = []
    for part in tier_str.split(","):
        part = part.strip().lower()
        if part == "tier1":
            languages.extend(TIER1_LANGUAGES)
        elif part == "tier2":
            languages.extend(TIER2_LANGUAGES)
        else:
            # Treat as a single language name
            languages.append(part)
    return languages


def main():
    parser = argparse.ArgumentParser(description="Generate prompt variant banks")
    parser.add_argument(
        "--template", required=True, type=Path,
        help="Path to template definition YAML",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: data/prompt_variants/{name}.json)",
    )
    parser.add_argument(
        "--num-variants", type=int, default=20,
        help="Number of English variation-axis variants to generate",
    )
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated language tiers or names (e.g., 'tier1,tier2')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print meta-prompts without calling claude",
    )
    args = parser.parse_args()

    template = load_template(args.template)
    print(f"Loaded template: {template['name']}")

    if args.dry_run:
        # Print a sample meta-prompt
        prompt = build_meta_prompt(
            template,
            axis="verbosity", axis_value="terse",
            axis_description="Very brief and concise",
        )
        print("\n--- Sample meta-prompt ---")
        print(prompt)
        return

    languages = parse_language_tiers(args.languages) if args.languages else None
    variants = generate_variants(
        template,
        num_english_variants=args.num_variants,
        languages=languages,
    )

    output_path = args.output
    if output_path is None:
        output_path = Path("data/prompt_variants") / f"{template['name']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(variants, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(variants)} variants -> {output_path}")


if __name__ == "__main__":
    main()
