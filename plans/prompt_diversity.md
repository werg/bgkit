# Prompt Diversity Pipeline Upgrade

**Status:** Draft
**Depends on:** Existing compression training pipeline (ICE → Joint Block → Step 1 → Step 2)
**Blocks:** Phase 2b trajectory distillation (agentic harness)

## Problem

BgKIT's encoder accepts `prompt_embeddings` that steer what information is preserved during compression. This is a core architectural feature — the agent asks `read_file_summary(path, "What are the public methods?")` and the encoder should compress the file with that question in mind.

**But prompt conditioning is broken across the entire training pipeline:**

1. **Only verbatim prompts exist.** `configs/prompt_variants/file_read_repro.json` is the sole variant bank — 25 variants, all saying "Return the file contents verbatim" in different languages/styles. Per-objective banks (`description_gen.json`, `structural_repro.json`, `commit_repro.json`) don't exist on disk, so all objectives fall back to verbatim prompts.

2. **Joint block trainer skips prompts entirely.** `joint_block_trainer.py:213` uses `build_encoder_user_only_prefix_ids()` — bare `<|im_start|>user\n` with no system prompt, no compression prompt. Pre-computed once, frozen for all training. The encoder's earliest training phase never sees prompt conditioning.

3. **No question-conditioned targets.** Even if we had diverse compression prompts, the decoder targets are always verbatim file content or pre-existing descriptions. A prompt like "What are the error handling patterns?" paired with a target of "reproduce the full file" teaches the encoder to ignore the prompt.

## Goal

After this upgrade:
- The encoder learns that different prompts produce different compressions of the same file
- The decoder learns that different prompts produce different outputs from the same survivors
- Training data includes (question, focused-answer) pairs, not just (verbatim-prompt, full-file) pairs
- The compression prompt is exercised in every training phase (except ICE, which is prompt-independent)

## Existing data inventory

All existing generated data is preserved — nothing is discarded or regenerated.

| Data | Location | Volume | Status |
|---|---|---|---|
| File descriptions (JSONL) | `$DATA_DIR/descriptions/` | 3,180 repos | Generated, used by `description_generation` objective |
| Structural skeletons (JSONL) | `$DATA_DIR/structural/` | 8,365 repos | Generated, used by `structural_relational` objective |
| Structural mmap | `$DATA_DIR/mmap/structural/` | tokens.npy + metadata | Converted, ready for training |
| File tokens (parquet shards) | `$DATA_DIR/processed/tokens/` | 93 shards | Converted, used by `data_reconstruction` |
| Commit reproduction | `$DATA_DIR/processed/commit_reproduction/` | 13 shards + mmap | Converted, used by `commit_reproduction` |
| ICE labels | `$DATA_DIR/processed/ice_labels/` | 44 shards + mmap | Converted, used by ICE trainer |

The new `query_conditioned` objective is **purely additive** — it generates new QA pair data alongside existing datasets. The QA pipeline targets the same repos already in the training corpus and joins to the same source file tokens via `(repo_path, file_path, commit_sha)`.

## Architecture

### New training objective: `query_conditioned`

Rather than bolting question-answer pairs onto the existing 4 objectives, we add a 5th objective to the compression dataset. This keeps existing objectives stable while adding the new capability.

```
OBJECTIVE_ORDER = [
    "data_reconstruction",      # 35% (was 40%)
    "description_generation",   # 15% (was 20%)
    "structural_relational",    # 12% (was 15%)
    "commit_reproduction",      # 20% (was 25%)
    "query_conditioned",        # 18% (NEW)
]
```

The `query_conditioned` objective:
- **Compression prompt:** A natural-language question about the file (e.g., "What are the public API methods and their signatures?")
- **Decoder target:** An LLM-generated answer to that question, given the file content
- **Source:** Same file token mmap as `data_reconstruction`
- **Chat template:** Uses a new `file_read_query` tool config — same `bgkit_read_file` tool name and parameter schema, but `content_in_code_fence=False` because answers are prose, not code. This is the single source of truth for the template shape; it determines suffix tokens, loss mask boundaries, and overhead calculations via `tokenize_with_sentinel()`.
- **Subset wrapper:** A `QAConditionedSubset` class (like `DescriptionSubset`, `StructuralSubset`, etc.) that joins `MmapQAConditionedDataset` to the source file `MmapTokenDataset` via `(repo_path, file_path, commit_sha)` and returns `FileCompressionSample`. This is how it integrates into `CompressionDataset.from_config()`.

### Query-answer pair generation pipeline

New script: `scripts/generate_qa_pairs.py`

For each file in the training corpus, generate N (question, answer) pairs using the vLLM pipeline. Stored as JSONL alongside existing descriptions.

**Question diversity is the core quality metric.** The generator is NOT given a fixed category list — it uses freeform categories and is instructed to maximize diversity. Questions should name specific identifiers, vary in granularity (single expression → whole module), and vary in expected output format (prose, structured list, specific value, rewritten code). See Milestone 1 for the full prompt design rules and rationale.

**Generation approach:** Two-stage prompting via vLLM:
1. **Question generation:** Given a file, generate 3-5 diverse questions. No example questions in the prompt — examples cause parroting. The generator chooses its own freeform category labels.
2. **Answer generation:** Given a file + question, generate a focused answer. Target length: 100-500 tokens (enough detail to be useful, short enough to be a compression target).

**Storage format:**
```
data/qa_pairs/
├── {owner}/{repo}/qa_pairs.jsonl   # per-repo, like descriptions
```

Each line (see Milestone 2 for full schema with provenance fields):
```json
{
  "repo_path": "owner/repo",
  "file_path": "src/main.py",
  "commit_sha": "abc123",
  "question": "What are the public API methods and their signatures?",
  "answer": "The file exports three public functions: ...",
  "category": "api_interface",
  "question_tokens": 12,
  "answer_tokens": 187,
  "model_id": "gpt-oss-20b",
  "generation_tier": "primary",
  "prompt_version": 1
}
```

**Join key:** `(repo_path, file_path, commit_sha)` — same composite key used by all existing datasets (`MmapDescriptionDataset`, `DescriptionSubset`) to join auxiliary targets to source file tokens. Must be carried through JSONL → mmap `metadata.parquet`.

**Provenance:** `model_id`, `generation_tier`, `prompt_version` enable safe resumption across tier switches and reproducibility, matching the `prompt_version` field already stored by the description pipeline (`convert_descriptions_to_npy.py:43`).

**Scale:** ~5K repos x ~20 files/repo x 3 QA pairs/file = ~300K pairs. At ~200 tokens average answer length = ~60M answer tokens. Generation time: ~2 days on vLLM fast tier (Qwen3.5-0.8B) or ~1 day on primary tier (GPT-OSS-20B).

### Variant banks for existing objectives

Create the missing per-objective variant banks. These don't need question-conditioned targets — they just need diverse compression prompts that match the objective's task:

**`configs/prompt_variants/description_gen.json`** — prompts like:
- "Generate a concise description of this code's purpose and functionality"
- "Summarize the key components and their relationships"
- "Describe what this code does at a high level"
- (+ multilingual variants)

**`configs/prompt_variants/structural_repro.json`** — prompts like:
- "Extract the structural elements: classes, functions, imports, and their relationships"
- "List all defined symbols with their types and signatures"
- "Map the code structure: what defines what, what calls what"

**`configs/prompt_variants/commit_repro.json`** — prompts like:
- "Reproduce the changes made in this commit"
- "Show the diff introduced by this commit"
- "Reconstruct the code modifications from this commit"

These are generated using the existing `scripts/generate_prompt_variants.py` infrastructure — it already supports template YAMLs, variation axes, and multilingual generation. The template YAMLs already exist in `configs/templates/`.

## Implementation Plan

The plan is structured as iterative milestones. Each milestone produces a usable result and informs decisions for subsequent milestones. Do not plan ahead beyond the current milestone — complete it, evaluate, and adjust.

---

### Milestone 0: Variant bank generation (zero-risk, start immediately)

This uses entirely existing tooling (`scripts/generate_prompt_variants.py` + template YAMLs in `configs/templates/`). No code changes required.

**Step 0a.** Generate per-objective variant banks:
```bash
python scripts/generate_prompt_variants.py \
    --template configs/templates/description_gen.yaml \
    --num-variants 20 --languages tier1
python scripts/generate_prompt_variants.py \
    --template configs/templates/structural_repro.yaml \
    --num-variants 20 --languages tier1
python scripts/generate_prompt_variants.py \
    --template configs/templates/commit_repro.yaml \
    --num-variants 20 --languages tier1
```

Output goes to `configs/prompt_variants/{name}.json`. Git-tracked.

**IMPORTANT:** This script calls `claude -p --model haiku` and cannot run inside a Claude Code session (the script checks for `$CLAUDECODE` and exits). Run it in a separate terminal. See `scripts/generate_prompt_variants.py:343-361`.

**Step 0b.** Verify all variant banks load correctly in `CompressionDataset.from_config()`. The loading code at `compression_dataset.py:766-803` already looks for per-objective JSON files — once they exist on disk, they'll be picked up automatically with zero code changes.

```python
# Quick smoke test — run in host venv
from pathlib import Path
import json

for name in ["description_gen", "structural_repro", "commit_repro"]:
    p = Path(f"configs/prompt_variants/{name}.json")
    assert p.exists(), f"Missing: {p}"
    bank = json.loads(p.read_text())
    assert len(bank) >= 20, f"{name}: only {len(bank)} variants"
    # Check compression_prompt varies (not all identical)
    prompts = {v["compression_prompt"] for v in bank}
    assert len(prompts) > 1, f"{name}: all compression_prompts identical"
    print(f"{name}: {len(bank)} variants, {len(prompts)} unique compression_prompts")
```

**Exit criteria:** 3 new variant banks exist, each with 20+ variants, diverse compression prompts. Existing training pipeline still works unchanged (variant banks are additive).

---

### Milestone 1: QA prompt pilot (50 files, manual review)

Before building the full QA generation pipeline, we need to validate that our prompts produce useful (question, answer) pairs. This is the highest-risk part of the plan — the prompt design must be iterated on by looking at real outputs.

**Step 1a.** Write a small standalone pilot script `scripts/pilot_qa_generation.py` (NOT the full pipeline yet). It should:
- Take a list of ~50 diverse files (mix of languages, sizes, complexity levels)
- For each file, call vLLM with the question-generation prompt
- For each generated question, call vLLM with the answer-generation prompt
- Write all outputs to a single JSONL for easy manual review
- Include the raw file content in the output so reviewer can judge answer quality

Starting prompts for the pilot (these WILL need iteration):

**Key principle: maximize semantic diversity.** The whole point of this pipeline is to teach the encoder that different prompts produce different compressions. If the questions are all variations of "summarize this file", we've failed. The question generator must produce wildly diverse questions — different semantic focuses, different levels of granularity, different output formats. This is the single most important quality criterion.

**Question generation prompt design rules:**
1. **No examples in the prompt.** Listing example questions (even as categories) causes the generator to parrot them back with minor rewording. The categories below are for the *plan reader's* understanding, NOT for inclusion in the generation prompt.
2. **Explicitly invite creative/unexpected questions.** The generator should ask for specific semantic content ("What does the `retry_with_backoff` decorator do?"), not just generic categories ("What patterns are used?").
3. **Encourage questions about specific identifiers.** Questions that name actual functions, classes, variables, or constants from the file force the encoder to learn fine-grained content selection.
4. **Vary granularity.** Some questions should be about a single line or expression, others about the whole module's architecture.
5. **Vary output format.** Some questions ask for prose ("Explain..."), others for structured output ("List the methods with their signatures"), others for specific values ("What is the default timeout?").

**Desired question diversity** (for plan reader — NOT included in prompt):
- Purpose/summary, API/interface, control flow, error handling, dependencies, data structures, testing, patterns — the obvious categories
- But also: "What is the return type of `process_batch`?", "Which exception is raised when the connection times out?", "What would break if you removed the `@cache` decorator?", "Rewrite the docstring for `validate_input`", "What assumptions does this code make about the input format?", "How would you refactor the nested if-else on line 45?", "What security vulnerabilities exist in this code?", "Translate the core algorithm into pseudocode"

**Question generation prompt (v1):**
```
You are generating training data for a code comprehension system that
learns to compress source code relative to a question. Your job is to
generate diverse, specific questions about the code below.

CRITICAL: Maximize diversity. Each question should focus on a DIFFERENT
aspect of the code. Ask about specific functions, variables, types,
algorithms, edge cases, assumptions, security properties, performance
characteristics, or anything else you find interesting. Be creative —
ask unexpected questions that require deep understanding of the code.
Name specific identifiers from the code in your questions when possible.
Vary the granularity (single line → whole module) and the expected
answer format (prose explanation, structured list, specific value,
rewritten code, etc.).

Do not repeat the same question type. If you already asked about
purpose, ask about something completely different next.

Generate 3-5 questions.

File: {file_path}
```
{content}
```

Return ONLY a JSON array: [{"question": "...", "category": "..."}]
The "category" is a short label you choose (freeform, not from a fixed list).
Do not include any other text.
```

**Answer generation prompt (v1):**
```
Answer the following question about a source code file.
Be specific — reference actual function names, class names, and logic
from the code. Target 100-400 tokens. Do not reproduce the entire file.

File: {file_path}
```
{content}
```

Question: {question}

Answer:
```

**Step 1b.** Run the pilot on ~50 files:
```bash
# Select 50 diverse files from the training corpus
python scripts/pilot_qa_generation.py \
    --repos-dir $DATA_DIR/repos/ \
    --output pilot_qa_output.jsonl \
    --num-files 50 \
    --server-url http://localhost:8090/v1
```

Use the primary tier (GPT-OSS-20B) for the pilot — we want to see the best-case output quality before deciding if the fast tier (0.8B) is sufficient.

**Step 1c.** Manual review checklist (evaluate all 50 files, ~150 QA pairs):

| Check | Pass criteria |
|---|---|
| Question relevance | >80% of questions are answerable from the file content |
| Question specificity | >50% of questions name a specific identifier from the file |
| Semantic diversity | Across 50 files (~150 questions), at least 30 distinct freeform categories |
| Within-file diversity | No two questions per file have the same category label |
| Answer accuracy | >70% of answers are factually correct w.r.t. the code |
| Answer specificity | >70% of answers reference actual identifiers from the code |
| Answer length | 80% of answers are 50-500 tokens (not trivially short or bloated) |
| JSON parsing | >95% of question-generation outputs parse as valid JSON |
| No parroting | <10% of questions are generic "What does this code do?" / "Summarize this file" |

**Step 1d.** Iterate on prompts based on review. Common failure modes to watch for:
- Questions too generic ("What does this code do?") → the prompt already says "do not repeat question types" and "name specific identifiers", but may need stronger language or a negative example
- All questions cluster into 3-4 categories despite freeform labels → strengthen the diversity instruction, possibly add "Your questions should be surprising"
- Answers just paraphrase the question → add "Reference specific code elements" instruction
- JSON parsing failures → try structured output / constrained decoding if available, or add few-shot examples to the *answer* prompt (but NOT the question prompt)
- Answers too long → adjust token guidance, consider separate max_tokens parameter
- Questions about trivial files (empty `__init__.py`, config JSON) → add file complexity filter (min lines, min AST nodes, etc.)

**Exit criteria:** Prompts produce >70% usable QA pairs on the pilot set. Document the final prompt versions in the script for the full pipeline.

**Decision point:** If GPT-OSS-20B quality is good, test the same prompts on the fast tier (0.8B). If 0.8B quality is acceptable (>50% usable), use it for bulk generation (10x faster). If not, use primary for all files or only for complex files (same two-tier routing as description generation).

---

### Milestone 2: Full QA generation pipeline

Only start this after Milestone 1 prompts are validated.

**Step 2a.** Create `scripts/generate_qa_pairs.py` — full pipeline script:
- Reuses `bgkit.inference.LlamaClient` and `bgkit.data.repo_processing.extract_repo_snapshot()`
- Same two-tier routing as `scripts/generate_descriptions.py`: primary for complex files, fast for simple
- Idempotent and resumable (skip repos with existing output files)
- Uses the final prompt versions from Milestone 1
- Filters out trivial files: skip files <10 lines, binary files, generated files, `__init__.py` with <5 lines
- **Required: skip files exceeding `max_seq_len` tokens** (default 8192). `MmapTokenDataset` chunks long files into multiple token indices, and `QAConditionedSubset` (Milestone 4) only joins to single-chunk files for correctness — the encoder must see the full file that the QA answer references. Generating QA pairs for multi-chunk files wastes generation budget on data that will be discarded. The token count can be estimated from character count (~3.5 chars/token for code) or by running the tokenizer on each file during extraction.
- Validates JSON parsing of question-generation output, retries once on failure
- Stores per-repo JSONL files
- **Prioritize repos without existing descriptions** (`$DATA_DIR/descriptions/`) for higher data diversity — the 3,180 repos with descriptions are already well-represented in training via the `description_generation` objective. QA pairs from under-represented repos add more new signal.
- All target repos must have source file tokens in `$DATA_DIR/processed/tokens/` (the join key must resolve)

**Storage format:**
```
data/qa_pairs/
├── {owner}/{repo}/qa_pairs.jsonl   # per-repo, like descriptions
```

Each line:
```json
{
  "repo_path": "owner/repo",
  "file_path": "src/main.py",
  "commit_sha": "abc123",
  "question": "What are the public API methods and their signatures?",
  "answer": "The file exports three public functions: ...",
  "category": "api_interface",
  "question_tokens": 12,
  "answer_tokens": 187,
  "model_id": "gpt-oss-20b",
  "generation_tier": "primary",
  "prompt_version": 1
}
```

`repo_path` is required for the `(repo_path, file_path, commit_sha)` join key. Provenance fields (`model_id`, `generation_tier`, `prompt_version`) enable safe resumption and mixed-tier detection.

**Step 2b.** Run QA pair generation over the training corpus:
```bash
python scripts/generate_qa_pairs.py \
    --repos-dir $DATA_DIR/repos/ \
    --output-dir $DATA_DIR/qa_pairs/ \
    --max-files-per-repo 30 \
    --max-qa-per-file 3 \
    --server-url-primary http://localhost:8090/v1 \
    --server-url-fast http://localhost:8091/v1 \
    --workers 8
```

**Scale:** ~5K repos x ~20 files/repo x 3 QA pairs/file = ~300K pairs. At ~200 tokens average answer length = ~60M answer tokens. Generation time: ~2 days on fast tier, ~1 day on primary tier.

**Step 2c.** Post-generation quality spot-check:
- Sample 100 random QA pairs from the output
- Run the same manual review checklist from Milestone 1
- If quality dropped significantly vs pilot (e.g., due to fast tier), re-generate failing repos with primary tier

**Step 2d.** Convert QA pairs to mmap format:
```bash
python scripts/convert_qa_pairs_to_npy.py \
    --input-dir $DATA_DIR/qa_pairs/ \
    --output-dir $DATA_DIR/models/qa_conditioned/ \
    --tokenizer Qwen/Qwen3.5-0.8B-Base
```

This produces the following artifacts:

**Memory-mapped arrays** (fast random access via `BaseMmapDataset`):
- `tokens.npy` / `offsets.npy` — answer token sequences (decoder targets)
- `question_tokens.npy` / `question_offsets.npy` — question token sequences (compression prompts). These are variable-length and accessed every sample, so they MUST be in dedicated mmap arrays, not in parquet. Parquet column access is row-by-row deserialization, not memory-mapped — it would be a bottleneck at training time.

**Metadata** (`metadata.parquet`):
- **Join key columns:** `repo_path`, `file_path`, `commit_sha` (for joining to source file tokens)
- **QA-specific columns:** `question` (raw text for debugging/analysis), `category`
- **Provenance columns:** `model_id`, `generation_tier`, `prompt_version` (matching `convert_descriptions_to_npy.py:42-45`)

**Other:** `manifest.json` (dataset stats and config)

**Exit criteria:** `data/qa_pairs/` populated, converted to mmap, spot-check passes.

---

### Milestone 3: Joint block trainer fix

This is independent of QA generation — it only needs the variant banks from Milestone 0.

**Step 3a.** Fix `joint_block_trainer.py` to use compression prompts:

Current (broken — `joint_block_trainer.py:213-216`):
```python
prefix_ids = build_encoder_user_only_prefix_ids(tokenizer).to(device)
# Pre-computed once, frozen, no compression prompt
```

Fixed — minimal approach (per-epoch rotation):
```python
# In __init__:
self._variant_bank = load_all_variant_banks("configs/prompt_variants/")
self._tokenizer = tokenizer  # keep reference for recomputing embeddings

# In training loop, at epoch boundary:
variant = select_variant(self._variant_bank, epoch, epoch_seed=self._seed + epoch)
prefix_ids = build_encoder_prefix_ids(self._tokenizer, variant["compression_prompt"]).to(self.device)
# CRITICAL: must recompute the cached embeddings, not just the token IDs.
# The current code converts prefix_ids → _prompt_embeddings once in __init__
# (joint_block_trainer.py:214-216) and _get_prompt_embeddings() returns that
# cache forever (line 244-246). Without this line, the rotation is a no-op.
with torch.no_grad():
    self._prompt_embeddings = self._get_input_embeddings(prefix_ids.unsqueeze(0))
logger.info("epoch_prompt_rotation", epoch=epoch, prompt=variant["compression_prompt"][:60])
```

**Step 3b.** Implement `load_all_variant_banks()` utility — concatenate all `*.json` files from `configs/prompt_variants/` into one pool. Extract just the `compression_prompt` field from each. Deduplicate. This gives the joint block trainer access to prompts from all objectives.

**Step 3c.** The prompt is still frozen within an epoch (same for every batch), but varies between epochs. This is a deliberate tradeoff — the joint block phase trains on raw token chunks (no file boundaries), so per-sample prompts don't make semantic sense. Per-epoch rotation is sufficient to teach the encoder that the prompt prefix exists and varies.

**Step 3d (important).** The epoch boundary hook must be placed where the trainer transitions between epochs. Check the trainer's `train()` method for the epoch loop and add the recomputation there. The `_get_prompt_embeddings()` method (line 244-246) returns `self._prompt_embeddings` via `getattr`, so updating the attribute is sufficient — no method signature changes needed.

**Step 3d.** Add unit test: verify that `build_encoder_prefix_ids(tokenizer, prompt)` produces different token sequences for different prompts, and that the joint block trainer calls it with varying prompts across epochs.

**Exit criteria:** Joint block trainer uses compression prompts. Existing unit tests still pass. No training regression (loss curve should be similar to before — the prompt prefix adds a few tokens but shouldn't change convergence behavior).

---

### Milestone 4: Dataset integration (5th objective)

Depends on: Milestone 2 (QA mmap data exists).

**Step 4a.** Create `src/bgkit/data/datasets/qa_conditioned_dataset.py` with two classes:

1. **`MmapQAConditionedDataset(BaseMmapDataset)`** — raw mmap access layer. Loads:
   - `tokens.npy` / `offsets.npy` — answer token sequences (decoder targets)
   - `question_tokens.npy` / `question_offsets.npy` — question token sequences (compression prompts)
   - `metadata.parquet` — `repo_path`, `file_path`, `commit_sha`, `question`, `category`, provenance fields

   **Index-based, not key-based.** Unlike `MmapDescriptionDataset` which provides dict lookup by `(repo_path, file_path, commit_sha)` (one description per file), the QA dataset has multiple rows per file (3 QA pairs each). There is no unique key per row. Instead, `MmapQAConditionedDataset` is iterated by index. Each row exposes `repo_path`, `file_path`, `commit_sha` from metadata so the *subset wrapper* can build the reverse join to source file tokens.

   `__getitem__(idx)` returns: `answer_token_ids`, `question_token_ids` (both from mmap), plus metadata fields.

2. **`QAConditionedSubset(Dataset)`** — training-side wrapper that joins QA rows to source file tokens. This is the class that `CompressionDataset.from_config()` constructs and registers as the 5th objective.

   **Join direction is QA→file** (opposite of `DescriptionSubset` which joins file→description). The subset iterates QA rows and looks up the corresponding source file tokens:

```python
class QAConditionedSubset(Dataset):
    """Sub-dataset for Objective 5: query-conditioned compression.

    Iterates QA rows and joins each to its source file tokens via
    (repo_path, file_path, commit_sha). Returns FileCompressionSample.

    Join direction: QA→file (iterate QA rows, look up source file).
    This is the reverse of DescriptionSubset (which iterates files,
    looks up descriptions) because there are multiple QA rows per file.
    """

    def __init__(self, token_dataset, qa_dataset, tokenizer, config, seed=42):
        self._token_ds = token_dataset
        self._qa_ds = qa_dataset
        self._tokenizer = tokenizer
        self._config = config  # TOOL_CONFIGS["file_read_query"]
        self._base_seed = seed
        self._epoch_seed = seed

        # Build reverse index: for each QA row, find the source file token index.
        # IMPORTANT: MmapTokenDataset chunks long files into multiple token_idx
        # values (mmap_token_dataset.py:90). A QA answer is about the whole file,
        # so we must only join to single-chunk files. Multi-chunk files would
        # give the encoder a partial view of the file while the decoder target
        # (answer) references the whole file — a correctness bug.
        #
        # Strategy: build file key → (first_chunk_idx, n_chunks) lookup.
        # Only keep files where n_chunks == 1.
        file_key_to_chunk_info: dict[tuple[str, str, str], tuple[int, int]] = {}
        for tok_idx in range(len(token_dataset)):
            key = token_dataset.file_key(tok_idx)  # returns (repo_path, file_path, commit_sha) or None
            if key is None:
                continue
            if key not in file_key_to_chunk_info:
                file_key_to_chunk_info[key] = (tok_idx, 1)
            else:
                first, count = file_key_to_chunk_info[key]
                file_key_to_chunk_info[key] = (first, count + 1)

        # Keep only single-chunk files (file fits in max_seq_len)
        single_chunk_keys = {
            k: first for k, (first, count) in file_key_to_chunk_info.items()
            if count == 1
        }
        # Most multi-chunk files are already excluded by the max_seq_len
        # filter in M2's generation pipeline. This is a safety net for any
        # that slip through (e.g., files near the boundary after tokenization).

        # For each QA row, resolve its source file. Drop QA rows with no match
        # or where the source file is multi-chunk.
        self._joined_indices: list[tuple[int, int]] = []  # (token_idx, qa_idx)
        for qa_idx in range(len(qa_dataset)):
            key = qa_dataset.file_key(qa_idx)  # MmapQAConditionedDataset exposes same file_key() API
            if key is None:
                continue
            tok_idx = single_chunk_keys.get(key)
            if tok_idx is not None:
                self._joined_indices.append((tok_idx, qa_idx))

        # Precompute overhead from file_read_query config
        self._max_overhead = ...
        self._cached_lengths = ...

    def __getitem__(self, idx) -> FileCompressionSample:
        tok_idx, qa_idx = self._joined_indices[idx]
        source = self._token_ds[tok_idx]
        qa = self._qa_ds[qa_idx]

        # Question from QA row → compression_prompt_ids (from question mmap)
        compression_prompt_ids = qa["question_token_ids"]

        # Build chat template with file_read_query config
        # Answer tokens are the decoder target (in place of file content)
        result = tokenize_with_sentinel(
            self._tokenizer, self._build_variant(qa), self._config,
            source["file_path"], source.get("language", ""),
            qa["answer_token_ids"],
        )

        return FileCompressionSample(
            objective="query_conditioned",
            content_token_ids=source["token_ids"],  # source file for encoder
            content_attention_mask=...,
            target_token_ids=result["token_ids"],    # answer for decoder
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=compression_prompt_ids,
            ...
        )
```

The question is pre-tokenized at mmap conversion time (Step 2d) so there's no tokenization overhead at training time.

**Step 4b.** Add `file_read_query` tool config to `TOOL_CONFIGS` in `chat_template.py`. This is the single source of truth for the query-conditioned template shape — it determines suffix tokens, loss mask boundaries, and overhead calculations:
```python
"file_read_query": ChatTemplateConfig(
    tool_name="bgkit_read_file",
    tool_description="Read and analyze file contents from BgKIT compressed context.",
    tool_parameters={...},  # same schema as file_read_repro
    content_in_code_fence=False,  # answers are prose, not code
),
```

**Step 4c.** Register `query_conditioned` in `CompressionDataset.from_config()`:
- Add to `OBJECTIVE_ORDER` and `DEFAULT_WEIGHTS`
- Construct `QAConditionedSubset` (not raw `MmapQAConditionedDataset`) in `from_config()`, following the same pattern as objectives 2-4 (`compression_dataset.py:878-914`)
- Re-balance weights:

```python
DEFAULT_WEIGHTS = {
    "data_reconstruction": 0.35,     # was 0.40
    "description_generation": 0.15,  # was 0.20
    "structural_relational": 0.12,   # was 0.15
    "commit_reproduction": 0.20,     # was 0.25
    "query_conditioned": 0.18,       # NEW
}
```

**Note on weights:** These are starting values. The right balance is empirical — tune based on per-objective loss curves in Milestone 6. The key constraint is that `data_reconstruction` stays the largest objective (verbatim reproduction is the foundation), and `query_conditioned` gets enough weight to actually learn prompt sensitivity.

**Step 4d.** Add Hydra config support: new fields in the training config YAML for `qa_data_dir` and `query_conditioned_weight`. The weight should be overridable without code changes.

**Step 4e.** Unit tests:
- `test_qa_conditioned_dataset.py` — loading, indexing, token shapes, compression_prompt_ids correctness
- `test_compression_dataset_5obj.py` — verify 5-objective construction, weight balancing, index ranges

**Exit criteria:** `CompressionDataset.from_config()` returns a dataset with 5 objectives. Iteration over the QA objective produces valid batches with question-derived `compression_prompt_ids` and answer-derived targets.

---

### Milestone 5: Decoder init (Step 1) QA mixing

Depends on: Milestone 4 (QA dataset class exists).

**Step 5a.** The existing `ChatReproDataset` (`chat_repro_dataset.py`) is deeply wired to a single template and cannot simply gain a QA branch in `__getitem__`:
- Hardcoded `_FILE_READ_CONFIG = TOOL_CONFIGS["file_read_repro"]` at module level (line 30)
- Precomputed `_max_template_overhead` assuming one template's token count (line 65-66)
- Cached `suffix_ids` from that single config (line 71-74), used as generation stopping criteria by `decoder_init.py:559`
- QA samples use `content_in_code_fence=False` (answers aren't code), producing a different suffix, different overhead, and different stopping tokens

**Do NOT try to make `ChatReproDataset` multi-template.** Instead, create a separate `QAChatReproDataset` that follows the same interface but uses the `file_read_query` config:

```python
class QAChatReproDataset(Dataset):
    """Chat-formatted QA dataset for decoder init training.

    Like ChatReproDataset but uses question-conditioned prompts and
    answer targets instead of verbatim file content. Uses file_read_query
    template config (content_in_code_fence=False).
    """
    def __init__(self, qa_mmap_dataset, file_token_dataset, tokenizer, seed=42):
        self._qa_ds = qa_mmap_dataset
        self._file_ds = file_token_dataset
        self._tokenizer = tokenizer
        self._config = TOOL_CONFIGS["file_read_query"]
        # Precompute overhead and suffix_ids from THIS config
        self._suffix_ids = compute_suffix_ids(
            tokenizer, self._build_variant_stubs(), self._config,
        )
        ...
```

**Step 5b.** Use **separate dataloaders** for verbatim and QA, alternating batches at the trainer level. Do NOT try to mix suffix types within a batch — `collate_chat_repro()` (`collators.py:62-103`) returns a fixed set of keys and does not pass through `suffix_ids`. Modifying the collator to handle heterogeneous suffix types is fragile and unnecessary.

Architecture:
```python
# In decoder_init.py setup:
self.verbatim_dataset = ChatReproDataset(...)       # existing
self.qa_dataset = QAChatReproDataset(...)            # new

# Separate samplers (both need .lengths for TokenBudgetBatchSampler)
verbatim_lengths = self.verbatim_dataset.lengths[train_indices_v]
qa_lengths = self.qa_dataset.lengths[train_indices_q]
self.verbatim_sampler = TokenBudgetBatchSampler(verbatim_lengths, ...)
self.qa_sampler = TokenBudgetBatchSampler(qa_lengths, ...)

self.verbatim_loader = DataLoader(
    ..., batch_sampler=self.verbatim_sampler, collate_fn=collate_chat_repro,
)
self.qa_loader = DataLoader(
    ..., batch_sampler=self.qa_sampler, collate_fn=collate_chat_repro,
)

# In training loop — sample-count-based ratio scheduling:
# TokenBudgetBatchSampler produces variable-size batches, so batch count
# does NOT correspond to sample count or token count. We must track the
# actual number of samples drawn from each loader and switch sources
# when the ratio drifts.
verbatim_samples_seen = 0
qa_samples_seen = 0
verbatim_iter = iter(self.verbatim_loader)
qa_iter = iter(self.qa_loader)

for step in range(total_steps):
    total = verbatim_samples_seen + qa_samples_seen + 1  # avoid div-by-zero
    current_qa_frac = qa_samples_seen / total
    if current_qa_frac < self.qa_ratio and qa_iter is not None:
        batch = next(qa_iter)
        qa_samples_seen += batch["token_ids"].size(0)
    else:
        batch = next(verbatim_iter)
        verbatim_samples_seen += batch["token_ids"].size(0)
    # ... training step

# Loader exhaustion policy: when one loader runs out, recreate its
# iterator (new epoch for that loader). An "epoch" ends when the
# *verbatim* loader is exhausted (it's the larger dataset and the
# primary training signal). The QA loader may cycle multiple times
# per epoch or not finish — that's fine.
# Reset sample counters at each epoch boundary for ratio tracking.
```

**`qa_ratio` is defined in terms of samples** (not batches, not tokens). This is the most intuitive unit — "30% of training samples are QA" — and it's straightforward to track since `batch["token_ids"].size(0)` gives the sample count per batch.

This is cleaner than an interleaved dataset wrapper because:
- Each loader uses `collate_chat_repro` unchanged — no collator modifications
- `TokenBudgetBatchSampler` gets a homogeneous `.lengths` array per loader
- Suffix types never mix within a batch — each loader's dataset has one `suffix_ids`
- The ratio is controlled by actual sample counts, not batch counts

**Step 5c.** Update `decoder_init.py` evaluation: use separate eval dataloaders for each dataset type. Each eval pass uses its own `suffix_ids` for generation stopping criteria. Report metrics per-type. Combined metric is weighted by **content token count** (not sample count), matching how `decoder_init.py:521-530` already computes loss internally as a content-token-weighted average. QA answers (~200 tokens) and verbatim targets (~500-2000 tokens) have very different lengths — sample-count weighting would bias toward the shorter-target dataset.

```python
# _eval_loop returns (metrics_dict, total_content_tokens)
v_metrics, v_tokens = self._eval_loop(self.verbatim_eval_loader,
                                       self.verbatim_dataset.suffix_ids)
q_metrics, q_tokens = self._eval_loop(self.qa_eval_loader,
                                       self.qa_dataset.suffix_ids)
# Weight by content token count — matches internal loss computation
total_tokens = v_tokens + q_tokens
combined_loss = (v_tokens * v_metrics["loss"] + q_tokens * q_metrics["loss"]) / total_tokens
```

**Step 5d.** Update `decoder_init.py` to construct both datasets and the alternating batch scheduler from Hydra config. Add config fields `data.qa_data_dir` and `training.qa_ratio` (default 0.3).

**Step 5e.** Unit tests:
- `test_qa_chat_repro_dataset.py` — verify `QAChatReproDataset` produces correct template shape, `suffix_ids` differ from verbatim, `.lengths` is exposed
- `test_decoder_init_dual_loader.py` — verify alternating batch scheduling achieves target ratio, eval uses correct suffix_ids per loader

**Exit criteria:** Dual-loader training produces alternating verbatim/QA batches at the configured ratio. Each loader uses `collate_chat_repro` unchanged. Eval reports per-type metrics with correct suffix_ids per dataset type.

---

### Milestone 6: Validation and tuning

Depends on: All previous milestones.

**Step 6a.** Integration smoke test:
- Run 100 steps of Step 2 compression training with 5 objectives
- Verify loss decreases on all objectives including `query_conditioned`
- Verify the `query_conditioned` loss starts higher than `data_reconstruction` (expected — questions are harder than verbatim)

**Step 6b.** Prompt sensitivity ablation:
- Encode the same 50 files with 5 different compression prompts each
- Measure cosine similarity of survivor embeddings across prompts for the same file
- **Before this upgrade:** similarity should be ~1.0 (encoder ignores prompt)
- **After this upgrade:** similarity should be measurably lower (encoder is prompt-sensitive)
- This is the key metric that validates the entire plan worked

**Step 6c.** Weight tuning:
- Monitor per-objective loss curves during a full Step 2 training run
- If `query_conditioned` loss plateaus early, increase its weight
- If `data_reconstruction` loss regresses, increase its weight back
- Document final weights in the Hydra config

**Step 6d.** Decoder init validation:
- Run Step 1 (decoder init) with QA mixing enabled
- Compare reconstruction quality on held-out files vs without QA mixing
- The verbatim reconstruction quality should not degrade significantly (the 70% verbatim ratio preserves it)

**Exit criteria:** Prompt sensitivity ablation shows measurable improvement. Per-objective losses converge. No regression on existing objectives.

## Data pipeline integration

The QA generation and conversion must be integrated into the existing data pipeline system (`scripts/prepare-data.sh`, `Makefile`). Without this, the QA data is a one-off artifact that doesn't get rebuilt when repos change.

### Makefile targets (add to `Makefile`)

```makefile
# NOTE: `make generate-variants` already generates all 4 template banks
# (file_read_repro, description_gen, structural_repro, commit_repro) —
# see Makefile:158-166. No new variant target is needed for M0.
# The only change is to add --languages tier1 to the existing invocations
# if multilingual variants are desired (currently they run without --languages).

# QA pair generation (needs vLLM, like generate-descriptions)
generate-qa-pairs: vllm-server
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set"; exit 1; }
	.venv/bin/python scripts/generate_qa_pairs.py \
		--repos-dir $(DATA_DIR)/repos/ \
		--output-dir $(DATA_DIR)/qa_pairs/ \
		--server-url-primary http://localhost:$(VLLM_PORT_PRIMARY)/v1 \
		--server-url-fast http://localhost:$(VLLM_PORT_FAST)/v1

# QA pair conversion to mmap
convert-qa-pairs:
	@test -n "$(DATA_DIR)" || { echo "ERROR: DATA_DIR not set"; exit 1; }
	.venv/bin/python scripts/convert_qa_pairs_to_npy.py \
		--input-dir $(DATA_DIR)/qa_pairs/ \
		--output-dir $(DATA_DIR)/mmap/qa_conditioned/
```

### Pipeline orchestrator (`scripts/prepare-data.sh`)

Add two new stages to the pipeline. Current stages are 1-8; add:

```
#   9  generate-qa-pairs       opt-in (--with-qa), after 1+2 (Wave 1), needs vLLM, marker-gated
#  10  convert-qa-pairs        opt-in, after 9, always re-runs
```

Changes to `prepare-data.sh`:
1. Add `--with-qa` flag (like `--with-descriptions`)
2. Add `WITH_QA=false` variable
3. Update `FROM_STAGE` validation to accept 1-10
4. Stages 4 (descriptions) and 9 (QA) both need vLLM and can run in parallel. Stage 4 depends on stage 1 (structural output), stage 9 depends on stage 2 (tokenized files). Both dependencies complete in Wave 1 before this wave starts. Add stage 9 to Wave 1b as a background job when either is enabled:
   ```bash
   # Wave 1b/1c: LLM generation (needs vLLM)
   if [[ "$WITH_DESCRIPTIONS" == true ]] || [[ "$WITH_QA" == true ]]; then
       log "Wave 1b: LLM generation (needs vLLM)"
       PIDS=(); NAMES=()

       if [[ "$WITH_DESCRIPTIONS" == true ]]; then
           run_stage 4 "generate-descriptions" generate-descriptions &
           PIDS+=($!); NAMES+=("generate-descriptions")
       fi
       if [[ "$WITH_QA" == true ]]; then
           run_stage 9 "generate-qa-pairs" generate-qa-pairs &
           PIDS+=($!); NAMES+=("generate-qa-pairs")
       fi

       FAIL=0
       for i in "${!PIDS[@]}"; do
           if ! wait "${PIDS[$i]}"; then
               echo "FAILED: ${NAMES[$i]}"; FAIL=1
           fi
       done
       (( FAIL == 0 )) || die "LLM generation wave failed"
   fi
   ```
   Both jobs hit the same vLLM servers concurrently. The servers already handle concurrent requests via batching — `generate_descriptions.py` and `generate_qa_pairs.py` both use `LlamaClient` with connection pooling. If resource contention is a problem, the user can run them sequentially with `--from`.
5. Add stage 10 in Wave 2 (parallel with other conversions):
   ```bash
   if [[ "$WITH_QA" == true ]]; then
       run_stage_always 10 "convert-qa-pairs" convert-qa-pairs &
       PIDS+=($!); NAMES+=("convert-qa-pairs")
   fi
   ```
6. Add dependency validation: stage 10 needs stage 9 output (`$DATA_DIR/qa_pairs/`)
7. Update `--force-all` to also clean `$DATA_DIR/qa_pairs/`

### `prepare-data-full` update

Add a new composite target:
```makefile
prepare-data-all:
	scripts/prepare-data.sh --with-descriptions --with-qa $(ARGS)
```

### Hydra training config updates

The training configs need new fields so trainers can find the QA data. Add to the `data:` section of the compression training config:

```yaml
data:
  # ... existing fields ...
  qa_data_dir: ${oc.env:DATA_DIR}/mmap/qa_conditioned/
  prompt_variants_dir: configs/prompt_variants/
```

And to the decoder init config:
```yaml
data:
  # ... existing fields ...
  qa_data_dir: ${oc.env:DATA_DIR}/mmap/qa_conditioned/
training:
  qa_ratio: 0.3
```

### CLAUDE.md updates

Add to the Key Commands table:
```
| Generate QA pairs | `make generate-qa-pairs` |
| Convert QA to mmap | `make convert-qa-pairs` |
| Generate all variant banks | `make generate-variants` (already generates all 4 templates) |
| Full data pipeline (all) | `make prepare-data-all` or `scripts/prepare-data.sh --with-descriptions --with-qa` |
```

Add to the Training phase pipeline description that prompt diversity is a prerequisite for Phase 2b.

## File Changes Summary

| Milestone | File | Change |
|---|---|---|
| M0 | `configs/prompt_variants/description_gen.json` | NEW — generated variant bank |
| M0 | `configs/prompt_variants/structural_repro.json` | NEW — generated variant bank |
| M0 | `configs/prompt_variants/commit_repro.json` | NEW — generated variant bank |
| M1 | `scripts/pilot_qa_generation.py` | NEW — small-scale QA pilot for prompt iteration |
| M2 | `scripts/generate_qa_pairs.py` | NEW — full QA pair generation pipeline |
| M2 | `scripts/convert_qa_pairs_to_npy.py` | NEW — convert QA JSONL to mmap format |
| M3 | `src/bgkit/training/joint_block_trainer.py` | Per-epoch prompt rotation via variant banks |
| M3 | `src/bgkit/data/chat_template.py` | Add `load_all_variant_banks()` utility |
| M4 | `src/bgkit/data/chat_template.py` | Add `file_read_query` to `TOOL_CONFIGS` |
| M4 | `src/bgkit/data/datasets/compression_dataset.py` | Add 5th objective via `QAConditionedSubset`, rebalance weights |
| M4 | `configs/templates/file_read_query.yaml` | NEW — template for query-conditioned reads |
| M4 | `src/bgkit/data/datasets/qa_conditioned_dataset.py` | NEW — `MmapQAConditionedDataset` (mmap access) + `QAConditionedSubset` (training wrapper for CompressionDataset) |
| M5 | `src/bgkit/data/datasets/qa_chat_repro_dataset.py` | NEW — `QAChatReproDataset` (for decoder init, uses `file_read_query` config) |
| M5 | `src/bgkit/training/phase1/decoder_init.py` | Dual dataloaders, alternating batch scheduler, per-type eval |
| M4 | `tests/unit/test_qa_conditioned_dataset.py` | NEW — unit tests |
| M4 | `tests/unit/test_compression_dataset_5obj.py` | NEW — 5-objective integration test |
| M3 | `tests/unit/test_joint_block_prompt_rotation.py` | NEW — prompt rotation + embedding recomputation test |
| M5 | `tests/unit/test_qa_chat_repro_dataset.py` | NEW — QA chat repro template shape test |
| M5 | `tests/unit/test_decoder_init_dual_loader.py` | NEW — alternating batch ratio, per-type eval with correct suffix_ids |
| M2 | `Makefile` | Add `generate-qa-pairs`, `convert-qa-pairs`, `prepare-data-all` targets |
| M2 | `scripts/prepare-data.sh` | Add stages 9-10 (QA generation + conversion), `--with-qa` flag |
| M0 | `Makefile` | Optionally add `--languages tier1` to existing `generate-variants` invocations |
| M4 | Hydra training configs | Add `qa_data_dir`, `prompt_variants_dir` fields |
| M5 | Hydra training configs | Add `qa_ratio` field to decoder init config |
| All | `CLAUDE.md` | Add QA pipeline commands to Key Commands table |

## Dependency Graph

```
M0 (variant banks) ──────────► M3 (joint block fix)
        │
        ▼
M1 (QA pilot, 50 files) ─── iterate prompts until quality passes
        │
        ▼
M2 (full QA generation) ───► M4 (5th objective dataset)
                                      │
                                      ▼
                              M5 (decoder init QA mix)
                                      │
                                      ▼
                              M6 (validation & tuning)
```

- **M0 and M1** can start in parallel (M0 is zero-risk, M1 is exploratory)
- **M3** only needs M0 (variant banks exist) — can run in parallel with M1/M2
- **M2** is blocked on M1 (prompt quality must be validated before bulk generation)
- **M4-M6** are sequential and depend on M2 completing
- **Critical path:** M1 → M2 → M4 → M5 → M6 (QA prompt iteration is the bottleneck)

## Sequencing guidance for the implementing agent

1. Start M0 immediately (run `generate_prompt_variants.py` in a separate terminal)
2. Start M1 in parallel (write and run the pilot script)
3. M1 will likely require 2-3 iterations of prompt refinement — do not skip the manual review
4. Once M0 is done, start M3 (joint block fix) — this is independent of QA work
5. Once M1 passes quality gates, proceed to M2 (bulk generation)
6. M4/M5/M6 are straightforward implementation once the data exists

## Open design decisions

These should be resolved during implementation, not pre-decided in this plan:

1. **Objective weights** (35/15/12/20/18 split) — starting values, tune empirically in M6
2. **QA generation tier** — pilot on GPT-OSS-20B (M1), decide in M1 exit whether 0.8B is sufficient for bulk
3. **QA pairs per file** — start with 3, adjust based on pilot quality and total dataset size
4. **Decoder init QA ratio** — start with 0.3, tune in M6 based on reconstruction quality
5. **Pre-tokenized vs on-the-fly question tokenization** — pre-tokenize at mmap conversion time (faster training, less flexible). If we need to change tokenizers, re-run conversion.

## Risks

1. **QA prompt quality requires iteration.** The biggest risk is that the first prompt design produces unusable QA pairs. This is why M1 exists as a separate pilot milestone with explicit quality gates. Budget 2-3 iterations of prompt refinement before the full pipeline run.

2. **QA answer quality from 0.8B model.** The fast tier may produce shallow or incorrect answers for complex files. Mitigation: pilot on primary tier first (M1), use two-tier routing in bulk generation (M2), spot-check after bulk run.

3. **Target length mismatch.** QA answers (~200 tokens) are much shorter than verbatim file content (~500-2000 tokens). The decoder may learn a length bias. Mitigation: the objective weight (18%) is moderate, and the decoder still sees 35% verbatim targets. Monitor average generation length per objective during training.

4. **Question relevance.** Generated questions may be too generic or too specific. Mitigation: enforce category balance during generation, reject questions shorter than 8 tokens, validate diversity in M1 pilot review.

5. **`generate_prompt_variants.py` cannot run inside Claude Code.** The script shells out to `claude -p` which doesn't nest. The implementing agent must instruct the user to run M0 commands in a separate terminal, or use `--dry-run` to preview and then batch-execute.
