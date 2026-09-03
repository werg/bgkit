# Reset data inventory — 2026-09-03

Prepared by session `fdf5f66b` at bgkit-c7's request, for §6 M0/M1 of
`plans/reset_2026_09_03.md`. Written as a separate file because that plan is
open in another session; inline or link it as you prefer.

## The finding that shapes M0: every pre-tokenized corpus is Qwen-tokenized

The reset moves to **LFM2.5** (`LFM2.5-Encoder-350M`, `LFM2.5-350M-Base` —
both confirmed in `~/.cache/huggingface/hub`). Every prepared corpus on disk
stores `token_ids` and **no text column**, under `Qwen/Qwen3.5-0.8B`:

| corpus | rows | Qwen tokens | size | disk | schema |
|---|---|---|---|---|---|
| `fineweb_edu_v1/tokens` | 4,897,152 | 5.00e9 | 93 G | **NVMe** | `repo_path, file_path, language, token_ids, commit_sha` |
| `processed_v2/tokens` (code) | 1,727,760 | 2.31e9 | 11 G | external HDD | `repo_path, file_path, language, token_ids` |

So they are not directly usable, but they are **recoverable**: decode with the
Qwen tokenizer and re-tokenize with LFM2.5. Verified end to end on real
shards — 794 Qwen tokens decoded to 3,665 characters of clean prose, no
artifacts.

Measured on 200 FineWeb documents:

- LFM2.5 vocab **64,400** vs Qwen **248,044** (3.85x smaller)
- token ratio **0.985** LFM2.5 per Qwen token — near 1:1 despite the smaller
  vocab, so the corpus does not inflate
- **4.64 characters per LFM2.5 token**
- => `fineweb_edu_v1` yields **~4.92e9 LFM2.5 tokens**, code **~2.28e9**

Round-trip cost is CPU-only and parallelizable per shard (984 FineWeb shards,
101 code shards), so it fits the "while the GPU is busy" window. The one thing
to check before committing to it: Qwen decode is not guaranteed byte-exact for
non-UTF8 or unusual whitespace, which matters more for code than prose.

## Raw text, tokenizer-independent (no round-trip needed)

| source | size | form |
|---|---|---|
| `$DATA_DIR/repos` | 349 G | 11,473 **bare git repos**; read via `bgkit.data.repo_files.iter_repo_files` (source-only by default, `exts=DATA_EXTS` for config/data files) |
| `$DATA_DIR/arxiv_v1/raw` | 41 G | raw + `raw_ccdv_flat` |
| `$DATA_DIR/pubmed_v1/raw` | 13 G | raw + `raw_ccdv_flat` |
| `$DATA_DIR/multi_news_v1/raw` | 722 M | raw |

For M0 autoencoding the repos are the cleanest code source: raw bytes at HEAD,
no tokenizer commitment, and the filters that matter (minified, generated,
lock files) already live in `iter_repo_files`.

**Operational note:** FineWeb is on NVMe, but `processed_v2/tokens` and
`repos` are on `/mnt/external` (the HDD). Streaming 349 G of git objects from
that drive is the setup that produced the checkpoint-stall incident; stage what
M0 needs onto NVMe first (250 G free there as of writing).

## M1's passage-QA sets are NOT on disk

`plans/reset_2026_09_03.md` §6 M1 names NQ, TriviaQA, HotpotQA, SQuAD.

- absent: `kilt_nq`, `kilt_hotpotqa`, `kilt_triviaqa`, `squad`
- present but **Qwen-tokenized**: `narrativeqa`, `msmarco_passage`, `newsqa`,
  `searchqa`

All four M1 sets are network-gated downloads (still unchecked boxes in
CLAUDE.md's runbook). This is the long pole for M1, not the trainer.

## BenchPress

Cloned to
`bgkit-data-nvme/capability_packaging/benchmarks/benchpress`
(`lil-lab/benchpress`, depth 1). Paper: *No Mean Feat: Simple, Strong
Baselines for Context Compression*, arXiv 2510.20797 — i.e. the baseline
paper the reset wants to sit beside.

**Format.** A small pure-Python package (`src/benchpress/`: `datasets.py`,
`metrics.py`, `templates.py`, `templates/`). Dependencies are only
`datasets>=3.0.0`, `transformers>=4.40.0`, `regex` — no torch, so it installs
into the host venv and needs no container.

**The data is NOT on disk.** It lives on the Hub at `yairfeldman/benchpress`
and is not in the HF cache; it needs one download. `scripts/prepare_dataset.py`
can rebuild it from sources instead.

**Protocol.**

```python
ds     = benchpress.load(path, subset="short"|"mid_range", split_type="in_domain"|"out_of_domain",
                         datasets=[...], max_context_tokens=N, tokenizer="<hf id>")
prompt = benchpress.prepare_prompt(sample["dataset_name"], sample["context"], sample["question"])
scores = benchpress.evaluate(predictions, references)   # references are LISTS of accepted answers
benchpress.aggregate(scores)  # -> {"M", "EM", "F1", "Precision", "Recall"}
```

10 datasets in two subsets. Short: `squad`, `narrativeqa`, `hotpotqa`,
`adversarial_qa`, `triviaqa_verified`, `paraphrase_rc` (first three in-domain).
Mid-range (LongBench): `longbench_qasper_e`, `longbench_multifieldqa_en_e`,
`longbench_hotpotqa_e`, `longbench_2wikimqa_e`.

Templates are a real part of the protocol, not decoration: extractive-QA sets
carry 101 templates each, `qa` 96, LongBench 1, and `sample_template(dataset,
context, question)` picks one **deterministically** from the content — so a
prompt is reproducible without storing it, and template variance is part of
the measured quantity. Use `prepare_prompt`/`sample_template` rather than
writing our own prompt string, or the numbers stop being comparable.

Two things that map straight onto the plan:

- `max_context_tokens` + `tokenizer` gives the matched-budget filtering the
  ratio sweep needs, counted with **our** tokenizer.
- `metrics.teacher_normalized_score` already implements "fraction of
  full-context performance retained", which is exactly the M1 gate ("retain
  >= 90% of full-context EM"). Use theirs rather than defining our own.

## Suggested order of CPU work

1. Download `yairfeldman/benchpress` and pin the revision (one download,
   unblocks writing the M1 eval script before any GPU is free).
2. Stage a FineWeb subset to NVMe as **text**: decode->re-tokenize a few
   hundred shards, enough for M0 rather than all 4.92e9 tokens.
3. Code text for the M0 mix straight from `repos` via `iter_repo_files`
   (skip `processed_v2` entirely — it is Qwen-tokenized AND on the HDD).
4. Only then the M1 passage-QA downloads.
