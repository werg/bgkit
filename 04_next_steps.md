# BgKIT: Next Steps

First prototyping work beyond environment setup. Three parallel tracks.

---

## Track 1: Data Pipeline (CPU-bound, unblocked)

Build against the existing raw git repo dataset on the dev server.

1. **Repo file extraction** — enumerate and extract file contents + metadata (path, language, size) from bare repos. Output: per-repo file manifests.
2. **Commit extraction + filtering** — parse commit histories, implement filters (exclude merges, trivial formatting, auto-generated, too-large commits). Tag commits requiring cross-file reasoning.
3. **Tokenization + corpus statistics** — run Qwen3 tokenizer over extracted files. Produce per-file token counts, repo-level size distributions, and estimated level 1 sequence lengths. These numbers inform compression ratios, batch sizes, and curriculum design.
4. **Commit reproduction data** — serialize filtered commits as standalone documents (message + paths + diff hunks) for direct level 1 compression training. No repo checkout needed.

## Track 2: ICE Prerequisites (first GPU work)

ICE is a hard dependency for all compression training. Also serves as the first end-to-end validation of the DGX Spark environment.

1. **ICE label generation** — run Qwen3-0.6B in causal mode over a training corpus slice to produce per-token cross-entropy values. Pure inference job, good smoke test.
2. **ICE training** — train the lightweight 1D CNN to regress cross-entropy from token embedding sequences, with uniformity regularizer. Output: frozen ICE model.

## Track 3: Architecture Prototyping

Derisk the novel components before full training.

1. **Auto-reproduction experiment** — retrain BgKIT's (Qwen3-Embedding-0.6B) last transformer block to reproduce per-position input embeddings (all other layers frozen). Optionally test SLERP/linear merges with Qwen3-0.6B. Select the best BgKIT base. Cheap experiment, one block trains.
2. **Drop-flag mechanism prototype** — implement survive/doomed labeling and consolidation on single files with random survivor selection (no ICE dependency). Validate gradient flow and basic mechanics.

---

## Open Questions

- **Dataset scale** — how large is the repo collection? Shapes whether ICE label generation is overnight or multi-day, and whether data processing needs sharding.
- **Pipeline formality** — build as proper `src/bgkit/data/` modules from the start, or quick standalone scripts first?
- **DGX Spark readiness** — timeline determines how much to front-load Track 1 vs. starting Track 2 immediately.
