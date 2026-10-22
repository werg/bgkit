"""Phase 2 knowledge-retrieval training.

Phase 2 is unified around a single trainer (:class:`KRKBTrainer`) that
handles every dataset via the trajectory framework. The legacy parallel
KRTrainer (single-doc / flat retrieval) has been removed; flat datasets
are now expressed as zero-browse-turn trajectories that issue one
``bgkit`` call directly on the gold article. Hierarchical datasets
(KILT, PubMedQA via MeSH, NarrativeQA per book) emit
``browse → bgkit → answer`` trajectories.

Stages:
- **Stage A** — live-L0 bootstrap on a small mixed corpus (one epoch).
  Trains both L0 LoRA and L1 LoRA.
- **Stage B** — full-scale training with cached L0 (built by
  ``scripts/precompute_l0_subset.py`` using the Stage A LoRA). L0 LoRA
  frozen; L1 LoRA + decoder train.

The decoder is Qwen3.5-0.8B throughout. There is no larger "target LLM"
swap.
"""
