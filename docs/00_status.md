# Current Project Status

Last reconciled with the implementation: 2026-08-02.

## Implemented

- Packed independent L0/L1 compression with query prompts, bridges, discrete
  survivorship, exact-ratio selection, and decoder-family projection.
- Phase-1 training families and Qwen/Falcon summarization handoffs.
- Phase-2 browse-tree datasets, cached/live L0 paths, live query-conditioned L1,
  interleaved dense decoder splices, recursive git-history experiments, and
  complete Stage-A/prompt-fit → Stage-B model handoff.
- Git-history schema-v2 extraction with real commit/parent SHAs, rename-aware
  file lineages, exact hunk replay, reconstruction anchors, repository-level
  held-out splits, opaque model-facing IDs, and fingerprinted tree/mmap/
  trajectory generations.
- Free-running Phase-2 evaluation that executes generated `bgkit` calls,
  rejects unsurfaced IDs, and reports route completion, evidence recall, and
  literal full-state exact match plus answer F1. Teacher-forced metrics remain
  separate diagnostics.
- Content-fingerprinted L0 cache manifests for checkpoint files or directories.
- Run-scoped checkpoint save names, resume, registry selection, pruning, NVMe
  recovery, and HDD retention.
- Phase-3 external-trajectory CE prototype with persistent packed filesystem
  encoding, bounded survivor-array caching, and paired likelihood ablation.

## Experimental or incomplete

- Hard selection masks still need explicit survivorship auxiliary objectives;
  the git-reproduction production preset enables them, while other presets may
  intentionally retain a frozen inherited selector.
- Phase-3 git-history and prior-session consumers have leakage-safe contracts,
  but their cache producers are not shipped and the sources are disabled.
- Phase-3 trains on external trajectory tokens; frozen-teacher KL/hint designs
  in older documents are unimplemented proposals.
- Recursive full-backprop repo training is specialized, expensive, and not a
  general large-tree solution.

## Not implemented

- Production encoder serving/export.
- Standard llama.cpp or vLLM support for arbitrary BgKIT vector injection.
- A complete free-running quality gate against matched BM25, dense retrieval,
  and reranker baselines.
- SWE-bench patch generation/execution integrated into the Phase-3 trainer.

## Recommended product direction

**Decision 2026-08-20:** the navigation-based knowledge-retrieval formulation
is judged a dead end (reconstruction trained well; navigation ID-accuracy
never left zero across two ID redesigns). Active direction is now
`plans/capability_packaging_2026_08_20.md`: package soft-prompt injection +
memory — trajectory compaction, targeted compression of wide tool results,
compressed cross-session memory, and staged whole-repo ambient background.
The paragraphs below predate this decision.

The strongest near-term use case is repository state and change-impact
reasoning: reconstructing file state from histories, API migrations,
dependency changes, and cross-file invariants. It naturally exercises the
hierarchy and has clear unseen-repository splits.

Other promising targets are multi-document synthesis with weak distributed
evidence, chronology-safe long-term technical memory, and a hybrid system with
cached ambient repository context plus query-conditioned drill-down. Simple
single-passage QA is a poor primary target because conventional retrieval is
cheaper and exact.

## Immediate engineering priorities

1. Add a golden end-to-end handoff/cache test using small real model fixtures.
2. Add matched-decoder BM25/dense/reranker baselines and acceptance thresholds
   around the free-running Phase-2 evaluator.
3. Measure the context contribution on unseen repositories before expanding
   Phase 3.
4. Split the Phase-2 trainer monolith along cache, tree encode, decoder assembly,
   evaluation, and optimizer-cadence boundaries.
5. Add CI gates for CPU tests and source lint, then reduce the existing typing
   backlog incrementally.
