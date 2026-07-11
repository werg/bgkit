# Current Project Status

Last reconciled with the implementation: 2026-07-11.

## Implemented

- Packed independent L0/L1 compression with query prompts, bridges, discrete
  survivorship, exact-ratio selection, and decoder-family projection.
- Phase-1 training families and Qwen/Falcon summarization handoffs.
- Phase-2 browse-tree datasets, cached/live L0 paths, live query-conditioned L1,
  interleaved dense decoder splices, recursive git-history experiments, and
  complete Stage-A/prompt-fit → Stage-B model handoff.
- Content-fingerprinted L0 cache manifests for checkpoint files or directories.
- Run-scoped checkpoint save names, resume, registry selection, pruning, NVMe
  recovery, and HDD retention.
- Phase-3 external-trajectory CE prototype with persistent packed filesystem
  encoding, bounded survivor-array caching, and paired likelihood ablation.

## Experimental or incomplete

- Phase-2 survivorship heads are frozen in current presets because auxiliary
  losses are disabled; task loss cannot directly optimize the hard mask.
- Phase-2 evaluation is teacher-forced, not an autonomous tool loop.
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
2. Build a free-running Phase-2 tool executor and paired matched-decoder RAG
   baseline.
3. Measure the context contribution on unseen repositories before expanding
   Phase 3.
4. Split the Phase-2 trainer monolith along cache, tree encode, decoder assembly,
   evaluation, and optimizer-cadence boundaries.
5. Add CI gates for CPU tests and source lint, then reduce the existing typing
   backlog incrementally.
