# Documentation Map

Documentation is separated by authority so an experiment journal cannot be
mistaken for a runtime contract.

## Current and normative

- `00_status.md` — implemented, experimental, blocked, and deferred features.
- `01_overview.md` — model and checkpoint contracts reflected by current code.
- `02_training_plan.md` — runnable phase workflows and known evaluation limits.
- `runbook.md` — setup, validation, cache, checkpoint, and launch operations.
- `survivorship_design.md` — selector mechanics and loss menu; Phase-2 config
  switches still determine which losses are active.
- `taxonomies.md` — browse-tree/taxonomy data formats.

When these documents disagree with a Hydra config about a numeric setting, the
config wins. When they disagree with a tested implementation contract, the code
and test win and the document should be corrected.

## Proposals

- `03_ideas_and_risks.md`
- `04_nvfp4_spark_training_plan.md`
- files under `plans/`

These describe candidate work and are not evidence of implementation.

## Historical investigations

`04_next_steps.md`, dated performance reviews, FA/FLA fork reviews, and theta or
wall-clock investigations record decisions and measurements at a point in time.
They may refer to removed configs, earlier kernels, or superseded architecture.
Their dates and experiment context must accompany any reused conclusion.
