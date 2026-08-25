# Architecture and Runtime Contracts

This document describes the code that exists now. Proposed serving systems and
experiments belong in `03_ideas_and_risks.md` or `plans/`.

## Data flow

```text
document tokens
  → independent L0 compressor
  → gathered L0 survivors
  → L0 auto-reproduction bridge
  → live query-conditioned L1 compressor
  → gathered L1 survivors
  → decoder-family projection block
  → dense splice inside a causal decoder sequence
```

The packed representation uses flat tensors plus `cu_seqlens`; samples are not
padded at the model boundary. Prompt packing is performed with on-device index
maps, not per-sample CPU slicing.

## Encoder levels

`BgKITEncoder` constructs L0 first and deep-copies its initialized backbone to
construct L1. These are independent modules after construction—there is no
runtime weight sharing. L0 has a learned auto-reproduction head because its
survivors must be mapped into L1 input space. Recursive tree experiments also
use a separate L1→L1 bridge.

Both levels contain a survivorship head at an intermediate backbone boundary.
The chosen positions receive a learned survive embedding before the remainder
of the backbone. After the final normalization, selected positions are gathered.
The projection block sees those gathered survivors; it does not attend to
discarded positions.

L0 supports prompts. L1 also supports query prompts in the current
implementation, despite older source comments that described L1 as prompt-free.

## Selection

The threshold controller is a monotone curve `theta(retention)`, represented by
anchors in linear, log, or logit ratio space. It is not one global scalar shared
across every requested retention. Threshold mode uses organic selections plus
optional pinned positions and an optional per-sample floor. Exact-ratio mode
uses segment-local top-k.

Segmented top-k is implemented as one stable global sort over an exact int64
composite key `(segment, ordered-float-bits)`. The floor trigger metric is the
fraction of non-empty eligible samples on which an enabled floor actually
activated; it is zero when the floor is disabled.

The selection mask is discrete and detached from the main task loss. Therefore
the head learns only when survivorship auxiliary losses are enabled. The
git-reproduction production preset enables relevance and utility supervision;
presets that disable those losses intentionally freeze the inherited policy and
must be treated as ablations, not evidence that the selector is task-optimal.

Pinned ID rows are forced protocol carriers, not controllable content choices.
Exact-ratio quotas and selector auxiliary ratios therefore exclude them from
the eligible population.

## Phase-2 handoff contract

Phase 2 has two checkpoint identities:

1. `phase1_checkpoint` constructs the Phase-1 base and initial decoders.
2. `stage_a_checkpoint` overlays the complete Phase-2 state after optional task
   prompts and LoRA/direct-training module topology exists.

The overlay includes L0, L0→L1 bridge, L1, L1→L1 bridge, threshold buffers,
projection blocks, task prompts, and every active decoder family. Stage B refuses
to start from a Phase-1 checkpoint alone. A legacy config may use a Phase-2
checkpoint as `phase1_checkpoint`, but new configs should keep the lineage
explicit.

Cached L0 survivors must be built from the same two checkpoint sources. Each
dataset manifest contains content fingerprints for checkpoint files or complete
checkpoint directories, adapter shape, and retention. If Stage B has a Phase-2
handoff, a missing manifest is fatal rather than a warning.

## Decoder injection

Phase 2 frames dense vectors between tokenized `bgkit` tool-response boundaries.
The vectors are directly spliced into the decoder embedding sequence; they are
not serialized as text. Phase 3 uses the same splice primitive. Its baseline
preservation arm omits both the vectors and the synthetic BgKIT wrapper.

No production inference adapter is shipped. Statements that standard LLaVA
paths in llama.cpp or vLLM can consume BgKIT vectors without integration work
are proposals, not current capabilities.

## Checkpoint identity

Checkpoint metadata and registry entries both store `phase` and `run_name`.
Auto-resume, manager pruning, fast-directory recovery, NVMe pruning, and HDD
archive pruning use both fields. Legacy checkpoints lacking `run_name` are not
adopted by a named run automatically.
