#!/usr/bin/env bash
# Usage: scripts/prepare-data.sh [OPTIONS]
#
# Single idempotent data pipeline orchestrator. Runs extraction, conversion,
# and label generation stages in the correct order with parallelism where safe.
#
# Stages (execution order by wave):
#   Wave 1:
#     extract-structural          CPU, parallel, marker-gated
#     process-repos               CPU, parallel, marker-gated
#     extract-commits             CPU, parallel, marker-gated
#     extract-commit-encoding     CPU, parallel, marker-gated, --with-commit-encoding
#   Wave 1b:
#     generate-descriptions       opt-in (--with-descriptions), marker-gated
#     generate-qa-pairs           opt-in (--with-qa), marker-gated
#   Wave 2:
#     convert-structural          fast, always re-runs
#     convert-descriptions        opt-in (--with-descriptions), always re-runs
#     convert-commits             fast, always re-runs
#     convert-commit-encoding     opt-in (--with-commit-encoding), always re-runs
#     convert-qa-pairs            opt-in (--with-qa), always re-runs
#     convert-tokens              fast, always re-runs
#   Wave 3:
#     ice-labels                  GPU/Docker, marker-gated
#
# Options:
#   --with-descriptions       Include LLM description generation (slow, needs vLLM)
#   --with-qa                 Include QA pair generation (slow, needs vLLM)
#   --with-commit-encoding    Include commit encoding extraction and conversion
#   --from <stage-name>       Resume from named stage (clears markers for that
#                             stage's wave and all later waves)
#   --force                   Delete tokenized artifacts + markers, re-run
#                             tokenizer-dependent stages
#   --force-all               Like --force, but also deletes structural/ and
#                             descriptions/ (for when repo collection changed)
#   --dry-run                 Print plan without executing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Wave definitions ---
# Wave ordering determines --from semantics: clearing a wave clears all later waves.
WAVE_ORDER=("1" "1b" "2" "3")

# Stage-to-wave mapping
declare -A STAGE_WAVE=(
    [extract-structural]="1"
    [process-repos]="1"
    [extract-commits]="1"
    [extract-commit-encoding]="1"
    [generate-descriptions]="1b"
    [generate-qa-pairs]="1b"
    [convert-structural]="2"
    [convert-descriptions]="2"
    [convert-commits]="2"
    [convert-commit-encoding]="2"
    [convert-qa-pairs]="2"
    [convert-tokens]="2"
    [ice-labels]="3"
)

ALL_STAGES=(
    extract-structural process-repos extract-commits extract-commit-encoding
    generate-descriptions generate-qa-pairs
    convert-structural convert-descriptions convert-commits
    convert-commit-encoding convert-qa-pairs convert-tokens
    ice-labels
)

# --- Parse arguments ---
WITH_DESCRIPTIONS=false
WITH_QA=false
WITH_COMMIT_ENCODING=false
FROM_STAGE=""
FORCE=false
FORCE_ALL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-descriptions) WITH_DESCRIPTIONS=true; shift ;;
        --with-qa) WITH_QA=true; shift ;;
        --with-commit-encoding) WITH_COMMIT_ENCODING=true; shift ;;
        --from)
            FROM_STAGE="${2:?--from requires a stage name}"
            if [[ -z "${STAGE_WAVE[$FROM_STAGE]+x}" ]]; then
                echo "ERROR: --from requires a valid stage name, got: $FROM_STAGE"
                echo "Valid stages: ${ALL_STAGES[*]}"
                exit 1
            fi
            shift 2
            ;;
        --force) FORCE=true; shift ;;
        --force-all) FORCE_ALL=true; FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '2,/^[^#]/{ s/^# \?//p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Load .env ---
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "ERROR: .env not found. Copy .env.example to .env and set DATA_DIR."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
set +a

if [[ -z "${DATA_DIR:-}" ]]; then
    echo "ERROR: DATA_DIR not set in .env"
    exit 1
fi

# Resolve relative DATA_DIR against project root
if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$PROJECT_ROOT/$DATA_DIR"
fi

PIPELINE_DIR="$DATA_DIR/.pipeline"
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$PIPELINE_DIR"
fi

# --- Helpers ---
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

log() { echo "==> $*"; }

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    local secs=$(( now - start ))
    printf '%dm%02ds' $(( secs / 60 )) $(( secs % 60 ))
}

marker_path() { echo "$PIPELINE_DIR/${1}.done"; }

# Markers simulated as cleared during dry-run (space-separated stage names)
DRYRUN_CLEARED_MARKERS=""

stage_done() {
    # In dry-run, respect simulated marker clears
    if [[ "$DRY_RUN" == true && " $DRYRUN_CLEARED_MARKERS " == *" $1 "* ]]; then
        return 1
    fi
    [[ -f "$(marker_path "$1")" ]];
}

mark_done() { date -Iseconds > "$(marker_path "$1")"; }

require_output() {
    local stage=$1 path=$2 hint=$3
    if [[ ! -d "$path" ]] || [[ -z "$(ls -A "$path" 2>/dev/null)" ]]; then
        die "Stage '$stage' was skipped but its output is missing: $path — $hint"
    fi
}

# wave_index returns the numeric index of a wave in WAVE_ORDER (0-based)
wave_index() {
    local target=$1
    for i in "${!WAVE_ORDER[@]}"; do
        if [[ "${WAVE_ORDER[$i]}" == "$target" ]]; then
            echo "$i"
            return
        fi
    done
    die "Unknown wave: $target"
}

# wave_ge returns 0 (true) if wave $1 >= wave $2 in execution order
wave_ge() {
    local a b
    a=$(wave_index "$1")
    b=$(wave_index "$2")
    (( a >= b ))
}

# Determine if a stage should be skipped due to --from
# Returns 0 (true) if the stage should be skipped
stage_skipped_by_from() {
    local stage=$1
    [[ -z "$FROM_STAGE" ]] && return 1  # no --from, don't skip

    local from_wave="${STAGE_WAVE[$FROM_STAGE]}"
    local stage_wave="${STAGE_WAVE[$stage]}"
    local from_idx stage_idx
    from_idx=$(wave_index "$from_wave")
    stage_idx=$(wave_index "$stage_wave")

    # Skip stages in earlier waves
    if (( stage_idx < from_idx )); then
        return 0
    fi

    # Within the same wave, skip stages that come before --from in alphabetical
    # order? No — within the same wave, all stages run (markers were cleared).
    return 1
}

run_stage() {
    local name=$1 target=$2
    shift 2

    if stage_skipped_by_from "$name"; then
        log "Stage $name: SKIP (--from $FROM_STAGE)"
        return 0
    fi

    if stage_done "$name"; then
        log "Stage $name: SKIP (marker exists)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "Stage $name: WOULD RUN -> make $target"
        return 0
    fi

    log "Stage $name: running..."
    local start
    start=$(date +%s)
    make -C "$PROJECT_ROOT" "$target" "$@"
    mark_done "$name"
    log "Stage $name: done ($(elapsed_since "$start"))"
}

# Like run_stage but always re-runs (no marker check/write).
# Used for fast conversion stages that re-tokenize from text.
run_stage_always() {
    local name=$1 target=$2
    shift 2

    if stage_skipped_by_from "$name"; then
        log "Stage $name: SKIP (--from $FROM_STAGE)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "Stage $name: WOULD RUN -> make $target"
        return 0
    fi

    log "Stage $name: running..."
    local start
    start=$(date +%s)
    make -C "$PROJECT_ROOT" "$target" "$@"
    log "Stage $name: done ($(elapsed_since "$start"))"
}

# --- Pre-flight checks ---
if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    die ".venv not found — run 'make install' first"
fi

if [[ ! -d "$DATA_DIR/repos" ]]; then
    die "No repos directory at $DATA_DIR/repos — download repos first"
fi

# --- Force cleanup ---
if [[ "$FORCE" == true ]]; then
    log "Force mode: deleting tokenized artifacts and pipeline markers"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would remove: $DATA_DIR/processed/tokens/"
        echo "  Would remove: $DATA_DIR/processed/ice_labels/"
        echo "  Would remove: $DATA_DIR/processed/commit_reproduction/"
        echo "  Would remove: $DATA_DIR/processed/commit_encoding/"
        echo "  Would remove: $PIPELINE_DIR/"
        DRYRUN_CLEARED_MARKERS="${ALL_STAGES[*]}"
    else
        rm -rf "$DATA_DIR/processed/tokens/"
        rm -rf "$DATA_DIR/processed/ice_labels/"
        rm -rf "$DATA_DIR/processed/commit_reproduction/"
        rm -rf "$DATA_DIR/processed/commit_encoding/"
        rm -rf "$PIPELINE_DIR/"
        mkdir -p "$PIPELINE_DIR"
    fi
fi

if [[ "$FORCE_ALL" == true ]]; then
    log "Force-all mode: also deleting structural/, descriptions/, qa_pairs/"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would remove: $DATA_DIR/structural/"
        echo "  Would remove: $DATA_DIR/descriptions/"
        echo "  Would remove: $DATA_DIR/qa_pairs/"
    else
        rm -rf "$DATA_DIR/structural/"
        rm -rf "$DATA_DIR/descriptions/"
        rm -rf "$DATA_DIR/qa_pairs/"
    fi
fi

# --- Clear markers for --from <stage-name> ---
if [[ -n "$FROM_STAGE" ]]; then
    FROM_WAVE="${STAGE_WAVE[$FROM_STAGE]}"
    FROM_WAVE_IDX=$(wave_index "$FROM_WAVE")
    log "Clearing markers for wave $FROM_WAVE and later (--from $FROM_STAGE)"

    for stage in "${ALL_STAGES[@]}"; do
        stage_wave="${STAGE_WAVE[$stage]}"
        stage_wave_idx=$(wave_index "$stage_wave")
        if (( stage_wave_idx >= FROM_WAVE_IDX )); then
            marker="$(marker_path "$stage")"
            if [[ "$DRY_RUN" == true ]]; then
                if [[ -f "$marker" ]] || [[ " $DRYRUN_CLEARED_MARKERS " != *" $stage "* ]]; then
                    echo "  Would clear marker: $stage"
                    DRYRUN_CLEARED_MARKERS="$DRYRUN_CLEARED_MARKERS $stage"
                fi
            else
                rm -f "$marker"
            fi
        fi
    done
fi

# --- Validate skipped stages' outputs ---
# Only check dependencies for stages that will actually run
# but whose producer was skipped by --from.
#
# Dependency graph:
#   generate-descriptions    needs extract-structural output (structural/)
#   convert-structural       needs extract-structural output (structural/)
#   convert-descriptions     needs generate-descriptions output (descriptions/)
#   convert-commits          needs extract-commits output (commit_reproduction/)
#   convert-commit-encoding  needs extract-commit-encoding output (commit_encoding/)
#   ice-labels               needs process-repos output (tokens/)
#   generate-qa-pairs        needs process-repos output (tokens/)

if [[ -n "$FROM_STAGE" ]]; then
    FROM_WAVE_IDX=$(wave_index "${STAGE_WAVE[$FROM_STAGE]}")

    # extract-structural output needed by generate-descriptions (1b) and convert-structural (2)
    if stage_skipped_by_from "extract-structural" && ! stage_skipped_by_from "convert-structural"; then
        require_output "extract-structural" "$DATA_DIR/structural/" \
            "Run without --from or use --from extract-structural"
    fi

    # process-repos output needed by ice-labels (3) and generate-qa-pairs (1b)
    if stage_skipped_by_from "process-repos"; then
        if ! stage_skipped_by_from "ice-labels" || ! stage_skipped_by_from "generate-qa-pairs"; then
            require_output "process-repos" "$DATA_DIR/processed/tokens/" \
                "Tokens needed for ICE labels / QA. Run without --from or use --from process-repos"
        fi
    fi

    # extract-commits output needed by convert-commits (2)
    if stage_skipped_by_from "extract-commits" && ! stage_skipped_by_from "convert-commits"; then
        require_output "extract-commits" "$DATA_DIR/processed/commit_reproduction/" \
            "Commit data needed for convert-commits. Run without --from or use --from extract-commits"
    fi

    # extract-commit-encoding output needed by convert-commit-encoding (2)
    if stage_skipped_by_from "extract-commit-encoding" && ! stage_skipped_by_from "convert-commit-encoding"; then
        if [[ "$WITH_COMMIT_ENCODING" == true ]]; then
            require_output "extract-commit-encoding" "$DATA_DIR/processed/commit_encoding/" \
                "Commit encoding data needed. Run without --from or use --from extract-commit-encoding"
        fi
    fi

    # generate-descriptions output needed by convert-descriptions (2)
    if stage_skipped_by_from "generate-descriptions" && ! stage_skipped_by_from "convert-descriptions"; then
        if [[ "$WITH_DESCRIPTIONS" == true ]]; then
            require_output "generate-descriptions" "$DATA_DIR/descriptions/" \
                "Descriptions needed for convert-descriptions. Run without --from or use --from generate-descriptions"
        fi
    fi
fi

TOTAL_START=$(date +%s)

# ============================================================
# Wave 1: Parallel extraction (CPU, host)
#   extract-structural
#   process-repos
#   extract-commits
#   extract-commit-encoding (opt-in)
# ============================================================
log "Wave 1: Parallel extraction"

if [[ "$DRY_RUN" == true ]]; then
    run_stage "extract-structural" extract-structural
    run_stage "process-repos" process-repos
    run_stage "extract-commits" extract-commits
    if [[ "$WITH_COMMIT_ENCODING" == true ]]; then
        run_stage "extract-commit-encoding" prepare-commit-encoding
    fi
else
    PIDS=()
    NAMES=()

    run_stage "extract-structural" extract-structural &
    PIDS+=($!); NAMES+=("extract-structural")

    run_stage "process-repos" process-repos &
    PIDS+=($!); NAMES+=("process-repos")

    run_stage "extract-commits" extract-commits &
    PIDS+=($!); NAMES+=("extract-commits")

    if [[ "$WITH_COMMIT_ENCODING" == true ]]; then
        run_stage "extract-commit-encoding" prepare-commit-encoding &
        PIDS+=($!); NAMES+=("extract-commit-encoding")
    fi

    FAIL=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "FAILED: ${NAMES[$i]}"
            FAIL=1
        fi
    done
    (( FAIL == 0 )) || die "Wave 1 failed — see errors above"
fi

# ============================================================
# Wave 1b: LLM generation (opt-in, needs vLLM)
#   generate-descriptions (after extract-structural)
#   generate-qa-pairs (after process-repos)
# ============================================================
if [[ "$WITH_DESCRIPTIONS" == true ]] || [[ "$WITH_QA" == true ]]; then
    log "Wave 1b: LLM generation (needs vLLM)"

    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$WITH_DESCRIPTIONS" == true ]]; then
            run_stage "generate-descriptions" generate-descriptions
        fi
        if [[ "$WITH_QA" == true ]]; then
            run_stage "generate-qa-pairs" generate-qa-pairs
        fi
    else
        PIDS=(); NAMES=()

        if [[ "$WITH_DESCRIPTIONS" == true ]]; then
            run_stage "generate-descriptions" generate-descriptions &
            PIDS+=($!); NAMES+=("generate-descriptions")
        fi
        if [[ "$WITH_QA" == true ]]; then
            run_stage "generate-qa-pairs" generate-qa-pairs &
            PIDS+=($!); NAMES+=("generate-qa-pairs")
        fi

        FAIL=0
        for i in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$i]}"; then
                echo "FAILED: ${NAMES[$i]}"
                FAIL=1
            fi
        done
        (( FAIL == 0 )) || die "Wave 1b (LLM generation) failed"
    fi
fi

# ============================================================
# Wave 2: Parallel conversion (fast, minutes)
#   convert-structural       (depends on extract-structural)
#   convert-descriptions     (depends on generate-descriptions, opt-in)
#   convert-commits          (depends on extract-commits)
#   convert-commit-encoding  (depends on extract-commit-encoding, opt-in)
#   convert-qa-pairs         (depends on generate-qa-pairs, opt-in)
#   convert-tokens           (depends on process-repos)
# ============================================================
log "Wave 2: Parallel conversion"

if [[ "$DRY_RUN" == true ]]; then
    run_stage_always "convert-structural" convert-structural
    run_stage_always "convert-commits" convert-commits
    run_stage_always "convert-tokens" convert-tokens
    if [[ "$WITH_DESCRIPTIONS" == true ]]; then
        run_stage_always "convert-descriptions" convert-descriptions
    fi
    if [[ "$WITH_COMMIT_ENCODING" == true ]]; then
        run_stage_always "convert-commit-encoding" convert-commit-encoding
    fi
    if [[ "$WITH_QA" == true ]]; then
        run_stage_always "convert-qa-pairs" convert-qa-pairs
    fi
else
    PIDS=()
    NAMES=()

    run_stage_always "convert-structural" convert-structural &
    PIDS+=($!); NAMES+=("convert-structural")

    run_stage_always "convert-commits" convert-commits &
    PIDS+=($!); NAMES+=("convert-commits")

    run_stage_always "convert-tokens" convert-tokens &
    PIDS+=($!); NAMES+=("convert-tokens")

    if [[ "$WITH_DESCRIPTIONS" == true ]]; then
        run_stage_always "convert-descriptions" convert-descriptions &
        PIDS+=($!); NAMES+=("convert-descriptions")
    fi

    if [[ "$WITH_COMMIT_ENCODING" == true ]]; then
        run_stage_always "convert-commit-encoding" convert-commit-encoding &
        PIDS+=($!); NAMES+=("convert-commit-encoding")
    fi

    if [[ "$WITH_QA" == true ]]; then
        run_stage_always "convert-qa-pairs" convert-qa-pairs &
        PIDS+=($!); NAMES+=("convert-qa-pairs")
    fi

    FAIL=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "FAILED: ${NAMES[$i]}"
            FAIL=1
        fi
    done
    (( FAIL == 0 )) || die "Wave 2 failed — see errors above"
fi

# ============================================================
# Wave 3: ICE labels (GPU, Docker)
#   ice-labels (depends on process-repos)
# ============================================================
log "Wave 3: ICE label generation (GPU)"
run_stage "ice-labels" ice-labels

# ============================================================
# Summary
# ============================================================
log "Pipeline complete ($(elapsed_since "$TOTAL_START") total)"
