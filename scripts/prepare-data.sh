#!/usr/bin/env bash
# Usage: scripts/prepare-data.sh [OPTIONS]
#
# Single idempotent data pipeline orchestrator. Runs extraction, conversion,
# and label generation stages in the correct order with parallelism where safe.
#
# Stages (execution order):
#   1  extract-structural      CPU, parallel with 2/3, marker-gated
#   2  process-repos           CPU, parallel with 1/3, marker-gated
#   3  extract-commits         CPU, parallel with 1/2, marker-gated
#   4  generate-descriptions   opt-in (--with-descriptions), after 1, marker-gated
#   5  convert-structural      fast, after 1, always re-runs
#   6  convert-descriptions    opt-in, after 4, always re-runs
#   7  convert-commits         fast, after 3, always re-runs
#   8  ice-labels              GPU/Docker, after 2, marker-gated
#
# Options:
#   --with-descriptions   Include LLM description generation (slow, needs vLLM)
#   --from N              Resume from stage N (clears markers for N onward,
#                         validates that skipped stages' outputs exist)
#   --force               Delete tokenized artifacts + markers, re-run
#                         tokenizer-dependent stages
#   --force-all           Like --force, but also deletes structural/ and
#                         descriptions/ (for when repo collection changed)
#   --dry-run             Print plan without executing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Parse arguments ---
WITH_DESCRIPTIONS=false
FROM_STAGE=0
FORCE=false
FORCE_ALL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-descriptions) WITH_DESCRIPTIONS=true; shift ;;
        --from)
            FROM_STAGE="${2:?--from requires a stage number}"
            if ! [[ "$FROM_STAGE" =~ ^[1-8]$ ]]; then
                echo "ERROR: --from requires a number between 1 and 8, got: $FROM_STAGE"
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

marker_path() { echo "$PIPELINE_DIR/stage${1}.done"; }

# Markers simulated as cleared during dry-run (space-separated stage numbers)
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
        die "Stage $stage was skipped but its output is missing: $path — $hint"
    fi
}

run_stage() {
    local num=$1 name=$2 target=$3
    shift 3

    if (( num < FROM_STAGE )); then
        log "Stage $num ($name): SKIP (--from $FROM_STAGE)"
        return 0
    fi

    if stage_done "$num"; then
        log "Stage $num ($name): SKIP (marker exists)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "Stage $num ($name): WOULD RUN → make $target"
        return 0
    fi

    log "Stage $num ($name): running..."
    local start
    start=$(date +%s)
    make -C "$PROJECT_ROOT" "$target" "$@"
    mark_done "$num"
    log "Stage $num ($name): done ($(elapsed_since "$start"))"
}

# Like run_stage but always re-runs (no marker check/write).
# Used for fast conversion stages that re-tokenize from text.
run_stage_always() {
    local num=$1 name=$2 target=$3
    shift 3

    if (( num < FROM_STAGE )); then
        log "Stage $num ($name): SKIP (--from $FROM_STAGE)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "Stage $num ($name): WOULD RUN → make $target"
        return 0
    fi

    log "Stage $num ($name): running..."
    local start
    start=$(date +%s)
    make -C "$PROJECT_ROOT" "$target" "$@"
    log "Stage $num ($name): done ($(elapsed_since "$start"))"
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
        echo "  Would remove: $PIPELINE_DIR/"
        DRYRUN_CLEARED_MARKERS="1 2 3 4 5 6 7 8"
    else
        rm -rf "$DATA_DIR/processed/tokens/"
        rm -rf "$DATA_DIR/processed/ice_labels/"
        rm -rf "$DATA_DIR/processed/commit_reproduction/"
        rm -rf "$PIPELINE_DIR/"
        mkdir -p "$PIPELINE_DIR"
    fi
fi

if [[ "$FORCE_ALL" == true ]]; then
    log "Force-all mode: also deleting structural/ and descriptions/"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Would remove: $DATA_DIR/structural/"
        echo "  Would remove: $DATA_DIR/descriptions/"
    else
        rm -rf "$DATA_DIR/structural/"
        rm -rf "$DATA_DIR/descriptions/"
    fi
fi

# --- Clear markers for --from N ---
if (( FROM_STAGE > 0 )); then
    log "Clearing markers for stages >= $FROM_STAGE"
    for f in "$PIPELINE_DIR"/stage*.done; do
        [[ -f "$f" ]] || continue
        num="${f##*stage}"; num="${num%.done}"
        if (( num >= FROM_STAGE )); then
            if [[ "$DRY_RUN" == true ]]; then
                echo "  Would clear marker: $f"
                DRYRUN_CLEARED_MARKERS="$DRYRUN_CLEARED_MARKERS $num"
            else
                rm -f "$f"
            fi
        fi
    done
fi

# --- Validate skipped stages' outputs ---
# Only check dependencies for stages that will actually run (num >= FROM_STAGE)
# but whose producer was skipped (producer < FROM_STAGE).
#
# Stage dependency graph:
#   4 (descriptions)        needs stage 1 output (structural/)
#   5 (convert-structural)  needs stage 1 output (structural/)
#   6 (convert-descriptions) needs stage 4 output (descriptions/)
#   7 (convert-commits)     needs stage 3 output (commit_reproduction/)
#   8 (ice-labels)          needs stage 2 output (tokens/)

# Stage 1 output needed by stages 4 and 5
if (( FROM_STAGE > 1 && FROM_STAGE <= 5 )); then
    require_output 1 "$DATA_DIR/structural/" \
        "Run without --from or use --from 1"
fi

# Stage 2 output needed by stage 8
if (( FROM_STAGE > 2 )); then
    require_output 2 "$DATA_DIR/processed/tokens/" \
        "Tokens needed for ICE labels. Run without --from or use --from 2"
fi

# Stage 3 output needed by stage 7
if (( FROM_STAGE > 3 && FROM_STAGE <= 7 )); then
    require_output 3 "$DATA_DIR/processed/commit_reproduction/" \
        "Commit data needed for convert-commits. Run without --from or use --from 3"
fi

# Stage 4 output needed by stage 6
if (( FROM_STAGE > 4 && FROM_STAGE <= 6 )) && [[ "$WITH_DESCRIPTIONS" == true ]]; then
    require_output 4 "$DATA_DIR/descriptions/" \
        "Descriptions needed for convert-descriptions. Run without --from or use --from 4"
fi

TOTAL_START=$(date +%s)

# ============================================================
# Wave 1: Parallel extraction (CPU, host)
#   Stage 1: extract-structural
#   Stage 2: process-repos (tokenize)
#   Stage 3: extract-commits
# ============================================================
log "Wave 1: Parallel extraction"

if [[ "$DRY_RUN" == true ]]; then
    run_stage 1 "extract-structural" extract-structural
    run_stage 2 "process-repos" process-repos
    run_stage 3 "extract-commits" extract-commits
else
    PIDS=()
    NAMES=()

    run_stage 1 "extract-structural" extract-structural &
    PIDS+=($!); NAMES+=("extract-structural")

    run_stage 2 "process-repos" process-repos &
    PIDS+=($!); NAMES+=("process-repos")

    run_stage 3 "extract-commits" extract-commits &
    PIDS+=($!); NAMES+=("extract-commits")

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
# Wave 1b: Descriptions (opt-in, after structural completes)
#   Stage 4: generate-descriptions
# ============================================================
if [[ "$WITH_DESCRIPTIONS" == true ]]; then
    log "Wave 1b: Generate descriptions (needs vLLM)"
    run_stage 4 "generate-descriptions" generate-descriptions
fi

# ============================================================
# Wave 2: Parallel conversion (fast, minutes)
#   Stage 5: convert-structural   (depends on stage 1)
#   Stage 6: convert-descriptions (depends on stage 4, opt-in)
#   Stage 7: convert-commits      (depends on stage 3)
# ============================================================
log "Wave 2: Parallel conversion"

if [[ "$DRY_RUN" == true ]]; then
    run_stage_always 5 "convert-structural" convert-structural
    run_stage_always 7 "convert-commits" convert-commits
    if [[ "$WITH_DESCRIPTIONS" == true ]]; then
        run_stage_always 6 "convert-descriptions" convert-descriptions
    fi
else
    PIDS=()
    NAMES=()

    run_stage_always 5 "convert-structural" convert-structural &
    PIDS+=($!); NAMES+=("convert-structural")

    run_stage_always 7 "convert-commits" convert-commits &
    PIDS+=($!); NAMES+=("convert-commits")

    if [[ "$WITH_DESCRIPTIONS" == true ]]; then
        run_stage_always 6 "convert-descriptions" convert-descriptions &
        PIDS+=($!); NAMES+=("convert-descriptions")
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
#   Stage 8: ice-labels (depends on stage 2)
# ============================================================
log "Wave 3: ICE label generation (GPU)"
run_stage 8 "ice-labels" ice-labels

# ============================================================
# Summary
# ============================================================
log "Pipeline complete ($(elapsed_since "$TOTAL_START") total)"
