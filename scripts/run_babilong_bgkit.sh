#!/usr/bin/env bash
# BABILong bgkit arm: run the compressed-context evaluation at both retention
# points and score with BABILong's own metric (2026-08-25).
#
# The two stock arms are already measured (scripts/baseline_babilong.py ->
# $FAST/baselines/babilong_qwen35_0p8b{,_ctx4k}); this is the third arm.
# Reference to beat, truncate-4k @100 samples:
#   qa1  0.41 @16k  0.35 @32k     qa2  0.34 @16k  0.25 @32k
# (full-context stock, 32k decoder positions: qa1 0.48, qa2 0.15)
#
# Refuses to run alongside a GPU training container — one GPU job at a time.
#   setsid nohup scripts/run_babilong_bgkit.sh <ckpt_67x> <ckpt_6x> \
#     > /home/werg/bgkit-ckpt-fast/babilong_bgkit.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
FAST=/home/werg/bgkit-ckpt-fast
NVME=/home/werg/bgkit-data-nvme/capability_packaging
BL=$NVME/benchmarks/babilong
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
OUT=$FAST/babilong_bgkit
# Same directory as the container sees it: $FAST is bind-mounted there, and
# hydra writes the report from INSIDE the container (a host path under
# /home/werg is not writable there — PermissionError, 2026-08-25).
COUT=/workspace/checkpoints_fast/babilong_bgkit
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

CKPT_67X=${1:-}
CKPT_6X=${2:-}
[ -n "$CKPT_67X" ] || { log "usage: $0 <ckpt_for_67x_arm> [ckpt_for_6x_arm]"; exit 1; }

if [ -n "$(docker ps -q --filter name=phase2-kb)" ] || [ -n "$(docker ps -q --filter name=blob-sft)" ]; then
  log "a GPU training container is running — refusing to start a second GPU job"; exit 1
fi
for ds in babilong_qa1_16k babilong_qa1_32k babilong_qa2_16k babilong_qa2_32k; do
  [ -f "$DATA_DIR/browse_trees/$ds.parquet" ] || {
    log "missing browse tree for $ds — run build_babilong_phase2.py + build_browse_tree.py"; exit 1; }
done
mkdir -p "$OUT"

# Preflight: real routing on the new datasets before spending eval time on
# them. The 2026-08-22 audit found three silent routing defects that every
# metric-level check missed — a zero-rep splice looks like a normal run.
log "preflight: diag_flat_splice_counts on babilong_bgkit_67x"
$COMPOSE run --rm --name babilong-diag train-phase2-kb-widenet-v7-ratio6x \
  python scripts/diag_flat_splice_counts.py +experiment=babilong_bgkit_67x \
    +diag.n_samples=8 2>&1 | tee "$OUT/preflight_splice.txt" | tail -20
if ! grep -q "DIAG DONE" "$OUT/preflight_splice.txt"; then
  log "preflight did not complete — aborting"; exit 1
fi
if grep -qE "reps=0 " "$OUT/preflight_splice.txt"; then
  log "preflight found a ZERO-REP splice — aborting (decoder would see no reps)"; exit 1
fi

run_arm() {  # $1 = experiment, $2 = checkpoint, $3 = tag
  local exp=$1 ckpt=$2 tag=$3
  mkdir -p "$OUT/$tag"
  log "arm $tag: eval_phase2_kb ($exp) on $ckpt"
  $COMPOSE run --rm --name babilong-$tag train-phase2-kb-widenet-v7-ratio6x \
    python scripts/eval_phase2_kb.py "+experiment=$exp" \
      "+eval.checkpoint=$ckpt" +eval.per_sample=true +eval.max_samples=400 \
      +eval.free_running=true +eval.max_tool_calls=2 +eval.max_new_tokens=128 \
      +eval.force_first_call=true \
      "+eval.output_dir=$COUT/$tag" 2>&1 | tee "$OUT/$tag/arm.log" | tail -40
  local report
  report=$(ls -t "$OUT/$tag"/eval_phase2_kb_stage_*.json 2>/dev/null | head -1)
  if [ -z "$report" ]; then log "arm $tag: NO REPORT WRITTEN"; return 1; fi
  log "arm $tag: scoring $report"
  .venv/bin/python scripts/score_babilong_bgkit.py --report "$report" \
    --repo-dir "$BL/babilong-repo" --traj-dir "$DATA_DIR/trajectories" \
    --out-json "$OUT/$tag/babilong_scores.json"
}

run_arm babilong_bgkit_67x "$CKPT_67X" 67x
if [ -n "$CKPT_6X" ]; then run_arm babilong_bgkit_6x "$CKPT_6X" 6x; fi
log "DONE -> $OUT"
