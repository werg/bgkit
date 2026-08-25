#!/usr/bin/env bash
# v7-ratio6x post-run: wait for training_complete, then the generative
# per-qtype eval on the FINAL checkpoint and the analysis vs the v6 (67×)
# report — the ratio-vs-quality Pareto read (does verbatim recall appear
# at ~6×?).
#   setsid nohup scripts/post_run_widenet_v7.sh > /home/werg/bgkit-ckpt-fast/post_run_v7.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
C=docker-train-phase2-kb-widenet-v7-ratio6x-1
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for training_complete on $C"
while true; do
  if docker logs "$C" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "training_complete" > /dev/null; then break; fi
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    log "container state=$st without training_complete — crash/stop; re-checking in 10 min"
    sleep 600
    continue
  fi
  sleep 120
done
log "training_complete seen"
while [ "$(pgrep -f '[t]rain.py' | wc -l)" != "0" ]; do sleep 30; done
sleep 30

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v7_ratio6x 2>/dev/null | head -1)
if [ -z "$CK" ]; then log "no v7 checkpoint found in $FAST"; exit 1; fi
NAME=$(basename "$CK")
log "final checkpoint: $NAME"
OUT=/workspace/checkpoints_fast/eval_reports_widenet_v7_ratio6x
log "generative eval (256 samples) -> $OUT"
$COMPOSE run --rm train-phase2-kb-widenet-v7-ratio6x python scripts/eval_phase2_kb.py \
  +experiment=phase2_kb_widenet_v7_ratio6x \
  "+eval.checkpoint=/workspace/checkpoints_fast/$NAME" \
  +eval.per_sample=true +eval.max_samples=256 +eval.max_new_tokens=512 +eval.max_tool_calls=4 \
  "+eval.output_dir=$OUT" \
  > "$FAST/eval_reports_widenet_v7.container.log" 2>&1
log "eval exit=$?"
REPORT="$FAST/eval_reports_widenet_v7_ratio6x/eval_phase2_kb_stage_A.json"
if [ -f "$REPORT" ]; then
  log "per-qtype analysis vs the v6 (67x) report"
  .venv/bin/python scripts/analyze_generative_eval.py "$REPORT" \
    --compare "$FAST/eval_reports_widenet_v6/eval_phase2_kb_stage_A.json" \
    > "$FAST/eval_reports_widenet_v7_ratio6x/analysis_vs_v6.txt" 2>&1
  log "analysis written"
else
  log "eval report missing — skipping analysis"
fi
log "POST-RUN V7 DONE"
