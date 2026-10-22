#!/usr/bin/env bash
# v6 post-run: wait for training_complete, then run the (now valid) generative
# eval on the FINAL checkpoint — free-running tool call + answer, per-sample
# rows — and the per-qtype analysis.
#   setsid nohup scripts/post_run_widenet_v6.sh > /home/werg/bgkit-ckpt-fast/post_run_v6.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
C=docker-train-phase2-kb-widenet-v6-1
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for training_complete on $C"
while true; do
  # ANSI-stripped, no `grep -q` (pipefail + early exit would report failure).
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

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v6 2>/dev/null | head -1)
if [ -z "$CK" ]; then log "no v6 checkpoint found in $FAST"; exit 1; fi
NAME=$(basename "$CK")
log "final checkpoint: $NAME"
OUT=/workspace/checkpoints_fast/eval_reports_widenet_v6
log "generative eval (256 samples, free-running + teacher-forced, per-sample rows) -> $OUT"
$COMPOSE run --rm train-phase2-kb-widenet-v6 python scripts/eval_phase2_kb.py \
  +experiment=phase2_kb_widenet_v6 \
  "+eval.checkpoint=/workspace/checkpoints_fast/$NAME" \
  +eval.per_sample=true +eval.max_samples=256 +eval.max_new_tokens=512 +eval.max_tool_calls=4 \
  "+eval.output_dir=$OUT" \
  > "$FAST/eval_reports_widenet_v6.container.log" 2>&1
log "eval exit=$?"
REPORT="$FAST/eval_reports_widenet_v6/eval_phase2_kb_stage_A.json"
if [ -f "$REPORT" ]; then
  log "per-qtype generative analysis (vs v5b report, whose free-running path was broken)"
  .venv/bin/python scripts/analyze_generative_eval.py "$REPORT" \
    --compare "$FAST/eval_reports_widenet_v5b/eval_phase2_kb_stage_A.json" \
    > "$FAST/eval_reports_widenet_v6/analysis_vs_v5b.txt" 2>&1
  log "analysis written"
else
  log "eval report missing — skipping analysis"
fi
log "POST-RUN DONE"
