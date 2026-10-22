#!/usr/bin/env bash
# Post-run chain for phase2_kb_widenet_v5b (2026-08-22):
#   1. wait for the trainer to log `training_complete` and exit,
#   2. generative per-qtype eval of the FINAL checkpoint (the headline gate),
#      compared against the v2 zero-rep report,
#   3. BABILong stock-model baseline (full-context vs truncation arms).
# Run detached from the agent harness:  setsid nohup scripts/post_run_widenet_v5b.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
C=docker-train-phase2-kb-widenet-v5-1
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for training_complete on $C"
until docker logs "$C" 2>&1 | grep -q "training_complete"; do
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    log "container state=$st without training_complete — not finishing the chain (crash/stop); re-checking in 10 min"
    sleep 600
    continue
  fi
  sleep 120
done
log "training_complete seen"
while [ "$(ps aux | grep -c '[t]rain.py')" != "0" ]; do sleep 30; done
sleep 30

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v5b 2>/dev/null | head -1)
if [ -z "$CK" ]; then log "no v5b checkpoint found in $FAST"; exit 1; fi
NAME=$(basename "$CK")
log "final checkpoint: $NAME"

OUT=/workspace/checkpoints_fast/eval_reports_widenet_v5b
log "generative eval (192 samples, per-sample rows) -> $OUT"
$COMPOSE run --rm train-phase2-kb-widenet-v5 python scripts/eval_phase2_kb.py \
  +experiment=phase2_kb_widenet_v5 \
  "+eval.checkpoint=/workspace/checkpoints_fast/$NAME" \
  +eval.per_sample=true +eval.max_samples=192 "+eval.output_dir=$OUT" \
  > "$FAST/eval_reports_widenet_v5b.container.log" 2>&1
log "eval exit=$?"
REPORT="$FAST/eval_reports_widenet_v5b/eval_phase2_kb_stage_A.json"
if [ -f "$REPORT" ]; then
  log "per-qtype generative analysis (vs v2 zero-rep report)"
  .venv/bin/python scripts/analyze_generative_eval.py "$REPORT" \
    --compare "$FAST/eval_reports_widenet_v2/eval_phase2_kb_stage_A.json" \
    > "$FAST/eval_reports_widenet_v5b/analysis_vs_v2.txt" 2>&1
  log "analysis written"
else
  log "eval report missing — skipping analysis"
fi

log "BABILong stock baseline (Qwen3.5-0.8B, qa1-3, 0k-32k, 100 samples)"
$COMPOSE run --rm train-phase2-kb-widenet-v5 python scripts/baseline_babilong.py \
  --data-dir /workspace/capability_packaging/benchmarks/babilong/babilong-1k-samples \
  --repo-dir /workspace/capability_packaging/benchmarks/babilong/babilong-repo \
  --out-dir /workspace/checkpoints_fast/baselines/babilong_qwen35_0p8b --use-chat-template \
  > "$FAST/baselines_babilong.container.log" 2>&1
log "babilong exit=$?"
log "POST-RUN DONE"
