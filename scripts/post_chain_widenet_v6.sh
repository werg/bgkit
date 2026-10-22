#!/usr/bin/env bash
# Stage 2 of the 2026-08-22 overnight chain. Waits for post_run_widenet_v5b.sh
# to log POST-RUN DONE, then:
#   1. RULER stock baseline for Qwen3.5-0.8B via the vllm-fast service
#      (predictions with scripts/baseline_ruler_predict.py, scored with
#      RULER's evaluate.py per context length), vLLM stopped afterwards,
#   2. launches the v6 clean-baseline training run.
# Run detached:  setsid nohup scripts/post_chain_widenet_v6.sh > $FAST/post_chain_v6.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
FAST=/home/werg/bgkit-ckpt-fast
NVME=/home/werg/bgkit-data-nvme/capability_packaging
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for POST-RUN DONE in $FAST/post_run_v5b.log"
until grep -q "POST-RUN DONE" "$FAST/post_run_v5b.log" 2>/dev/null; do sleep 120; done
log "stage 1 chain done"
# No GPU job may overlap: wait until no bgkit train/eval/baseline container runs.
while [ -n "$(docker ps -q --filter name=phase2-kb)" ]; do sleep 30; done

# ---------------- RULER baseline (vLLM) ----------------
RULER=$NVME/benchmarks/ruler
PRED=$RULER/pred_qwen35_0p8b
log "starting vllm-fast (Qwen3.5-0.8B, max-model-len 40960)"
VLLM_MAX_MODEL_LEN_FAST=40960 $COMPOSE up -d vllm-fast
for i in $(seq 1 90); do
  if curl -sf http://localhost:${VLLM_PORT_FAST:-8091}/v1/models >/dev/null 2>&1; then break; fi
  sleep 10
done
if curl -sf http://localhost:${VLLM_PORT_FAST:-8091}/v1/models >/dev/null 2>&1; then
  log "vllm-fast healthy; predicting RULER grid"
  .venv/bin/python scripts/baseline_ruler_predict.py --data-root "$RULER/data" --out-root "$PRED" \
    --model Qwen/Qwen3.5-0.8B --base-url "http://localhost:${VLLM_PORT_FAST:-8091}/v1" \
    > "$FAST/baselines_ruler_predict.log" 2>&1
  log "predict exit=$?"
  for L in "$PRED"/L*; do
    [ -d "$L" ] || continue
    log "scoring $(basename "$L")"
    (cd "$RULER/RULER/scripts/eval" && /home/werg/bgkit/.venv/bin/python evaluate.py \
      --data_dir "$L" --benchmark synthetic >> "$FAST/baselines_ruler_eval.log" 2>&1)
  done
  log "RULER scored (summaries under $PRED/L*/summary*.csv)"
else
  log "vllm-fast did not become healthy — skipping RULER"
fi
$COMPOSE stop vllm-fast >/dev/null 2>&1 || true
$COMPOSE rm -f vllm-fast >/dev/null 2>&1 || true
while [ -n "$(docker ps -q --filter name=vllm)" ]; do sleep 10; done
log "vllm-fast stopped"

# ---------------- v6 launch ----------------
if [ -n "$(docker ps -q --filter name=phase2-kb)" ]; then
  log "a phase2-kb container is running — NOT launching v6"; exit 1
fi
log "launching train-phase2-kb-widenet-v6"
scripts/run-train.sh --no-follow train-phase2-kb-widenet-v6
sleep 10
docker inspect -f '{{.State.Status}} {{.State.StartedAt}}' docker-train-phase2-kb-widenet-v6-1
log "V6 LAUNCHED"
