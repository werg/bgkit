#!/usr/bin/env bash
# RULER stock baseline for Qwen3.5-0.8B via the vllm-fast service.
# The grid (5 tasks x 4 lengths x 100 samples) is already generated under
# $NVME/benchmarks/ruler/data. Requires the vllm-node image — built locally
# from ~/spark-vllm-docker (eugr's repo), NOT pullable from any registry;
# see plans/capability_packaging_2026_08_20.md §2026-08-24.
# Refuses to start while any phase2-kb GPU container is running (HARD RULE:
# never a second GPU job alongside a live trainer/eval).
#   setsid nohup scripts/run_ruler_baseline.sh > /home/werg/bgkit-ckpt-fast/ruler_baseline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
FAST=/home/werg/bgkit-ckpt-fast
NVME=/home/werg/bgkit-data-nvme/capability_packaging
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if ! docker image inspect vllm-node:latest >/dev/null 2>&1; then
  log "vllm-node:latest missing — build it first: ~/spark-vllm-docker/build-and-copy.sh"
  exit 1
fi
if [ -n "$(docker ps -q --filter name=phase2-kb)" ]; then
  log "a phase2-kb GPU container is running — refusing to start vLLM alongside it"
  exit 1
fi

RULER=$NVME/benchmarks/ruler
PRED=$RULER/pred_qwen35_0p8b
log "starting vllm-fast (Qwen3.5-0.8B, max-model-len 40960 for the 32K prompts)"
VLLM_MAX_MODEL_LEN_FAST=40960 $COMPOSE up -d vllm-fast
healthy=0
for _ in $(seq 1 90); do
  if curl -sf "http://localhost:${VLLM_PORT_FAST:-8091}/v1/models" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 10
done
if [ "$healthy" = "1" ]; then
  log "vllm-fast healthy; predicting RULER grid"
  .venv/bin/python scripts/baseline_ruler_predict.py \
    --data-root "$RULER/data" --out-root "$PRED" \
    --model Qwen/Qwen3.5-0.8B \
    --base-url "http://localhost:${VLLM_PORT_FAST:-8091}/v1" \
    > "$FAST/baselines_ruler_predict.log" 2>&1
  log "predict exit=$?"
  # RULER's own evaluate.py needs the NeMo toolkit just for jsonl IO;
  # scripts/score_ruler.py applies RULER's exact metric fns without it.
  log "scoring all lengths"
  /home/werg/bgkit/.venv/bin/python /home/werg/bgkit/scripts/score_ruler.py \
    --pred-root "$PRED" >> "$FAST/baselines_ruler_eval.log" 2>&1
  log "RULER scored (summaries under $PRED/L*/summary.csv)"
else
  log "vllm-fast did not become healthy after 15 min — check: $COMPOSE logs vllm-fast"
fi
$COMPOSE stop vllm-fast >/dev/null 2>&1 || true
$COMPOSE rm -f vllm-fast >/dev/null 2>&1 || true
log "vllm-fast stopped; RULER BASELINE DONE"
