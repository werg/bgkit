#!/usr/bin/env bash
# Post-run chain for 2026-08-25: wait out blob_sft_v1, then take the two
# measurements that are blocked only on the GPU being free.
#
#   1. Family-A gate — scripts/eval_blob_sft.py on the FINAL blob_sft
#      checkpoint. The in-flight process predates the probe-zeroed-gap
#      metric, so `eval/probe_zeroed_gap` gets its first real reading here:
#      how much of the ~0.30 recall-probe EM is actually READ from the
#      compacted reps versus guessed from SWE-Zero's templated priors.
#      (The pooled `eval/zeroed_gap` ~0.005 does NOT answer this — 98 of 128
#      eval samples are continuation, which barely needs the blob.)
#   2. BABILong bgkit arm at both retention points (see run_babilong_bgkit.sh).
#
# One GPU job at a time: each step starts only after the previous container
# is gone.
#   setsid nohup scripts/post_run_blob_sft_then_babilong.sh \
#     > /home/werg/bgkit-ckpt-fast/post_run_2026_08_25.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
CKPT_V6=/workspace/checkpoints_fast/phase2_kb_step2629_20260824_002147_192565_run-phase2_kb_widenet_v6
CKPT_V7=/workspace/checkpoints_fast/phase2_kb_step999_20260824_233911_572282_run-phase2_kb_widenet_v7_ratio6x
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for docker-train-blob-sft-v1-1 to exit (max 6h)"
for _ in $(seq 1 360); do
  [ -z "$(docker ps -q --filter name=blob-sft)" ] && break
  sleep 60
done
if [ -n "$(docker ps -q --filter name=blob-sft)" ]; then
  log "blob_sft still running after 6h — aborting chain rather than sharing the GPU"; exit 1
fi
log "blob_sft container gone; last log lines:"
docker logs --tail 5 docker-train-blob-sft-v1-1 2>&1 | sed 's/\x1b\[[0-9;]*m//g'

CKPT=$(ls -1dt "$FAST"/blob_sft_step*_run-blob_sft_v1 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  log "no blob_sft checkpoint found — skipping the Family-A gate"
else
  NAME=$(basename "$CKPT")
  log "Family-A gate on $NAME"
  $COMPOSE run --rm --name blob-sft-eval train-blob-sft-v1 \
    python scripts/eval_blob_sft.py +experiment=blob_sft_v1 \
      "+eval.checkpoint=/workspace/checkpoints_fast/$NAME" \
      ++training.max_eval_samples=256 \
      "+eval.output_json=/workspace/checkpoints_fast/eval_blob_sft_v1_final.json" 2>&1 | tail -40
  log "Family-A metrics:"
  .venv/bin/python - <<'PY' || true
import json, pathlib
p = pathlib.Path("/home/werg/bgkit-ckpt-fast/eval_blob_sft_v1_final.json")
if not p.exists():
    print("  (no report written)")
else:
    m = json.loads(p.read_text())
    for k in sorted(m):
        if "probe" in k or "zeroed" in k or k in ("eval/loss", "eval/token_accuracy"):
            print(f"  {k} = {m[k]}")
PY
fi

log "BABILong bgkit arms"
scripts/run_babilong_bgkit.sh "$CKPT_V6" "$CKPT_V7"
log "CHAIN DONE"
