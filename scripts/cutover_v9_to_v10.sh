#!/usr/bin/env bash
# Stop v9, take its verdict, swap in the rebuilt fileneedle, launch v10.
#
# ONE SCRIPT because the pieces have to happen in this order and the manual
# version already went wrong once: a `compose stop` was still working when the
# relaunch went in, and its SIGTERM killed the new container during setup. So:
# stop, WAIT for the container to actually be gone, then act.
#
# The fileneedle swap happens AFTER the probe and BEFORE v10. The probe runs
# the v9 experiment, which reads fileneedle; replacing files under an open
# mmap is not something to find out about the hard way.
#
#   setsid nohup bash scripts/cutover_v9_to_v10.sh \
#     > /home/werg/bgkit-ckpt-fast/cutover_v10.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
C=docker-train-phase2-kb-widenet-v9-interface-1
V9=train-phase2-kb-widenet-v9-interface
V10=train-phase2-kb-widenet-v10-reconstruct
FAST=/home/werg/bgkit-ckpt-fast
SP=/tmp/claude-1000/-home-werg-bgkit/fdf5f66b-3f97-4e4a-bb54-48aa51fc0d51/scratchpad
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
BASE=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459
V8=/workspace/checkpoints/phase2_kb_step2999_20260829_222528_014366_run-phase2_kb_widenet_v8
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "stopping v9 (graceful shutdown writes an emergency save)"
$COMPOSE stop "$V9" >/dev/null 2>&1
for _ in $(seq 1 180); do
  [ -z "$(docker ps -q -f name=$C)" ] && break
  sleep 5
done
if [ -n "$(docker ps -q -f name=$C)" ]; then
  log "v9 still running after 15 min — aborting rather than forcing"
  exit 1
fi
while [ "$(pgrep -f '[t]rain.py' | wc -l)" != "0" ]; do sleep 10; done
log "v9 stopped"

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v9_interface 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  NAME=$(basename "$CK")
  log "STEP 1/3 probe $NAME against the base and v8, same documents"
  $COMPOSE run --rm "$V9" scripts/probe_rep_distinguishability.py \
    +experiment=phase2_kb_widenet_v9_interface \
    "+diag.checkpoints=[$BASE,$V8,/workspace/checkpoints_fast/$NAME]" \
    +diag.n_samples=128 \
    +diag.out=/workspace/checkpoints/repdist_v9_final.json \
    > "$FAST/repdist_v9_final.container.log" 2>&1
  log "probe exit=$?"
  sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/repdist_v9_final.container.log" \
    | grep -E "^checkpoint:|^documents:|^raw |^l1_input|^l1_in@k|^reps " || true
  JSON="$CHECKPOINT_DIR/repdist_v9_final.json"
  [ -f "$JSON" ] && .venv/bin/python scripts/verdict_repdist.py "$JSON" \
    --treatment v9_interface | tee "$FAST/verdict_v9_final.txt"
else
  log "no v9 checkpoint — skipping the probe, v9 never saved"
fi

log "STEP 2/3 swap in the rebuilt fileneedle (balanced presence class)"
STAGE="$SP/fileneedle_staging"
if [ -d "$STAGE/mmap/fileneedle" ] && [ -f "$STAGE/traj/fileneedle.parquet" ]; then
  BK="$DATA_DIR/mmap/phase2/fileneedle.bak_unbalanced_$(date -u +%Y%m%d)"
  [ -d "$BK" ] || mv "$DATA_DIR/mmap/phase2/fileneedle" "$BK"
  cp -r "$STAGE/mmap/fileneedle" "$DATA_DIR/mmap/phase2/fileneedle"
  cp "$DATA_DIR/trajectories/fileneedle.parquet" \
     "$DATA_DIR/trajectories/fileneedle.parquet.bak_unbalanced" 2>/dev/null
  cp "$STAGE/traj/fileneedle.parquet" "$DATA_DIR/trajectories/fileneedle.parquet"
  # The rebuild drops negative_spans_json; the contrastive term needs it back.
  .venv/bin/python scripts/add_contrastive_spans.py --datasets fileneedle 2>&1 | tail -2
  log "fileneedle swapped (old kept at $BK)"
else
  log "no fileneedle staging — leaving the deployed one in place"
fi

log "STEP 3/3 launch v10"
scripts/run-train.sh --no-follow "$V10" 2>&1 | tail -2
log "CUTOVER DONE"
