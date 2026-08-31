#!/usr/bin/env bash
# Early read on the v9 arm: stop training, probe the latest checkpoint against
# the base and v8, resume. ~20 minutes, against ~10 hours of continuing a dead
# arm.
#
# WHY A SCRIPT. The last manual stop left a `compose stop` still working when
# the relaunch went in, and its SIGTERM killed the new container during setup.
# Stop, WAIT for the container to actually be gone, then act.
#
#   bash scripts/early_probe_v9.sh
set -uo pipefail
cd "$(dirname "$0")/.."
C=docker-train-phase2-kb-widenet-v9-interface-1
SVC=train-phase2-kb-widenet-v9-interface
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
BASE=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459
V8=/workspace/checkpoints/phase2_kb_step2999_20260829_222528_014366_run-phase2_kb_widenet_v8
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v9_interface 2>/dev/null | head -1)
if [ -z "$CK" ]; then log "no v9 checkpoint yet — nothing to probe"; exit 1; fi
log "probing $(basename "$CK")"

log "stopping training"
$COMPOSE stop "$SVC" >/dev/null 2>&1
# The stop must COMPLETE before anything else touches this service.
for _ in $(seq 1 120); do
  [ -z "$(docker ps -q -f name=$C)" ] && break
  sleep 5
done
if [ -n "$(docker ps -q -f name=$C)" ]; then
  log "container still running after 10 min — aborting rather than forcing"
  exit 1
fi
while [ "$(pgrep -f '[t]rain.py' | wc -l)" != "0" ]; do sleep 10; done
log "training stopped"

# A graceful shutdown writes an emergency save; probe the NEWEST checkpoint,
# which may now be that one rather than the scheduled save.
CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v9_interface | head -1)
NAME=$(basename "$CK")
log "newest checkpoint after shutdown: $NAME"

$COMPOSE run --rm "$SVC" scripts/probe_rep_distinguishability.py \
  +experiment=phase2_kb_widenet_v9_interface \
  "+diag.checkpoints=[$BASE,$V8,/workspace/checkpoints_fast/$NAME]" \
  +diag.n_samples=128 \
  +diag.out=/workspace/checkpoints/repdist_v9_early.json \
  > "$FAST/repdist_v9_early.container.log" 2>&1
log "probe exit=$?"
sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/repdist_v9_early.container.log" \
  | grep -E "^checkpoint:|^documents:|^raw |^l1_input|^l1_in@k|^reps " || true

JSON="${CHECKPOINT_DIR:-/mnt/external/bgkit-checkpoints}/repdist_v9_early.json"
if [ -f "$JSON" ]; then
  .venv/bin/python scripts/verdict_repdist.py "$JSON" --treatment v9_interface \
    | tee "$FAST/verdict_v9_early.txt"
else
  log "no probe JSON — no verdict"
fi

log "resuming training (auto-resume is run-scoped and the config is unchanged,"
log "so the fingerprint check passes and it picks up $NAME)"
scripts/run-train.sh --no-follow "$SVC" 2>&1 | tail -2
log "EARLY PROBE DONE"
