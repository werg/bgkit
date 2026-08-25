#!/usr/bin/env bash
# Wait for v6's checkpoint at a given step (its eval has then completed), then
# run the redeploy chain (stop -> regen -> gates -> relaunch).
#   setsid nohup scripts/after_step_redeploy_v6.sh 1000 > LOG 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
STEP="${1:?step}"
C=docker-train-phase2-kb-widenet-v6-1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "waiting for checkpoint_saved step=$STEP on $C"
while true; do
  # structlog colors values: strip ANSI escapes before matching step=N.
  # No `grep -q`: under pipefail its early exit SIGPIPEs the producers and
  # the pipeline reports failure even on a match.
  if docker logs "$C" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "checkpoint_saved.*step=$STEP( |$)" > /dev/null; then break; fi
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then log "container state=$st before step $STEP — proceeding with redeploy anyway"; break; fi
  sleep 60
done
log "trigger reached; running redeploy chain"
exec scripts/redeploy_widenet_v6_scope_ids.sh
