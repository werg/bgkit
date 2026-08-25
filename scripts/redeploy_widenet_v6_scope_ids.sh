#!/usr/bin/env bash
# 2026-08-23: stop v6 (rescue save) -> regenerate the four flat datasets with
# the entrypoint id named in scope (flat_phase2_writer) -> relaunch v6
# (auto-resumes). Run detached:  setsid nohup scripts/redeploy_widenet_v6_scope_ids.sh > LOG 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "stopping v6 (graceful, rescue save)"
timeout 900 $COMPOSE stop -t 840 train-phase2-kb-widenet-v6
st=$(docker inspect -f '{{.State.Status}}' docker-train-phase2-kb-widenet-v6-1 2>/dev/null || echo gone)
log "v6 state=$st"
if [ "$st" = "running" ]; then log "FATAL: v6 still running; not touching data"; exit 1; fi

log "regenerating lognav fileneedle grepset swerecall"
if ! scripts/rebuild_widenet_data_v2.sh; then
  log "FATAL: rebuild failed; v6 NOT relaunched"; exit 1
fi

log "scope-id liveness: every row's call id must be named in its scope_description"
if ! .venv/bin/python - <<'EOF'
import os, sys, json, pyarrow.parquet as pq
from bgkit.eval.kb_trajectory_eval import scope_entrypoint_ids
d = os.environ["DATA_DIR"] + "/trajectories"; bad = 0
for ds in ("lognav", "fileneedle", "grepset", "swerecall"):
    t = pq.read_table(f"{d}/{ds}.parquet", columns=["trajectory_json", "scope_description"]).to_pylist()
    miss = sum(1 for r in t if json.loads(r["trajectory_json"])[0]["args"]["ids"][0] not in scope_entrypoint_ids(r["scope_description"]))
    print(f"{ds}: {miss}/{len(t)} rows whose call id is NOT named in scope")
    bad += miss
sys.exit(1 if bad else 0)
EOF
then log "FATAL: scope-id liveness failed; v6 NOT relaunched"; exit 1; fi

log "relaunching v6"
scripts/run-train.sh --no-follow train-phase2-kb-widenet-v6
log "REDEPLOY DONE"
