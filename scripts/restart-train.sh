#!/usr/bin/env bash
# Restart ANY training service from its latest checkpoint, losing nothing.
#
#   scripts/restart-train.sh <compose-service> [--glob '<checkpoint-glob>']
#
# Use this instead of calling run-train.sh directly on a service that is
# already mid-run. run-train.sh is the right entry point for a COLD start; it
# is not safe for a restart, for four reasons this script exists to handle.
# All four were observed in production on 2026-08-28/29 and none of them
# errored — each silently discarded work.
#
#  1. STALE PIN. A compose service carries a hard-coded +resume_checkpoint=
#     pin that is correct only until the next checkpoint is written. A restart
#     on a stale pin rewinds training to it: a verification chain restarted a
#     run pinned at step286 that had reached step 701, losing ~400 steps.
#     Dropping the pin is worse — auto-resume does not match on run_name, so it
#     cold-started from the Phase-1 base and discarded 1258 steps.
#
#  2. THE RESCUE SAVE WAS BEING THROWN AWAY. BaseTrainer._graceful_shutdown_save
#     writes a checkpoint on SIGTERM and compose honours stop_grace_period, so
#     it completes. But resolving the checkpoint BEFORE stopping picks the last
#     scheduled save, and the rescue save — written seconds later, and strictly
#     newer — is ignored. Observed on every restart of one run: steps 799, 1032,
#     1256 and 1541 were all rescue saves, all discarded, ~130 steps of GPU.
#     So: stop FIRST, resolve AFTER.
#
#  3. LOGS DIED WITH THE CONTAINER. `compose rm` destroys the container and its
#     logs. A free-running eval failure at step 1250 was unrecoverable for
#     exactly this reason — the traceback existed only in a container that had
#     been removed. Archive before removing.
#
#  4. NEWEST-MTIME IS THE WRONG ORDER. The rescue save lands on the fast NVMe
#     dir first; the archival copy on the slow disk is written later. Ordering
#     by mtime can therefore select an OLDER step. Resolve by STEP NUMBER
#     across both roots, and copy across if the winner is fast-dir-only, since
#     the pin is a /workspace/checkpoints path the container must be able to
#     reach.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

SVC="${1:-}"
if [ -z "$SVC" ]; then
  echo "usage: $0 <compose-service> [--glob '<checkpoint-glob>']" >&2
  exit 2
fi
shift
GLOB=""
STOP_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --glob) GLOB="${2:-}"; shift 2 ;;
    # Free the GPU for a probe WITHOUT restarting. Same archive-then-graceful-
    # stop discipline as a restart, so the SIGTERM rescue save still lands and
    # the logs still survive; the caller restarts afterwards with a normal
    # invocation, which then resolves that rescue save as the resume target.
    --stop-only) STOP_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

COMPOSE_FILE=docker/docker-compose.yaml
CONTAINER="docker-${SVC}-1"
EXT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR unset — see .env}"   # -> /workspace/checkpoints
FAST_DIR="${BGKIT_FAST_CHECKPOINT_HOST_DIR:-/home/werg/bgkit-ckpt-fast}"
LOG_ARCHIVE="$FAST_DIR/run_logs"
STOP_TIMEOUT="${RESTART_STOP_TIMEOUT:-600}"

# The service's current pin tells us which checkpoint family belongs to it,
# so the glob does not have to be hard-coded per service.
#
# Parsed out of the SERVICE's own command, not grepped from the file: the
# compose file defines many training services and each carries its own pin. A
# plain grep returns whichever appears first (caught in a dry run — it returned
# control_armb's pin while restarting v8), and a plain regex rewrite would have
# repointed EVERY service's pin at this run's checkpoint.
PIN=$(python3 - "$COMPOSE_FILE" "$SVC" <<'PY'
import re, sys
import yaml
compose_file, svc = sys.argv[1], sys.argv[2]
with open(compose_file) as fh:
    doc = yaml.safe_load(fh)
service = (doc.get("services") or {}).get(svc) or {}
blob = " ".join(
    x if isinstance(x, str) else " ".join(map(str, x))
    for key in ("command", "entrypoint")
    for x in ([service[key]] if isinstance(service.get(key), str) else service.get(key) or [])
)
m = re.search(r"\+resume_checkpoint=/workspace/checkpoints/([A-Za-z0-9_.-]+)", blob)
print(m.group(1) if m else "")
PY
)
if [ -z "$GLOB" ]; then
  # phase2_kb_step1541_20260829_083226_114443_run-phase2_kb_widenet_v8
  #                                           ^^^^^^^^^^^^^^^^^^^^^^^^ identity
  SUFFIX=$(echo "$PIN" | sed -n 's/.*_\(run-.*\)$/\1/p')
  [ -n "$SUFFIX" ] && GLOB="*${SUFFIX}"
fi
if [ -z "$GLOB" ]; then
  echo "cannot infer a checkpoint glob for $SVC (pin='$PIN'); pass --glob" >&2
  exit 2
fi

mkdir -p "$LOG_ARCHIVE"

# --- 1. Archive logs, then stop gracefully so the rescue save completes ------
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  LAST_STEP=$(docker logs --tail 4000 "$CONTAINER" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
              | grep -a train_step | tail -1 | grep -oE '(^|[^a-z_])step=[0-9]+' \
              | grep -oE '[0-9]+' | tail -1)
  LOG_OUT="$LOG_ARCHIVE/${SVC}_step${LAST_STEP:-unknown}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "==> archiving container logs -> ${LOG_OUT}.gz"
  docker logs "$CONTAINER" > "$LOG_OUT" 2>&1 || true
  gzip -f "$LOG_OUT" 2>/dev/null || true

  echo "==> graceful stop (timeout ${STOP_TIMEOUT}s) so the SIGTERM rescue save finishes"
  docker stop -t "$STOP_TIMEOUT" "$CONTAINER" >/dev/null 2>&1 || true
  for _ in $(seq 1 "$STOP_TIMEOUT"); do
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || break
    sleep 1
  done
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "REFUSING: $CONTAINER still running after ${STOP_TIMEOUT}s"
    echo "  a forced kill here would lose the rescue save; investigate instead"
    exit 1
  fi
  echo "==> stopped"
fi

if [ "$STOP_ONLY" = "1" ]; then
  echo "==> stop-only: GPU is free; pin left untouched"
  exit 0
fi

# --- 2. NOW resolve the latest checkpoint (may be the rescue save just written)
resolve_latest() {
  local best_step=-1 best_path="" d p step
  for d in "$EXT_DIR" "$FAST_DIR"; do
    [ -d "$d" ] || continue
    for p in "$d"/$GLOB; do
      [ -d "$p" ] || continue
      [ -f "$p/metadata.json" ] || continue          # skip half-written saves
      step=$(basename "$p" | sed -n 's/.*step\([0-9]*\)_.*/\1/p')
      [ -n "$step" ] || continue
      if [ "$step" -gt "$best_step" ]; then best_step=$step; best_path=$p; fi
    done
  done
  [ -n "$best_path" ] && echo "$best_step|$best_path"
}

RESOLVED=$(resolve_latest)
if [ -z "$RESOLVED" ]; then
  echo "no checkpoint matching '$GLOB' — starting fresh from the configured base"
  scripts/run-train.sh --no-follow "$SVC"
  exit 0
fi
NEW="${RESOLVED%%|*}"
CK="${RESOLVED#*|}"
NAME=$(basename "$CK")

if [ ! -d "$EXT_DIR/$NAME" ]; then
  echo "==> $NAME exists only on the fast dir; copying to $EXT_DIR"
  cp -a "$CK" "$EXT_DIR/$NAME.partial" && mv "$EXT_DIR/$NAME.partial" "$EXT_DIR/$NAME"
fi

# --- 3. Never pin BACKWARDS ---------------------------------------------------
CUR=$(echo "$PIN" | sed -n 's/.*step\([0-9]*\)_.*/\1/p')
if [ -n "${CUR:-}" ] && [ "${NEW:-0}" -lt "${CUR:-0}" ] 2>/dev/null; then
  echo "REFUSING: latest checkpoint step$NEW is OLDER than the current pin step$CUR"
  echo "  that would rewind training; investigate before restarting"
  exit 1
fi

# Replace THIS service's pin only. Checkpoint directory names are unique per
# run (they carry step, timestamp and run_name), so swapping the exact old
# string cannot touch another service's pin — unlike a pattern rewrite, which
# would repoint all of them.
python3 - "$PIN" "$NAME" "$COMPOSE_FILE" <<'PY'
import sys
from pathlib import Path
old, new, compose_file = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path(compose_file)
s = p.read_text()
if old == new:
    print(f"pin already at {new}")
    raise SystemExit(0)
if old not in s:
    raise SystemExit(f"REFUSING: current pin '{old}' not found verbatim in {compose_file}")
n = s.count(old)
p.write_text(s.replace(old, new))
print(f"pin -> {new}" + (f"  ({n} occurrences)" if n > 1 else ""))
PY
if [ $? -ne 0 ]; then echo "pin rewrite failed; not starting"; exit 1; fi

echo "==> starting $SVC"
scripts/run-train.sh --no-follow "$SVC"
