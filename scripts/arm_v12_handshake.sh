#!/usr/bin/env bash
# Touch the bgkit2 reset handshake when the v12 post-run chain is finished.
#
#   setsid nohup bash scripts/arm_v12_handshake.sh \
#     > /home/werg/bgkit-ckpt-fast/v12_handshake.log 2>&1 &
#
# WHY A SEPARATE PROCESS. The obvious place for this is the last line of
# post_run_arm.sh, but that script is RUNNING: bash reads a script by byte
# offset, so appending to it makes the live shell resume parsing mid-token.
# That killed the 2026-08-25 chain on its final line. Never edit a running
# bash script; watch its log instead.
#
# Two exits, and they are deliberately different:
#   CLEAN  the chain logged "POST-RUN DONE"      -> touch
#   DEAD   the chain process is gone, no bgkit GPU container and no bgkit GPU
#          process remain, sustained across DEAD_CONFIRM probes, and the log
#          never said DONE                        -> touch, loudly
# The dead path is what bgkit-c7 asked for ("touch it anyway after confirming
# no GPU job remains"). It is mechanical confirmation, not a judgement call:
# the only consequence of touching is that the reset run starts, which is the
# intended next step either way. Sustained confirmation avoids firing during
# the ~60s gap between the chain's own probe and sweep containers.

set -u
LOG=/home/werg/bgkit-ckpt-fast/post_run_v12.log
FLAG=/home/werg/bgkit-data-nvme/bgkit2/logs/v12_postrun_done
DEAD_CONFIRM=5          # consecutive quiet probes before believing the chain died
dead_streak=0

log() { echo "$(date -u +%FT%TZ) $*"; }

bgkit_gpu_busy() {
  # A bgkit GPU container, or a GPU compute process running a bgkit script.
  # ComfyUI (main.py, port 8188) is deliberately NOT counted -- it is the
  # user's own app, it holds ~17 GB indefinitely, and requiring it to exit is
  # what made the first version of the reset watcher unsatisfiable.
  local n_c n_p
  n_c=$(docker ps --format '{{.Names}}' | grep -cE 'train-|eval-|profile-|parity-' ; true)
  [ "${n_c:-0}" -gt 0 ] && return 0
  n_p=0
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if ps -o args= -p "${pid%,}" 2>/dev/null \
        | grep -qE 'scripts/(train|eval_|probe_|diag_)|bgkit2_train'; then
      n_p=$((n_p + 1))
    fi
  done
  [ "$n_p" -gt 0 ]
}

log "armed; watching $LOG for POST-RUN DONE -> $FLAG"
while true; do
  if [ -f "$FLAG" ]; then
    log "flag already present; nothing to do"; exit 0
  fi
  if [ -f "$LOG" ] && grep -q "POST-RUN .* DONE" "$LOG"; then
    touch "$FLAG"
    log "CLEAN: chain logged POST-RUN DONE; touched $FLAG"
    exit 0
  fi
  if [ "$(pgrep -f '[p]ost_run_arm.sh' | wc -l)" -eq 0 ] && ! bgkit_gpu_busy; then
    dead_streak=$((dead_streak + 1))
    log "chain process absent and no bgkit GPU work ($dead_streak/$DEAD_CONFIRM)"
    if [ "$dead_streak" -ge "$DEAD_CONFIRM" ]; then
      touch "$FLAG"
      log "DEAD: chain gone without POST-RUN DONE, GPU quiet; touched $FLAG anyway"
      log "DEAD: v12 post-run results are INCOMPLETE -- read $LOG before trusting any number"
      exit 0
    fi
  else
    dead_streak=0
  fi
  sleep 60
done
