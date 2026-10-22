#!/bin/bash
# Memory guard for the teacher/sub-agent inference container running ALONGSIDE training/evals
# (user 2026-09-21: allowed "with close monitoring, pausing and evacuating the eval server if memory
# gets tight"). Every 10 s: MemAvailable < PAUSE_GB -> docker pause; < STOP_GB -> docker stop
# (evacuate, frees its GPU/unified memory); paused and > RESUME_GB for 60 s -> unpause.
# Training always wins. Usage: teacher_mem_guard.sh <container> [pause_gb] [stop_gb] [resume_gb]
C="$1"; PAUSE=${2:-22}; STOP=${3:-14}; RESUME=${4:-35}
ok=0
while true; do
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null) || { echo "$(date -u +%T) GONE $C"; exit 0; }
  [ "$st" = exited ] && { echo "$(date -u +%T) EXITED $C"; exit 0; }
  av=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$av" -lt "$STOP" ] && [ "$st" != exited ]; then
    docker unpause "$C" >/dev/null 2>&1; docker stop -t 5 "$C" >/dev/null 2>&1
    echo "$(date -u +%T) EVACUATED $C at ${av}GB available"; exit 0
  elif [ "$av" -lt "$PAUSE" ] && [ "$st" = running ]; then
    docker pause "$C" >/dev/null && echo "$(date -u +%T) PAUSED $C at ${av}GB available"; ok=0
  elif [ "$st" = paused ]; then
    if [ "$av" -gt "$RESUME" ]; then ok=$((ok+1)); else ok=0; fi
    if [ "$ok" -ge 6 ]; then docker unpause "$C" >/dev/null && echo "$(date -u +%T) RESUMED $C at ${av}GB available"; ok=0; fi
  fi
  sleep 10
done
