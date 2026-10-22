#!/usr/bin/env bash
# Reusable bgkit training monitor (DGX Spark, 121 GB unified memory).
#
# Encodes the 2026-06-28 Stage-A lessons: the two worst failures were INVISIBLE
# to GPU-peak/crash monitoring —
#   * a HOST OOM froze the whole machine (GPU process + driver/OS > 121 GB host),
#   * a memory THRASH caused a 10x slowdown with no crash.
# So this watches HOST RAM and STEP-TIME, plus RestartCount, GPU peak, and the
# crash signatures we've actually hit (incl. NV_ERR_NO_MEMORY and the bf16
# resume 'expected dtype float' lerp crash).
#
# Usage:
#   scripts/monitor_training.sh <service> [host_alert_gb=8] [peak_alert_gb=90] [poll_s=120] [iters=30]
# Exit codes: 0 = clean check-in / eval event; 1 = crash/bad-state; 2 = host-RAM
# critical (run stopped to prevent a freeze).
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env 2>/dev/null; set +a

SVC="${1:?usage: monitor_training.sh <service> [host_alert_gb] [peak_alert_gb] [poll_s] [iters]}"
HOST_ALERT="${2:-5}"     # stop the run only if host available RAM (GB) stays below this for 3+ consecutive polls (a single dip is the adaptive flush mid-reclaim, not a freeze)
PEAK_ALERT="${3:-90}"    # warn if cuda_max_allocated_gb exceeds this
POLL="${4:-120}"
ITERS="${5:-30}"
CN="docker-${SVC}-1"
C="docker compose -f docker/docker-compose.yaml"
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }

peak=0; minhost=9999; last=-1; lastt=$(date +%s); lowstreak=0
for i in $(seq 1 "$ITERS"); do
  st=$(docker inspect "$CN" --format '{{.State.Status}}' 2>/dev/null || echo missing)
  rc=$(docker inspect "$CN" --format '{{.RestartCount}}' 2>/dev/null || echo 0)
  host=$(free -g | awk 'NR==2{print $7}')
  [ -n "$host" ] && [ "$host" -lt "$minhost" ] 2>/dev/null && minhost=$host

  # HOST-OOM guard (freeze prevention): the guard caps the GPU *process*, not
  # host-side dataloader buffers, so watch host RAM directly. But a SINGLE low
  # reading is normal — at a tight guard the adaptive CUDA flush lets host-avail
  # dip (~7GB at a ~79GB GPU peak) then reclaims GPU cache and bounces it back to
  # ~20GB. So only bail on a SUSTAINED low (3+ consecutive polls) = the flush
  # can't keep up = a real freeze trajectory. (2026-06-28: a flat single-dip stop
  # false-alarms on the managed oscillation.)
  if [ -n "$host" ] && [ "$host" -lt "$HOST_ALERT" ] 2>/dev/null; then lowstreak=$((lowstreak+1)); else lowstreak=0; fi
  if [ "$lowstreak" -ge 3 ]; then
    echo "ALERT[host] available=${host}GB < ${HOST_ALERT}GB sustained x${lowstreak} — FREEZE RISK; stopping ${SVC}"
    timeout 40 $C stop -t 10 "$SVC" 2>&1 | tail -1
    exit 2
  fi

  L=$($C logs --tail 1200 "$SVC" 2>/dev/null | strip)
  if printf '%s\n' "$L" | grep -iE 'OutOfMemoryError|CUDA out of memory|NV_ERR_NO_MEMORY|Traceback \(most recent|CheckpointError|expected dtype float' \
       | grep -viE 'HTTP|expand=false|\.no_exist|404' | grep -q .; then
    echo "ALERT[crash] rc=${rc} host=${host}GB peak=${peak}GB"
    printf '%s\n' "$L" | grep -iE 'OutOfMem|NV_ERR|expected dtype|Error:|Traceback' | grep -vi HTTP | tail -6
    exit 1
  fi
  if [ "$st" != "running" ] || [ "${rc:-0}" -gt 0 ]; then
    echo "ALERT[state] status=${st} restarts=${rc}"
    exit 1
  fi

  m=$(printf '%s' "$L" | grep train_step | tail -1 | grep -oE 'cuda_max_allocated_gb=[0-9.]+' | grep -oE '[0-9.]+')
  if [ -n "$m" ]; then
    awk "BEGIN{exit !($m>$peak)}" 2>/dev/null && peak=$m
    awk "BEGIN{exit !($m>$PEAK_ALERT)}" 2>/dev/null && echo "WARN[peak] ${m}GB > ${PEAK_ALERT}GB"
  fi

  # eval / checkpoint event → report and stop this cycle
  if printf '%s' "$L" | grep -qiE 'eval/token_f1|eval/loss=|checkpoint_saved'; then
    echo "EVENT[eval/save]:"
    printf '%s' "$L" | grep -oE 'eval/token_f1=[0-9.]+|eval/loss=[0-9.]+|eval/[a-z]+/token_f1=[0-9.]+|checkpoint_saved|step=[0-9]+' | tail -8
    exit 0
  fi

  step=$(printf '%s' "$L" | grep train_step | grep -oE 'step=[0-9]+' | tail -1 | grep -oE '[0-9]+')
  if [ -n "$step" ] && [ "$step" != "$last" ] 2>/dev/null; then
    now=$(date +%s); ds=$((step - last)); [ "$last" -lt 0 ] && ds=1; [ "$ds" -lt 1 ] && ds=1
    rate=$(( (now - lastt) / ds ))
    echo "[$(date +%H:%M:%S)] step=${step} peak=${m}GB host=${host}GB ~${rate}s/step"
    last=$step; lastt=$now
  fi
  sleep "$POLL"
done
echo "CHECKIN status=${st} restarts=${rc} step=${last} peak=${peak}GB min_host=${minhost}GB"
