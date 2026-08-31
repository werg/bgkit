#!/usr/bin/env bash
# v9 post-run: wait for training_complete, then answer the ONE question the
# arm was launched to answer -- do the emitted reps still identify their own
# document? -- and only then spend GPU on task metrics.
#
#   setsid nohup scripts/post_run_widenet_v9.sh \
#     > /home/werg/bgkit-ckpt-fast/post_run_v9.log 2>&1 &
#
# Order matters. The probe is the arm's verdict and costs minutes; the eval
# sweep costs an hour and is only interpretable if the probe passed. The
# probe runs on the v9 checkpoint ALONGSIDE the Phase-1 base and widenet v8
# in one process, because those two are the numbers it has to be read
# against (reps top-1: base 0.898, v8 0.031) and a separate invocation would
# not guarantee the same eval documents.
set -uo pipefail
cd "$(dirname "$0")/.."
C=docker-train-phase2-kb-widenet-v9-interface-1
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
SVC=train-phase2-kb-widenet-v9-interface
BASE=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459
V8=/workspace/checkpoints/phase2_kb_step2999_20260829_222528_014366_run-phase2_kb_widenet_v8
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for training_complete on $C"
while true; do
  if docker logs "$C" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -q "training_complete"; then break; fi
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    log "container state=$st without training_complete — crash or stop. NOT running the"
    log "analysis: a partial run's checkpoint would be reported as the arm's result."
    exit 1
  fi
  sleep 120
done
log "training_complete seen"
# One GPU job at a time, and `compose run` one-offs are invisible to
# `compose ps` -- wait on the process, not on the service.
while [ "$(pgrep -f '[t]rain.py' | wc -l)" != "0" ]; do sleep 30; done
sleep 30

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-phase2_kb_widenet_v9_interface 2>/dev/null | head -1)
if [ -z "$CK" ]; then
  CK=$(ls -dt "${CHECKPOINT_DIR:-/mnt/external/bgkit-checkpoints}"/phase2_kb_step*_run-phase2_kb_widenet_v9_interface 2>/dev/null | head -1)
  MOUNT=/workspace/checkpoints
else
  MOUNT=/workspace/checkpoints_fast
fi
if [ -z "$CK" ]; then log "no v9 checkpoint found"; exit 1; fi
NAME=$(basename "$CK")
log "final checkpoint: $NAME (mounted at $MOUNT)"

OUT=/workspace/checkpoints/eval_reports_widenet_v9
log "STEP 1/2 rep distinguishability: v9 vs the base vs v8, same documents"
$COMPOSE run --rm "$SVC" scripts/probe_rep_distinguishability.py \
  +experiment=phase2_kb_widenet_v9_interface \
  "+diag.checkpoints=[$BASE,$V8,$MOUNT/$NAME]" \
  +diag.n_samples=128 \
  +diag.out=/workspace/checkpoints/repdist_v9_vs_base_v8.json \
  > "$FAST/repdist_v9.container.log" 2>&1
log "probe exit=$?"
sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/repdist_v9.container.log" \
  | grep -E "^checkpoint:|^documents:|^stage|^raw |^l1_input|^l1_in@k|^reps " || true

# The verdict is a threshold committed BEFORE the numbers arrived, not a
# reading made after them. See scripts/verdict_repdist.py.
JSON="${CHECKPOINT_DIR:-/mnt/external/bgkit-checkpoints}/repdist_v9_vs_base_v8.json"
if [ -f "$JSON" ]; then
  .venv/bin/python scripts/verdict_repdist.py "$JSON" --treatment v9_interface \
    | tee "$FAST/verdict_v9.txt"
else
  log "no probe JSON at $JSON — cannot state a verdict"
fi

log "STEP 2/2 floor / reps / ceiling on the same samples"
$COMPOSE run --rm "$SVC" scripts/eval_phase2_kb.py \
  +experiment=phase2_kb_widenet_v9_interface \
  "+eval.checkpoint=$MOUNT/$NAME" \
  "+eval.ablation_sweep=[none,zeroed,full_text]" \
  +eval.per_sample=true +eval.max_samples=192 \
  +eval.max_new_tokens=512 +eval.max_tool_calls=4 \
  "+eval.output_dir=$OUT" \
  > "$FAST/eval_v9_sweep.container.log" 2>&1
log "sweep exit=$?"
sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/eval_v9_sweep.container.log" \
  | grep -E "kb_eval_ceiling" || true
log "POST-RUN V9 DONE"
