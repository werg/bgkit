#!/usr/bin/env bash
# Post-run analysis for a wide-net arm: verdict first, task metrics second.
#
#   setsid nohup bash scripts/post_run_arm.sh <experiment> <run_name> <service> \
#     > /home/werg/bgkit-ckpt-fast/post_run_<run_name>.log 2>&1 &
#
# GENERATION RUNS ON THE ZEROED ARM TOO, which doubles its cost and is not
# optional. A family whose answer is a long verbatim span cannot be measured
# teacher forced: the gold prefix is handed to both arms and determines most
# of the continuation. Measured on `reconstruct` at v10 step 500, same
# checkpoint and samples -- teacher-forced token_f1 0.404, free-running 0.012,
# against a cross-document floor of 0.021. The teacher-forced number WAS the
# prefix, so the only real rep_gain for that family is a generative one.
#
# Generic because this is the third arm to need it. Order is the point: the
# probe is the arm's verdict and costs minutes, the floor/reps/ceiling sweep
# costs an hour and is only interpretable if the probe passed. A container that
# stops WITHOUT training_complete aborts rather than reporting a partial run's
# checkpoint as the result.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
EXP="${1:?experiment name, e.g. phase2_kb_widenet_v10_reconstruct}"
RUN="${2:?run_name, e.g. phase2_kb_widenet_v10_reconstruct}"
SVC="${3:?compose service}"
C="docker-${SVC}-1"
FAST=/home/werg/bgkit-ckpt-fast
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
BASE=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459
V8=/workspace/checkpoints/phase2_kb_step2999_20260829_222528_014366_run-phase2_kb_widenet_v8
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "waiting for training_complete on $C"
while true; do
  if docker logs "$C" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -q "training_complete"; then break; fi
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null || echo gone)
  if [ "$st" != "running" ]; then
    log "container state=$st without training_complete — crash or stop. NOT running"
    log "the analysis: a partial run's checkpoint is not this arm's result."
    exit 1
  fi
  sleep 120
done
log "training_complete seen"
while [ "$(pgrep -f '[t]rain.py' | wc -l)" != "0" ]; do sleep 30; done
sleep 30

CK=$(ls -dt "$FAST"/phase2_kb_step*_run-"$RUN" 2>/dev/null | head -1)
MOUNT=/workspace/checkpoints_fast
if [ -z "$CK" ]; then
  CK=$(ls -dt "$CHECKPOINT_DIR"/phase2_kb_step*_run-"$RUN" 2>/dev/null | head -1)
  MOUNT=/workspace/checkpoints
fi
if [ -z "$CK" ]; then log "no checkpoint for $RUN"; exit 1; fi
NAME=$(basename "$CK")
log "final checkpoint: $NAME (mounted at $MOUNT)"

log "STEP 1/2 probe against the base and v8, same documents"
$COMPOSE run --rm "$SVC" scripts/probe_rep_distinguishability.py \
  "+experiment=$EXP" \
  "+diag.checkpoints=[$BASE,$V8,$MOUNT/$NAME]" \
  +diag.n_samples=128 \
  "+diag.out=/workspace/checkpoints/repdist_${RUN}.json" \
  > "$FAST/repdist_${RUN}.container.log" 2>&1
log "probe exit=$?"
sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/repdist_${RUN}.container.log" \
  | grep -E "^checkpoint:|^documents:|^raw |^l1_input|^l1_in@k|^reps " || true
JSON="$CHECKPOINT_DIR/repdist_${RUN}.json"
if [ -f "$JSON" ]; then
  .venv/bin/python scripts/verdict_repdist.py "$JSON" --treatment "$RUN" \
    | tee "$FAST/verdict_${RUN}.txt"
else
  log "no probe JSON at $JSON — cannot state a verdict"
fi

log "STEP 2/2 floor / reps / ceiling on the same samples"
$COMPOSE run --rm "$SVC" scripts/eval_phase2_kb.py \
  "+experiment=$EXP" \
  "+eval.checkpoint=$MOUNT/$NAME" \
  "+eval.ablation_sweep=[none,zeroed,full_text]" \
  "+eval.free_running_arms=[zeroed]" \
  +eval.per_sample=true +eval.max_samples=192 \
  +eval.max_new_tokens=512 +eval.max_tool_calls=4 \
  "+eval.output_dir=/workspace/checkpoints/eval_reports_${RUN}" \
  > "$FAST/eval_${RUN}_sweep.container.log" 2>&1
log "sweep exit=$?"
sed -E 's/\x1b\[[0-9;]*m//g' "$FAST/eval_${RUN}_sweep.container.log" \
  | grep -E "kb_eval_ceiling" || true
log "POST-RUN $RUN DONE"
