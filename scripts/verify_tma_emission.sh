#!/bin/bash
# verify_tma_emission.sh — confirm Triton 3.6 emits TMA (cp.async.bulk) on
# sm_121 for the hot DeltaNet kernels. The Blackwell autotune configs
# (num_stages=3-5) hint at TMA-friendly patterns; this script confirms the
# compiler actually emitted them.
#
# Usage:
#   bash scripts/verify_tma_emission.sh
#
# Procedure:
#   1. Stop the running training container.
#   2. Restart with TRITON_KERNEL_DUMP=1 (kernel dump dir at $TRITON_DUMP_DIR).
#   3. Wait for the first DeltaNet kernel to compile (~3-5 min after first step).
#   4. grep the dumped PTX for `cp.async.bulk` and TMA descriptor ops.
#   5. Print a summary.
#
# After verification, restart without TRITON_KERNEL_DUMP to keep the
# kernel-compile fast on subsequent runs.
#
# Requires the container to be runnable (the same compose service that's
# already in use). Saves results under $CHECKPOINT_DIR/tma-verification/.
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env so DATA_DIR / CHECKPOINT_DIR resolve.
set -a
# shellcheck disable=SC1091
source .env
set +a

OUT_DIR="${CHECKPOINT_DIR}/tma-verification"
mkdir -p "$OUT_DIR"
TS=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
SUMMARY="$OUT_DIR/tma_${TS}.md"

DC="docker compose --env-file ${PWD}/.env -f docker/docker-compose.yaml"
SERVICE="train-phase1-step3"

echo "==> Stopping current container..."
$DC stop "$SERVICE" || true
$DC rm -f "$SERVICE" || true

echo "==> Restarting with TRITON_KERNEL_DUMP=1..."
TRITON_KERNEL_DUMP=1 BGKIT_ALLOW_PEER_CUDA=1 $DC up -d "$SERVICE"

echo "==> Waiting for first train_step (kernels will compile + dump on first call)..."
until $DC logs --no-color --no-log-prefix --since 1m "$SERVICE" 2>&1 | \
    sed 's/\x1b\[[0-9;]*m//g' | grep -qF "train_step                  "; do
  sleep 30
done
echo "==> First step landed. Waiting 60s more for kernel dumps to flush..."
sleep 60

echo "==> Inspecting Triton kernel dump..."
DUMP_IN_CONTAINER="/workspace/.cache/triton-dump"
{
  echo "# TMA emission verification — $TS"
  echo
  echo "## Kernel dump location"
  echo "\`$DUMP_IN_CONTAINER\` (inside container)"
  echo
  echo "## DeltaNet PTX files found"
  $DC exec -T "$SERVICE" bash -c "find $DUMP_IN_CONTAINER -name '*.ptx' 2>/dev/null | xargs -I{} grep -l 'chunk_gated\|chunk_h\|chunk_o\|wy_fast\|delta' {} 2>/dev/null" || true
  echo
  echo "## TMA / cp.async.bulk occurrences in DeltaNet PTX"
  $DC exec -T "$SERVICE" bash -c "find $DUMP_IN_CONTAINER -name '*.ptx' 2>/dev/null | xargs -I{} bash -c 'if grep -lq chunk_gated\\\|chunk_h\\\|chunk_o\\\|wy_fast {}; then echo \"=== \$(basename {}) ===\"; grep -cE \"cp\\.async\\.bulk|tma\\.\" {} 2>/dev/null || echo 0; fi'" || echo "(no PTX dumps found — kernel compile may not have completed)"
  echo
  echo "## Bulk-async patterns (cp.async.bulk*) — first 5 across all DeltaNet kernels"
  $DC exec -T "$SERVICE" bash -c "find $DUMP_IN_CONTAINER -name '*.ptx' 2>/dev/null | xargs grep -h 'cp\\.async\\.bulk' 2>/dev/null | sort -u | head -5" || echo "(none — likely no TMA emitted)"
} | tee "$SUMMARY"

echo
echo "==> Done. Report at $SUMMARY"
echo
echo "==> To restore normal (no-dump) behavior, restart without TRITON_KERNEL_DUMP:"
echo "    $DC stop $SERVICE && $DC rm -f $SERVICE && BGKIT_ALLOW_PEER_CUDA=1 $DC up -d $SERVICE"
