#!/usr/bin/env bash
# Thin wrapper kept because operator runbooks and monitor prompts name it.
# The logic is generic and lives in scripts/restart-train.sh — every lesson in
# it (stop before resolving so the SIGTERM rescue save is picked up, archive
# logs before the container is removed, resolve by step number across both
# checkpoint roots, refuse to pin backwards) applies to every training
# service, not just this one run.
set -uo pipefail
exec "$(dirname "$0")/restart-train.sh" train-phase2-kb-widenet-v8 "$@"
