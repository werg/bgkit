#!/usr/bin/env bash
# Usage: scripts/run-train.sh [--no-follow] <service> [extra docker-compose args...]
# Example: scripts/run-train.sh train-ice
#          scripts/run-train.sh --no-follow train-phase1-step1
#
# Ensures a clean container start so host source changes are always picked up.
# Use --no-follow to start the container without tailing logs.
set -euo pipefail

FOLLOW=true
if [[ "${1:-}" == "--no-follow" ]]; then
    FOLLOW=false
    shift
fi

SERVICE="${1:?Usage: $0 [--no-follow] <service-name>}"
shift
COMPOSE_FILE="docker/docker-compose.yaml"

echo "==> Stopping and removing old container for ${SERVICE}..."
docker compose -f "$COMPOSE_FILE" rm -fs "$SERVICE" 2>/dev/null || true

echo "==> Starting ${SERVICE} (detached)..."
docker compose -f "$COMPOSE_FILE" up -d "$SERVICE" "$@"

echo "==> Container started: ${SERVICE}"

if [[ "$FOLLOW" == true ]]; then
    echo "==> Tailing logs (Ctrl-C to detach, container keeps running)..."
    docker compose -f "$COMPOSE_FILE" logs -f "$SERVICE"
else
    echo "==> Logs: docker compose -f ${COMPOSE_FILE} logs -f ${SERVICE}"
fi
