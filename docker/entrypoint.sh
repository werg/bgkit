#!/usr/bin/env bash
set -euo pipefail

# Fail fast if bind mounts are missing or empty
for required in src/bgkit/__init__.py scripts/train.py configs/config.yaml; do
    if [ ! -f "/workspace/bgkit/$required" ]; then
        echo "FATAL: /workspace/bgkit/$required not found." >&2
        echo "Are docker-compose bind mounts configured?" >&2
        exit 1
    fi
done

# Print source hash so logs always show which code is running
hash=$(find /workspace/bgkit/src -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
echo "bgkit source hash: $hash"
python -c "import bgkit; print(f'bgkit {bgkit.__version__}')"

# If the first arg is a .py script, run it with python.
# Otherwise exec as-is (supports: bash, python -c "...", etc.)
if [[ "${1-}" == *.py ]]; then
    exec python "$@"
else
    exec "$@"
fi
