#!/usr/bin/env bash
# Restart v8 from its LATEST checkpoint. Use this instead of calling
# run-train.sh directly.
#
# WHY THIS EXISTS. The compose service carries a hard-coded
# +resume_checkpoint= pin. That pin is correct only until the next checkpoint
# is written, after which every restart silently rewinds training to it. On
# 2026-08-28 a verification chain restarted v8 while the pin still said
# step286; v8 had reached step 701, so it rewound ~400 steps. Nothing errored —
# it simply retrained ground it had already covered.
#
# The alternative, dropping the pin and relying on auto-resume, is worse: that
# cold-started a run from the Phase-1 base and discarded 1258 steps, because
# auto-resume does not match on run_name.
#
# So: resolve the latest checkpoint, rewrite the pin, then start.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
COMPOSE="docker compose --env-file .env -f docker/docker-compose.yaml"
SVC=train-phase2-kb-widenet-v8

CK=$(ls -1dt /mnt/external/bgkit-checkpoints/*widenet_v8 2>/dev/null | head -1)
if [ -z "$CK" ]; then
  echo "no v8 checkpoint found — starting fresh from the configured base"
  scripts/run-train.sh --no-follow "$SVC"
  exit 0
fi
NAME=$(basename "$CK")

# Sanity: never pin to a checkpoint OLDER than one already pinned, or a
# restart could rewind training even with this script in the loop.
CUR=$(grep -oE "resume_checkpoint=/workspace/checkpoints/phase2_kb_step[0-9]+[^\"]*widenet_v8" \
      docker/docker-compose.yaml | sed 's/.*step\([0-9]*\)_.*/\1/' | head -1)
NEW=$(echo "$NAME" | sed 's/.*step\([0-9]*\)_.*/\1/')
if [ -n "${CUR:-}" ] && [ "${NEW:-0}" -lt "${CUR:-0}" ] 2>/dev/null; then
  echo "REFUSING: latest checkpoint step$NEW is OLDER than the current pin step$CUR"
  echo "  that would rewind training; investigate before restarting"
  exit 1
fi

python3 - "$NAME" <<'PY'
import re, sys
from pathlib import Path
name = sys.argv[1]
p = Path("docker/docker-compose.yaml")
s = p.read_text()
s2 = re.sub(r'(\+resume_checkpoint=/workspace/checkpoints/)phase2_kb_step\d+_[^"]*widenet_v8',
            r'\g<1>' + name, s)
p.write_text(s2)
print(f"pin -> {name}")
PY

echo "==> starting $SVC"
scripts/run-train.sh --no-follow "$SVC"
