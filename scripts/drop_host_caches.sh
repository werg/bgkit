#!/bin/bash
# drop_host_caches.sh — periodically free reclaimable kernel page cache so the
# bgkit cuda-mem-guard's "peer CUDA usage" estimate stays low.
#
# On DGX Spark unified memory, kernel page cache from data file reads counts
# against the per-process CUDA memory budget at container startup time. Over
# a long training run, page cache can grow to 25+ GB, triggering the auto-
# shrink in ``scripts/train.py`` and reducing the container's effective
# allocation cap. Dropping reclaimable cache periodically prevents this.
#
# **Requires sudo (writes to /proc/sys/vm/drop_caches).** Add to root crontab:
#
#     # /etc/cron.d/bgkit-drop-caches
#     # m h dom mon dow user command
#     0 */6 * * * root /home/werg/bgkit/scripts/drop_host_caches.sh
#
# Logs to /var/log/bgkit-drop-caches.log so you can confirm runs.
#
# Safe to run while training is in progress: only reclaimable pages are
# freed; in-use pages (model weights, allocator-held memory) are untouched.
set -euo pipefail

LOG=/var/log/bgkit-drop-caches.log
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

before=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
sync
echo 3 > /proc/sys/vm/drop_caches
after=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)

echo "$TS  freed=$((after - before)) MiB  available_now=$after MiB" >> "$LOG"
