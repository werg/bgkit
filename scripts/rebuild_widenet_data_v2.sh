#!/usr/bin/env bash
# Data-v2 rebuild for the capability-packaging Family A+B mix: all four flat
# datasets regenerated with gold_span_json (v5 span-level supervision) and the
# error-centered lognav windows. Run from the repo root with .env present and
# NO trainer/eval reading these artifacts (mmap files are replaced via
# tmp+rename, but trajectory parquets are rewritten in place).
#
#   scripts/rebuild_widenet_data_v2.sh                      # all four
#   DATASETS="fileneedle grepset" scripts/rebuild_widenet_data_v2.sh   # subset
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
NVME=/home/werg/bgkit-data-nvme/capability_packaging
PY=.venv/bin/python
NICE="ionice -c3 nice -n 15"
DATASETS="${DATASETS:-lognav fileneedle grepset swerecall}"

want() { case " $DATASETS " in *" $1 "*) return 0;; *) return 1;; esac; }
tree() { $PY scripts/build_browse_tree.py --dataset "$1" --phase2-dir "$DATA_DIR/mmap/phase2/$1" --output-dir "$DATA_DIR/browse_trees" --leaf-cap 1000 | tail -1; }

if want lognav; then
echo "== lognav (error-centered windows + gold spans)"
$NICE $PY scripts/build_lognav_phase2.py --loghub-dir "$NVME/loghub" \
  --mmap-out "$DATA_DIR/mmap/phase2" --traj-out "$DATA_DIR/trajectories" 2>&1 | grep -E "^(mmap|trajectories|by split)"
tree lognav
fi

if want fileneedle; then
echo "== fileneedle"
$NICE $PY scripts/build_fileneedle_phase2.py --repos-dir "$DATA_DIR/repos" \
  --mmap-out "$DATA_DIR/mmap/phase2" --traj-out "$DATA_DIR/trajectories" 2>&1 | grep -E "^(mmap|by split)"
tree fileneedle
fi

if want grepset; then
echo "== grepset"
$NICE $PY scripts/build_grepset_phase2.py --repos-dir "$DATA_DIR/repos" \
  --mmap-out "$DATA_DIR/mmap/phase2" --traj-out "$DATA_DIR/trajectories" 2>&1 | grep -E "^(mmap|by split)"
tree grepset
fi

if want swerecall; then
echo "== swerecall"
$NICE $PY scripts/build_swerecall_phase2.py --swezero-dir "$NVME/trajectories_ext/swe_zero" \
  --mmap-out "$DATA_DIR/mmap/phase2" --traj-out "$DATA_DIR/trajectories" 2>&1 | grep -E "^(mmap|by split)"
tree swerecall
fi

echo "== gold-answer length tails (needles must be human-scale lines)"
$PY - <<'EOF0'
import os, sys, pyarrow.parquet as pq
d = os.environ["DATA_DIR"] + "/trajectories"; bad = 0
for ds in ("lognav", "fileneedle", "grepset", "swerecall"):
    col = pq.read_table(f"{d}/{ds}.parquet", columns=["gold_answer"]).column("gold_answer").to_pylist()
    lens = sorted(len(a) for a in col); n = len(lens)
    over = sum(1 for L in lens if L > 2000)
    print(f"{ds}: n={n} median={lens[n//2]} p99={lens[int(0.99*(n-1))]} max={lens[-1]} >2000chars={over}")
    bad += over
if bad:
    sys.exit("FATAL: whole-file gold answers present (minified sources leaked through)")
EOF0

echo "== gold_span coverage"
$PY - <<'EOF'
import os, pyarrow.parquet as pq
d = os.environ["DATA_DIR"] + "/trajectories"
for ds in ("lognav", "fileneedle", "grepset", "swerecall"):
    t = pq.read_table(f"{d}/{ds}.parquet", columns=["gold_span_json"])
    col = t.column("gold_span_json").to_pylist()
    n = len(col); has = sum(1 for x in col if x)
    print(f"{ds}: {has}/{n} rows with gold_span ({100*has/max(n,1):.0f}%)")
EOF
echo "== trajectory -> browse-tree coverage (every retrieval id must be a tree article)"
$PY - <<'EOF2'
import json, os, sys, pyarrow.parquet as pq
from bgkit.data.browse_tree import BrowseTree
d = os.environ["DATA_DIR"]; bad = 0
for ds in ("lognav", "fileneedle", "grepset", "swerecall"):
    t = pq.read_table(f"{d}/trajectories/{ds}.parquet", columns=["trajectory_json"]).to_pylist()
    ids = [json.loads(r["trajectory_json"])[0]["args"]["ids"][0] for r in t]
    tree = BrowseTree.load(f"{d}/browse_trees/{ds}.parquet", dataset=ds)
    miss = sum(1 for i in ids if not (i in tree and tree.get(i).is_article))
    print(f"{ds}: {miss}/{len(ids)} trajectories reference an article missing from the tree")
    bad += miss
if bad:
    sys.exit("FATAL: browse-tree coverage gap (flat --leaf-cap silently truncates; raise it)")
EOF2
echo "DATA-V2 DONE"
