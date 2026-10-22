#!/usr/bin/env python
"""GPU parity check for cons-compaction spines on a REAL checkpoint (guarded; not run by CI).

The trainer (``BgKIT2Core.encode_spine_slots``, from-scratch forward) and the student server
(``StudentModel._encode_spine_slot``, incremental with its node cache: the same spine grown one
fold at a time) must give the same TOP-node reps for the same spine. The unit test
tests/unit/bgkit2/models/test_spine.py checks this on a tiny CPU model; this script repeats it
on a real encoder + spine adapter for spine rows of a task store.

Guarded: needs --i-know-the-gpu-is-free, caps CUDA memory (bgkit_core.host_memory_guard) and
bounds the work with --rows / --max-depth. Run inside the GPU container:

    python scripts/bgkit2_spine_parity.py --experiment <exp> --checkpoint <dir> \\
        --store /workspace/bgkit2_data/tasks/web_openresearcher.arrow --rows 8 \\
        --i-know-the-gpu-is-free
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--atol", type=float, default=2e-2, help="bf16 autocast tolerance")
    ap.add_argument("--cuda-fraction", type=float, default=0.25)
    ap.add_argument("--i-know-the-gpu-is-free", action="store_true")
    args = ap.parse_args()
    if not args.i_know_the_gpu_is_free:
        sys.exit("refusing to touch the GPU without --i-know-the-gpu-is-free")

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from bgkit2.agent.context import compaction_leaves, compaction_payload
    from bgkit2.agent.student import StudentModel
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import cap_cuda

    cap_cuda(args.cuda_fraction)
    core = load_models(args.experiment, args.checkpoint)
    if not core.cfg2.compaction_spine.enabled:
        sys.exit("the experiment must set compaction_spine.enabled: true")
    renderer = core.chat_renderer()  # spine leaves / prompts: the encoder's tokenizer
    student = StudentModel.__new__(StudentModel)
    student.core, student.renderer = core, renderer

    if args.store.endswith(".arrow"):
        table = pa.ipc.open_file(pa.memory_map(args.store)).read_all()
    else:
        table = pq.read_table(args.store, columns=["context", "prompt"])
    worst, n = 0.0, 0
    for ctx_json, prm_json in zip(table.column("context").to_pylist(),
                                  table.column("prompt").to_pylist(), strict=True):
        if n >= args.rows or not ctx_json:
            continue
        ctx, prm = json.loads(ctx_json), json.loads(prm_json)
        sp = compaction_payload(ctx[0])
        if sp is None:
            continue
        leaves = compaction_leaves(sp)[: args.max_depth]
        prompt = prm[0]
        student._spine_cache = None
        node = None
        for leaf in leaves:  # grow the spine fold by fold, as an episode does
            node = {"compaction": 1, "child": node, "leaf": leaf}
            r_eval, _m, _ = student._encode_spine_slot(node, prompt)
        ids = [renderer.encode(x) for x in leaves]
        with torch.no_grad(), core.autocast():
            ((r_train, _mk),) = core.encode_spine_slots([ids], [renderer.encode(prompt)])
        diff = float((r_eval.float() - r_train.float()).abs().max())
        worst = max(worst, diff)
        n += 1
        print(json.dumps({"row": n, "depth": len(leaves), "max_abs_diff": diff}), flush=True)
    print(json.dumps({"rows": n, "worst_max_abs_diff": worst, "ok": worst <= args.atol}))
    sys.exit(0 if n and worst <= args.atol else 1)


if __name__ == "__main__":
    main()
