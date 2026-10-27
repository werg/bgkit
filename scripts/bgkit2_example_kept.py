#!/usr/bin/env python
"""One BABILong item, encoded as in the bgkit arm of scripts/bgkit2_bench.py: which tokens the
encoder keeps around the bAbI facts that mention the person asked about. Prints, for every such
fact sentence, the fact and the kept tokens inside it -- the paper's worked example of what the
kept-tokens arm sees (the vectors arm sees encoder vectors at the same positions).

    python scripts/bgkit2_example_kept.py --experiment bgkit2_s2_showcase --checkpoint <ckpt> \\
        --items .../babilong_16k.jsonl --id babilong:16k:qa1:38 --budget 1024
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NAMES = ("Mary", "John", "Sandra", "Daniel")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--slot-tokens", type=int, default=2048)
    ap.add_argument("--max-ratio", type=float, default=64.0)
    args = ap.parse_args()

    from bgkit2_bench import pack_chunks
    from hydra import compose, initialize_config_dir

    from bgkit2.agent.student import StudentModel, slot_ratio

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    student = StudentModel(cfg, Path(args.checkpoint), torch.device("cuda"))
    tok = student.renderer.ctx_tok
    it = next(d for d in map(json.loads, open(args.items)) if d["id"] == args.id)
    chunks, lens = pack_chunks(tok, it["segments"], args.slot_tokens)
    ratio = min(max(sum(lens) / args.budget, 1.0), args.max_ratio)
    person = next(n for n in NAMES if n in it["question"])
    fact_re = re.compile(rf"\b{person}\b[^.\n]*\.")
    n_kept = 0
    print(json.dumps({"question": it["question"], "answer": it["answers"], "context_tokens": sum(lens),
                      "chunks": len(chunks), "ratio": round(ratio, 1)}))
    for c_i, (text, n) in enumerate(zip(chunks, lens, strict=True)):
        ids = student._encode_text_slot(text)
        p = student.renderer.encode_ctx(it["question"])
        n_reps = max(4, round(n / ratio))
        with student.core.autocast():
            out = student.encoder(ids[None].to(student.device),
                                  torch.ones(1, ids.shape[0], dtype=torch.long, device=student.device),
                                  ratio=torch.tensor([slot_ratio(n_reps, int(ids.shape[0]))], device=student.device),
                                  prompt_ids=p[None].to(student.device),
                                  prompt_mask=torch.ones(1, p.shape[0], dtype=torch.long, device=student.device))
        kept = set(out.positions[0][out.rep_mask[0]].tolist())
        n_kept += len(kept)
        # character span of every token, to place the fact sentences on token positions
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == ids.tolist()
        for m in fact_re.finditer(text):
            toks = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < m.end() and b > m.start()]
            kept_here = [tok.decode([enc["input_ids"][j]]) for j in toks if j in kept]
            print(json.dumps({"chunk": c_i, "fact": m.group(0), "tokens": len(toks), "kept": kept_here}))
    print(json.dumps({"kept_total": n_kept}))


if __name__ == "__main__":
    main()
