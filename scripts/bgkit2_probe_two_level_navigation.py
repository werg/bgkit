#!/usr/bin/env python
"""Can level-1 reps answer WHICH FILE — the question the tree exists for?

The continuation probe found level-1 reps worth ~1.79 nats over a zeroed floor,
but a RANDOM bridge scored ~1.69 of that. Two readings: either the backbone is
robust to the input distribution (so the repro head is a refinement, not a
prerequisite), or predicting a file's continuation leans on generic code
statistics that survive any projection.

This separates them. The repo's real files are encoded two-level, and the target
is the ANSWER PATH to a navigation question about that repo -- which cannot be
produced from generic statistics, only from knowing which file holds what. Same
three arms, same zeroed floor.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygit2

from bgkit2.data.repo_corpus import blob_text, is_test_path, python_repo_candidates


def repo_files(repo: pygit2.Repository, want: str, limit: int, max_bytes: int) -> list[str]:
    """The evidence file first, then other modules, as SOURCE TEXT."""
    try:
        tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
    except Exception:
        return []
    paths, stack = [], [("", tree)]
    while stack and len(paths) < limit * 4:
        prefix, node = stack.pop()
        for e in node:
            p = f"{prefix}{e.name}"
            if e.type_str == "tree":
                if not e.name.startswith(".") and e.name != "__pycache__":
                    stack.append((f"{p}/", repo[e.id]))
            elif p.endswith(".py") and not is_test_path(p):
                paths.append(p)
    if want not in paths:
        return []
    ordered = [want] + [p for p in paths if p != want]
    out = []
    for p in ordered[:limit]:
        t = blob_text(repo, tree, p, max_bytes=max_bytes)
        if t:
            try:
                ast.parse(t)
            except (SyntaxError, ValueError, RecursionError):
                continue
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True,
                    help="experiment the checkpoint was trained with: the models are built and "
                    "loaded by the trainer's code (bgkit2.training.standalone.load_models), so an "
                    "encoder-adapter checkpoint loads WITH its adapter and nothing is dropped")
    ap.add_argument("--cuda-fraction", type=float, default=0.5,
                    help="per-process CUDA cap: a runaway raises here, not a host OOM")
    ap.add_argument(
        "--tasks",
        default="/home/werg/bgkit-data-nvme/bgkit2/tasks/repo_qa_tasks_v3.jsonl",
    )
    ap.add_argument("--n-tasks", type=int, default=24)
    ap.add_argument("--files-per-repo", type=int, default=6)
    ap.add_argument("--file-tokens", type=int, default=384)
    ap.add_argument("--ratio-lower", type=float, default=0.25)
    ap.add_argument("--ratio-upper", type=float, default=0.25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bgkit2.models.multilevel import encode_two_level
    from bgkit2.training.standalone import load_models

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from bgkit_core.host_memory_guard import arm, cap_cuda

        arm(10.0)
        cap_cuda(args.cuda_fraction)
    # the run's own models, loaded strictly (adapters included) by the trainer's
    # code -- the old hand-built encoder + strict=False dropped adapter tensors
    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc, dec = core.encoder, core.decoder
    trained = {k: v.clone() for k, v in enc.repro.state_dict().items()}
    tok = AutoTokenizer.from_pretrained(core.cfg2.model.decoder_id)

    cands = {c.full_name: c.path for c in python_repo_candidates(os.environ["DATA_DIR"])}
    samples = []
    with Path(args.tasks).open() as fh:
        for ln in fh:
            d = json.loads(ln)
            ev = d.get("evidence_paths") or ""
            if isinstance(ev, str) and ev.startswith("["):
                try:
                    ev = ast.literal_eval(ev)
                except (ValueError, SyntaxError):
                    continue
            paths = [str(x) for x in (ev if isinstance(ev, list) else [ev]) if x]
            if not paths or d.get("repo") not in cands:
                continue
            # Drop questions that already NAME the answer path: 58.3% of
            # repo_qa questions read "the signature of X in <path>", so the
            # answer is in the prompt and the row measures copying rather than
            # retrieval. The first run of this probe did not filter them.
            base = paths[0].rsplit("/", 1)[-1]
            q0 = d["question"]
            if paths[0] in q0 or base in q0 or base.rsplit(".", 1)[0] in q0:
                continue
            texts = repo_files(pygit2.Repository(cands[d["repo"]]), paths[0],
                               args.files_per_repo, 200_000)
            if len(texts) < 2:
                continue
            samples.append((d["question"], paths[0], texts))
            if len(samples) >= args.n_tasks:
                break
    print(f"{len(samples)} navigation samples")
    if not samples:
        raise SystemExit("no usable samples")

    enc_tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-Encoder-350M")
    flat, sizes, prefixes, suffixes = [], [], [], []
    for q, want, texts in samples:
        for t in texts:
            v = enc_tok(t, add_special_tokens=False)["input_ids"][: args.file_tokens]
            v = v + [0] * (args.file_tokens - len(v))
            flat.append(torch.tensor(v, dtype=torch.long))
        sizes.append(len(texts))
        prefixes.append(torch.tensor(
            tok(f"{q}\nWhich file? ", add_special_tokens=False)["input_ids"], dtype=torch.long))
        suffixes.append(torch.tensor(
            tok(want, add_special_tokens=False)["input_ids"], dtype=torch.long))
    ids = torch.stack(flat).to(dev)
    mask = (ids != 0).to(dev)
    prefixes = [p.to(dev) for p in prefixes]
    suffixes = [s.to(dev) for s in suffixes]

    def ce(reps, rep_mask) -> float:
        with torch.no_grad(), core.autocast():
            return float(dec.forward(dec.build_batch(prefixes, reps, rep_mask, suffixes)))

    out: dict[str, float] = {}
    for label in ("pretrained", "random"):
        if label == "random":
            for m in enc.repro.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        # a group of files is a small tree: the trainer's tree role rule
        with torch.no_grad(), enc.role(core.tree_role(max(sizes))), core.autocast():
            r = encode_two_level(enc, ids, mask, ratio_lower=args.ratio_lower,
                                 ratio_upper=args.ratio_upper, group_sizes=sizes)
        out[label] = ce(r.top.reps, r.top.rep_mask)
        if label == "pretrained":
            out["zeroed"] = ce(torch.zeros_like(r.top.reps), r.top.rep_mask)
        print(f"{label:>11}: CE on the answer path {out[label]:.4f}")
    enc.repro.load_state_dict(trained)
    print(f"{'zeroed':>11}: CE on the answer path {out['zeroed']:.4f}   (floor)")
    out["gain_pretrained"] = out["zeroed"] - out["pretrained"]
    out["gain_random"] = out["zeroed"] - out["random"]
    print(f"gain over floor — pretrained {out['gain_pretrained']:+.4f}, "
          f"random {out['gain_random']:+.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
