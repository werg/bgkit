#!/usr/bin/env python
"""Does the pre-warmed bridge make a SECOND level carry anything?

Stage A trains `encoder.repro` to map survivors into the encoder's own input
space, and its training metric (cos_to_token ~0.7) says the bridged vectors look
like input embeddings. That is necessary, not sufficient: level 1 could read
them happily and still collapse across documents, which is exactly the
rank-collapse pathology that ended the bgkit1 lineage
(project_rep_rank_collapse_2026_08_31).

So this measures the thing that matters directly, with no decoder in the loop:
split each document's LEVEL-1 reps into disjoint halves, pool, and ask whether
half A retrieves half B (the protocol of scripts/probe_rep_distinguishability.py).
Run it with the pre-warmed bridge and with a RANDOM one; if the two scores match,
Stage A bought nothing and Stage B has no foundation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def top1(reps: torch.Tensor, mask: torch.Tensor) -> float:
    """Parity-split retrieval: does half A of a document retrieve half B?"""
    a, b = [], []
    for i in range(reps.shape[0]):
        r = reps[i][mask[i]]
        if r.shape[0] < 4:
            continue
        a.append(torch.nn.functional.normalize(r[0::2].mean(0), dim=-1))
        b.append(torch.nn.functional.normalize(r[1::2].mean(0), dim=-1))
    if len(a) < 2:
        return float("nan")
    left = torch.stack(a).float()
    right = torch.stack(b).float()
    hit = (left @ right.T).argmax(dim=-1) == torch.arange(len(a), device=left.device)
    return float(hit.float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="M1 checkpoint with a trained repro head")
    ap.add_argument("--experiment", required=True,
                    help="experiment the checkpoint was trained with: the models are built and "
                    "loaded by the trainer's code (bgkit2.training.standalone.load_models), so an "
                    "encoder-adapter checkpoint loads WITH its adapter and nothing is dropped")
    ap.add_argument("--cuda-fraction", type=float, default=0.5,
                    help="per-process CUDA cap: a runaway raises here, not a host OOM")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--files-per-doc", type=int, default=4)
    ap.add_argument("--file-tokens", type=int, default=256)
    ap.add_argument("--ratio-lower", type=float, default=0.25)
    ap.add_argument("--ratio-upper", type=float, default=0.25)
    ap.add_argument(
        "--corpus",
        default="/home/werg/bgkit-data-nvme/bgkit2/corpus/code_eval",
        help="REAL documents. The first version of this probe used random token "
        "soup and both arms scored 1.000 -- unrelated noise is trivially "
        "separable, so the test could not fail and measured nothing.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    enc = core.encoder
    if getattr(enc, "repro", None) is None:
        raise SystemExit(f"{args.checkpoint}: the experiment builds no repro head")
    trained = {k: v.clone() for k, v in enc.repro.state_dict().items()}
    print(f"loaded {args.checkpoint} ({args.experiment})")

    from bgkit2.data.token_store import TokenStore

    store = TokenStore(args.corpus)
    n = args.docs * args.files_per_doc
    if len(store) < n:
        raise SystemExit(f"{args.corpus} has {len(store)} docs, need {n}")
    rows = []
    for i in range(n):
        doc = torch.as_tensor(store[i][: args.file_tokens], dtype=torch.long)
        if doc.numel() < args.file_tokens:
            doc = torch.cat([doc, doc.new_zeros(args.file_tokens - doc.numel())])
        rows.append(doc)
    ids = torch.stack(rows).to(dev)
    mask = (ids != 0).to(dev)
    sizes = [args.files_per_doc] * args.docs
    print(f"{n} real documents from {args.corpus}")

    out: dict[str, float] = {}
    for label in ("pretrained", "random"):
        if label == "random":
            for m in enc.repro.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        # a group of files is a small tree: the trainer's tree role rule
        with torch.no_grad(), enc.role(core.tree_role(args.files_per_doc)), core.autocast():
            r = encode_two_level(
                enc, ids, mask, ratio_lower=args.ratio_lower, ratio_upper=args.ratio_upper,
                group_sizes=sizes,
            )
        out[label] = top1(r.top.reps, r.top.rep_mask)
        print(f"{label:>11} bridge: level-1 parity top-1 = {out[label]:.3f} "
              f"(chance {1 / args.docs:.3f})")
    enc.repro.load_state_dict(trained)
    out["chance"] = 1.0 / args.docs
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
