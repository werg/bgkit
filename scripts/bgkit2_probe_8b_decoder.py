#!/usr/bin/env python
"""Feasibility probe for the LFM2.5-8B-A1B decoder (2026-09-25): load it bf16, attach LoRA the way
SpliceDecoder would, and time forward+backward on packed inputs_embeds of several lengths, with
and without gradient checkpointing. Reports peak CUDA and s/iter per length. Guarded (host floor +
CUDA cap); one line of progress per measurement.

  python scripts/bgkit2_probe_8b_decoder.py --lengths 2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from bgkit_core.host_memory_guard import arm, cap_cuda

M8 = "/models/hf/LFM2.5-8B-A1B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=M8)
    ap.add_argument("--lengths", default="2048,4096,8192,16384")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--cuda-fraction", type=float, default=0.6)
    ap.add_argument("--host-floor-gb", type=float, default=15.0)
    args = ap.parse_args()
    arm(args.host_floor_gb)
    cap_cuda(args.cuda_fraction)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    for p in lm.parameters():
        p.requires_grad_(False)
    names = sorted({n.split(".")[-1] for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)})
    print(json.dumps({"event": "loaded", "s": round(time.time() - t0, 1), "linear_names": names,
                      "cuda_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)
    # attention + conv projections + the dense layers' MLP; NOT the routed experts
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3") if n in names]
    lm = get_peft_model(lm, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules=targets,
                                       bias="none", task_type="CAUSAL_LM"))
    ntr = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(json.dumps({"event": "lora", "targets": targets, "trainable_m": round(ntr / 1e6, 1)}), flush=True)
    inner = lm.base_model.model.model
    d = lm.base_model.model.config.hidden_size
    for ckpt in (True, False):
        if ckpt:
            lm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            lm.base_model.model.gradient_checkpointing_disable()
        lm.train()
        for L in [int(x) for x in args.lengths.split(",")]:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                ts = []
                for _ in range(args.iters):
                    emb = torch.randn(1, L, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    torch.cuda.synchronize(); t = time.time()
                    h = inner(inputs_embeds=emb, use_cache=False).last_hidden_state
                    h.float().pow(2).mean().backward()
                    torch.cuda.synchronize(); ts.append(time.time() - t)
                    lm.zero_grad(set_to_none=True)
                print(json.dumps({"event": "fwd_bwd", "ckpt": ckpt, "len": L, "s_iter": round(min(ts), 3),
                                  "tok_per_s": round(L / min(ts)),
                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
            except torch.OutOfMemoryError:
                print(json.dumps({"event": "oom", "ckpt": ckpt, "len": L}), flush=True)
                torch.cuda.empty_cache()
                break


if __name__ == "__main__":
    main()
