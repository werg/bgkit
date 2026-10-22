#!/usr/bin/env python
"""Parity + speed of the ``bgkit_triton`` experts implementation (src/bgkit2/models/moe_kernels.py)
against transformers' ``eager`` and ``grouped_mm`` experts on the REAL LFM2.5-8B-A1B.

Parity (one model, implementations switched in place, identical inputs = real text tokens):
  * per-layer hidden-state relative error and final-hidden / input-gradient relative error of
    each bf16 implementation against bf16 eager;
  * the same against an fp32-eager TRUTH (the same bf16 weights upcast), so the triton error can
    be read against the bf16 noise floor (eager-bf16 vs truth, grouped_mm vs eager).
Speed: forward+backward w.r.t. the input embeddings with gradient checkpointing (weights frozen,
as in training) per implementation and length; then a torch.profiler op table for bgkit_triton.

Guarded (host-memory floor + CUDA cap). One JSON line per measurement.

  python scripts/bgkit2_moe_parity.py --model /models/hf/LFM2.5-8B-A1B
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import time

from bgkit_core.host_memory_guard import arm, cap_cuda

arm(15.0)
cap_cuda(0.6)

import torch


def _emit(**kw) -> None:
    print(json.dumps(kw), flush=True)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _text_ids(tok, length: int) -> torch.Tensor:
    text = "\n".join(
        pathlib.Path(p).read_text()
        for p in sorted(glob.glob("/workspace/bgkit/src/bgkit2/**/*.py", recursive=True))
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < length:
        ids = ids + ids
    return torch.tensor(ids[:length])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/hf/LFM2.5-8B-A1B")
    ap.add_argument("--parity-len", type=int, default=2048)
    ap.add_argument("--speed-lengths", default="4096,16384")
    ap.add_argument("--impls", default="eager,grouped_mm,bgkit_triton,bgkit_triton_norm")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--profile-len", type=int, default=8192)
    ap.add_argument("--skip-parity", action="store_true")
    # whole-model fp32 truth upcasts the whole 8B (~33 GB more); opt-in. The per-layer isolated
    # parity always has an fp32 truth (one experts block upcast at a time).
    ap.add_argument("--with-truth", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bgkit2.models import moe_kernels
    from bgkit2.models.fused_rmsnorm import fuse_rmsnorm, unfuse_rmsnorm

    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, experts_implementation="grouped_mm"
    ).cuda()
    lm.requires_grad_(False)
    embed = lm.get_input_embeddings()

    def set_impl(name: str) -> None:
        # "bgkit_triton_norm" = bgkit_triton experts + fused RMSNorm (fused_rmsnorm.fuse_rmsnorm)
        unfuse_rmsnorm(lm)
        if name.startswith("bgkit_triton"):
            moe_kernels.enable(lm)
            if name == "bgkit_triton_norm":
                fuse_rmsnorm(lm)
        else:
            lm.config._experts_implementation = (
                name  # built-in names; same path as set_experts_impl
            )

    def fwd_bwd(ids: torch.Tensor, gout_seed: int = 0, hidden: bool = True):
        e = embed(ids[None].cuda()).detach().requires_grad_(True)
        out = lm.model(inputs_embeds=e, use_cache=False, output_hidden_states=hidden)
        h = out.last_hidden_state
        g = torch.randn(h.shape, generator=torch.Generator().manual_seed(gout_seed)).to(h.device)
        (h.float() * g).sum().backward()
        hs = [x.detach().float() for x in out.hidden_states] if hidden else None
        return h.detach().float(), e.grad.detach().float(), hs

    # ---------------------------------------------------------------- per-layer isolated parity
    # The whole-model comparison compounds bf16 noise through 24 layers and discrete top-k
    # routing flips, so its floor is large. This isolates the kernel: every MoE layer's experts
    # block gets the SAME captured real inputs (hidden, top-k index, top-k weights from one eager
    # forward) and is scored against an fp32 eager evaluation of the same bf16 weights.
    if not args.skip_parity:
        lm.eval()
        ids = _text_ids(tok, args.parity_len)
        experts = [
            m for m in lm.modules() if hasattr(m, "gate_up_proj") and hasattr(m, "is_transposed")
        ]
        caps = []
        hooks = [
            m.register_forward_pre_hook(
                lambda mod, a: caps.append(
                    (mod, a[0].detach().clone(), a[1].clone(), a[2].detach().clone())
                )
            )
            for m in experts
        ]
        set_impl("eager")
        with torch.no_grad():
            lm.model(inputs_embeds=embed(ids[None].cuda()), use_cache=False)
        for h_ in hooks:
            h_.remove()

        def run_block(mod, x, idx, w, dtype):
            xg = x.to(dtype).clone().requires_grad_(True)
            wg = w.to(dtype).clone().requires_grad_(True)
            out = mod(xg, idx, wg)
            g = torch.randn(out.shape, generator=torch.Generator().manual_seed(3)).to(out.device)
            (out.float() * g).sum().backward()
            return out.detach().float(), xg.grad.float(), wg.grad.float()

        agg = {}
        for li, (mod, x, idx, w) in enumerate(caps):
            got = {}
            for impl in args.impls.split(","):
                if impl == "bgkit_triton_norm":  # same experts path as bgkit_triton
                    continue
                set_impl(impl)
                got[impl] = run_block(mod, x, idx, w, torch.bfloat16)
            mod.float()
            lm.config._experts_implementation = "eager"
            truth = run_block(mod, x, idx, w, torch.float32)
            mod.to(torch.bfloat16)
            row = {}
            for impl, vals in got.items():
                row[impl] = [_rel(a, b) for a, b in zip(vals, truth, strict=True)]
            if "grouped_mm" in got and "bgkit_triton" in got:
                row["triton_vs_grouped_mm"] = [
                    _rel(a, b) for a, b in zip(got["bgkit_triton"], got["grouped_mm"], strict=True)
                ]
            if "eager" in got and "grouped_mm" in got:
                row["grouped_mm_vs_eager"] = [
                    _rel(a, b) for a, b in zip(got["grouped_mm"], got["eager"], strict=True)
                ]
            for k, v in row.items():
                agg.setdefault(k, []).append(v)
            _emit(
                event="layer_parity",
                layer=li,
                **{k: [round(x_, 5) for x_ in v] for k, v in row.items()},
            )
        for k, v in agg.items():
            _emit(
                event="layer_parity_max",
                which=k,
                fields=["out", "d_hidden", "d_route_w"],
                max=[round(max(r[i] for r in v), 5) for i in range(3)],
                mean=[round(statistics.mean(r[i] for r in v), 5) for i in range(3)],
            )
        del caps

    # ---------------------------------------------------------------- whole-model parity
    if not args.skip_parity:
        lm.eval()
        ids = _text_ids(tok, args.parity_len)
        res = {}
        for impl in args.impls.split(","):
            set_impl(impl)
            res[impl] = fwd_bwd(ids)
            _emit(event="parity_run", impl=impl, len=args.parity_len)
        ref = res["eager"]
        if "grouped_mm" in res and "bgkit_triton" in res:
            a, b = res["bgkit_triton"], res["grouped_mm"]
            _emit(
                event="parity_triton_vs_grouped_mm",
                rel_final_hidden=_rel(a[0], b[0]),
                rel_input_grad=_rel(a[1], b[1]),
                per_layer=[round(_rel(x, y), 5) for x, y in zip(a[2], b[2], strict=True)],
            )
        for impl, (h, g, hs) in res.items():
            per_layer = [round(_rel(a, b), 5) for a, b in zip(hs, ref[2], strict=True)]
            _emit(
                event="parity_vs_eager_bf16",
                impl=impl,
                rel_final_hidden=_rel(h, ref[0]),
                rel_input_grad=_rel(g, ref[1]),
                max_layer_rel=max(per_layer),
                per_layer=per_layer,
            )
        if args.with_truth:
            # remember per-tensor dtypes (fp32 buffers: expert_bias, inv_freq must stay fp32)
            dtypes = {
                (id(m), k): t.dtype
                for m in lm.modules()
                for k, t in list(m._parameters.items()) + list(m._buffers.items())
                if t is not None
            }
            lm.float()
            lm.config._experts_implementation = "eager"
            th, tg, ths = fwd_bwd(ids)
            for impl, (h, g, hs) in res.items():
                per_layer = [round(_rel(a, b), 5) for a, b in zip(hs, ths, strict=True)]
                _emit(
                    event="parity_vs_fp32_truth",
                    impl=impl,
                    rel_final_hidden=_rel(h, th),
                    rel_input_grad=_rel(g, tg),
                    max_layer_rel=max(per_layer),
                    per_layer=per_layer,
                )
            with torch.no_grad():
                for m in lm.modules():
                    for k, t in m._parameters.items():
                        if t is not None:
                            t.data = t.data.to(dtypes[(id(m), k)])
                    for k, t in m._buffers.items():
                        if t is not None:
                            m._buffers[k] = t.to(dtypes[(id(m), k)])
        del res
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- speed
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.train()
    for impl in args.impls.split(","):
        set_impl(impl)
        for length in [int(x) for x in args.speed_lengths.split(",")]:
            ids = _text_ids(tok, length)
            fwd_bwd(ids, hidden=False)  # warm-up (+ triton autotune)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for i in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fwd_bwd(ids, gout_seed=i, hidden=False)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            _emit(
                event="speed",
                impl=impl,
                len=length,
                s_fwd_bwd=round(med, 3),
                tok_per_s=round(length / med),
                all_s=[round(t, 3) for t in ts],
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            )

    # ---------------------------------------------------------------- profile
    if args.profile_len > 0:
        from torch.profiler import ProfilerActivity, profile

        prof_impl = args.impls.split(",")[-1]
        set_impl(prof_impl)
        ids = _text_ids(tok, args.profile_len)
        fwd_bwd(ids, hidden=False)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fwd_bwd(ids, hidden=False)
            torch.cuda.synchronize()
        _emit(event="profile", impl=prof_impl, len=args.profile_len)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=60
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
