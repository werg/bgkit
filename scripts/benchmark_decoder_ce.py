#!/usr/bin/env python3
"""Benchmark decoder LM CE implementations on the frozen packed-splice contract."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from transformers import AutoModelForCausalLM

from bgkit.models.decoder import LM_CE_IMPLS, ReconstructionDecoder
from bgkit.utils.attention_backend import resolve_decoder_attention_implementation


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _make_splice_targets(
    *,
    batch_size: int,
    seq_len: int,
    survivor_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len = seq_len + survivor_len
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, total_len),
        device=device,
        dtype=torch.long,
    )
    loss_mask = torch.ones((batch_size, total_len), device=device, dtype=torch.bool)
    # Synthetic packed-splice layout: one prefix token, survivor embeddings,
    # then token suffix. Prefix/survivor positions are not CE targets.
    protected = min(total_len, 1 + max(0, survivor_len))
    loss_mask[:, :protected] = False
    attention_mask = torch.ones_like(loss_mask)
    return token_ids, attention_mask, loss_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--survivor-len", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--ce-impl", choices=sorted(LM_CE_IMPLS), default="cce")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.survivor_len < 0:
        raise ValueError("--survivor-len must be non-negative")

    torch.manual_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    attn_impl = resolve_decoder_attention_implementation("auto", decoder_family="qwen35")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).to(device)
    model.requires_grad_(False)
    decoder = ReconstructionDecoder(model, hidden_dim=1024, decoder_family="qwen35")
    decoder.set_lm_ce_impl(args.ce_impl)
    decoder.set_lm_ce_strict(bool(args.strict))
    lm_head = model.lm_head
    hidden_size = int(lm_head.weight.shape[1])
    vocab_size = int(lm_head.weight.shape[0])
    total_len = int(args.seq_len) + int(args.survivor_len)
    target_tokens = int(args.batch_size) * max(0, int(args.seq_len) - 1)
    token_ids, attention_mask, loss_mask = _make_splice_targets(
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        survivor_len=int(args.survivor_len),
        vocab_size=vocab_size,
        device=device,
    )

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(int(args.warmup) + int(args.steps)):
        hidden = torch.randn(
            int(args.batch_size),
            total_len,
            hidden_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        start = torch.cuda.Event(enable_timing=True)
        fwd_done = torch.cuda.Event(enable_timing=True)
        bwd_done = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = decoder._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids,
            attention_mask=attention_mask,
            loss_mask_full=loss_mask,
            chunk_size=args.chunk_size,
        )
        fwd_done.record()
        loss.backward()
        bwd_done.record()
        torch.cuda.synchronize()
        phase = "warmup" if step < int(args.warmup) else "measure"
        elapsed_fwd = start.elapsed_time(fwd_done)
        elapsed_bwd = fwd_done.elapsed_time(bwd_done)
        elapsed_total = start.elapsed_time(bwd_done)
        loss_value = float(loss.detach())
        print(
            f"step={step:03d} phase={phase} loss={loss_value:.6f} "
            f"fwd_ms={elapsed_fwd:.3f} bwd_ms={elapsed_bwd:.3f} "
            f"total_ms={elapsed_total:.3f}",
            flush=True,
        )
        if step >= int(args.warmup):
            fwd_ms.append(elapsed_fwd)
            bwd_ms.append(elapsed_bwd)
            total_ms.append(elapsed_total)
            losses.append(loss_value)

    summary = {
        "model": args.model,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "survivor_len": int(args.survivor_len),
        "target_tokens": target_tokens,
        "ce_impl": args.ce_impl,
        "chunk_size": args.chunk_size,
        "strict": bool(args.strict),
        "warmup": int(args.warmup),
        "steps": int(args.steps),
        "fwd_ms": _stats(fwd_ms),
        "bwd_ms": _stats(bwd_ms),
        "total_ms": _stats(total_ms),
        "target_tokens_per_second": target_tokens
        / (statistics.median(total_ms) / 1000.0),
        "loss_mean": statistics.mean(losses),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print("summary=" + json.dumps(summary, sort_keys=True))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
