#!/usr/bin/env python3
"""Benchmark Luce hidden prefill against BgKIT ReconstructionDecoder forward.

This is a forward-only training-surface benchmark. It times the tensor that the
future backward kernel must differentiate through: final hidden states plus the
shifted LM CE path for a B=1 packed splice.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.inference.luce_megakernel import LuceSingleSpliceForward, load_decoder_from_hf_model
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.utils.attention_backend import resolve_decoder_attention_implementation

DEFAULT_TEXT = """
BgKIT compresses long code and document contexts into compact survivor
embeddings. The decoder reconstructs the target stream from a splice of normal
token embeddings and continuous survivor embeddings.
"""


@dataclass
class Timing:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    loss_mean: float


def _read_text(path: str | None, text: str | None) -> str:
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    if text:
        return text
    return DEFAULT_TEXT


def _dense_ids(tokenizer, text: str, seq_len: int, device: torch.device) -> torch.Tensor:
    unit = text.strip()
    repeated = unit
    while len(tokenizer(repeated, truncation=False)["input_ids"]) < seq_len:
        repeated = repeated + "\n\n" + unit
    encoded = tokenizer(
        repeated,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"][0].to(device=device, dtype=torch.long)


def _time_forward(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    steps: int,
) -> Timing:
    times: list[float] = []
    losses: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for step in range(warmup + steps):
        start.record()
        loss = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        phase = "warmup" if step < warmup else "measure"
        print(f"{name} step={step:03d} phase={phase} loss={float(loss):.6f} ms={elapsed:.3f}")
        if step >= warmup:
            times.append(elapsed)
            losses.append(float(loss.detach().cpu()))
    return Timing(
        name=name,
        median_ms=statistics.median(times),
        mean_ms=statistics.mean(times),
        min_ms=min(times),
        max_ms=max(times),
        loss_mean=statistics.mean(losses),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument(
        "--ce-impl",
        default="chunked",
        help=(
            "LM CE implementation to use through ReconstructionDecoder "
            "(for example chunked, cce, cce_exact, or auto)."
        ),
    )
    parser.add_argument("--ce-strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    attn_impl = resolve_decoder_attention_implementation("auto", decoder_family="qwen35")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_decoder = ReconstructionDecoder(model, hidden_dim=1024, decoder_family="qwen35")
    if hasattr(hf_decoder, "set_lm_ce_impl"):
        hf_decoder.set_lm_ce_impl(args.ce_impl)
    if hasattr(hf_decoder, "set_lm_ce_strict"):
        hf_decoder.set_lm_ce_strict(args.ce_strict)
    luce_decoder = load_decoder_from_hf_model(
        model,
        tokenizer=tokenizer,
        backend="nvfp4",
        verbose=not args.quiet,
    )

    ids = _dense_ids(tokenizer, _read_text(args.text_file, args.text), args.seq_len, device)
    prefix_ids = ids[:1].contiguous()
    suffix_ids = ids[1:].contiguous()
    survivor_embeddings = torch.empty(0, 1024, dtype=torch.bfloat16, device=device)
    survivor_cu = torch.zeros(2, dtype=torch.int32, device=device)
    luce_forward = LuceSingleSpliceForward(decoder=luce_decoder)

    def hf_forward() -> torch.Tensor:
        out = hf_decoder.forward_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=[prefix_ids],
            suffix_ids=[suffix_ids],
            return_hidden_states=True,
        )
        return out.loss

    def luce_forward_ce() -> torch.Tensor:
        out = luce_forward.forward_with_single_splice_hidden(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
        )
        return hf_decoder._compute_lm_ce(
            lm_head=model.lm_head,
            hidden_states=out.hidden_states,
            token_ids_full=out.token_ids,
            attention_mask=torch.ones_like(out.loss_mask, dtype=torch.bool),
            loss_mask_full=out.loss_mask,
            chunk_size=None,
        )

    def luce_hidden_only() -> torch.Tensor:
        out = luce_forward.forward_with_single_splice_hidden(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
        )
        return out.hidden_states[0, -1, 0].float()

    print(
        json.dumps(
            {
                "env": {
                    "MEGAKERNEL_PREFILL_GRAPH": os.environ.get(
                        "MEGAKERNEL_PREFILL_GRAPH", "<default>"
                    ),
                    "MEGAKERNEL_PREFILL_MODE": os.environ.get(
                        "MEGAKERNEL_PREFILL_MODE", "<default>"
                    ),
                    "MEGAKERNEL_PREFILL_TC": os.environ.get(
                        "MEGAKERNEL_PREFILL_TC", "<default>"
                    ),
                    "MEGAKERNEL_PREFILL_TC_GATE_UP": os.environ.get(
                        "MEGAKERNEL_PREFILL_TC_GATE_UP", "<default>"
                    ),
                    "MEGAKERNEL_PREFILL_TC_PROJ": os.environ.get(
                        "MEGAKERNEL_PREFILL_TC_PROJ", "<default>"
                    ),
                },
                "model": args.model,
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "seq_len": args.seq_len,
                "steps": args.steps,
                "warmup": args.warmup,
                "ce_impl": args.ce_impl,
                "ce_strict": args.ce_strict,
            },
            sort_keys=True,
        )
    )
    hf_t = _time_forward("hf_reconstruction", hf_forward, warmup=args.warmup, steps=args.steps)
    luce_hidden_t = _time_forward(
        "luce_hidden_only",
        luce_hidden_only,
        warmup=args.warmup,
        steps=args.steps,
    )
    luce_t = _time_forward("luce_hidden_ce", luce_forward_ce, warmup=args.warmup, steps=args.steps)
    summary = {
        "env": {
            "MEGAKERNEL_PREFILL_GRAPH": os.environ.get(
                "MEGAKERNEL_PREFILL_GRAPH", "<default>"
            ),
            "MEGAKERNEL_PREFILL_MODE": os.environ.get(
                "MEGAKERNEL_PREFILL_MODE", "<default>"
            ),
            "MEGAKERNEL_PREFILL_TC": os.environ.get(
                "MEGAKERNEL_PREFILL_TC", "<default>"
            ),
            "MEGAKERNEL_PREFILL_TC_GATE_UP": os.environ.get(
                "MEGAKERNEL_PREFILL_TC_GATE_UP", "<default>"
            ),
            "MEGAKERNEL_PREFILL_TC_PROJ": os.environ.get(
                "MEGAKERNEL_PREFILL_TC_PROJ", "<default>"
            ),
        },
        "hf_reconstruction": asdict(hf_t),
        "luce_hidden_only": asdict(luce_hidden_t),
        "luce_hidden_ce": asdict(luce_t),
        "hidden_only_speedup": hf_t.median_ms / luce_hidden_t.median_ms,
        "hidden_ce_speedup": hf_t.median_ms / luce_t.median_ms,
        "loss_abs": abs(hf_t.loss_mean - luce_t.loss_mean),
        "ce_impl": args.ce_impl,
        "ce_strict": args.ce_strict,
    }
    print("summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
