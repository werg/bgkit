#!/usr/bin/env python3
"""Benchmark Qwen3.5 decoder forward/backward speed on real text.

This is a small training jig for DeltaNet/GDR kernel work. It loads the same
Qwen decoder backbone configured for BgKIT, applies BgKIT's DeltaNet patch, and
times causal-LM forward/backward passes over tokenized text.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import textwrap
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXT = """
BgKIT compresses long code and document contexts into a compact memory that a
decoder can query while reconstructing or answering questions. The benchmark
batch repeats this text to create stable Qwen3.5 DeltaNet sequence lengths for
forward and backward timing on the target hardware.
"""


def _set_env_default(name: str, value: str) -> None:
    if os.environ.get(name) is None:
        os.environ[name] = value


def _set_toggle(name: str, value: str) -> None:
    if value == "default":
        return
    elif value == "on":
        os.environ[name] = "1"
    elif value == "off":
        os.environ[name] = "0"
    else:
        raise ValueError(f"Unknown toggle value for {name}: {value}")


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return DEFAULT_TEXT


def _dense_repeated_text(tokenizer, text: str, seq_len: int) -> str:
    unit = text.strip()
    repeated = unit
    while len(tokenizer(repeated, truncation=False)["input_ids"]) < seq_len:
        repeated = repeated + "\n\n" + unit
    return repeated


def _build_token_batch(
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    repeated = _dense_repeated_text(tokenizer, text, seq_len)
    encoded = tokenizer(
        repeated,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].repeat(batch_size, 1).to(device)
    attention_mask = encoded["attention_mask"].repeat(batch_size, 1).to(device)
    return input_ids, attention_mask


def _build_hf_batch(
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids, attention_mask = _build_token_batch(
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _build_packed_splice_batch(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, object]:
    input_ids, attention_mask = _build_token_batch(
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    if not torch.all(attention_mask):
        raise RuntimeError("packed-splice benchmark expects dense, unpadded token batches.")
    if seq_len < 2:
        raise ValueError("packed-splice benchmark requires --seq-len >= 2.")

    if hasattr(model, "_get_inner_model_and_head"):
        inner_model, _lm_head = model._get_inner_model_and_head()
    else:
        backbone = getattr(model, "backbone", model)
        inner_model = backbone.model if hasattr(backbone, "model") else backbone
    embed = inner_model.get_input_embeddings()
    hidden_dim = int(embed.weight.shape[-1])
    return {
        "survivor_embeddings": torch.empty(
            0,
            hidden_dim,
            dtype=embed.weight.dtype,
            device=device,
        ),
        "survivor_cu_seqlens": torch.zeros(batch_size + 1, dtype=torch.int32, device=device),
        "prefix_ids": [row[:1].contiguous() for row in input_ids],
        "suffix_ids": [row[1:].contiguous() for row in input_ids],
    }


def _ms_stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _run_step(
    model: torch.nn.Module,
    forward_loss: Callable[[], torch.Tensor],
) -> tuple[float, float, float, float]:
    model.zero_grad(set_to_none=True)
    start = torch.cuda.Event(enable_timing=True)
    fwd_done = torch.cuda.Event(enable_timing=True)
    bwd_done = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = forward_loss()
    fwd_done.record()
    loss.backward()
    bwd_done.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(fwd_done),
        fwd_done.elapsed_time(bwd_done),
        start.elapsed_time(bwd_done),
        float(loss.detach().cpu()),
    )


def _count_gdr_layers(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "A_log")
    )


def _active_backend(requested: str, resolved: str | None) -> str:
    if resolved is not None:
        return resolved
    if requested == "fla":
        return "fla"
    return "<unresolved>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--backend", choices=["fla", "flashqla", "auto"], default="fla")
    parser.add_argument(
        "--loss-path",
        choices=["packed-splice", "hf"],
        default="packed-splice",
        help="Use bgkit's packed decoder loss path, or raw HF causal-LM loss.",
    )
    parser.add_argument("--ce-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--ce-impl",
        choices=[
            "auto",
            "chunked",
            "liger",
            "cce",
            "cce_exact",
            "cce_kahan_full",
            "cce_kahan_full_c",
            "cce_kahan_full_e",
            "cce_kahan_full_c_full_e",
            "torch_compile",
        ],
        default=os.environ.get("BGKIT_DECODER_CE_IMPL", "auto"),
        help="Decoder LM CE implementation for the packed-splice path.",
    )
    parser.add_argument(
        "--decoder-lora",
        action="store_true",
        help="Apply bgkit's default decoder LoRA setup.",
    )
    parser.add_argument("--lora-implementation", choices=["peft", "native"], default="peft")
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--decoder-nvfp4",
        action="store_true",
        help="Convert decoder base linears to an NVFP4 backend.",
    )
    parser.add_argument(
        "--decoder-nvfp4-backend",
        choices=["te", "native-frozen"],
        default="te",
        help="NVFP4 backend; native-frozen is BgKIT's packed frozen-base reference path.",
    )
    parser.add_argument("--save-intermediates", choices=["default", "on", "off"], default="default")
    parser.add_argument(
        "--save-local-attention",
        choices=["default", "on", "off"],
        default="default",
    )
    parser.add_argument("--recompute-wy-dw", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-wy-dg-cumsum", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-dqkg-wy", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-gate-bwd", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-kkt-wu", choices=["default", "on", "off"], default="default")
    parser.add_argument("--sm121-output", choices=["default", "on", "off"], default="off")
    parser.add_argument("--dqkwg-warps", type=int, default=None)
    parser.add_argument("--dqkwg-bk", type=int, default=None)
    parser.add_argument("--dqkwg-bv", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-liger", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile one extra step after timing.",
    )
    parser.add_argument("--profile-topk", type=int, default=30)
    parser.add_argument("--profile-trace", default=None, help="Optional Chrome trace output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    os.environ["BGKIT_GDN_BACKEND"] = args.backend
    _set_toggle("FLA_GDR_SAVE_INTERMEDIATES", args.save_intermediates)
    _set_toggle("FLA_GDR_SAVE_LOCAL_ATTENTION", args.save_local_attention)
    _set_toggle("FLA_GDR_RECOMPUTE_WY_DW", args.recompute_wy_dw)
    _set_toggle("FLA_GDR_FUSE_WY_DG_CUMSUM", args.fuse_wy_dg_cumsum)
    _set_toggle("FLA_GDR_FUSE_DQKG_WY", args.fuse_dqkg_wy)
    _set_toggle("FLA_GDR_FUSE_GATE_BWD", args.fuse_gate_bwd)
    _set_toggle("FLA_GDR_FUSE_KKT_WU", args.fuse_kkt_wu)
    _set_toggle("FLA_USE_SM121_CUSTOM_KERNEL", args.sm121_output)
    if args.ce_impl not in {"auto", "chunked", "liger"}:
        _set_env_default("BGKIT_DECODER_CE_STRICT", "1")
    if args.dqkwg_warps is not None:
        os.environ["FLA_DQKWG_TL_NUM_WARPS"] = str(args.dqkwg_warps)
    if args.dqkwg_bk is not None:
        os.environ["FLA_DQKWG_TL_BK"] = str(args.dqkwg_bk)
    if args.dqkwg_bv is not None:
        os.environ["FLA_DQKWG_TL_BV"] = str(args.dqkwg_bv)
    _set_env_default("TOKENIZERS_PARALLELISM", "false")

    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
    from bgkit.utils.gdn_backend import describe_backend_environment, resolved_backend_name

    patch_gated_delta_rule_numerics(model=None)

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_gated_delta_rule_numerics(model=backbone)

    if args.loss_path == "packed-splice":
        from bgkit.models.decoder import ReconstructionDecoder

        hidden_dim = backbone.get_input_embeddings().weight.shape[1]
        model = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)
        if args.decoder_lora:
            model.apply_lora(
                {
                    "r": 16,
                    "alpha": 32,
                    "dropout": args.lora_dropout,
                    "implementation": args.lora_implementation,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                }
            )
        if args.decoder_nvfp4:
            if args.decoder_nvfp4_backend == "native-frozen":
                model.enable_native_frozen_nvfp4()
            else:
                model.enable_nvfp4()
        if hasattr(model, "set_lm_ce_impl"):
            model.set_lm_ce_impl(args.ce_impl)
    else:
        if args.decoder_lora:
            raise ValueError("--decoder-lora currently requires --loss-path packed-splice.")
        if args.decoder_nvfp4:
            raise ValueError("--decoder-nvfp4 currently requires --loss-path packed-splice.")
        model = backbone

    if args.gradient_checkpointing:
        target = model.backbone if hasattr(model, "backbone") else model
        target.gradient_checkpointing_enable()
        if hasattr(target.config, "use_cache"):
            target.config.use_cache = False
    else:
        target = model.backbone if hasattr(model, "backbone") else model
        if hasattr(target.config, "use_cache"):
            target.config.use_cache = False

    if args.use_liger:
        from bgkit.utils.liger_integration import apply_liger_to_qwen35

        apply_liger_to_qwen35(
            model,
            patch_rmsnorm=False,
            patch_swiglu=True,
            patch_rope=True,
        )
        if hasattr(model, "enable_liger_ce"):
            model.enable_liger_ce(True)

    model.train()
    text = _read_text(args)
    if args.loss_path == "packed-splice":
        batch = _build_packed_splice_batch(
            model=model,
            tokenizer=tokenizer,
            text=text,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )

        def forward_loss() -> torch.Tensor:
            return model.forward_with_single_splice(**batch, chunk_size=args.ce_chunk_size)

    else:
        batch = _build_hf_batch(
            tokenizer=tokenizer,
            text=text,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )

        def forward_loss() -> torch.Tensor:
            return model(**batch).loss

    env = describe_backend_environment()
    active_backend = _active_backend(args.backend, resolved_backend_name())
    n_gdr = _count_gdr_layers(model)
    tokens = args.batch_size * args.seq_len
    env_save_intermediates = os.environ.get("FLA_GDR_SAVE_INTERMEDIATES", "<default>")
    env_save_local_attention = os.environ.get("FLA_GDR_SAVE_LOCAL_ATTENTION", "<default>")
    env_recompute_wy_dw = os.environ.get("FLA_GDR_RECOMPUTE_WY_DW", "<default>")
    env_fuse_wy_dg_cumsum = os.environ.get("FLA_GDR_FUSE_WY_DG_CUMSUM", "<default>")
    env_fuse_dqkg_wy = os.environ.get("FLA_GDR_FUSE_DQKG_WY", "<default>")
    env_fuse_gate_bwd = os.environ.get("FLA_GDR_FUSE_GATE_BWD", "<default>")
    env_fuse_kkt_wu = os.environ.get("FLA_GDR_FUSE_KKT_WU", "<default>")
    env_sm121_output = os.environ.get("FLA_USE_SM121_CUSTOM_KERNEL", "<default>")
    env_dqkwg_warps = os.environ.get("FLA_DQKWG_TL_NUM_WARPS", "<default>")
    env_dqkwg_bk = os.environ.get("FLA_DQKWG_TL_BK", "<default>")
    env_dqkwg_bv = os.environ.get("FLA_DQKWG_TL_BV", "<default>")
    env_ce_strict = os.environ.get("BGKIT_DECODER_CE_STRICT", "<default>")
    print(
        textwrap.dedent(
            f"""
            qwen_decoder_gdr_benchmark
              model={args.model}
              device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}
              dtype={args.dtype} batch={args.batch_size} seq_len={args.seq_len} tokens/step={tokens}
              loss_path={args.loss_path} ce_impl={args.ce_impl} ce_chunk_size={args.ce_chunk_size}
              requested_backend={args.backend} active_backend={active_backend}
              gdr_layers={n_gdr}
              FLA_GDR_SAVE_INTERMEDIATES={env_save_intermediates}
              FLA_GDR_SAVE_LOCAL_ATTENTION={env_save_local_attention}
              FLA_GDR_RECOMPUTE_WY_DW={env_recompute_wy_dw}
              FLA_GDR_FUSE_WY_DG_CUMSUM={env_fuse_wy_dg_cumsum}
              FLA_GDR_FUSE_DQKG_WY={env_fuse_dqkg_wy}
              FLA_GDR_FUSE_GATE_BWD={env_fuse_gate_bwd}
              FLA_GDR_FUSE_KKT_WU={env_fuse_kkt_wu}
              FLA_USE_SM121_CUSTOM_KERNEL={env_sm121_output}
              FLA_DQKWG_TL_NUM_WARPS={env_dqkwg_warps}
              FLA_DQKWG_TL_BK={env_dqkwg_bk}
              FLA_DQKWG_TL_BV={env_dqkwg_bv}
              BGKIT_DECODER_CE_STRICT={env_ce_strict}
              gradient_checkpointing={args.gradient_checkpointing} use_liger={args.use_liger}
              decoder_lora={args.decoder_lora} lora_impl={args.lora_implementation}
              lora_dropout={args.lora_dropout}
              decoder_nvfp4={args.decoder_nvfp4} decoder_nvfp4_backend={args.decoder_nvfp4_backend}
            """
        ).strip()
    )
    print(f"backend_env={json.dumps(env, sort_keys=True)}")

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        fwd, bwd, total, loss = _run_step(model, forward_loss)
        if step >= args.warmup:
            fwd_ms.append(fwd)
            bwd_ms.append(bwd)
            total_ms.append(total)
            losses.append(loss)
        print(
            f"step={step:03d} phase={'warmup' if step < args.warmup else 'measure'} "
            f"loss={loss:.4f} fwd_ms={fwd:.3f} bwd_ms={bwd:.3f} total_ms={total:.3f}",
            flush=True,
        )

    summary = {
        "model": args.model,
        "backend": args.backend,
        "active_backend": _active_backend(args.backend, resolved_backend_name()),
        "dtype": args.dtype,
        "loss_path": args.loss_path,
        "ce_impl": args.ce_impl,
        "ce_chunk_size": args.ce_chunk_size,
        "decoder_lora": args.decoder_lora,
        "lora_implementation": args.lora_implementation,
        "lora_dropout": args.lora_dropout,
        "decoder_nvfp4": args.decoder_nvfp4,
        "decoder_nvfp4_backend": args.decoder_nvfp4_backend,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": tokens,
        "steps": args.steps,
        "warmup": args.warmup,
        "fwd_ms": _ms_stats(fwd_ms),
        "bwd_ms": _ms_stats(bwd_ms),
        "total_ms": _ms_stats(total_ms),
        "tokens_per_second": tokens / (statistics.median(total_ms) / 1000.0),
        "loss_mean": statistics.mean(losses),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gdr_layers": n_gdr,
        "env": {
            "FLA_GDR_SAVE_INTERMEDIATES": env_save_intermediates,
            "FLA_GDR_SAVE_LOCAL_ATTENTION": env_save_local_attention,
            "FLA_GDR_RECOMPUTE_WY_DW": env_recompute_wy_dw,
            "FLA_GDR_FUSE_WY_DG_CUMSUM": env_fuse_wy_dg_cumsum,
            "FLA_GDR_FUSE_DQKG_WY": env_fuse_dqkg_wy,
            "FLA_GDR_FUSE_GATE_BWD": env_fuse_gate_bwd,
            "FLA_GDR_FUSE_KKT_WU": env_fuse_kkt_wu,
            "FLA_USE_SM121_CUSTOM_KERNEL": env_sm121_output,
            "FLA_DQKWG_TL_NUM_WARPS": env_dqkwg_warps,
            "FLA_DQKWG_TL_BK": env_dqkwg_bk,
            "FLA_DQKWG_TL_BV": env_dqkwg_bv,
            "BGKIT_DECODER_CE_STRICT": env_ce_strict,
        },
    }
    print("summary=" + json.dumps(summary, sort_keys=True))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        print("profile_start=1", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            fwd, bwd, total, loss = _run_step(model, forward_loss)
        print(
            f"profiled_step loss={loss:.4f} fwd_ms={fwd:.3f} "
            f"bwd_ms={bwd:.3f} total_ms={total:.3f}"
        )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.profile_topk))
        if args.profile_trace:
            prof.export_chrome_trace(args.profile_trace)
            print(f"profile_trace={args.profile_trace}")


if __name__ == "__main__":
    main()
