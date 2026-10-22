#!/usr/bin/env python3
"""Check packed Qwen3.5 layerwise-split semantics against per-sample split."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _ids(tokenizer, *, length: int, device: torch.device, offset: int) -> torch.Tensor:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 512:
        vocab_size = 151936
    ids = (torch.arange(length, device=device, dtype=torch.long) * 131 + offset) % (
        vocab_size - 512
    )
    return ids + 512


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().detach().cpu())


def _rms_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).square().mean().sqrt().detach().cpu())


def _relative_rms_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.float().square().mean().sqrt().detach().cpu())
    if denom == 0.0:
        return 0.0
    return _rms_diff(a, b) / denom


@contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _sync_time(fn, *, steps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(max(steps, 1))


def _sync_fwd_bwd_time(fn, *, steps: int) -> float:
    for _ in range(max(1, steps)):
        loss = fn()
        loss.backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        loss = fn()
        loss.backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(max(steps, 1))


def _load_decoder(args):
    from bgkit.models.decoder import ReconstructionDecoder
    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

    patch_gated_delta_rule_numerics(model=None)
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
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
    backbone.config.use_cache = True
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    hidden_dim = int(backbone.get_input_embeddings().weight.shape[1])
    decoder = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)
    decoder.eval()
    decoder._use_liger_ce = False
    decoder._lm_ce_impl = "chunked"
    return tokenizer, decoder, device, dtype, hidden_dim


def _make_batch(tokenizer, args, *, device: torch.device, dtype: torch.dtype, hidden_dim: int):
    prefix_ids = [
        _ids(
            tokenizer,
            length=args.prefix_len,
            device=device,
            offset=19 + sample_idx * 7919,
        )
        for sample_idx in range(args.batch)
    ]
    suffix_ids = [
        _ids(
            tokenizer,
            length=args.suffix_len,
            device=device,
            offset=911 + sample_idx * 3571,
        )
        for sample_idx in range(args.batch)
    ]
    survivor_cu_values = [0]
    for _sample_idx in range(args.batch):
        survivor_cu_values.append(survivor_cu_values[-1] + args.survivor_len)
    survivor_cu = torch.tensor(survivor_cu_values, dtype=torch.int32, device=device)
    loss_parts = []
    for _sample_idx in range(args.batch):
        loss_parts.append(
            torch.cat(
                [
                    torch.zeros(
                        args.prefix_len + args.survivor_len,
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.ones(args.suffix_len, dtype=torch.bool, device=device),
                ],
                dim=0,
            )
        )
    loss_mask = torch.cat(loss_parts, dim=0)
    base_survivors = (
        torch.randn(
            args.batch * args.survivor_len,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )
    return prefix_ids, suffix_ids, survivor_cu, loss_mask, base_survivors


def _reference_per_sample_loss(
    decoder,
    *,
    survivor_embeddings: torch.Tensor,
    survivor_len: int,
    prefix_ids: list[torch.Tensor],
    suffix_ids: list[torch.Tensor],
    ce_chunk_size: int,
) -> torch.Tensor:
    losses = []
    for sample_idx, (prefix, suffix) in enumerate(zip(prefix_ids, suffix_ids, strict=True)):
        start = sample_idx * survivor_len
        end = start + survivor_len
        sample_survivors = survivor_embeddings[start:end]
        sample_loss_mask = torch.cat(
            [
                torch.zeros(
                    prefix.shape[0] + survivor_len,
                    dtype=torch.bool,
                    device=prefix.device,
                ),
                torch.ones(suffix.shape[0], dtype=torch.bool, device=prefix.device),
            ],
            dim=0,
        )
        sample_cu = torch.tensor([0, survivor_len], dtype=torch.int32, device=prefix.device)
        losses.append(
            decoder.forward_with_single_splice(
                survivor_embeddings=sample_survivors,
                survivor_cu_seqlens=sample_cu,
                survivor_cu_seqlens_cpu=[0, survivor_len],
                prefix_ids=[prefix],
                suffix_ids=[suffix],
                loss_mask=sample_loss_mask,
                chunk_size=ce_chunk_size,
                return_hidden_states=False,
            )
        )
    return torch.stack(losses).mean()


def _production_split_loss(
    decoder,
    *,
    survivor_embeddings: torch.Tensor,
    survivor_cu: torch.Tensor,
    prefix_ids: list[torch.Tensor],
    suffix_ids: list[torch.Tensor],
    loss_mask: torch.Tensor,
    ce_chunk_size: int,
    packed_deltanet: bool,
) -> torch.Tensor:
    with (
        _env("BGKIT_QWEN35_LAYERWISE_SPLIT", "1"),
        _env("BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET", "1" if packed_deltanet else "0"),
    ):
        return decoder.forward_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            survivor_cu_seqlens_cpu=survivor_cu.detach().cpu().tolist(),
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=loss_mask,
            chunk_size=ce_chunk_size,
            return_hidden_states=False,
        )


def _reference_per_sample_loss_from_base(
    decoder,
    *,
    base_survivors: torch.Tensor,
    survivor_len: int,
    prefix_ids: list[torch.Tensor],
    suffix_ids: list[torch.Tensor],
    ce_chunk_size: int,
) -> torch.Tensor:
    survivors = base_survivors.detach().clone().requires_grad_(True)
    return _reference_per_sample_loss(
        decoder,
        survivor_embeddings=survivors,
        survivor_len=survivor_len,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        ce_chunk_size=ce_chunk_size,
    )


def _production_split_loss_from_base(
    decoder,
    *,
    base_survivors: torch.Tensor,
    survivor_cu: torch.Tensor,
    prefix_ids: list[torch.Tensor],
    suffix_ids: list[torch.Tensor],
    loss_mask: torch.Tensor,
    ce_chunk_size: int,
    packed_deltanet: bool,
) -> torch.Tensor:
    survivors = base_survivors.detach().clone().requires_grad_(True)
    return _production_split_loss(
        decoder,
        survivor_embeddings=survivors,
        survivor_cu=survivor_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
        ce_chunk_size=ce_chunk_size,
        packed_deltanet=packed_deltanet,
    )


def _layer_probe(decoder, args, *, device: torch.device, dtype: torch.dtype, hidden_dim: int):
    from bgkit.models.decoder import (
        _qwen35_deltanet_layer_split_packed,
        _qwen35_deltanet_layer_split_single,
    )

    inner_model, _lm_head = decoder._get_inner_model_and_head()
    layer_types = list(getattr(inner_model.config, "layer_types", []))
    layer_idx = layer_types.index("linear_attention")
    layer = inner_model.layers[layer_idx]
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1000)
    prefix_parts = [
        (
            torch.randn(
                1,
                args.prefix_len,
                hidden_dim,
                device=device,
                dtype=dtype,
                generator=gen,
            )
            * 0.02
        )
        for _ in range(args.batch)
    ]
    cont_parts = [
        (
            torch.randn(
                1,
                args.survivor_len + args.suffix_len,
                hidden_dim,
                device=device,
                dtype=dtype,
                generator=gen,
            )
            * 0.02
        )
        for _ in range(args.batch)
    ]
    single_prefix = []
    single_cont = []
    with torch.no_grad():
        for prefix, cont in zip(prefix_parts, cont_parts, strict=True):
            prefix_out, cont_out = _qwen35_deltanet_layer_split_single(layer, prefix, cont)
            single_prefix.append(prefix_out)
            single_cont.append(cont_out)
        packed_prefix, packed_cont = _qwen35_deltanet_layer_split_packed(
            layer,
            prefix_parts,
            cont_parts,
        )
    return {
        "layer_index": layer_idx,
        "prefix_max_abs_diffs": [
            _max_abs_diff(a, b) for a, b in zip(single_prefix, packed_prefix, strict=True)
        ],
        "prefix_rms_diffs": [
            _rms_diff(a, b) for a, b in zip(single_prefix, packed_prefix, strict=True)
        ],
        "cont_max_abs_diffs": [
            _max_abs_diff(a, b) for a, b in zip(single_cont, packed_cont, strict=True)
        ],
        "cont_rms_diffs": [
            _rms_diff(a, b) for a, b in zip(single_cont, packed_cont, strict=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("BGKIT_QWEN_MODEL", "Qwen/Qwen3.5-0.8B"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=1536)
    parser.add_argument("--survivor-len", type=int, default=128)
    parser.add_argument("--suffix-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ce-chunk-size", type=int, default=2048)
    parser.add_argument("--timing-steps", type=int, default=0)
    parser.add_argument("--skip-layer-probe", action="store_true")
    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.prefix_len <= 0 or args.survivor_len <= 0 or args.suffix_len <= 0:
        raise ValueError("all sequence lengths must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer, decoder, device, dtype, hidden_dim = _load_decoder(args)
    batch = _make_batch(tokenizer, args, device=device, dtype=dtype, hidden_dim=hidden_dim)
    prefix_ids, suffix_ids, survivor_cu, loss_mask, base_survivors = batch

    layer_result = None
    if not args.skip_layer_probe:
        layer_result = _layer_probe(
            decoder,
            args,
            device=device,
            dtype=dtype,
            hidden_dim=hidden_dim,
        )

    survivors_ref = base_survivors.detach().clone().requires_grad_(True)
    ref_loss = _reference_per_sample_loss(
        decoder,
        survivor_embeddings=survivors_ref,
        survivor_len=args.survivor_len,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        ce_chunk_size=args.ce_chunk_size,
    )
    ref_loss.backward()
    ref_grad = survivors_ref.grad.detach().clone()

    survivors_single = base_survivors.detach().clone().requires_grad_(True)
    split_single_loss = _production_split_loss(
        decoder,
        survivor_embeddings=survivors_single,
        survivor_cu=survivor_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
        ce_chunk_size=args.ce_chunk_size,
        packed_deltanet=False,
    )
    split_single_loss.backward()
    split_single_grad = survivors_single.grad.detach().clone()

    survivors_packed = base_survivors.detach().clone().requires_grad_(True)
    split_packed_loss = _production_split_loss(
        decoder,
        survivor_embeddings=survivors_packed,
        survivor_cu=survivor_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
        ce_chunk_size=args.ce_chunk_size,
        packed_deltanet=True,
    )
    split_packed_loss.backward()
    split_packed_grad = survivors_packed.grad.detach().clone()

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "batch": args.batch,
        "prefix_len": args.prefix_len,
        "survivor_len": args.survivor_len,
        "suffix_len": args.suffix_len,
        "reference_loss": float(ref_loss.detach().cpu()),
        "split_single_loss": float(split_single_loss.detach().cpu()),
        "split_packed_loss": float(split_packed_loss.detach().cpu()),
        "split_single_loss_abs_diff": abs(
            float(ref_loss.detach().cpu()) - float(split_single_loss.detach().cpu())
        ),
        "split_packed_loss_abs_diff": abs(
            float(ref_loss.detach().cpu()) - float(split_packed_loss.detach().cpu())
        ),
        "split_single_grad_max_abs_diff": _max_abs_diff(ref_grad, split_single_grad),
        "split_single_grad_relative_rms_diff": _relative_rms_diff(ref_grad, split_single_grad),
        "split_packed_grad_max_abs_diff": _max_abs_diff(ref_grad, split_packed_grad),
        "split_packed_grad_relative_rms_diff": _relative_rms_diff(ref_grad, split_packed_grad),
    }
    if layer_result is not None:
        result["layer_probe"] = layer_result

    if args.timing_steps > 0:
        with torch.no_grad():
            result["timing_ms"] = {
                "reference_per_sample": _sync_time(
                    lambda: _reference_per_sample_loss(
                        decoder,
                        survivor_embeddings=base_survivors,
                        survivor_len=args.survivor_len,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        ce_chunk_size=args.ce_chunk_size,
                    ),
                    steps=args.timing_steps,
                ),
                "split_single": _sync_time(
                    lambda: _production_split_loss(
                        decoder,
                        survivor_embeddings=base_survivors,
                        survivor_cu=survivor_cu,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        loss_mask=loss_mask,
                        ce_chunk_size=args.ce_chunk_size,
                        packed_deltanet=False,
                    ),
                    steps=args.timing_steps,
                ),
                "split_packed": _sync_time(
                    lambda: _production_split_loss(
                        decoder,
                        survivor_embeddings=base_survivors,
                        survivor_cu=survivor_cu,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        loss_mask=loss_mask,
                        ce_chunk_size=args.ce_chunk_size,
                        packed_deltanet=True,
                    ),
                    steps=args.timing_steps,
                ),
            }
        result["timing_fwd_bwd_ms"] = {
            "reference_per_sample": _sync_fwd_bwd_time(
                lambda: _reference_per_sample_loss_from_base(
                    decoder,
                    base_survivors=base_survivors,
                    survivor_len=args.survivor_len,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    ce_chunk_size=args.ce_chunk_size,
                ),
                steps=args.timing_steps,
            ),
            "split_single": _sync_fwd_bwd_time(
                lambda: _production_split_loss_from_base(
                    decoder,
                    base_survivors=base_survivors,
                    survivor_cu=survivor_cu,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    loss_mask=loss_mask,
                    ce_chunk_size=args.ce_chunk_size,
                    packed_deltanet=False,
                ),
                steps=args.timing_steps,
            ),
            "split_packed": _sync_fwd_bwd_time(
                lambda: _production_split_loss_from_base(
                    decoder,
                    base_survivors=base_survivors,
                    survivor_cu=survivor_cu,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    loss_mask=loss_mask,
                    ce_chunk_size=args.ce_chunk_size,
                    packed_deltanet=True,
                ),
                steps=args.timing_steps,
            ),
        }
        fwd = result["timing_ms"]
        fwd_bwd = result["timing_fwd_bwd_ms"]
        result["timing_speedup"] = {
            "split_single_vs_reference": fwd["reference_per_sample"] / fwd["split_single"],
            "split_packed_vs_reference": fwd["reference_per_sample"] / fwd["split_packed"],
        }
        result["timing_fwd_bwd_speedup"] = {
            "split_single_vs_reference": (
                fwd_bwd["reference_per_sample"] / fwd_bwd["split_single"]
            ),
            "split_packed_vs_reference": (
                fwd_bwd["reference_per_sample"] / fwd_bwd["split_packed"]
            ),
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
