"""Benchmark Falcon-H1 kernel candidates in the Docker training image.

This is intentionally stricter than a smoke test: it instruments the Falcon-H1
Mamba and attention call sites, disables non-flash SDPA fallbacks for Falcon,
and emits machine-readable JSON records for train/prefill/decode cases.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import importlib.util as importlib_util
import json
import os
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers.models.falcon_h1.modeling_falcon_h1 as falcon_h1
from transformers import AutoConfig
from transformers.models.falcon_h1 import FalconH1Config, FalconH1ForCausalLM

from bgkit.utils.attention_backend import resolve_decoder_attention_implementation
from bgkit.utils.falcon_h1_defaults import effective_falcon_h1_fast_env, falcon_h1_env_truthy
from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

DEFAULT_MODEL_ID = "tiiuae/Falcon-H1-Tiny-90M-Instruct"


@dataclass(frozen=True)
class BenchCase:
    mode: str
    shape: str
    batch_size: int
    seq_len: int
    num_hidden_layers: int
    hidden_size: int
    attention_heads: int
    key_value_heads: int
    attention_head_dim: int
    mamba_d_ssm: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_d_state: int
    mamba_chunk_size: int


def _shape_smoke(num_hidden_layers: int) -> FalconH1Config:
    return FalconH1Config(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=768,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        mamba_d_ssm=768,
        mamba_n_heads=24,
        mamba_d_head=32,
        mamba_n_groups=1,
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_chunk_size=128,
        mamba_rms_norm=False,
        mamba_norm_before_gate=False,
        use_cache=True,
    )


def _shape_tiny(model_id: str, num_hidden_layers: int | None) -> FalconH1Config:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if num_hidden_layers is not None:
        cfg.num_hidden_layers = num_hidden_layers
    cfg.vocab_size = min(int(cfg.vocab_size), 4096)
    cfg.use_cache = True
    return cfg


def build_config(shape: str, model_id: str, num_hidden_layers: int | None) -> FalconH1Config:
    if shape == "smoke":
        return _shape_smoke(num_hidden_layers or 1)
    if shape == "tiny-1l":
        return _shape_tiny(model_id, 1 if num_hidden_layers is None else num_hidden_layers)
    if shape == "tiny":
        return _shape_tiny(model_id, num_hidden_layers)
    raise ValueError(f"unknown shape {shape!r}")


def case_from_config(
    *,
    mode: str,
    shape: str,
    cfg: FalconH1Config,
    batch_size: int,
    seq_len: int,
) -> BenchCase:
    head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)
    return BenchCase(
        mode=mode,
        shape=shape,
        batch_size=batch_size,
        seq_len=seq_len,
        num_hidden_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        attention_heads=int(cfg.num_attention_heads),
        key_value_heads=int(cfg.num_key_value_heads),
        attention_head_dim=head_dim,
        mamba_d_ssm=int(cfg.mamba_d_ssm),
        mamba_n_heads=int(cfg.mamba_n_heads),
        mamba_d_head=int(cfg.mamba_d_head),
        mamba_d_state=int(cfg.mamba_d_state),
        mamba_chunk_size=int(cfg.mamba_chunk_size),
    )


def env_record() -> dict[str, Any]:
    import transformers

    def package_version(name: str) -> str | None:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    def module_file(name: str) -> str | None:
        try:
            spec = importlib_util.find_spec(name)
        except ModuleNotFoundError:
            return None
        if spec is None or spec.origin is None:
            return None
        return str(Path(spec.origin).resolve())

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "mamba_ssm_version": package_version("mamba-ssm"),
        "causal_conv1d_version": package_version("causal-conv1d"),
        "mamba_ssm_file": module_file("mamba_ssm"),
        "ssd_combined_file": module_file("mamba_ssm.ops.triton.ssd_combined"),
        "selective_state_update_file": module_file(
            "mamba_ssm.ops.triton.selective_state_update"
        ),
        "causal_conv1d_file": module_file("causal_conv1d"),
        "kernels_installed": bool(importlib_util.find_spec("kernels")),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": (
            torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        ),
        "mamba_sm121_safe_autotune": os.environ.get("MAMBA_SM121_SAFE_AUTOTUNE"),
        "mamba_sm121_static_configs": os.environ.get("MAMBA_SM121_STATIC_CONFIGS"),
        "mamba_falcon_sm121_fastpath": os.environ.get("MAMBA_FALCON_SM121_FASTPATH"),
        "falcon_h1_fast_env": effective_falcon_h1_fast_env(),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }


def triton_best_configs() -> dict[str, Any]:
    try:
        import mamba_ssm.ops.triton.layernorm_gated as layernorm_gated
        import mamba_ssm.ops.triton.ssd_bmm as ssd_bmm
        import mamba_ssm.ops.triton.ssd_chunk_scan as ssd_chunk_scan
        import mamba_ssm.ops.triton.ssd_chunk_state as ssd_chunk_state
        import mamba_ssm.ops.triton.ssd_combined as ssd_combined
        import mamba_ssm.ops.triton.ssd_state_passing as ssd_state_passing
    except ModuleNotFoundError:
        return {}

    modules = [
        ssd_combined,
        ssd_bmm,
        ssd_chunk_state,
        ssd_chunk_scan,
        ssd_state_passing,
        layernorm_gated,
    ]
    configs: dict[str, Any] = {}
    for module in modules:
        module_name = module.__name__.rsplit(".", maxsplit=1)[-1]
        for name, obj in vars(module).items():
            best_config = getattr(obj, "best_config", None)
            if best_config is None:
                continue
            configs[f"{module_name}.{name}"] = {
                "kwargs": dict(getattr(best_config, "kwargs", {}) or {}),
                "num_warps": getattr(best_config, "num_warps", None),
                "num_stages": getattr(best_config, "num_stages", None),
            }
    return configs


@contextmanager
def kernel_counters(*, fail_on_fallback: bool) -> Iterable[dict[str, int]]:
    counts = {
        "mixer_torch_forward": 0,
        "mixer_cuda_kernels_forward": 0,
        "mamba_split_conv1d_scan_combined": 0,
        "mamba_chunk_scan_combined": 0,
        "causal_conv1d_fn": 0,
        "causal_conv1d_update": 0,
        "selective_state_update": 0,
        "scaled_dot_product_attention": 0,
    }

    originals = {
        "torch_forward": falcon_h1.FalconH1Mixer.torch_forward,
        "cuda_kernels_forward": falcon_h1.FalconH1Mixer.cuda_kernels_forward,
        "mamba_split_conv1d_scan_combined": falcon_h1.mamba_split_conv1d_scan_combined,
        "mamba_chunk_scan_combined": falcon_h1.mamba_chunk_scan_combined,
        "causal_conv1d_fn": falcon_h1.causal_conv1d_fn,
        "causal_conv1d_update": falcon_h1.causal_conv1d_update,
        "selective_state_update": falcon_h1.selective_state_update,
        "scaled_dot_product_attention": F.scaled_dot_product_attention,
    }

    def counted_torch_forward(self, *args: Any, **kwargs: Any) -> Any:
        counts["mixer_torch_forward"] += 1
        if fail_on_fallback:
            raise AssertionError("FalconH1Mixer.torch_forward fallback was called")
        return originals["torch_forward"](self, *args, **kwargs)

    def counted_cuda_forward(self, *args: Any, **kwargs: Any) -> Any:
        counts["mixer_cuda_kernels_forward"] += 1
        return originals["cuda_kernels_forward"](self, *args, **kwargs)

    def wrap_global(name: str) -> Callable[..., Any] | None:
        original = originals[name]
        if original is None:
            return None

        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return counted

    def counted_sdpa(*args: Any, **kwargs: Any) -> Any:
        counts["scaled_dot_product_attention"] += 1
        return originals["scaled_dot_product_attention"](*args, **kwargs)

    falcon_h1.FalconH1Mixer.torch_forward = counted_torch_forward
    falcon_h1.FalconH1Mixer.cuda_kernels_forward = counted_cuda_forward
    falcon_h1.mamba_split_conv1d_scan_combined = wrap_global(
        "mamba_split_conv1d_scan_combined"
    )
    falcon_h1.mamba_chunk_scan_combined = wrap_global("mamba_chunk_scan_combined")
    falcon_h1.causal_conv1d_fn = wrap_global("causal_conv1d_fn")
    falcon_h1.causal_conv1d_update = wrap_global("causal_conv1d_update")
    falcon_h1.selective_state_update = wrap_global("selective_state_update")
    F.scaled_dot_product_attention = counted_sdpa

    try:
        yield counts
    finally:
        falcon_h1.FalconH1Mixer.torch_forward = originals["torch_forward"]
        falcon_h1.FalconH1Mixer.cuda_kernels_forward = originals["cuda_kernels_forward"]
        falcon_h1.mamba_split_conv1d_scan_combined = originals[
            "mamba_split_conv1d_scan_combined"
        ]
        falcon_h1.mamba_chunk_scan_combined = originals["mamba_chunk_scan_combined"]
        falcon_h1.causal_conv1d_fn = originals["causal_conv1d_fn"]
        falcon_h1.causal_conv1d_update = originals["causal_conv1d_update"]
        falcon_h1.selective_state_update = originals["selective_state_update"]
        F.scaled_dot_product_attention = originals["scaled_dot_product_attention"]


def reset_counts(counts: dict[str, int]) -> None:
    for key in counts:
        counts[key] = 0


def validate_counts(
    mode: str,
    counts: dict[str, int],
    *,
    patch_report: dict[str, int],
) -> None:
    patched_attention = bool(patch_report.get("attention", 0))
    patched_mixer = bool(patch_report.get("mixer", 0))
    if counts["mixer_torch_forward"] != 0:
        raise AssertionError(f"Mamba torch fallback was called: {counts}")
    if not patched_mixer and counts["mixer_cuda_kernels_forward"] <= 0:
        raise AssertionError(f"Mamba CUDA fast path was not called: {counts}")
    if not patched_attention and counts["scaled_dot_product_attention"] <= 0:
        raise AssertionError(f"SDPA attention was not called: {counts}")
    if (
        mode == "train"
        and not patched_mixer
        and counts["mamba_split_conv1d_scan_combined"] <= 0
    ):
        raise AssertionError(f"training fused Mamba kernel was not called: {counts}")
    if (
        mode == "prefill"
        and (counts["causal_conv1d_fn"] <= 0 or counts["mamba_chunk_scan_combined"] <= 0)
    ):
        raise AssertionError(f"prefill Mamba kernels were not called: {counts}")
    if (
        mode == "decode"
        and (counts["causal_conv1d_update"] <= 0 or counts["selective_state_update"] <= 0)
    ):
        raise AssertionError(f"decode Mamba kernels were not called: {counts}")


def cuda_measure(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def make_model(cfg: FalconH1Config, dtype: torch.dtype) -> FalconH1ForCausalLM:
    cfg._attn_implementation = resolve_decoder_attention_implementation(
        "auto",
        decoder_family="falcon_h1",
    )
    model = FalconH1ForCausalLM(cfg)
    model = model.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)
    if falcon_h1_env_truthy("BGKIT_FALCON_H1_PATCH"):
        model._bgkit_falcon_h1_patch_report = patch_falcon_h1_decoder(model)  # type: ignore[attr-defined]
    return model


def patch_report_for(model: FalconH1ForCausalLM) -> dict[str, int]:
    report = getattr(model, "_bgkit_falcon_h1_patch_report", None)
    if report is None:
        return {}
    return dict(report.as_dict())


def reset_internal_profiles() -> None:
    try:
        from bgkit.kernels.falcon_h1_mamba import reset_falcon_h1_mamba_profile

        reset_falcon_h1_mamba_profile()
    except Exception:
        pass
    try:
        from bgkit.kernels.falcon_h1_mlp import reset_falcon_h1_mlp_profile

        reset_falcon_h1_mlp_profile()
    except Exception:
        pass


def summarize_internal_profiles() -> dict[str, Any]:
    profiles: dict[str, Any] = {"mamba": [], "mlp": []}
    try:
        from bgkit.kernels.falcon_h1_mamba import summarize_falcon_h1_mamba_profile

        profiles["mamba"] = summarize_falcon_h1_mamba_profile()
    except Exception as exc:
        profiles["mamba_error"] = repr(exc)
    try:
        from bgkit.kernels.falcon_h1_mlp import summarize_falcon_h1_mlp_profile

        profiles["mlp"] = summarize_falcon_h1_mlp_profile()
    except Exception as exc:
        profiles["mlp_error"] = repr(exc)
    return profiles


def assert_fast_handles() -> dict[str, bool]:
    handles = {
        "is_fast_path_available": bool(falcon_h1.is_fast_path_available),
        "causal_conv1d_fn": falcon_h1.causal_conv1d_fn is not None,
        "causal_conv1d_update": falcon_h1.causal_conv1d_update is not None,
        "selective_state_update": falcon_h1.selective_state_update is not None,
        "mamba_chunk_scan_combined": falcon_h1.mamba_chunk_scan_combined is not None,
        "mamba_split_conv1d_scan_combined": (
            falcon_h1.mamba_split_conv1d_scan_combined is not None
        ),
    }
    if not all(handles.values()):
        raise AssertionError(f"Falcon-H1 fast handles are incomplete: {handles}")
    return handles


def run_train_case(
    model: FalconH1ForCausalLM,
    cfg: FalconH1Config,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.train()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    def step() -> float:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = out.loss
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss: {loss}")
        loss.backward()
        return float(loss.detach().cpu())

    last_loss = step()
    elapsed = cuda_measure(step, warmup=warmup, iters=iters)
    return elapsed, last_loss


def run_prefill_case(
    model: FalconH1ForCausalLM,
    cfg: FalconH1Config,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    @torch.no_grad()
    def step() -> float:
        out = model(input_ids=input_ids, use_cache=True)
        return float(out.logits[..., -1, :].float().mean().detach().cpu())

    last_value = step()
    elapsed = cuda_measure(step, warmup=warmup, iters=iters)
    return elapsed, last_value


def run_decode_case(
    model: FalconH1ForCausalLM,
    cfg: FalconH1Config,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        cache = prefill.past_key_values

    @torch.no_grad()
    def step() -> float:
        next_ids = torch.randint(0, cfg.vocab_size, (batch_size, 1), device=device)
        out = model(input_ids=next_ids, past_key_values=cache, use_cache=True)
        return float(out.logits[..., -1, :].float().mean().detach().cpu())

    last_value = step()
    elapsed = cuda_measure(step, warmup=warmup, iters=iters)
    return elapsed, last_value


def _zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def _run_attention_backend(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    def step() -> float:
        _zero_grads(q, k, v)
        out = fn(q, k, v)
        loss = out.float().square().mean()
        loss.backward()
        return float(loss.detach().cpu())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    value = step()
    elapsed_ms = cuda_measure(step, warmup=warmup, iters=iters)
    return {
        "backend": name,
        "status": "ok",
        "elapsed_ms_per_iter": elapsed_ms,
        "value": value,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }


def run_attention_case(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_config(args.shape, args.model_id, args.num_hidden_layers)
    resolve_decoder_attention_implementation("auto", decoder_family="falcon_h1")
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    case = case_from_config(
        mode="attention",
        shape=args.shape,
        cfg=cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(cfg.hidden_size) // heads
    q_bhsd = torch.randn(
        args.batch_size,
        heads,
        args.seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_bhsd = torch.randn(
        args.batch_size,
        kv_heads,
        args.seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_bhsd = torch.randn_like(k_bhsd, requires_grad=True)
    q_bshd = torch.randn(
        args.batch_size,
        args.seq_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_bshd = torch.randn(
        args.batch_size,
        args.seq_len,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v_bshd = torch.randn_like(k_bshd, requires_grad=True)

    def sdpa(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_in,
            k_in,
            v_in,
            is_causal=True,
            enable_gqa=heads != kv_heads,
        )

    results = [
        _run_attention_backend(
            "torch_sdpa_flash_only_bhsd",
            sdpa,
            q=q_bhsd,
            k=k_bhsd,
            v=v_bhsd,
            warmup=args.warmup,
            iters=args.iters,
        )
    ]

    try:
        from flash_attn import flash_attn_func

        def flash_attn_bshd(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            return flash_attn_func(q_in, k_in, v_in, causal=True)

        def flash_attn_from_bhsd(
            q_in: torch.Tensor,
            k_in: torch.Tensor,
            v_in: torch.Tensor,
        ) -> torch.Tensor:
            out = flash_attn_bshd(
                q_in.transpose(1, 2).contiguous(),
                k_in.transpose(1, 2).contiguous(),
                v_in.transpose(1, 2).contiguous(),
            )
            return out.transpose(1, 2).contiguous()

        with torch.no_grad():
            q_ref = q_bshd.detach().transpose(1, 2)
            k_ref = k_bshd.detach().transpose(1, 2)
            v_ref = v_bshd.detach().transpose(1, 2)
            reference = sdpa(q_ref, k_ref, v_ref).transpose(1, 2)
            candidate = flash_attn_bshd(
                q_bshd.detach(),
                k_bshd.detach(),
                v_bshd.detach(),
            )
            max_abs_diff = float((reference - candidate).abs().max().detach().cpu())
        flash_nocopy_result = _run_attention_backend(
            "flash_attn_func_bshd_nocopy",
            flash_attn_bshd,
            q=q_bshd,
            k=k_bshd,
            v=v_bshd,
            warmup=args.warmup,
            iters=args.iters,
        )
        flash_nocopy_result["max_abs_diff_vs_sdpa"] = max_abs_diff
        results.append(flash_nocopy_result)
        flash_copy_result = _run_attention_backend(
            "flash_attn_func_from_bhsd_with_copy",
            flash_attn_from_bhsd,
            q=q_bhsd,
            k=k_bhsd,
            v=v_bhsd,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(flash_copy_result)
    except Exception as exc:
        results.append(
            {
                "backend": "flash_attn_func_dense",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )

    return {
        "case": asdict(case),
        "attention_results": results,
        "sdpa_flags": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
    }


def run_case(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    if mode == "attention":
        return run_attention_case(args)

    cfg = build_config(args.shape, args.model_id, args.num_hidden_layers)
    dtype = getattr(torch, args.dtype)
    model = make_model(cfg, dtype)
    patch_report = patch_report_for(model)
    handles = assert_fast_handles()
    case = case_from_config(
        mode=mode,
        shape=args.shape,
        cfg=cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with kernel_counters(fail_on_fallback=not args.allow_fallback) as counts:
        if mode == "train":
            elapsed_ms, value = run_train_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=0,
                iters=1,
            )
        elif mode == "prefill":
            elapsed_ms, value = run_prefill_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=0,
                iters=1,
            )
        elif mode == "decode":
            elapsed_ms, value = run_decode_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=0,
                iters=1,
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")

        reset_counts(counts)
        reset_internal_profiles()
        if mode == "train":
            elapsed_ms, value = run_train_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        elif mode == "prefill":
            elapsed_ms, value = run_prefill_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        else:
            elapsed_ms, value = run_decode_case(
                model,
                cfg,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        validate_counts(mode, counts, patch_report=patch_report)
        call_counts = dict(counts)

    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    return {
        "case": asdict(case),
        "elapsed_ms_per_iter": elapsed_ms,
        "value": value,
        "peak_cuda_memory_bytes": peak_memory,
        "fast_handles": handles,
        "patch_report": patch_report,
        "call_counts": call_counts,
        "internal_profiles": summarize_internal_profiles(),
        "sdpa_flags": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "triton_best_configs": triton_best_configs(),
    }


def parse_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    valid = {"train", "prefill", "decode", "attention"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    return modes


def parse_int_csv(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [fallback]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or [fallback]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--shape", choices=["smoke", "tiny-1l", "tiny"], default="tiny-1l")
    parser.add_argument("--modes", default="train,prefill,decode")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch-size sweep. Overrides --batch-size when set.",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--seq-lens",
        default=None,
        help="Comma-separated sequence-length sweep. Overrides --seq-len when set.",
    )
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Falcon-H1 kernel benchmark requires CUDA")

    modes = parse_modes(args.modes)
    batch_sizes = parse_int_csv(args.batch_sizes, args.batch_size)
    seq_lens = parse_int_csv(args.seq_lens, args.seq_len)
    env = env_record()
    records = [{"kind": "environment", **env}]
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            args.batch_size = batch_size
            args.seq_len = seq_len
            for mode in modes:
                result = run_case(args, mode)
                records.append({"kind": "benchmark", **result})

    for record in records:
        print(json.dumps(record, sort_keys=True), flush=True)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
