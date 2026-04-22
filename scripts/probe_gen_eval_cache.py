#!/usr/bin/env python
"""Reproduce the gen_eval cache shape mismatch.

Exercises the exact failing call path: ``ReconstructionDecoder.generate_with_single_splice``
with a tiny prefix/survivors/suffix, while instrumenting
``bgkit_flash_attention_4_forward`` to dump Q/K/V and output shapes on every
call. The goal is to expose exactly where rank changes between the prefill
and the first cached-decode step.

Run inside the container with a small memory cap so the live training isn't
disturbed:

    docker compose -f docker/docker-compose.yaml run --rm \
        --name probe-gen-eval \
        -e BGKIT_CUDA_MEM_FRACTION=0.10 \
        train-phase1-step3 python scripts/probe_gen_eval_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _install_shape_probe() -> None:
    """Monkey-patch bgkit_flash_attention_4_forward to log shapes."""
    import bgkit.utils.attention_backend as ab

    _orig = ab.bgkit_flash_attention_4_forward
    counter = {"n": 0}

    def wrapped(module, query, key, value, *args, **kwargs):
        counter["n"] += 1
        n = counter["n"]
        layer_idx = getattr(module, "layer_idx", "?")
        cu_q = kwargs.get("cu_seqlens_q") or kwargs.get("cu_seq_lens_q")
        cu_k = kwargs.get("cu_seqlens_k") or kwargs.get("cu_seq_lens_k")
        m_q = kwargs.get("max_seqlen_q") or kwargs.get("max_length_q")
        m_k = kwargs.get("max_seqlen_k") or kwargs.get("max_length_k")
        print(
            f"[fa4 #{n:03d}] layer={layer_idx} "
            f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)} "
            f"cu_q={None if cu_q is None else cu_q.tolist()} "
            f"cu_k={None if cu_k is None else cu_k.tolist()} "
            f"m_q={m_q} m_k={m_k}",
            flush=True,
        )
        out, weights = _orig(module, query, key, value, *args, **kwargs)
        print(f"[fa4 #{n:03d}] out={tuple(out.shape)} dim={out.dim()}", flush=True)
        return out, weights

    ab.bgkit_flash_attention_4_forward = wrapped

    # Also re-register with transformers so the new wrapper is what HF calls.
    from transformers import AttentionInterface

    AttentionInterface.register(ab.BGKIT_FA4_ATTENTION_IMPL, wrapped)


def _install_model_forward_probe() -> None:
    """Log hidden_states shape entering each decoder layer."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    _orig = Qwen3_5DecoderLayer.forward
    counter = {"n": 0}

    def wrapped(self, hidden_states, *args, **kwargs):
        counter["n"] += 1
        n = counter["n"]
        layer_idx = getattr(getattr(self, "self_attn", None) or getattr(self, "linear_attn", None), "layer_idx", "?")
        print(
            f"[layer #{n:03d}] layer_type={self.layer_type} layer_idx={layer_idx} "
            f"hidden_states={tuple(hidden_states.shape)} dim={hidden_states.dim()}",
            flush=True,
        )
        return _orig(self, hidden_states, *args, **kwargs)

    Qwen3_5DecoderLayer.forward = wrapped


def _install_cache_probe() -> None:
    """Monkey-patch DynamicLayer.update to log stored cache shapes."""
    from transformers.cache_utils import DynamicLayer

    _orig = DynamicLayer.update
    counter = {"n": 0}

    def wrapped(self, key_states, value_states, *args, **kwargs):
        counter["n"] += 1
        n = counter["n"]
        prior = tuple(self.keys.shape) if getattr(self, "is_initialized", False) else "uninit"
        print(
            f"[cache #{n:03d}] prior_keys={prior} "
            f"new_keys={tuple(key_states.shape)}",
            flush=True,
        )
        try:
            out_k, out_v = _orig(self, key_states, value_states, *args, **kwargs)
        except RuntimeError as e:
            print(f"[cache #{n:03d}] UPDATE FAILED: {e}", flush=True)
            raise
        print(f"[cache #{n:03d}] post_keys={tuple(out_k.shape)}", flush=True)
        return out_k, out_v

    DynamicLayer.update = wrapped


def main() -> None:
    _install_shape_probe()
    _install_model_forward_probe()
    _install_cache_probe()

    from bgkit.models.decoder import ReconstructionDecoder
    from bgkit.utils.attention_backend import resolve_attention_implementation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    attn_impl = resolve_attention_implementation("auto")
    print(f"attn_impl={attn_impl}", flush=True)

    model_name = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    decoder = ReconstructionDecoder(backbone, hidden_dim=1024)
    decoder.to(device).eval()

    # Tiny input.
    surv_dim = 1024
    survivors = torch.randn(4, surv_dim, dtype=torch.bfloat16, device=device)
    surv_cu = torch.tensor([0, 4], dtype=torch.int32, device=device)
    prefix = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    suffix = torch.tensor([4, 5], dtype=torch.long, device=device)

    print("\n=== calling generate_with_single_splice(max_new_tokens=3) ===\n", flush=True)
    try:
        out = decoder.generate_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=surv_cu,
            prefix_ids=prefix,
            suffix_ids=suffix,
            tokenizer=tok,
            max_new_tokens=3,
            temperature=0.0,
        )
        print("\n=== SUCCESS ===", flush=True)
        print(f"content_ids={[ids.tolist() for ids in out.content_ids]}", flush=True)
    except RuntimeError as e:
        print(f"\n=== FAILED: {e} ===", flush=True)
        raise


if __name__ == "__main__":
    main()
