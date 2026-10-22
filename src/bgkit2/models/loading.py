"""Load the two pristine LFM2.5 models without the checkpoint's remote code."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

from .lfm2_bidirectional import Lfm2BidirectionalBackbone

ENCODER_ID = "LiquidAI/LFM2.5-Encoder-350M"
DECODER_ID = "LiquidAI/LFM2.5-350M-Base"

_PREFIXES = ("lfm2.", "model.")


def resolve_snapshot(model_id_or_path: str) -> Path:
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.json", "*.safetensors"]))


def _strip_prefix(key: str) -> str:
    for pre in _PREFIXES:
        if key.startswith(pre):
            return key[len(pre) :]
    return key


def load_encoder_backbone(
    model_id_or_path: str = ENCODER_ID,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "sdpa",
) -> Lfm2BidirectionalBackbone:
    """Build ``Lfm2Model`` from the encoder checkpoint's weights, then wrap it.

    The encoder repo stores weights under ``lfm2.*`` for its remote-code class;
    we read the safetensors directly, drop the tied ``lm_head`` if present, and
    refuse any other mismatch.
    """
    snap = resolve_snapshot(model_id_or_path)
    cfg_dict = json.loads((snap / "config.json").read_text())
    cfg_dict.pop("auto_map", None)
    cfg_dict["architectures"] = ["Lfm2Model"]
    cfg_dict["use_cache"] = False
    cfg = Lfm2Config(**{k: v for k, v in cfg_dict.items() if k not in {"transformers_version"}})
    cfg._attn_implementation = attn_implementation
    model = Lfm2Model(cfg)
    raw = load_file(str(snap / "model.safetensors"))
    state = {}
    for k, v in raw.items():
        if k.startswith("lm_head."):
            continue
        state[_strip_prefix(k)] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "encoder weights mismatch: "
            f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
        )
    model = model.to(dtype)
    return Lfm2BidirectionalBackbone(model)


def load_decoder_lm(
    model_id_or_path: str = DECODER_ID,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "sdpa",
    experts_implementation: str | None = None,
    fused_rmsnorm: bool = False,
    device: torch.device | str | None = None,
):
    """The decoder LM: ``Lfm2ForCausalLM`` for the dense LFM2.5 models, the Auto class for any
    other family (LFM2.5-8B-A1B is ``Lfm2MoeForCausalLM``; ``experts_implementation`` selects
    its experts kernel, ``fused_rmsnorm`` swaps its RMSNorms for the fused Triton kernel).

    ``device`` (a CUDA device) loads the weights straight onto it: on unified memory a CPU-staged
    load of the 8B (16 GB bf16) followed by ``.to(cuda)`` left ~20 GB of freed-but-mapped glibc
    arenas in the trainer, inherited by every dataloader worker (measured 2026-09-26)."""
    snap = resolve_snapshot(model_id_or_path)
    place = (
        {"device_map": {"": str(device)}}
        if device is not None and torch.device(device).type == "cuda"
        else {}
    )
    mtype = json.loads((snap / "config.json").read_text()).get("model_type")
    if mtype == "lfm2":
        lm = Lfm2ForCausalLM.from_pretrained(
            str(snap), dtype=dtype, attn_implementation=attn_implementation, **place
        )
    else:
        from transformers import AutoModelForCausalLM

        # from_pretrained only accepts transformers' built-in experts names; a registered one
        # (bgkit_triton) is switched on after loading
        triton = experts_implementation == "bgkit_triton"
        impl = "grouped_mm" if triton else experts_implementation
        kw = {"experts_implementation": impl} if impl else {}
        lm = AutoModelForCausalLM.from_pretrained(
            str(snap), dtype=dtype, attn_implementation=attn_implementation, **kw, **place
        )
        if triton:
            from . import moe_kernels

            moe_kernels.enable(lm)
    if fused_rmsnorm:
        from .fused_rmsnorm import fuse_rmsnorm

        fuse_rmsnorm(lm)
    lm.config.use_cache = False
    return lm


def load_embedding_table(model_id_or_path: str) -> torch.Tensor:
    """A model's input embedding table (float32, CPU) read straight from its safetensors."""
    from safetensors import safe_open

    snap = resolve_snapshot(model_id_or_path)
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k.endswith("embed_tokens.weight"):
                    return h.get_tensor(k).float()
    raise ValueError(f"no embed_tokens.weight in {snap}")


def load_tokenizer(model_id_or_path: str = DECODER_ID):
    """The DECODER's tokenizer is the one the whole pipeline uses.

    The encoder's tokenizer differs only in three added tokens (``<|mask|>``,
    ``<think>``, ``</think>``); merges and vocabulary are identical, so decoder
    ids are valid encoder ids.
    """
    return AutoTokenizer.from_pretrained(str(resolve_snapshot(model_id_or_path)))
