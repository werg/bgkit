#!/usr/bin/env python
"""Export a bgkit2 checkpoint as a self-contained Hugging Face release (merged weights).

    encoder/                LFM2.5-Encoder-350M layout (``lfm2.*`` keys, Liquid's config and
                            remote code), the encoder's "slots" LoRA merged into the backbone
    decoder/                a plain Lfm2ForCausalLM: LFM2.5-350M with the decoder LoRA merged,
                            tokenizer + chat template
    compressor.safetensors  the slot-role parts: head, survive embedding, prompt separator,
                            projection block, projection MLP (keys as in release/huggingface/modeling_bgkit.py)
    bgkit_config.json       head layer, ratio limits, interface norm, slot token, chunking
    modeling_bgkit.py, README.md  from release/huggingface/
    LICENSE, NOTICE         LFM Open License v1.0 and the modification notice it requires

The parts exported are the ones text slots read (``BgKIT2Encoder.part`` under the "slots" role).
The repository-tree path (base parts, repro heads, the spine adapter) is not exported.

    PYTHONPATH=src python scripts/bgkit2_export_hf.py --experiment bgkit2_s2_showcase \\
        --checkpoint /mnt/external/bgkit-checkpoints/<ckpt> --out /path/to/release
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SRC = ROOT / "release" / "huggingface"
SLOT_PARTS = ("head", "survive_embedding", "prompt_separator", "projection_block", "projection")


def merged_encoder_state(encoder) -> dict[str, torch.Tensor]:
    """The backbone's weights with the "slots" LoRA folded in, keyed as Liquid's checkpoint."""
    model = encoder.backbone.model
    merged: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "slots" in mod.lora_A:
            w = mod.base_layer.weight.detach().float()
            delta = mod.lora_B["slots"].weight.float() @ mod.lora_A["slots"].weight.float()
            merged[f"{name}.weight"] = w + delta * float(mod.scaling["slots"])
    state = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            continue
        clean = k.replace(".base_layer", "")
        state["lfm2." + clean] = merged.get(clean, v).detach().float().contiguous()
    n_merged = len(merged)
    if n_merged == 0:
        raise RuntimeError("no 'slots' LoRA found in the encoder")
    return state, n_merged


def compressor_state(encoder) -> dict[str, torch.Tensor]:
    out = {}
    for name in SLOT_PARTS:
        part = encoder.part(name)
        if isinstance(part, torch.nn.Parameter):
            out[name] = part.detach().float().contiguous()
            continue
        for k, v in part.state_dict().items():
            key = f"{name}.{k}"
            if name == "head":
                key = f"head.{k.removeprefix('net.')}"
            out[key] = v.detach().float().contiguous()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from hydra import compose, initialize_config_dir

    from bgkit2.agent.student import MAX_RATIO, MIN_RATIO, StudentModel
    from bgkit2.data.templates import SPLICE_SENTINEL

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    st = StudentModel(cfg, Path(args.checkpoint), torch.device("cpu"))
    enc, m = st.encoder, st.cfg2.model
    if enc.default_role != "slots":
        raise RuntimeError("expected an encoder with the slot adapter (role 'slots')")
    out = Path(args.out)
    (out / "encoder").mkdir(parents=True, exist_ok=True)

    # encoder: merged backbone in Liquid's layout, its config and remote code unchanged
    state, n_merged = merged_encoder_state(enc)
    liquid = {f: hf_hub_download(m.encoder_id, f) for f in ("config.json", "modeling_lfm2_bidirectional.py")}
    from safetensors import safe_open

    with safe_open(hf_hub_download(m.encoder_id, "model.safetensors"), "pt") as f:
        want = set(f.keys())
    if set(state) != want:
        raise RuntimeError(f"encoder keys differ from {m.encoder_id}: "
                           f"{sorted(set(state) ^ want)[:8]}")
    save_file(state, out / "encoder" / "model.safetensors", metadata={"format": "pt"})
    for f, src in liquid.items():
        shutil.copy(src, out / "encoder" / f)

    # compressor parts + config
    save_file(compressor_state(enc), out / "compressor.safetensors", metadata={"format": "pt"})
    inorm = enc.part("interface_norm")
    if not inorm.rms_only:
        raise RuntimeError("modeling_bgkit.py implements the rms_only interface norm only")
    bcfg = {
        "head_layer": int(m.head_layer),
        "head_inner_dim": int(m.head_inner_dim),
        "projection_mult": int(m.projection_mult),
        "projection_block_layer": int(enc._projection_block_src),
        "min_survivors": int(m.min_survivors),
        "min_ratio": MIN_RATIO,
        "max_ratio": MAX_RATIO,
        "interface_row_norm": float(inorm.target_row_norm),
        "slot_token": SPLICE_SENTINEL,
        "default_prompt": "Compress the tool result.\n",
        "max_slot_tokens": int(st.max_ctx_tokens),
        "chunk_tokens": 2048,
        "max_compression": 64.0,
        "encoder_base": m.encoder_id,
        "decoder_base": m.decoder_id,
        "source_checkpoint": Path(args.checkpoint).name,
    }
    (out / "bgkit_config.json").write_text(json.dumps(bcfg, indent=2) + "\n")

    # decoder: LoRA merged in fp32, saved in the base model's dtype
    base_dtype = getattr(torch, json.loads(Path(hf_hub_download(m.decoder_id, "config.json")).read_text())["dtype"])
    lm = st.decoder.lm.float().merge_and_unload()
    lm = lm.to(base_dtype)
    lm.save_pretrained(out / "decoder")
    st.tok.save_pretrained(out / "decoder")

    for f in ("modeling_bgkit.py", "README.md", "NOTICE"):
        if (RELEASE_SRC / f).exists():
            shutil.copy(RELEASE_SRC / f, out / f)
    shutil.copy(hf_hub_download(m.decoder_id, "LICENSE"), out / "LICENSE")
    print(json.dumps({"out": str(out), "encoder_lora_merged": n_merged,
                      "decoder_dtype": str(base_dtype)}, indent=1))


if __name__ == "__main__":
    main()
