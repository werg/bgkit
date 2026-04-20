#!/usr/bin/env python
"""One-off: migrate legacy positional optimizer state to name-keyed format.

Motivation
----------
phase1_step3_step2000_20260418_153212 was saved when a bug was still
live in ``DecoderInitTrainer._configure_trainable_state``: after
applying LoRA to the decoder, the old code did a blanket
``self.decoder.requires_grad_(True)``, which un-froze the full 758 M
base model alongside the LoRA adapters. The saved optimizer therefore
had 6 param groups / 648 state entries:

  0/1 projection_block  (14 params,  proj_lr)
  2/3 decoder           (512 params, decoder_lr)      <- bug: all of it
  4/5 encoder compressor(129 params, encoder_lr)

After the 2026-04-18 fix (``decoder_unfreeze_lora_only``) the decoder
group shrinks to ~192 LoRA-only params, so the current trainer builds
5 param groups / ~335 state entries. Positional load-state-dict then
fails ("different number of parameter groups"), and the legacy
fallback gives up.

This script reconstructs the save-time topology, loads the legacy
state into it, and re-keys by stable module-path name. The resulting
``optimizer_state_by_name.pt`` file is picked up by
``_restore_optimizer_state_by_name`` on resume -- only the 192
LoRA-adapter names present in the current topology are restored, the
rest are silently dropped (which is what we want: the base-decoder
moments would not be applied anyway since those params are now
frozen).

Usage
-----
Run inside the pre-built Step 3 training image so torch / transformers
/ peft / bgkit src are all on the same versions that produced the
checkpoint::

    docker compose --env-file .env -f docker/docker-compose.yaml run --rm \\
        --entrypoint python train-phase1-step3 \\
        scripts/migrate_step3_optimizer_state.py \\
        /workspace/checkpoints/phase1_step3_step2000_20260418_153212

The script writes ``optimizer_state_by_name.pt`` into the same
checkpoint directory. Safe to re-run (the file is overwritten).

Validation
----------
The script logs three numbers that together prove the migration
worked:

* ``legacy_groups_loaded`` — should match the 6 saved groups.
* ``names_with_state`` — should be ~648, matching legacy state entries.
* ``names_matched_on_resume_estimate`` — rough estimate of how many of
  those names will match under the CURRENT (post-fix) trainer topology.
  Expect ~192 (the LoRA adapters) plus projection + compressor params.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import structlog
import torch

logger = structlog.get_logger()


def main(ckpt_dir: Path) -> None:
    if not ckpt_dir.is_dir():
        raise SystemExit(f"not a directory: {ckpt_dir}")
    legacy_path = ckpt_dir / "optimizer.pt"
    enc_path = ckpt_dir / "encoder.pt"
    dec_path = ckpt_dir / "decoder.pt"
    for p in (legacy_path, enc_path, dec_path):
        if not p.exists():
            raise SystemExit(f"missing expected file: {p}")

    # Load legacy optimizer state (pickled Python dict; weights_only=False
    # is fine -- we fully control the file).
    legacy_opt = torch.load(legacy_path, map_location="cpu", weights_only=False)
    n_saved_groups = len(legacy_opt["param_groups"])
    n_state_entries = len(legacy_opt["state"])
    saved_counts = [len(g["params"]) for g in legacy_opt["param_groups"]]
    logger.info(
        "legacy_state_loaded",
        n_groups=n_saved_groups,
        counts_per_group=saved_counts,
        n_state_entries=n_state_entries,
    )

    # Import torch-dependent bgkit modules only after torch is in scope
    # so the error message from a torch import failure isn't buried.
    from transformers import AutoModelForCausalLM

    from bgkit.models.decoder import ReconstructionDecoder
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.utils.attention_backend import resolve_attention_implementation

    attention_impl = resolve_attention_implementation("auto")

    # -------- Encoder --------
    encoder_sd = torch.load(enc_path, map_location="cpu", weights_only=True)
    logger.info("encoder_state_loaded", n_keys=len(encoder_sd))
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        "Qwen/Qwen3.5-0.8B-Base",
        encoder_sd,
        hidden_dim=1024,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attention_impl,
        bidi_warmup_steps=0,
    )

    # -------- Decoder + LoRA --------
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attention_impl,
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=1024)
    decoder.apply_lora(
        {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
    )
    decoder_sd = torch.load(dec_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(decoder_sd)
    logger.info("decoder_state_loaded", n_keys=len(decoder_sd))

    # -------- Replicate save-time unfreeze topology --------
    # Before the 2026-04-18 fix, the trainer called a blanket
    # ``decoder.requires_grad_(True)`` AFTER ``apply_lora``. That's what
    # produced the 512-param decoder group in the saved state. Replay it.
    encoder.projection_block.requires_grad_(True)
    encoder.compressor.requires_grad_(True)
    decoder.requires_grad_(True)
    encoder.projection_block.train()
    encoder.compressor.train()
    decoder.train()

    # -------- Build param groups in the SAME order as the trainer --------
    # DecoderInitTrainer._setup_optimizer order:
    #   1. projection_block (proj_lr)
    #   2. decoder         (decoder_lr)
    #   3. encoder.compressor (encoder_lr)
    # LRs from the saved param_groups metadata so we don't have to
    # re-derive them from config.
    proj_lr = saved_counts_to_lr(legacy_opt["param_groups"], group_idx=0)
    decoder_lr = saved_counts_to_lr(legacy_opt["param_groups"], group_idx=2)
    encoder_lr = saved_counts_to_lr(legacy_opt["param_groups"], group_idx=4)
    logger.info(
        "reconstructed_lrs",
        proj_lr=proj_lr,
        decoder_lr=decoder_lr,
        encoder_lr=encoder_lr,
    )

    proj_params = [p for p in encoder.projection_block.parameters() if p.requires_grad]
    decoder_params = [p for p in decoder.parameters() if p.requires_grad]
    compressor_params = [p for p in encoder.compressor.parameters() if p.requires_grad]

    logger.info(
        "current_param_counts",
        projection=len(proj_params),
        decoder=len(decoder_params),
        compressor=len(compressor_params),
        total=len(proj_params) + len(decoder_params) + len(compressor_params),
    )
    expected_total = sum(saved_counts)
    actual_total = len(proj_params) + len(decoder_params) + len(compressor_params)
    if actual_total != expected_total:
        raise SystemExit(
            f"param count mismatch: saved {expected_total} vs current "
            f"{actual_total}. Cannot migrate safely without knowing "
            f"exactly which params were in the saved state.",
        )

    param_groups = [
        {"params": proj_params, "lr": proj_lr, "base_lr": proj_lr},
        {"params": decoder_params, "lr": decoder_lr, "base_lr": decoder_lr},
        {"params": compressor_params, "lr": encoder_lr, "base_lr": encoder_lr},
    ]

    # -------- Build the exclude set for Muon (embed_tokens, lm_head) --------
    exclude_ids: set[int] = set()
    # Decoder embed + lm_head (1D-ish, but explicit exclusion matches the
    # trainer's Muon split logic in ``_muon_excluded_param_ids``).
    try:
        from peft import PeftModel

        backbone = decoder.backbone
        inner = backbone.base_model.model if isinstance(backbone, PeftModel) else backbone
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            for p in inner.model.embed_tokens.parameters():
                exclude_ids.add(id(p))
        if hasattr(inner, "lm_head"):
            for p in inner.lm_head.parameters():
                exclude_ids.add(id(p))
    except ImportError:
        pass
    # Encoder compressor's embed_tokens is also 2D but should be Muon-excluded
    # per ``PruningDistillTrainer._muon_excluded_param_ids`` convention;
    # DecoderInitTrainer doesn't add it explicitly, so don't add it here.

    # -------- Muon-split (Muon for 2D+ non-excluded, AdamW otherwise) --------
    from bgkit.training.muon import Muon

    split_groups: list[dict] = []
    for group in param_groups:
        meta = {k: v for k, v in group.items() if k != "params"}
        muon_params = [
            p for p in group["params"]
            if p.ndim >= 2 and id(p) not in exclude_ids
        ]
        adam_params = [
            p for p in group["params"]
            if p.ndim < 2 or id(p) in exclude_ids
        ]
        if muon_params:
            split_groups.append({"params": muon_params, "use_muon": True, **meta})
        if adam_params:
            split_groups.append({"params": adam_params, "use_muon": False, **meta})

    current_counts = [len(g["params"]) for g in split_groups]
    logger.info(
        "reconstructed_param_groups",
        n_groups=len(split_groups),
        counts_per_group=current_counts,
    )
    if current_counts != saved_counts:
        raise SystemExit(
            f"param-group topology doesn't match saved. Saved: "
            f"{saved_counts}, reconstructed: {current_counts}. Check the "
            f"Muon-split / exclude logic against the trainer at save time.",
        )

    # -------- Build the optimizer with exactly this topology --------
    optimizer = Muon(split_groups)

    # -------- Load legacy state (positional match should now succeed) --------
    optimizer.load_state_dict(legacy_opt)
    logger.info("legacy_state_load_ok", n_state_entries=len(optimizer.state))

    # -------- Walk named_parameters and re-key by name --------
    # Prefix matches DecoderInitTrainer._named_parameters_for_optimizer:
    #   "encoder.<...>" for encoder params
    #   "decoder.<...>" for decoder params
    state_by_name: dict = {}
    for name, param in itertools.chain(
        (("encoder." + n, p) for n, p in encoder.named_parameters()),
        (("decoder." + n, p) for n, p in decoder.named_parameters()),
    ):
        if param in optimizer.state:
            state_by_name[name] = dict(optimizer.state[param])

    logger.info(
        "name_keyed_state_built",
        names_with_state=len(state_by_name),
        legacy_state_entries=len(optimizer.state),
    )

    # -------- Estimate match rate under the current (post-fix) topology --------
    # LoRA-only decoder params contain "lora_" in their name.
    lora_names = sum(1 for n in state_by_name if ".lora_" in n)
    encoder_names = sum(1 for n in state_by_name if n.startswith("encoder."))
    logger.info(
        "current_resume_match_estimate",
        lora_adapter_names=lora_names,
        encoder_param_names=encoder_names,
        total_estimated_restore=lora_names + encoder_names,
        hint="LoRA adapters + all encoder params will be restored "
        "on resume; base-decoder moments are dropped (fine: those "
        "params are frozen under the 2026-04-18 fix).",
    )

    # -------- Write out --------
    out_path = ckpt_dir / "optimizer_state_by_name.pt"
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(state_by_name, tmp_path)
    tmp_path.rename(out_path)
    logger.info("wrote_migrated_state", path=str(out_path), n_keys=len(state_by_name))


def saved_counts_to_lr(param_groups: list, group_idx: int) -> float:
    """Pull the base_lr off a saved param_group entry (safer than config)."""
    g = param_groups[group_idx]
    return float(g.get("base_lr", g.get("lr")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(Path(sys.argv[1]))
