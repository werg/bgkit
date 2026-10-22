"""Build the encoder/decoder pair for a STANDALONE script (probe, cache, eval)
exactly as the trainer builds it: from the run's own experiment config, with
the trainer's construction and loading code. An encoder-adapter checkpoint
(optim.encoder_mode adapter) loads WITH its adapter; tree encodes must then
name their role (``encoder.role("base")`` = the frozen tree encoder whose
output the node caches hold).

Why this exists (2026-09-18): the tree probe and the root-cache script each
hand-built their models. The probe put the selection head at layer 8 (training:
``model.head_layer`` 6), skipped ``set_target_moments``, and loaded the decoder
with ``strict=False`` into a model with no LoRA installed -- so P28's adapter
was silently dropped and two "different" checkpoints scored identically to the
last digit. A standalone script must not re-derive the model; it asks for it
here.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from .config import Config, _validate
from .core import BgKIT2Core

_REPO = Path(__file__).resolve().parents[3]


def experiment_config(experiment: str) -> Config:
    """configs/training/bgkit2.yaml + configs/experiment/<experiment>.yaml,
    merged onto the dataclass schema (unknown keys raise)."""
    base = OmegaConf.load(_REPO / "configs/training/bgkit2.yaml").training.bgkit2
    exp = OmegaConf.load(_REPO / f"configs/experiment/{experiment}.yaml").training.bgkit2
    merged = OmegaConf.merge(
        OmegaConf.structured(Config), OmegaConf.to_container(base), OmegaConf.to_container(exp)
    )
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    _validate(cfg)
    return cfg


class StandaloneModels(BgKIT2Core):
    def __init__(self, cfg: Config, device: str = "cuda") -> None:
        self.cfg2 = cfg
        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if cfg.run.dtype == "bfloat16" else torch.float32


def load_models(
    experiment: str,
    checkpoint: str,
    *,
    decoder_checkpoint: str | None = None,
    device: str = "cuda",
) -> StandaloneModels:
    """Encoder (+ head, projection, interface) from ``checkpoint``; decoder
    adapter from ``decoder_checkpoint`` (default: the same checkpoint). Loads
    are STRICT in the trainer's sense: a dropped or missing tensor raises."""
    from bgkit.training.checkpointing import load_checkpoint

    core = StandaloneModels(experiment_config(experiment), device)
    # decoder + encoder tokenizers first: a decoder-space adapter (model.decoder_space_adapter)
    # is initialised over the two vocabularies at build time
    core.load_tokenizers()
    core.encoder = core.build_encoder()
    core.decoder = core.build_decoder()
    _m, state = load_checkpoint(Path(checkpoint))
    if core.cfg2.compaction_spine.enabled is None:
        core.resolve_spine_enabled(checkpoint_state=state["encoder"])
    # before the load: the checkpoint may carry the encoder adapter (served with
    # it whatever the experiment says) and / or a spine adapter
    core.resolve_encoder_adapter(state["encoder"])
    core.inject_encoder_adapter()
    core.inject_spine_adapter()
    core.load_encoder_state(core.encoder, state["encoder"])
    # as the trainer does after an encoder warm start
    for inorm in core.encoder.interface_norms():
        inorm.reseed_if_reference_out_of_contract()
    dstate = state.get("decoder")
    if decoder_checkpoint:
        dstate = load_checkpoint(Path(decoder_checkpoint))[1].get("decoder")
    has_adapter = bool(dstate) and any("lora" in k.lower() for k in dstate)
    if core.cfg2.model.lora.enabled and not has_adapter:
        raise ValueError("experiment enables a decoder adapter but the checkpoint has none")
    if has_adapter and not core.cfg2.model.lora.enabled:
        raise ValueError(
            "checkpoint carries a decoder adapter but the experiment has lora disabled -- "
            "pass the experiment the checkpoint was trained with"
        )
    if dstate:
        # adapter run: the adapter tensors; no-adapter run: the whole frozen LM
        # (what the trainer saved); model.decoder_save_trainable_only: the LoRA +
        # the decoder-space adapter over the base re-read from decoder_id. Either
        # way nothing may be dropped.
        core.load_decoder_state(dstate, source=decoder_checkpoint or checkpoint)
    core.finish_models()
    core.encoder.eval()
    core.decoder.eval()
    return core
