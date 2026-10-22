"""The bgkit2 training core, shared by the standalone CPU-test trainer and the
production ``BgKIT2Trainer(BaseTrainer)``.

A mixin: the host class provides ``cfg`` (bgkit2 ``Config``), ``device``,
``amp_dtype``, ``tok``, ``templates``, ``encoder``, ``decoder``, ``collate``,
``eval_ds``, ``task_eval``, ``ratio_schedule``, ``select_rng`` and a
``current_step`` property. Everything numerical lives here so the two hosts
cannot drift apart.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader

from bgkit_core.anchor import embed_anchor_losses
from bgkit_core.selection import (
    head_scale_anchor,
    listwise_kl,
    teacher_concentration,
    utility_targets,
)
from bgkit_core.splice_guard import SpliceGuard

from ..data.autoencode import AutoencodeDataset, Batch, Collator, task_indices
from ..data.framing_bank import FramingBank
from ..data.mixture import MixtureDataset
from ..data.tasks import ChatRenderer, TaskDataset
from ..data.templates import Templates, set_no_think_template
from ..data.token_store import TokenStore
from ..models.decoder import LoraSpec, SpliceDecoder
from ..models.encoder import BgKIT2Encoder, EncoderOutput
from ..models.interface_norm import rep_stats
from ..models.loading import load_decoder_lm, load_encoder_backbone, load_tokenizer
from ..models.survivorship import segment_zscore
from ..probes.rep_probe import rep_distinguishability
from .config import ADAPTER_MODES, Config, encoder_mode

try:  # project structlog wrapper
    from bgkit.utils.logging import get_logger

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover
    import structlog

    logger = structlog.get_logger(__name__)
from .metrics import corpus_bleu, exact_match, prefix_accuracy, token_f1
from .ratios import EVAL_RATIOS, RatioSchedule


def _prefix_cap(d, path: str) -> int:
    """Decoder-prefix cap for one store.

    ``task_max_prefix_tokens_by_store`` (keyed by file stem) overrides the
    global cap, so a store built for long-context anchoring keeps its long
    rows without lengthening decoder sequences for every other store.
    """
    return int(d.task_max_prefix_tokens_by_store.get(Path(path).stem, d.task_max_prefix_tokens))


class BgKIT2Core:
    cfg2: Config
    device: torch.device
    amp_dtype: torch.dtype
    tok: object
    templates: Templates
    encoder: BgKIT2Encoder
    decoder: SpliceDecoder
    collate: Collator
    eval_ds: AutoencodeDataset
    task_eval: dict[str, TaskDataset]
    ratio_schedule: RatioSchedule
    select_rng: random.Random

    @property
    def current_step(self) -> int:  # host overrides
        return 0

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def build_encoder(self) -> BgKIT2Encoder:
        m = self.cfg2.model
        backbone = load_encoder_backbone(m.encoder_id, dtype=torch.float32)
        return BgKIT2Encoder(
            backbone,
            head_layer=m.head_layer,
            head_inner_dim=m.head_inner_dim,
            projection_mult=m.projection_mult,
            interface_momentum=m.interface_momentum,
            selection=m.selection,
            min_survivors=m.min_survivors,
            gate_tau=m.gate_tau,
            interface_row_norm=m.interface_row_norm,
            interface_prenorm=m.interface_prenorm,
            interface_match_moments=m.interface_match_moments,
            summary_slot_frac=m.summary_slot_frac,
            projection_block=m.projection_block,
            projection_input_skip=m.projection_input_skip,
            interface_rms_only=m.interface_rms_only,
            prompt_separator=m.prompt_separator,
            prompt_conditioning=m.prompt_conditioning,
            repro_head=m.repro_head,
        )

    def build_decoder(self, attn_implementation: str = "sdpa", lora_cfg=None) -> SpliceDecoder:
        m = self.cfg2.model
        lm = load_decoder_lm(
            m.decoder_id, dtype=self.amp_dtype, attn_implementation=attn_implementation,
            experts_implementation=m.decoder_experts_implementation,
            fused_rmsnorm=m.decoder_fused_rmsnorm,
            device=getattr(self, "device", None),
        )
        lc = lora_cfg if lora_cfg is not None else m.lora
        lora = None
        if lc.enabled:
            lora = LoraSpec(
                rank=lc.rank,
                alpha=lc.alpha,
                dropout=lc.dropout,
                layers=list(lc.layers),
                rep_gated=bool(lc.rep_gated),
            )
        dec = SpliceDecoder(lm, lora=lora)
        if lora is not None:
            for p in dec.trainable_parameters():
                p.data = p.data.float()
        if m.decoder_space_adapter and lora_cfg is None:
            self.attach_space_adapter(dec)
        return dec

    def attach_space_adapter(self, dec: SpliceDecoder) -> None:
        """P8 (models.space_adapter): the encoder emits reps in ``rep_space_decoder_id``'s
        embedding space; ``dec`` reads them through a trained adapter whose linear part starts
        at the closed-form map between the two embedding tables over their shared token
        strings, and whose output keeps the interface contract toward ``dec``'s table."""
        from ..models.loading import load_embedding_table
        from ..models.space_adapter import DecoderSpaceAdapter, shared_token_pairs

        m = self.cfg2.model
        src = load_embedding_table(m.rep_space_decoder_id)
        self.rep_space_table = src
        ad = DecoderSpaceAdapter(
            src.shape[1], dec.hidden_size, hidden_mult=int(m.space_adapter_hidden_mult),
            momentum=m.interface_momentum, prenorm=m.interface_prenorm,
            match_moments=m.interface_match_moments,
        )
        dst = dec.embed_tokens.weight.detach().float().cpu()
        fit = ad.init_from_tables(src, dst, shared_token_pairs(load_tokenizer(m.rep_space_decoder_id),
                                                              self.tok))
        ad.contract.set_target_row_norm(dec.embed_row_norm)
        ad.contract.set_target_moments(dst)
        dec.space_adapter = ad
        logger.info("decoder_space_adapter_attached", rep_space=m.rep_space_decoder_id,
                    in_dim=int(src.shape[1]), out_dim=int(dec.hidden_size),
                    params=sum(p.numel() for p in ad.parameters()), **fit)

    @property
    def rep_row_norm(self) -> float:
        """Row norm of the space the ENCODER's reps live in (its decoder's table)."""
        t = getattr(self, "rep_space_table", None)
        return float(t.norm(dim=-1).mean()) if t is not None else self.decoder.embed_row_norm

    @property
    def rep_dim(self) -> int:
        t = getattr(self, "rep_space_table", None)
        return int(t.shape[1]) if t is not None else int(self.decoder.embed_tokens.weight.shape[1])

    # ------------------------------------------------------------------
    # tokenizers + decoder state: ONE source for the trainer and every
    # standalone reader of a checkpoint (student server, benches, probes)
    # ------------------------------------------------------------------
    def load_tokenizers(self) -> None:
        """``self.tok`` (the DECODER's) and ``self.ctx_tok`` (the ENCODER's on two-tokenizer
        runs, ``model.ctx_tokenizer_id``; None = ``self.tok`` serves both). Idempotent: a
        tokenizer the host already set is kept."""
        m = self.cfg2.model
        if getattr(self, "tok", None) is None:
            self.tok = load_tokenizer(m.decoder_id)
            if m.decoder_no_think:
                set_no_think_template(self.tok)
        if "ctx_tok" not in self.__dict__:
            self.ctx_tok = load_tokenizer(m.ctx_tokenizer_id) if m.ctx_tokenizer_id else None

    def chat_renderer(self) -> ChatRenderer:
        """The run's ``ChatRenderer``: decoder-side text with ``self.tok`` (its own sentinel),
        encoder-side text (``encode_ctx`` / ``ctx_tok``) with the encoder's tokenizer."""
        self.load_tokenizers()
        return ChatRenderer.from_tokenizer(self.tok, enc_tok=self.ctx_tok)

    def load_decoder_state(self, dec_state: dict, source: str = "") -> None:
        """Strict decoder load of a checkpoint's ``decoder`` state: the whole state dict, or --
        ``model.decoder_save_trainable_only`` (8B: the base is re-read from ``decoder_id``) --
        exactly the trainable tensors + the space adapter (``TrainableDecoderState``)."""
        target = self.decoder
        if self.cfg2.model.decoder_save_trainable_only:
            from ..models.decoder import TrainableDecoderState

            target = TrainableDecoderState(self.decoder)
        missing, unexpected = target.load_state_dict(dec_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"decoder state mismatch{f' ({source})' if source else ''}: "
                f"{list(missing)[:5]} {list(unexpected)[:5]}")

    # Buffers added after a checkpoint may legitimately be absent from it; they
    # are re-derived at setup (finish_models sets the target moments from the
    # decoder). Anything else missing or unexpected is a real mismatch.
    _ENCODER_OPTIONAL_KEYS: ClassVar[tuple[str, ...]] = (
        "interface_norm.target_mean",
        "interface_norm.target_std",
    )

    @staticmethod
    def load_head_state(encoder: BgKIT2Encoder, state: dict) -> int:
        """Copy only the selection head (``head.*``) and ``survive_embedding``
        from another encoder state dict; everything else is left as is."""
        sub = {k: v for k, v in state.items() if k.startswith("head.") or k == "survive_embedding"}
        if not sub:
            raise ValueError("checkpoint carries no selection head")
        _missing, unexpected = encoder.load_state_dict(sub, strict=False)
        if unexpected:
            raise ValueError(f"unexpected head tensors: {list(unexpected)[:5]}")
        return len(sub)

    @classmethod
    def load_encoder_state(
        cls, encoder: BgKIT2Encoder, state: dict, drop_prefixes: tuple[str, ...] = ()
    ) -> list[str]:
        """Strict load except for the optional buffers; returns the keys that
        were absent and left at their construction values. ``drop_prefixes``
        names sub-modules to leave at their fresh initialisation."""
        if drop_prefixes:
            state = {k: v for k, v in state.items() if not k.startswith(tuple(drop_prefixes))}
        state = cls._remap_adapter_keys(encoder, state)
        missing, unexpected = encoder.load_state_dict(state, strict=False)

        def optional(k: str) -> bool:
            if drop_prefixes and k.startswith(tuple(drop_prefixes)):
                return True
            return (
                k in cls._ENCODER_OPTIONAL_KEYS
                or k.startswith("repro.")  # the multi-level bridge post-dates every checkpoint
                or k.startswith("projection_block.")
                or k.startswith("input_skip.")
                or k == "prompt_separator"
                or k.startswith("prompt_cond_")
                # a checkpoint that predates the compaction-spine adapter: the
                # adapter starts at its zero delta, the bridge copy from repro
                or ".spine." in k
                or k.startswith("spine_repro.")
                # ... or the encoder adapter (optim.encoder_mode adapter): the
                # LoRA starts at its zero delta, the part copies from the base
                or ".slots." in k
                or k.startswith("slot_parts.")
            )

        tolerated = [k for k in missing if optional(k)]
        real_missing = [k for k in missing if not optional(k)]
        if real_missing or unexpected:
            raise RuntimeError(f"encoder state mismatch: {real_missing[:5]} {list(unexpected)[:5]}")
        if any(k.startswith("projection_block.") for k in tolerated):
            # The checkpoint predates the block: start it from the LOADED backbone.
            encoder.reset_projection_block_from_backbone()
            if any(k.startswith("slot_parts.projection_block.") for k in tolerated):
                encoder.resync_slot_parts(("projection_block",))
        return tolerated

    @staticmethod
    def _adapter_kind(k: str) -> str | None:
        if ".spine." in k or k.startswith("spine_repro."):
            return "spine"
        if ".slots." in k or k.startswith("slot_parts."):
            return "slots"
        return None

    @classmethod
    def _remap_adapter_keys(cls, encoder: BgKIT2Encoder, state: dict) -> dict:
        """peft renames a wrapped layer's weight to ``<layer>.base_layer.weight``.
        A checkpoint without an adapter (compaction spine or encoder adapter)
        loads into an encoder with it -- and the reverse, with the adapter's
        tensors dropped LOUDLY -- by mapping the names. Adapter copies the
        checkpoint lacks start from the LOADED base: ``spine_repro`` from
        ``repro``, ``slot_parts.X`` from ``X``. Used by every encoder load,
        including the trainer's resume (``BgKIT2Trainer._migrate_state_dict``)."""
        own = set(encoder.state_dict())
        out = {}
        dropped: dict[str, int] = {}
        for k, v in state.items():
            if k in own:
                out[k] = v
                continue
            kind = cls._adapter_kind(k)
            if kind is not None:
                dropped[kind] = dropped.get(kind, 0) + 1  # this encoder has no such adapter
                continue
            parts = k.rsplit(".", 1)
            with_base = f"{parts[0]}.base_layer.{parts[1]}" if len(parts) == 2 else k
            without = k.replace(".base_layer.", ".")
            out[with_base if with_base in own else without if without in own else k] = v
        import warnings

        if dropped.get("spine"):
            warnings.warn(f"checkpoint carries {dropped['spine']} compaction-spine tensors this "
                          "encoder has no adapter for; they are NOT loaded", stacklevel=3)
        if dropped.get("slots"):
            warnings.warn(f"checkpoint carries {dropped['slots']} encoder-adapter tensors "
                          "(optim.encoder_mode adapter) this encoder has no adapter for; they "
                          "are NOT loaded -- load it with the adapter "
                          "(BgKIT2Core.resolve_encoder_adapter)", stacklevel=3)
        if getattr(encoder, "spine_repro", None) is not None and not any(
            k.startswith("spine_repro.") for k in out
        ):
            # fresh adapter: its bridge starts as a copy of the LOADED repro head
            for k, v in list(out.items()):
                if k.startswith("repro."):
                    out["spine_" + k] = v
        for k in own:
            if k.startswith("slot_parts.") and k not in out:
                base = k[len("slot_parts."):]
                if base in out:
                    out[k] = out[base]  # a fresh slot adapter starts at the LOADED part
        return out

    # the pre-generalisation name (spine-only), kept for callers
    _remap_spine_keys = _remap_adapter_keys

    def build_teacher(self) -> SpliceDecoder | None:
        """A separate frozen decoder carrying the full-text adapter named by
        ``kd.teacher_checkpoint`` (same ``model.lora`` spec), or None."""
        src = self.cfg2.kd.teacher_checkpoint
        if not src:
            return None
        from pathlib import Path

        from bgkit.training.checkpointing import load_checkpoint

        tl = self.cfg2.kd.teacher_lora
        if tl is None and not self.cfg2.model.lora.enabled:
            raise ValueError(
                "kd.teacher_checkpoint needs an adapter spec: model.lora or kd.teacher_lora"
            )
        # sdpa; attention_importance switches to eager per pass
        teacher = self.build_decoder(lora_cfg=tl)
        _meta, state = load_checkpoint(Path(src))
        dec_state = state.get("decoder")
        if not dec_state:
            raise ValueError(f"{src} carries no decoder state")
        missing, unexpected = teacher.load_state_dict(dec_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"teacher load mismatch: {missing[:5]} {unexpected[:5]}")
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        teacher.eval()
        teacher.attention_importance_chunk = int(self.cfg2.kd.selection_chunk)
        return teacher.to(self.device)

    def finish_models(self) -> list:
        """Target norm, freeze policy, gradient checkpointing. Returns the
        trainable backbone layer list."""
        for inorm in self.encoder.interface_norms():
            # the ENCODER's contract is toward the space its reps live in (P8 maps them onward)
            inorm.set_target_row_norm(self.rep_row_norm)
            inorm.set_target_moments(
                getattr(self, "rep_space_table", None)
                if getattr(self, "rep_space_table", None) is not None
                else self.decoder.embed_tokens.weight
            )
        layers = self.encoder.set_backbone_trainable(list(self.cfg2.model.train_backbone_layers))
        el = self.cfg2.model.encoder_lora
        if el.enabled:
            n = self.encoder.inject_lora(el.rank, el.alpha, el.dropout, list(el.layers) or None)
            layers = [*layers, f"lora:{n}"]
        n = self.inject_encoder_adapter()
        if n:
            layers = [*layers, f"encoder_adapter:{n}"]
        n = self.inject_spine_adapter()
        if n:
            layers = [*layers, f"spine_adapter:{n}"]
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return layers

    def param_groups(self) -> list[dict]:
        o = self.cfg2.optim
        mode = encoder_mode(self.cfg2)
        if mode == "adapter":
            return self._adapter_param_groups()
        if mode in ("frozen", "frozen_adapter"):
            spine = (
                self.encoder.spine_parameters()
                if self.encoder.has_spine_adapter and mode == "frozen" else []
            )
            for prm in self.encoder.parameters():
                prm.requires_grad_(False)
            for prm in spine:  # the spine adapter trains even with a frozen encoder
                prm.requires_grad_(True)
            lora_params = self.decoder.trainable_parameters()
            if not lora_params and not spine and not self.decoder.space_adapter_parameters():
                raise ValueError(
                    "optim.freeze_encoder with a frozen decoder leaves nothing to train"
                )
            out = [{"params": lora_params, "lr": o.lr_lora, "name": "lora"}] if lora_params else []
            p8 = self.decoder.space_adapter_parameters()
            if p8:
                out.append({"params": p8, "lr": o.lr_space_adapter, "name": "space_adapter"})
            if spine:
                out.append({"params": spine, "lr": self.cfg2.compaction_spine.lr, "name": "spine"})
            for g in out:
                g["base_lr"] = g["lr"]
            return out
        groups = self.encoder.trainable_groups()
        pgs = [
            {"params": groups["backbone"], "lr": o.lr_backbone, "name": "backbone"},
            {
                "params": groups["head"] + groups["survive_embedding"],
                "lr": o.lr_head,
                "name": "head",
            },
            {"params": groups["projection"], "lr": o.lr_projection, "name": "projection"},
            {"params": groups["prompt_cond"], "lr": o.lr_head, "name": "prompt_cond"},
            {
                "params": groups["projection_block"],
                "lr": o.lr_projection_block,
                "name": "projection_block",
            },
        ]
        if groups.get("repro"):
            pgs.append({"params": groups["repro"], "lr": o.lr_repro, "name": "repro"})
        if bool(getattr(o, "train_only_repro", False)):
            pgs = [g for g in pgs if g["name"] == "repro"]
            if not pgs:
                raise ValueError("train_only_repro needs model.repro_head: true")
            for prm in self.encoder.parameters():
                prm.requires_grad_(False)
            for prm in self.encoder.repro.parameters():
                prm.requires_grad_(True)
            for prm in self.decoder.parameters():
                prm.requires_grad_(False)
            return pgs
        lora_params = self.decoder.trainable_parameters()
        if lora_params:
            pgs.append({"params": lora_params, "lr": o.lr_lora, "name": "lora"})
        if self.encoder.has_spine_adapter:
            spine = set(map(id, self.encoder.spine_parameters()))
            for g in pgs:  # the backbone group collects requires_grad params: keep them apart
                g["params"] = [p for p in g["params"] if id(p) not in spine]
            pgs.append({"params": self.encoder.spine_parameters(),
                        "lr": self.cfg2.compaction_spine.lr, "name": "spine"})
        pgs = [g for g in pgs if g["params"]]
        for g in pgs:
            g["base_lr"] = g["lr"]
        return pgs

    def _adapter_param_groups(self) -> list[dict]:
        """optim.encoder_mode adapter: the BASE encoder is frozen and out of the
        optimizer (it is the tree-cache encoder); the encoder adapter (slot
        LoRA + part copies), the spine bridge copy and the decoder adapter
        train. The invariant the tree cache relies on -- nothing the base role
        reads has requires_grad -- is established here and checked."""
        o, enc = self.cfg2.optim, self.encoder
        if not enc.has_slot_adapter:
            raise ValueError("optim.encoder_mode adapter but the encoder has no slot adapter "
                             "(finish_models installs it)")
        slot, spine = enc.slot_adapter_parameters(), enc.spine_parameters()
        train = {id(p) for p in [*slot, *spine]}
        for prm in enc.parameters():
            prm.requires_grad_(id(prm) in train)
        base = [n for n, p in enc.named_parameters() if p.requires_grad
                and not (n.startswith(("slot_parts.", "spine_repro.")) or ".slots." in n)]
        if base:  # by construction impossible; the tree cache depends on it
            raise RuntimeError(f"base encoder parameters would train in adapter mode: {base[:5]}")
        parts = enc.slot_part_groups()
        lora = [p for p in enc._lora_parameters(enc.SLOT_ADAPTER) if p.requires_grad]
        a = self.cfg2.model.encoder_adapter
        pgs = [
            {"params": lora, "lr": a.lr, "name": "encoder_adapter"},
            {"params": parts["head"], "lr": o.lr_head, "name": "encoder_adapter_head"},
            {"params": parts["prompt_cond"], "lr": o.lr_head,
             "name": "encoder_adapter_prompt_cond"},
            {"params": parts["projection"], "lr": o.lr_projection,
             "name": "encoder_adapter_projection"},
            {"params": parts["projection_block"], "lr": o.lr_projection_block,
             "name": "encoder_adapter_projection_block"},
            {"params": spine, "lr": self.cfg2.compaction_spine.lr, "name": "spine"},
            {"params": self.decoder.trainable_parameters(), "lr": o.lr_lora, "name": "lora"},
        ]
        pgs = [g for g in pgs if g["params"]]
        for g in pgs:
            g["base_lr"] = g["lr"]
        return pgs

    def inject_encoder_adapter(self) -> int:
        """Install the encoder adapter when ``optim.encoder_mode`` is adapter
        (idempotent). Must run BEFORE loading a checkpoint that carries it."""
        if encoder_mode(self.cfg2) not in ADAPTER_MODES or self.encoder.has_slot_adapter:
            return 0
        a = self.cfg2.model.encoder_adapter
        layers = self.encoder_adapter_layers()
        n = self.encoder.inject_slot_adapter(
            a.rank, a.alpha, a.dropout, layers, list(a.targets),
            train_parts=bool(a.train_parts),
        )
        logger.info("encoder_adapter_installed", params=n, rank=a.rank, alpha=a.alpha,
                    lr=a.lr, layers=layers or "all", train_parts=bool(a.train_parts),
                    grad_token_budget=int(self.cfg2.optim.encoder_grad_token_budget),
                    live_tree_max_files=int(self.cfg2.multilevel.live_tree_max_files))
        return n

    def encoder_adapter_layers(self) -> list[int] | None:
        """Backbone layers the encoder adapter wraps: ``layers`` if given, else
        the last ``top_layers`` (None = every layer)."""
        a = self.cfg2.model.encoder_adapter
        if a.layers:
            return [int(i) for i in a.layers]
        n = int(self.encoder.backbone.num_layers)
        k = int(a.top_layers)
        if k <= 0 or k >= n:
            return None
        return list(range(n - k, n))

    # ------------------------------------------------------------------
    # encoder gradient token budget (optim.encoder_grad_token_budget)
    # ------------------------------------------------------------------
    def encoder_grad_budget_active(self) -> bool:
        return (
            encoder_mode(self.cfg2) == "adapter"
            and int(self.cfg2.optim.encoder_grad_token_budget) > 0
            and self.encoder.training
            and torch.is_grad_enabled()
        )

    def _enc_grad_account(self, n_tokens: int, n_slots: int, n_grad_tokens: int,
                          n_grad_slots: int) -> None:
        acc = self.__dict__.setdefault("_enc_grad_acc", {})
        for k, v in (("slot_tokens", n_tokens), ("slots", n_slots),
                     ("grad_tokens", n_grad_tokens), ("grad_slots", n_grad_slots)):
            acc[k] = acc.get(k, 0) + int(v)

    def _enc_grad_reset(self) -> None:
        self.__dict__["_enc_grad_acc"] = {}

    def enc_grad_stats(self) -> dict[str, float]:
        acc = self.__dict__.get("_enc_grad_acc") or {}
        if not acc:
            return {}
        return {f"enc_grad/{k}": float(v) for k, v in acc.items()}

    def plan_extra_grad(self, batch: Batch) -> set[int] | None:
        """Extra-slot rows that encode WITH gradient under the budget (None =
        all of them). What the primary slots of this microbatch already used
        is subtracted first; the rest goes to the extra slots, the most recent
        slot of each sample first, round-robin over samples in a per-step
        random order, greedily (a slot that does not fit is skipped)."""
        if not self.encoder_grad_budget_active():
            return None
        n = int(batch.extra_input_ids.shape[0])
        lens = batch.extra_attention_mask.sum(-1).tolist()
        if batch.extra_prompt_mask is not None:
            plens = batch.extra_prompt_mask.sum(-1).tolist()
            lens = [int(a) + int(b) for a, b in zip(lens, plens, strict=True)]
        acc = self.__dict__.get("_enc_grad_acc") or {}
        left = int(self.cfg2.optim.encoder_grad_token_budget) - int(acc.get("grad_tokens", 0))
        per: dict[int, list[int]] = {}
        for r in range(n):
            per.setdefault(int(batch.extra_owner[r]), []).append(r)
        order = list(per)
        random.Random(self.current_step * 1_000_003 + n).shuffle(order)
        queues = {o: list(reversed(per[o])) for o in order}  # most recent slot first
        grad: set[int] = set()
        while any(queues.values()):
            for o in order:
                if not queues[o]:
                    continue
                r = queues[o].pop(0)
                if int(lens[r]) <= left:
                    grad.add(r)
                    left -= int(lens[r])
        return grad

    def _account_primary(self, batch: Batch) -> None:
        """Primary slots always take gradient; count them against the budget."""
        if not self.encoder_grad_budget_active():
            return
        t = int(batch.attention_mask.sum().item())
        if batch.prompt_mask is not None:
            t += int(batch.prompt_mask.sum().item())
        b = len(batch.tasks)
        self._enc_grad_account(t, b, t, b)

    def resolve_encoder_adapter(self, checkpoint_state: dict | None) -> bool:
        """A checkpoint that carries the encoder adapter is SERVED with it (eval
        server, standalone scripts) whatever the experiment says: switches
        ``optim.encoder_mode`` to adapter so train == eval. Returns whether
        the adapter is on."""
        has = bool(checkpoint_state) and any(
            self._adapter_kind(k) == "slots" for k in checkpoint_state
        )
        if has and encoder_mode(self.cfg2) not in ADAPTER_MODES:
            import logging

            logging.getLogger(__name__).warning(
                "checkpoint carries an encoder adapter: optim.encoder_mode %s -> adapter",
                encoder_mode(self.cfg2))
            self.cfg2.optim.encoder_mode = "adapter"
            self.cfg2.optim.freeze_encoder = False
        return encoder_mode(self.cfg2) in ADAPTER_MODES

    def build_datasets(self, seed: int):
        d = self.cfg2.data
        ctx_tok = getattr(self, "ctx_tok", None)
        prompts = (
            Templates.compression_prompt_ids(ctx_tok or self.tok) if d.compression_prompts else None
        )
        train = AutoencodeDataset(
            [TokenStore(p) for p in d.train_stores],
            weights=list(d.train_weights) if d.train_weights else None,
            ctx_min=d.ctx_min,
            ctx_max=d.ctx_max,
            cont_min=d.cont_min,
            cont_max=d.cont_max,
            p_continue=d.p_continue,
            seed=seed,
            epoch_size=d.epoch_size,
            task_prompt_ids=prompts,
        )
        dec_view = ChatRenderer.from_tokenizer(self.tok, enc_tok=ctx_tok).dec_view if ctx_tok else None
        train.dec_view = dec_view
        evald = AutoencodeDataset(
            [TokenStore(p) for p in d.eval_stores],
            ctx_min=d.ctx_min,
            ctx_max=d.ctx_max,
            cont_min=d.cont_min,
            cont_max=d.cont_max,
            p_continue=d.p_continue,
            seed=10_007,
            epoch_size=d.eval_samples,
            task_prompt_ids=prompts,
        )
        evald.dec_view = dec_view
        task_eval: dict[str, TaskDataset] = {}
        if d.task_stores:
            renderer = ChatRenderer.from_tokenizer(self.tok, enc_tok=ctx_tok)
            bank = (
                FramingBank("train", d.framing_p_system, d.framing_p_no_prompt)
                if d.framing_diversity
                else None
            )
            members = [
                TaskDataset(
                    path,
                    renderer,
                    split="train",
                    max_ctx_tokens=d.task_max_ctx_tokens,
                    max_target_tokens=d.task_max_target_tokens,
                    max_prefix_tokens=_prefix_cap(d, path),
                    seed=seed,
                    bank=bank,
                    sibling_prompts=Path(path).stem in set(d.alt_prompt_stores),
                )
                for path in d.task_stores
            ]
            tw = list(d.task_weights) if d.task_weights else [1.0] * len(members)
            tasks = MixtureDataset(members, tw, seed=seed + 7)
            n_epoch = d.epoch_size or (len(train) + len(tasks))
            train = MixtureDataset(
                [train, tasks],
                [1.0 - d.task_mix, d.task_mix],
                seed=seed + 11,
                epoch_size=n_epoch,
            )
            for path in d.task_stores:
                task_eval[Path(path).stem] = TaskDataset(
                    path,
                    renderer,
                    split="eval",
                    max_ctx_tokens=d.task_max_ctx_tokens,
                    max_target_tokens=d.task_max_target_tokens,
                    max_prefix_tokens=_prefix_cap(d, path),
                    seed=10_009,
                    epoch_size=d.task_eval_samples,
                )
                if (
                    d.framing_diversity
                    and d.task_eval_heldout_samples > 0
                    and (not d.heldout_stores or Path(path).stem in d.heldout_stores)
                ):
                    task_eval[Path(path).stem + "_heldout"] = TaskDataset(
                        path,
                        renderer,
                        split="eval",
                        max_ctx_tokens=d.task_max_ctx_tokens,
                        max_target_tokens=d.task_max_target_tokens,
                        max_prefix_tokens=_prefix_cap(d, path),
                        seed=10_009,
                        epoch_size=d.task_eval_heldout_samples,
                        bank=FramingBank("heldout", d.framing_p_system, d.framing_p_no_prompt),
                    )
        if d.eval_only_stores:
            renderer = ChatRenderer.from_tokenizer(self.tok, enc_tok=ctx_tok)
            for path in d.eval_only_stores:
                task_eval[Path(path).stem] = TaskDataset(
                    path,
                    renderer,
                    split="eval",
                    max_ctx_tokens=d.task_max_ctx_tokens,
                    max_target_tokens=d.task_max_target_tokens,
                    max_prefix_tokens=_prefix_cap(d, path),
                    seed=10_009,
                    epoch_size=d.task_eval_samples,
                )
        return train, evald, task_eval

    # ------------------------------------------------------------------
    # forward pieces
    # ------------------------------------------------------------------
    def autocast(self):
        if self.device.type == "cuda" and self.amp_dtype == torch.bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def encode(self, batch: Batch, ratio: float, selection: str | None = None):
        self._account_primary(batch)
        out = self.encoder(
            batch.input_ids,
            batch.attention_mask,
            ratio=ratio,
            selection=selection,
            prompt_ids=batch.prompt_ids,
            prompt_mask=batch.prompt_mask,
        )
        return self.finish_reps(batch.input_ids, out)

    def finish_reps(self, input_ids: torch.Tensor, out: EncoderOutput) -> EncoderOutput:
        """Post-encoder rep assembly shared by training, eval and the external
        evaluators: token residual on the survivor reps, then the summary
        slots appended after them."""
        import dataclasses

        m = self.cfg2.model
        if m.rep_token_residual or m.rep_token_residual_only:
            out = self.add_token_residual(input_ids, out)
        if out.slot_reps is not None:
            reps = torch.cat([out.reps, out.slot_reps.to(out.reps.dtype)], dim=1)
            mask = torch.cat([out.rep_mask, out.slot_mask], dim=1)
            pos = torch.cat(
                [out.positions, torch.zeros_like(out.slot_mask, dtype=out.positions.dtype)], dim=1
            )
            out = dataclasses.replace(out, reps=reps, rep_mask=mask, positions=pos)
        return out

    def add_token_residual(self, input_ids: torch.Tensor, out: EncoderOutput) -> EncoderOutput:
        """rep_i += scale * E_dec[token at the survivor's position] (masked);
        with ``rep_token_residual_only`` the encoder rep is dropped."""
        import dataclasses

        m = self.cfg2.model
        ids = input_ids.gather(1, out.positions.clamp_min(0))
        with torch.no_grad():
            emb = self.decoder.embed(ids).to(out.reps.dtype)
        emb = emb * out.rep_mask.unsqueeze(-1).to(emb.dtype) * float(m.rep_token_residual_scale)
        if out.gate is not None and m.rep_token_gate:
            # Straight-through gate (forward 1): the selection head receives the
            # decoder's gradient through the tokens it chose (P17).
            emb = emb * out.gate.to(emb.dtype).unsqueeze(-1)
        reps = emb if m.rep_token_residual_only else out.reps + emb
        return dataclasses.replace(out, reps=reps)

    def full_text_batch(self, batch: Batch, pad_side: str = "right", teacher_window: bool = False):
        """The ceiling arm: the context's own token embeddings in the rep slot.
        ``teacher_window`` (kd.teacher_gold_window): rows carrying an oracle
        window get THAT in the slot instead of the document. Extra slots get
        their own token embeddings."""
        ids, mask = batch.input_ids, batch.attention_mask
        e_ids, e_mask = batch.extra_input_ids, batch.extra_attention_mask
        if batch.dec_ctx is not None:
            # two tokenizers: input_ids are ENCODER ids; the decoder reads the same text in its own
            ids, mask = self._pad_ids(batch.dec_ctx)
            if batch.dec_extra_ctx is not None:
                e_ids, e_mask = self._pad_ids(batch.dec_extra_ctx)
        if teacher_window and batch.teacher_input_ids is not None:
            ids, mask = batch.teacher_input_ids, batch.teacher_attention_mask
        with torch.no_grad():
            reps = self.decoder.embed(ids)
            extra = None
            if e_ids is not None:
                e_reps = self.decoder.embed(e_ids)
                extra = self._group_extra(batch, e_reps, e_mask.bool())
        if extra is None:
            return self.splice(batch, reps, mask.bool(), pad_side=pad_side, decoder_space=True)
        return self.splice(batch, reps, mask.bool(), pad_side=pad_side, extra=extra,
                           decoder_space=True)

    def _pad_ids(self, seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-padded (N, L) ids + mask on the device (pad 0 is masked out; L >= 1)."""
        n = max(1, max((int(t.shape[0]) for t in seqs), default=1))
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, t in enumerate(seqs):
            ids[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = 1
        return ids.to(self.device), mask.to(self.device)

    def _group_extra(self, batch: Batch, e_reps, e_mask) -> list[list[tuple]]:
        """Stacked extra-slot reps (E, K, D) -> per sample, in slot order."""
        per: list[list[tuple]] = [[] for _ in batch.tasks]
        for r in range(int(batch.extra_owner.shape[0])):
            per[int(batch.extra_owner[r])].append((e_reps[r], e_mask[r]))
        return per

    # ------------------------------------------------------------------
    # cons-mode compaction spines (models.multilevel.encode_spines)
    # ------------------------------------------------------------------
    def resolve_spine_enabled(self, store_paths: list[str] | None = None,
                              checkpoint_state: dict | None = None) -> bool:
        """Settle ``compaction_spine.enabled`` when it is None (auto): on iff a
        store carries spine rows or the checkpoint's encoder state carries the
        spine adapter. Logged, so the decision is visible at launch."""
        c = self.cfg2.compaction_spine
        if c.enabled is not None:
            return bool(c.enabled)
        from ..data.tasks import store_has_spines

        why = None
        for p in store_paths or []:
            try:
                if store_has_spines(p):
                    why = f"store {p} carries compaction-spine rows"
                    break
            except (OSError, ValueError, KeyError):
                continue  # a missing / foreign store is the data loader's error to raise
        if why is None and checkpoint_state and any(
            ".spine." in k or k.startswith("spine_repro.") for k in checkpoint_state
        ):
            why = "the checkpoint carries a compaction-spine adapter"
        c.enabled = why is not None
        import logging

        logging.getLogger(__name__).warning(
            "compaction_spine.enabled auto -> %s (%s)", c.enabled,
            why or "no store carries spine rows and the checkpoint has no spine adapter")
        return c.enabled

    def inject_spine_adapter(self) -> int:
        """Install the compaction-spine adapter when configured (idempotent);
        must run BEFORE a checkpoint that carries it is loaded."""
        c = self.cfg2.compaction_spine
        if c.enabled is None:
            self.resolve_spine_enabled(
                [*self.cfg2.data.task_stores, *self.cfg2.data.eval_only_stores])
        if not c.enabled or c.trainable != "adapter" or self.encoder.has_spine_adapter:
            return 0
        if encoder_mode(self.cfg2) in ADAPTER_MODES:
            # ONE encoder adapter: the spine role runs the slot adapter; only
            # the bridge gets a spine copy (the shared repro serves the trees)
            self.encoder.add_spine_bridge(bool(c.train_bridge))
            return sum(p.numel() for p in self.encoder.spine_parameters())
        return self.encoder.inject_spine_adapter(
            c.rank, c.alpha, c.dropout, list(c.layers) or None, list(c.targets),
            train_bridge=bool(c.train_bridge),
        )

    @staticmethod
    def _prompt_row(ids, mask, i: int):
        if ids is None:
            return None
        m = mask[i].bool()
        return ids[i][m] if bool(m.any()) else None

    def spine_bridge(self):
        b = str(self.cfg2.multilevel.bridge)
        return (lambda x: x) if b == "identity" else None

    def spine_output(self, spines: list[list[torch.Tensor]], prompts: list, leaf_given=None,
                     cell_given=None):
        """``encode_spines`` with this run's settings -- the ONE spine encoder call
        shared by the trainer and the student server (which passes ``start`` from
        its node cache)."""
        from ..models.multilevel import encode_spines

        c = self.cfg2.compaction_spine
        if not c.enabled:
            raise ValueError(
                "a row carries a compaction spine but compaction_spine.enabled is off "
                "(auto found no spine rows in the configured stores at setup)")
        trains = c.trainable == "adapter" and self.encoder.training and torch.is_grad_enabled()
        return encode_spines(
            self.encoder, spines,
            tok_encode=lambda s: (getattr(self, "ctx_tok", None) or self.tok)(
                s, add_special_tokens=False)["input_ids"],
            leaf_ratio=float(c.leaf_ratio), cell_ratio=float(c.cell_ratio), prompts=prompts,
            grad_levels=int(c.grad_levels) if trains else 0,
            max_depth=int(c.max_depth), node_max_tokens=int(c.node_max_tokens),
            device=self.device, bridge=self.spine_bridge(),
            token_sq_budget=int(self.cfg2.data.extra_encode_token_sq_budget),
            leaf_given=leaf_given, cell_given=cell_given,
        )

    def encode_spine_slots(self, spines: list[list[torch.Tensor]], prompts: list) -> list[tuple]:
        """[(reps (K, D), mask (K,))] of each spine's TOP node."""
        out = self.spine_output(spines, prompts)
        return [(r, torch.ones(r.shape[0], dtype=torch.bool, device=r.device)) for r in out.reps]

    @staticmethod
    def spine_rows(batch: Batch) -> list[int]:
        sp = getattr(batch, "spine_slot", None)
        return [i for i, x in enumerate(sp) if x is not None] if sp else []

    def spine_primary(self, sub: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [self._prompt_row(sub.prompt_ids, sub.prompt_mask, i)
                   for i in range(len(sub.tasks))]
        enc = self.encode_spine_slots(list(sub.spine_slot), prompts)
        width = max(int(r.shape[0]) for r, _ in enc)
        reps = torch.zeros(len(enc), width, enc[0][0].shape[-1], dtype=enc[0][0].dtype,
                           device=self.device)
        mask = torch.zeros(len(enc), width, dtype=torch.bool, device=self.device)
        for i, (r, _m) in enumerate(enc):
            reps[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = True
        return reps, mask

    def spine_loss(self, sub: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """CE on rows whose PRIMARY slot is a compaction spine (the overview
        rows' shape): spine reps + the row's other slots through encode_extra."""
        ratio = self.ratio_schedule.sample(self.current_step)
        with self.autocast():
            fam_ratio, extra_ratio = self.slot_ratios(sub, ratio)
            reps, mask = self.spine_primary(sub)
            extra = self.encode_extra(sub, fam_ratio, None, extra_ratio=extra_ratio)
            sb = (self.splice(sub, reps, mask) if extra is None
                  else self.splice(sub, reps, mask, extra=extra))
            nll, valid = self.decoder.token_nll(sb)
        ce = self.pooled_ce(sub, nll, valid)
        stats = {f"loss/{f}_spine": float(ce.item()) for f in sorted(set(sub.tasks))}
        self.guard_reps("spine", reps, mask, stats)
        stats["spine_rows"] = float(len(sub.tasks))
        stats["spine_leaves"] = float(sum(len(x) for x in sub.spine_slot) / len(sub.tasks))
        return ce, stats

    @torch.no_grad()
    def spine_eval_metrics(self, batches: list[Batch]) -> dict[str, float]:
        """Teacher-forced CE on spine rows: with the spine, the spine ZEROED
        (other slots intact) and no slot at all."""
        tot = {"sp": 0.0, "zero": 0.0, "noslot": 0.0}
        n_tok = 0.0
        n_rows = 0
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                reps, mask = self.spine_primary(batch)
                extra = self.encode_extra(batch, 1.0 / 8.0, self.eval_selection)
                kw = {"extra": extra} if extra is not None else {}
                arms = {
                    "sp": self.splice(batch, reps, mask, **kw),
                    "zero": self.splice(batch, torch.zeros_like(reps), mask, **kw),
                    "noslot": self.splice(batch, None, None),
                }
                for k, sb in arms.items():
                    nll, valid = self.decoder.token_nll(sb)
                    tot[k] += float(nll.sum().item())
                    if k == "sp":
                        n_tok += float(valid.sum().item())
            n_rows += len(batch.tasks)
        n_tok = max(n_tok, 1.0)
        return {
            "spine/loss": tot["sp"] / n_tok,
            "spine/zeroed_spine_loss": tot["zero"] / n_tok,
            "spine/noslot_loss": tot["noslot"] / n_tok,
            "spine/gain_vs_zeroed_nats": (tot["zero"] - tot["sp"]) / n_tok,
            "spine/rows": float(n_rows),
        }

    @staticmethod
    def spine_row_indices(ds) -> list[int]:
        """Rows whose FIRST slot is a compaction spine."""
        row = getattr(ds, "row", None)
        if row is None:
            return []
        return [i for i in range(len(ds))
                if str(row(i).get("context") or "").startswith('["{\\"compaction')]

    def encode_extra(self, batch: Batch, ratio, selection: str | None = None,
                     extra_ratio: torch.Tensor | None = None):
        """Encode the extra slots of a multi-slot batch, LENGTH-SORTED and in
        chunks bounded by rows x maxlen^2.

        One call over all slots padded to the longest one is what killed P26 at
        step ~800: the encoder is bidirectional, so a chunk costs
        rows x heads x L^2 in attention scores, and a batch holding one
        8,192-token slot padded every short slot up to it — 36 rows x 16 heads
        x 8192^2 is 72 GiB of scores (the crash asked for 23.84 GiB inside
        softmax). Sorting by length keeps short slots cheap, and the budget
        bounds the peak regardless of what the sampler drew. Gradients are
        unaffected: each chunk is a normal forward and the results are placed
        back in slot order.
        """
        if batch.extra_input_ids is None:
            return None
        r = ratio
        if extra_ratio is not None:
            r = extra_ratio  # per-slot harness prices (slot_ratios)
        elif torch.is_tensor(ratio):
            r = ratio[batch.extra_owner]
        n_rows = int(batch.extra_input_ids.shape[0])
        lens = batch.extra_attention_mask.sum(-1).tolist()
        budget = int(self.cfg2.data.extra_encode_token_sq_budget)
        per: list[list[tuple]] = [[] for _ in batch.tasks]
        placed: list[list[int]] = [[] for _ in batch.tasks]
        # ``browse`` slots: a cached directory node of the repo's tree -- never
        # encoded here, read from the node cache (frozen tree encoder).
        e_cached = getattr(batch, "extra_cached", None) or [None] * n_rows
        e_spine = getattr(batch, "extra_spine", None) or [None] * n_rows
        sp_rows = [i for i in range(n_rows) if e_spine[i] is not None]
        if sp_rows:
            p_list = [self._prompt_row(batch.extra_prompt_ids, batch.extra_prompt_mask, i)
                      for i in sp_rows]
            enc = self.encode_spine_slots([e_spine[i] for i in sp_rows], p_list)
            for i, rm in zip(sp_rows, enc, strict=True):
                owner = int(batch.extra_owner[i])
                per[owner].append(rm)
                placed[owner].append(i)
        e_cached = [key if e_spine[i] is None else ("__spine__", "")
                    for i, key in enumerate(e_cached)]
        for row, key in enumerate(e_cached):
            if key == ("__spine__", ""):
                continue
            if key is None:
                continue
            cache = getattr(self, "tree_root_cache", None)
            if cache is None:
                raise ValueError("rows with a browse slot need data.repo_tree_root_cache")
            nr = cache.get(key[0], key[1])[0].to(self.device)
            owner = int(batch.extra_owner[row])
            per[owner].append((nr, torch.ones(nr.shape[0], dtype=torch.bool, device=self.device)))
            placed[owner].append(row)
        # An EMPTY slot (a tool call that returned nothing) has nothing to
        # encode: 0 reps. Sent to the encoder, a chunk of only empty slots is an
        # all-masked document with 0 survivors, which the projection block
        # cannot run on (S2 died at step ~2500, 2026-09-24).
        for row in range(n_rows):
            if e_cached[row] is None and int(lens[row]) == 0:
                w = self.decoder.embed_tokens.weight
                w = w.new_zeros(1, self.rep_dim)  # reps are in the ENCODER's rep space
                owner = int(batch.extra_owner[row])
                per[owner].append((w.new_zeros(0, w.shape[1]),
                                   torch.zeros(0, dtype=torch.bool, device=w.device)))
                placed[owner].append(row)
        live = [i for i in range(n_rows) if e_cached[i] is None and int(lens[i]) > 0]
        # optim.encoder_grad_token_budget: which live slots keep a gradient;
        # a chunk is all-grad or all-no_grad
        grad_rows = self.plan_extra_grad(batch) if live else None
        if grad_rows is not None:
            g_tok = sum(int(lens[i]) for i in live if i in grad_rows)
            self._enc_grad_account(sum(int(lens[i]) for i in live), len(live), g_tok,
                                   sum(1 for i in live if i in grad_rows))

        def wants_grad(i: int) -> bool:
            return grad_rows is None or i in grad_rows

        order = sorted(live, key=lambda i: (not wants_grad(i), int(lens[i])))
        chunk: list[int] = []
        for i in [*order, None]:
            if i is not None:
                cand = [*chunk, i]
                cost = len(cand) * max(int(lens[j]) for j in cand) ** 2
                if not chunk or (cost <= budget and wants_grad(i) == wants_grad(chunk[0])):
                    chunk = cand
                    continue
            if chunk:
                idx = torch.tensor(chunk, device=batch.extra_input_ids.device)
                width = max(1, max(int(lens[j]) for j in chunk))
                ids = batch.extra_input_ids[idx][:, :width]
                mask = batch.extra_attention_mask[idx][:, :width]
                p_ids = batch.extra_prompt_ids[idx] if batch.extra_prompt_ids is not None else None
                p_mask = (
                    batch.extra_prompt_mask[idx] if batch.extra_prompt_mask is not None else None
                )
                rr = r[idx] if torch.is_tensor(r) else r
                with contextlib.nullcontext() if wants_grad(chunk[0]) else torch.no_grad():
                    out = self.encoder(
                        ids, mask, ratio=rr, selection=selection, prompt_ids=p_ids,
                        prompt_mask=p_mask,
                    )
                    out = self.finish_reps(ids, out)
                for k, row in enumerate(chunk):
                    owner = int(batch.extra_owner[row])
                    per[owner].append((out.reps[k], out.rep_mask[k]))
                    placed[owner].append(row)
            chunk = [i] if i is not None else []
        # slots must reach splice() in their original order, not length order
        for owner, rows in enumerate(placed):
            if len(rows) > 1:
                per[owner] = [
                    x for _, x in sorted(zip(rows, per[owner], strict=True), key=lambda t: t[0])
                ]
        return per

    def splice(self, batch: Batch, reps, rep_mask, pad_side: str = "right", extra=None,
               decoder_space: bool = False):
        """``decoder_space``: ``reps`` / ``extra`` already are decoder embeddings (the full-text
        arm); otherwise they are encoder reps and go through the decoder's space adapter."""
        mid = None
        if batch.mid_ids is not None:
            mid = batch.mid_ids
            if extra is None:
                # no-context / zeroed arms keep the segments, drop the slots
                extra = [[(None, None)] * len(m) for m in mid]
        return self.decoder.build_batch(
            batch.prefix_ids,
            reps,
            rep_mask,
            batch.suffix_ids,
            pad_side=pad_side,
            suffix_label_start=batch.suffix_label_start,
            suffix_loss_mask=batch.suffix_loss_mask,
            extra=extra,
            mid_ids=mid,
            decoder_space=decoder_space,
        )

    def selection_rows(self, batch: Batch) -> list[int] | None:
        """Rows eligible for the attention-selection pass (None = every row).

        Two filters, both about the EAGER teacher pass in
        ``SpliceDecoder.attention_importance``, which materialises (chunk, H,
        T, T) attention weights:

        * ``kd.selection_max_ctx_tokens`` caps the row's context length.
        * MULTI-SLOT rows are excluded outright. The cap above is on the main
          slot's context, but a multi-slot row's DECODER sequence is the sum of
          its prefix and every slot, so a row with a short main context can
          still carry T > 10,000 — P26 died on a 12.78 GiB allocation there
          (2026-09-15). Excluding them is also more correct than capping:
          ``selection_loss`` reads ``tb.slot_start/slot_len``, which describe
          slot 0 only, so the other slots never received a target anyway.
        """
        cap = int(self.cfg2.kd.selection_max_ctx_tokens)
        owners: set[int] = set()
        if batch.extra_owner is not None and batch.extra_owner.numel():
            owners = {int(x) for x in batch.extra_owner.tolist()}
        if cap <= 0 and not owners:
            return None
        n = batch.attention_mask.sum(-1)
        rows = [
            i for i in range(len(batch.tasks)) if (cap <= 0 or int(n[i]) <= cap) and i not in owners
        ]
        return rows if len(rows) < len(batch.tasks) else None

    def selection_loss(
        self, batch: Batch, out: EncoderOutput, tb, rows: list[int] | None = None
    ) -> torch.Tensor:
        """Listwise KL(teacher-attention target || softmax(head z / T)) over each
        document's valid positions. The teacher's attention mass from the
        answer positions onto the context slot (the full-text batch) is the
        target; the slot holds exactly the valid context tokens in order.
        ``rows``: the subset of ``batch`` that ``tb`` was built from."""
        teacher = getattr(self, "teacher", None) or self.decoder
        imp = teacher.attention_importance(tb)  # (B, T_dec)
        scores = out.scores if rows is None else out.scores[rows]
        amask = batch.attention_mask if rows is None else batch.attention_mask[rows]
        bsz, seq = scores.shape
        target = torch.zeros(bsz, seq, dtype=torch.float32, device=scores.device)
        valid = amask.bool()
        gs = getattr(batch, "gold_span", None)
        spans = gs if rows is None else (None if gs is None else gs[rows])
        for b in range(bsz):
            n = int(valid[b].sum().item())
            if spans is not None and bool(spans[b].any()):
                # Oracle span target (rows with meta.gold_window): uniform over
                # the supporting facts' tokens — exact where the teacher's
                # attention over a long document is not.
                target[b, :n] = spans[b, :n].to(target.dtype)
                continue
            st = int(tb.slot_start[b].item())
            ln = int(tb.slot_len[b].item())
            if n == 0 or ln != n:
                continue
            target[b, :n] = imp[b, st : st + n]
        target = target * valid.to(target.dtype)
        tsum = target.sum(dim=-1, keepdim=True)
        ok = tsum.squeeze(-1) > 0
        if not bool(ok.any()):
            return scores.sum() * 0.0
        p = target / tsum.clamp_min(1e-12)
        logits = scores.float() / float(self.cfg2.kd.selection_temperature)
        logits = logits.masked_fill(~valid, float("-inf"))
        logq = torch.log_softmax(logits, dim=-1)
        kl = (p * (torch.log(p.clamp_min(1e-12)) - logq)).masked_fill(~valid, 0.0).sum(dim=-1)
        return kl[ok].mean()

    def prompt_contrast_backward(self) -> dict[str, float] | None:
        """Prompt contrast (kd.prompt_contrast_weight), run AFTER the main
        backward so the two encoder graphs never coexist: the rows that carry
        a sibling prompt are re-encoded under it (same ratio, same selection)
        and the true answer's nll under those reps is pushed ABOVE the
        right-prompt nll (detached from the main pass) by the margin. The
        decoder's adapter takes no gradient here: the pressure is on the
        encoder to make what it emits depend on the prompt, not on the
        reader to fail at mismatched reps."""
        cap = getattr(self, "_prompt_contrast_capture", None)
        self._prompt_contrast_capture = None
        if not cap:
            return None
        enc = getattr(self, "encoder", None)
        # The encoder-SIDE modules the contrast may train: the encoder, and on a
        # decoder-space-adapter run (model.decoder_space_adapter: a frozen encoder read by a
        # larger decoder, 8B-A) the adapter P8, which sits between them and is the one
        # trainable map from the encoder's reps to what the decoder reads.
        sa = self.decoder._modules.get("space_adapter")
        sa_params = [p for p in sa.parameters() if p.requires_grad] if sa is not None else []
        if (
            enc is not None
            and not any(p.requires_grad for p in enc.parameters())
            and not sa_params
        ):
            # Decoder-first phase (optim.encoder_freeze_steps): the contrast
            # trains the ENCODER side only (decoder params are detached below),
            # so with nothing trainable there it is skipped.
            return None
        batch: Batch = cap["batch"]
        idx = [i for i in range(len(batch.tasks)) if bool(batch.alt_present[i])]
        budget = int(self.cfg2.kd.prompt_contrast_max_tokens)
        if budget > 0:
            keep, used = [], 0
            for i in idx:
                n = int(batch.attention_mask[i].sum().item())
                if keep and used + n > budget:
                    break
                keep.append(i)
                used += n
            idx = keep
        if not idx:
            return None
        sub_right = batch.subset(idx)
        sub = sub_right.with_alt_prompts()
        r = cap["ratio"]
        if torch.is_tensor(r):
            r = r[idx]
        keep = {id(p) for p in sa_params}
        dec_params = [
            p for p in self.decoder.parameters() if p.requires_grad and id(p) not in keep
        ]
        for p in dec_params:
            p.requires_grad_(False)
        try:
            if self.cfg2.kd.prompt_contrast_paired:
                if batch.sibling is None:
                    return None
                # idx is a prefix of the present rows (the cap keeps order),
                # and the sibling batch is in present-row order.
                sub_b = batch.sibling.subset(list(range(len(idx))))
                return self._paired_contrast(sub_right, sub_b, r, cap["selection"])
            with self.autocast():
                out = self.encode(sub, r, cap["selection"])
                sb = self.splice(sub, out.reps, out.rep_mask)
                nll_alt, valid_alt = self.decoder.token_nll(sb)
                per_alt = (nll_alt * valid_alt).sum(-1) / valid_alt.sum(-1).clamp_min(1)
                if self.cfg2.kd.prompt_contrast_symmetric:
                    out_r = self.encode(sub_right, r, cap["selection"])
                    sb_r = self.splice(sub_right, out_r.reps, out_r.rep_mask)
                    nll_r, valid_r = self.decoder.token_nll(sb_r)
                    per_right = (nll_r * valid_r).sum(-1) / valid_r.sum(-1).clamp_min(1)
                else:
                    per_right = cap["nll_right"][idx]
            gap = per_alt - per_right
            margin = float(self.cfg2.kd.prompt_contrast_margin)
            hinge = torch.relu(margin - gap).mean()
            (float(self.cfg2.kd.prompt_contrast_weight) * hinge).backward()
        finally:
            for p in dec_params:
                p.requires_grad_(True)
        return {
            "loss/prompt_contrast": float(hinge.item()),
            "prompt_gap": float(gap.mean().item()),
            "prompt_gap_min": float(gap.min().item()),
            "prompt_contrast_n": float(len(idx)),
        }

    def _paired_contrast(self, sub_a: Batch, sub_b: Batch, r, selection) -> dict[str, float]:
        """kd.prompt_contrast_paired — see the config note. A = the batch's
        rows, B = their siblings (same context, other question + answer)."""
        margin = float(self.cfg2.kd.prompt_contrast_margin)

        def per_sample(sb):
            nll, valid = self.decoder.token_nll(sb)
            return (nll * valid).sum(-1) / valid.sum(-1).clamp_min(1)

        with self.autocast():
            out_a = self.encode(sub_a, r, selection)
            out_b = self.encode(sub_b, r, selection)
            l_aa = per_sample(self.splice(sub_a, out_a.reps, out_a.rep_mask))
            l_ab = per_sample(self.splice(sub_a, out_b.reps, out_b.rep_mask))
            l_bb = per_sample(self.splice(sub_b, out_b.reps, out_b.rep_mask))
            l_ba = per_sample(self.splice(sub_b, out_a.reps, out_a.rep_mask))
        gap_a = l_ab - l_aa
        gap_b = l_ba - l_bb
        hinge = 0.5 * (torch.relu(margin - gap_a).mean() + torch.relu(margin - gap_b).mean())
        ce_b = l_bb.mean()
        loss = float(self.cfg2.kd.prompt_contrast_weight) * hinge + ce_b
        stats_extra: dict[str, float] = {}
        if (
            getattr(self, "_sel_due_this_step", False)
            and self.cfg2.kd.selection_signal == "attention"
            and float(self.cfg2.kd.selection_weight) > 0
        ):
            # The head's target under the SIBLING prompt is the teacher's
            # attention for the sibling's answer: selection is asked to
            # follow the question, not the document.
            rows = self.selection_rows(sub_b)
            if rows is None:
                with self.autocast():
                    sel_b = self.selection_loss(sub_b, out_b, self.full_text_batch(sub_b))
            elif rows:
                with self.autocast():
                    sel_b = self.selection_loss(
                        sub_b, out_b, self.full_text_batch(sub_b.subset(rows)), rows=rows
                    )
            else:
                sel_b = None
            if sel_b is not None:
                loss = loss + float(self.cfg2.kd.selection_weight) * sel_b
                stats_extra["loss/selection_sibling"] = float(sel_b.item())
        loss.backward()
        gap = torch.cat([gap_a, gap_b])
        return {
            "loss/prompt_contrast": float(hinge.item()),
            "loss/sibling_ce": float(ce_b.item()),
            "prompt_gap": float(gap.mean().item()),
            "prompt_gap_min": float(gap.min().item()),
            "prompt_contrast_n": float(len(sub_a.tasks)),
            **stats_extra,
        }

    def utility_selection_backward(self) -> float | None:
        """Utility-gradient selection (old lineage, shared math in
        ``bgkit_core.selection``): after the main backward, the target is
        ``utility_targets(dL/dh, h)`` over EVERY eligible position; the head is
        recomputed on its cached input and trained by listwise KL toward it.
        Returns the loss value, or None if nothing was captured this step."""
        cap = getattr(self, "_utility_capture", None)
        self._utility_capture = None
        if not cap:
            return None
        try:
            if "grad" not in cap or "out" not in cap:
                return None
            out: EncoderOutput = cap["out"]
            if out.head_input is None or out.eligible is None:
                return None
            m = out.eligible
            target = utility_targets(cap["grad"], out.head_input, m)
            raw = self.encoder.part("head")(out.head_input).float()  # the slot head in adapter mode
            z = segment_zscore(raw, m)
            loss = listwise_kl(target, z, m, float(self.cfg2.kd.selection_temperature))
            if loss is None:
                return None
            (float(self.cfg2.kd.selection_weight) * loss).backward()
            conc = teacher_concentration(target, m)
            self._utility_teacher_rate = float(conc[m.sum(-1) > 1].mean().item())
            return float(loss.item())
        finally:
            # Break the hook -> dict -> output -> graph -> hooked-tensor cycle.
            handle = cap.pop("handle", None)
            if handle is not None:
                handle.remove()
            cap.clear()

    def head_scale_terms(self, out: EncoderOutput) -> tuple[torch.Tensor | None, float | None]:
        """Per-document std of the head's RAW scores (logged always) and the
        shared log-deviation anchor toward ``kd.head_scale_target``."""
        if out.raw is None or out.eligible is None:
            return None, None
        loss, std = head_scale_anchor(
            out.raw,
            out.eligible,
            float(self.cfg2.kd.head_scale_target),
            float(self.cfg2.kd.head_scale_anchor_weight),
        )
        return loss, float(std.mean().item())

    def build_ratio_schedule(self, seed: int) -> RatioSchedule:
        """THE construction of the curriculum from ``cfg2.ratio`` — every host
        (trainer, CPU test host) must build it here so no field is silently
        dropped (the deleted standalone trainer had drifted: no hold, no gate,
        no adaptive floor)."""
        r = self.cfg2.ratio
        return RatioSchedule(
            r.r_max,
            r.r_start,
            r.r_end,
            r.descent_steps,
            seed,
            r_max_start=r.r_max_start,
            ceiling_steps=r.ceiling_steps,
            hold_steps=r.hold_steps,
            adaptive=r.adaptive,
            rate_log2_per_step=r.rate_log2_per_step,
            rate_power=r.rate_power,
            mult_min=r.mult_min,
            band_top_prob=float(r.band_top_prob),
            band_top_lo=float(r.band_top_lo),
        )

    def anchor_weight(self, step: int) -> float:
        a = self.cfg2.anchor
        if a.weight <= 0:
            return 0.0
        if int(a.phase_out_steps) > 0:
            t = min(1.0, step / float(a.phase_out_steps))
            return float(a.floor + (a.weight - a.floor) * (1.0 - t))
        sched = getattr(self, "ratio_schedule", None)
        if sched is not None and sched.adaptive:
            # Decay with the floor's PROGRESS (log space), not wall-clock steps.
            return float(a.floor + (a.weight - a.floor) * (1.0 - sched.progress))
        if a.steps <= 0 or step >= a.steps:
            return float(a.floor)
        return float(a.floor + (a.weight - a.floor) * (1.0 - step / a.steps))

    def anchor_loss(self, batch: Batch, out) -> tuple[torch.Tensor, float]:
        """Embed anchor (shared: ``bgkit_core.anchor.embed_anchor_losses``) on
        the survivors' reps vs the decoder embedding of the token at their
        position."""
        ids = batch.input_ids.gather(1, out.positions)
        with torch.no_grad():
            target = self.decoder.embed(ids).float()
        m = out.rep_mask
        if int(m.sum()) == 0:
            return out.reps.sum() * 0.0, 0.0
        loss, stats = embed_anchor_losses(out.reps[m], target[m])
        return loss, stats["cos_to_token"]

    def frac_extras(self, step: int) -> float:
        """Curriculum: how many of the teacher's DOOMED positions survive level 0.

        Starts at 1.0 (level 0 keeps everyone, level 1 does all the compression)
        and slides down, moving load onto level 0 so level 1's job becomes real.
        """
        m = self.cfg2.multilevel
        n = max(1, int(m.curriculum_steps))
        t = min(1.0, max(0.0, step / n))
        return float(m.frac_extras_start + t * (m.frac_extras_end - m.frac_extras_start))

    def multilevel_distill(self, batch: Batch, step: int) -> tuple[torch.Tensor, dict] | None:
        """Stage B: the two-level student must reproduce the frozen single-level
        teacher's projection output on the SAME survivor set."""
        from ..models.multilevel import distill_loss, encode_two_level, teacher_forced_lower_keep

        m = self.cfg2.multilevel
        teacher = getattr(self, "encoder_teacher", None)
        if teacher is None:
            return None
        ids, mask = batch.input_ids, batch.attention_mask.bool()
        with torch.no_grad():
            t_out = teacher(ids, mask, ratio=m.teacher_ratio)
        if int(t_out.rep_mask.sum()) == 0:
            return None
        if float(torch.rand(())) < m.path_a_prob:
            # Path A: the single-level encode, so the projection keeps handling
            # the distribution it already serves instead of drifting to level 1's.
            s_out = self.encoder(ids, mask, ratio=m.teacher_ratio, forced_keep=t_out.keep)
            loss, st = distill_loss(
                s_out.reps, t_out.reps, t_out.rep_mask,
                mse_weight=m.mse_weight, cos_weight=m.cos_weight,
            )
            st["path"] = 0.0
            return loss, st
        fe = self.frac_extras(step)
        g = int(m.group_size)
        if g > 1:
            return self._grouped_path_b(batch, teacher, m, fe, g)
        lower_keep = teacher_forced_lower_keep(t_out.keep, mask, fe)
        two = encode_two_level(
            self.encoder, ids, mask,
            ratio_lower=1.0, ratio_upper=1.0,
            group_sizes=[1] * int(ids.shape[0]),
            forced_lower_keep=lower_keep,
            forced_upper_from=t_out.keep,
        )
        if two.top.reps.shape != t_out.reps.shape:
            return None  # a forced set went empty; skip rather than compare padding
        loss, st = distill_loss(
            two.top.reps, t_out.reps, t_out.rep_mask,
            mse_weight=m.mse_weight, cos_weight=m.cos_weight,
        )
        st["path"] = 1.0
        st["frac_extras"] = fe
        return loss, st

    def _grouped_path_b(self, batch: Batch, teacher, m, fe: float, g: int):
        """Path B over GROUPS: the teacher encodes the flat concatenation of
        ``g`` documents single-level; the student encodes them as a two-level
        tree; the final survivor sets are forced equal through the concat
        positions. This is the distribution the upper level actually serves --
        survivors from DIFFERENT files -- which the per-document port never
        showed it."""
        from ..models.multilevel import distill_loss, encode_two_level, teacher_forced_lower_keep

        ids, mask = batch.input_ids, batch.attention_mask.bool()
        n_rows, n_cols = ids.shape
        cap = int(self.cfg2.data.task_max_ctx_tokens) or 8192
        losses, stats_acc, n_groups = [], {"mse": 0.0, "cos": 0.0}, 0
        for start in range(0, n_rows - g + 1, g):
            rows = list(range(start, start + g))
            lens = [int(mask[i].sum()) for i in rows]
            if sum(lens) > cap or min(lens) == 0:
                continue
            cat = torch.cat([ids[i, : lens[j]] for j, i in enumerate(rows)])[None]
            cat_mask = torch.ones_like(cat, dtype=torch.bool)
            with torch.no_grad():
                t_out = teacher(cat, cat_mask, ratio=m.teacher_ratio)
            if int(t_out.rep_mask.sum()) == 0:
                continue
            # teacher's keep over the concat -> per-row (g, L) masks
            per_row = torch.zeros(g, n_cols, dtype=torch.bool, device=ids.device)
            off = 0
            for j, ln in enumerate(lens):
                per_row[j, :ln] = t_out.keep[0, off : off + ln]
                off += ln
            sub_ids, sub_mask = ids[rows], mask[rows]
            lower_keep = teacher_forced_lower_keep(per_row, sub_mask, fe)
            two = encode_two_level(
                self.encoder, sub_ids, sub_mask,
                ratio_lower=1.0, ratio_upper=1.0,
                group_sizes=[g],
                forced_lower_keep=lower_keep,
                forced_upper_from=per_row,
            )
            s_reps, s_mask = two.top.reps, two.top.rep_mask
            t_reps, t_mask = t_out.reps, t_out.rep_mask
            if int(s_mask.sum()) != int(t_mask.sum()):
                continue  # a forced set went empty
            k = max(int(s_reps.shape[1]), int(t_reps.shape[1]))

            def pad(x, w):
                if x.shape[1] == w:
                    return x
                z = torch.zeros((x.shape[0], w - x.shape[1], *x.shape[2:]),
                                dtype=x.dtype, device=x.device)
                return torch.cat([x, z], dim=1)

            loss, st = distill_loss(
                pad(s_reps, k), pad(t_reps, k), pad(s_mask, k),
                mse_weight=m.mse_weight, cos_weight=m.cos_weight,
            )
            losses.append(loss)
            stats_acc["mse"] += st["mse"]
            stats_acc["cos"] += st["cos"]
            n_groups += 1
        if not losses:
            return None
        st = {k: v / n_groups for k, v in stats_acc.items()}
        st["path"] = 1.0
        st["frac_extras"] = fe
        st["groups"] = float(n_groups)
        return torch.stack(losses).mean(), st

    def repro_loss(self, batch: Batch, out) -> tuple[torch.Tensor, float] | None:
        """Pre-warm the repro head: survivors -> the ENCODER's OWN input space.

        The multi-level tree feeds one level's survivors into the next, and the
        backbone was trained on token embeddings, so the bridge has to land in
        that distribution before a second level means anything. Same objective
        as `anchor_loss`, different target space: the ENCODER's embed_tokens,
        not the decoder's. This is bgkit1's joint-block pretrain of
        auto_repro_head (src/bgkit/training/joint_block_trainer.py).
        """
        if getattr(self.encoder, "repro", None) is None or out.survivors is None:
            return None
        m = out.rep_mask
        if int(m.sum()) == 0:
            return None
        ids = batch.input_ids.gather(1, out.positions)
        with torch.no_grad():
            target = self.encoder.backbone.embed_tokens(ids).float()
        pred = self.encoder.reproduce(out.survivors)
        loss, stats = embed_anchor_losses(pred[m].float(), target[m])
        return loss, stats["cos_to_token"]

    @staticmethod
    def decorrelation_loss(reps: torch.Tensor, rep_mask: torch.Tensor) -> torch.Tensor:
        """Mean squared off-diagonal CORRELATION over all valid rep rows in the
        batch: scale-free (the interface norm makes per-channel variance tiny),
        0 for whitened reps, ~1 for reps that share one direction."""
        x = reps[rep_mask].float()
        n, d = x.shape
        if n < 2:
            return reps.sum() * 0.0
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
        corr = x.T @ x / n
        off = corr - torch.diag(torch.diagonal(corr))
        return off.pow(2).sum() / (d * (d - 1))

    @staticmethod
    def within_doc_diversity_loss(reps: torch.Tensor, rep_mask: torch.Tensor) -> torch.Tensor:
        """Mean squared cosine over distinct survivor pairs within each document."""
        x = reps.float()
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        m = rep_mask.float()
        gram = torch.einsum("bkd,bjd->bkj", x, x)
        pair = m[:, :, None] * m[:, None, :]
        eye = torch.eye(gram.shape[1], device=gram.device)[None]
        pair = pair * (1.0 - eye)
        n_pairs = pair.sum()
        if float(n_pairs) == 0:
            return reps.sum() * 0.0
        return (gram.pow(2) * pair).sum() / n_pairs

    def guard_reps(self, tag: str, reps: torch.Tensor, rep_mask: torch.Tensor,
                   stats: dict[str, float]) -> None:
        """The splice guard at TRAIN time (norm band, within-document eff_rank,
        shared_frac, cosine to the payload-mean EMA): logs
        ``spliced_rep_out_of_contract`` and ``<tag>/*`` stats. One guard per
        tag, so payloads from different paths (primary slots, the extra slots
        of overview rows, compaction spines) keep separate EMAs; ``rep`` is the
        one persisted across resumes. P2-P5 (2026-09-04): an open encoder
        collapsed the reps within 1000 steps -- with the encoder training again
        (encoder_mode adapter) every trained path is watched."""
        reps = reps.detach()
        rs = rep_stats(reps, rep_mask, self.rep_row_norm)
        guards = self.__dict__.setdefault("_splice_guards", {})
        if tag == "rep":
            guard = getattr(self, "splice_guard", None)
            if guard is None:
                guard = self.splice_guard = SpliceGuard()
        else:
            guard = guards.get(tag)
            if guard is None:
                guard = guards[tag] = SpliceGuard()
        violated = guard.observe(reps, rep_mask, rs)
        if violated:
            logger.warning("spliced_rep_out_of_contract", step=self.current_step, path=tag,
                           **violated)
            stats[f"{tag}/out_of_contract"] = 1.0
        stats.update({f"{tag}/{k}": v for k, v in rs.items()})

    @staticmethod
    def stack_slots(extra) -> tuple[torch.Tensor, torch.Tensor] | None:
        """encode_extra's per-row slot lists -> (N, K, D) reps + mask."""
        items = [(r, m) for row in (extra or []) for (r, m) in row if r is not None]
        if not items:
            return None
        width = max(int(r.shape[0]) for r, _ in items)
        reps = items[0][0].new_zeros(len(items), width, items[0][0].shape[-1])
        mask = torch.zeros(len(items), width, dtype=torch.bool, device=reps.device)
        for i, (r, m) in enumerate(items):
            reps[i, : r.shape[0]] = r.detach()
            mask[i, : m.shape[0]] = m
        return reps, mask

    def pooled_ce(self, batch: Batch, nll: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """The batch CE. Default: token-pooled mean. ``kd.per_row_ce``: mean
        over rows of each row's mean nll, so a long document's single row
        weighs as much as any short row (token pooling lets the rows with
        the longest answers own the loss). ``kd.family_loss_weight``
        multiplies rows of the named families in either pooling."""
        kd = self.cfg2.kd
        w = torch.ones(len(batch.tasks), device=nll.device)
        if kd.family_loss_weight:
            for i, t in enumerate(batch.tasks):
                w[i] = float(kd.family_loss_weight.get(t, 1.0))
        v = valid.to(nll.dtype)
        if kd.per_row_ce:
            cnt = v.sum(-1)
            per_row = (nll * v).sum(-1) / cnt.clamp_min(1)
            has = cnt > 0
            if not bool(has.any()):
                return nll.sum() * 0.0
            return (per_row * w)[has].sum() / w[has].sum()
        wt = v * w[:, None]
        return (nll * wt).sum() / wt.sum().clamp_min(1)

    def slot_ratios(self, batch: Batch, ratio: float):
        """(primary ratio, per-extra-slot ratio or None) for a training step.

        ``ratio.slot_reps_mode`` "row": ``_family_ratio`` for the primary slots and every extra
        slot follows its row (``extra`` None). "meta" / "harness": every slot the eval harness
        prices (``Batch.slot_reps`` / ``extra_slot_reps``, from bgkit2.data.slot_pricing;
        "meta" accepts only rows whose prices come from ``meta.slot_reps``) is encoded at
        ``price x jitter`` reps -- jitter log-uniform in ``ratio.slot_reps_jitter``, per slot --
        i.e. ratio = reps / slot length, clamped like the student's ``slot_ratio``
        ([1/1024, 1/2]). Unpriced slots and rows keep the row ratio."""
        fam = self._family_ratio(batch, ratio)
        mode = self.cfg2.ratio.slot_reps_mode
        if mode == "row" or batch.slot_reps is None or not batch.slot_reps_src:
            return fam, None
        ok = {"meta"} if mode == "meta" else {"meta", "prompt"}
        rows_ok = [src in ok for src in batch.slot_reps_src]
        if not any(rows_ok):
            return fam, None
        lo, hi = (math.log(float(x)) for x in self.cfg2.ratio.slot_reps_jitter)
        rng = self.select_rng

        def priced(reps: float, n: int) -> float:
            k = reps * math.exp(rng.uniform(lo, hi))
            return float(min(0.5, max(1.0 / 1024, k / max(1, n))))

        n_rows = len(batch.tasks)
        base = (fam.to(self.device, torch.float32) if torch.is_tensor(fam)
                else torch.full((n_rows,), float(fam), device=self.device))
        vec = base.clone()
        n_priced = n_slots = 0
        prim = batch.slot_reps.tolist()
        n_valid = batch.attention_mask.sum(-1).tolist()
        for i in range(n_rows):
            n_slots += 1
            if rows_ok[i] and prim[i] > 0 and int(n_valid[i]) > 0:
                vec[i] = priced(prim[i], int(n_valid[i]))
                n_priced += 1
        extra = None
        if batch.extra_input_ids is not None and batch.extra_slot_reps is not None:
            owner = batch.extra_owner.to(self.device)
            extra = base[owner].clone()
            lens = batch.extra_attention_mask.sum(-1).tolist()
            ers = batch.extra_slot_reps.tolist()
            for r, o in enumerate(owner.tolist()):
                n_slots += 1
                if rows_ok[o] and ers[r] > 0 and int(lens[r]) > 0:
                    extra[r] = priced(ers[r], int(lens[r]))
                    n_priced += 1
        acc = self.__dict__.setdefault("_slot_price_acc", [0, 0])
        acc[0] += n_priced
        acc[1] += n_slots
        return vec, extra

    def _family_ratio(self, batch: Batch, ratio: float):
        """The step's ratio, except that families listed in
        ``ratio.family_bands`` draw their own log-uniform ratio per sample
        (pressure where it counts: needle rows at 32-128x while the rest of
        the batch follows the wide band). Returns a float or a (B,) tensor."""
        bands = self.cfg2.ratio.family_bands
        budgets = self.cfg2.ratio.family_rep_budget
        if not bands and not budgets:
            return ratio
        hit = [i for i, t in enumerate(batch.tasks) if t in bands or t in budgets]
        if not hit:
            return ratio
        vec = torch.full((len(batch.tasks),), float(ratio), device=self.device)
        n_valid = batch.attention_mask.sum(-1)
        for i in hit:
            fam = batch.tasks[i]
            if fam in budgets:
                lo, hi = budgets[fam]
                k = math.exp(self.select_rng.uniform(math.log(lo), math.log(hi)))
                vec[i] = float(min(0.5, max(1.0 / 1024, k / max(1, int(n_valid[i])))))
            else:
                lo, hi = bands[fam]
                vec[i] = float(math.exp(self.select_rng.uniform(math.log(lo), math.log(hi))))
        return vec

    def full_text_training_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """``kd.full_text_student``: CE on the answer with the context's token
        embeddings in the rep slot. No encoder forward, no reps, no anchor or
        selection terms — only the decoder adapter learns (at the slot
        positions when rep-gated). The matched ceiling / KD teacher."""
        batch = batch.to(self.device)
        with self.autocast():
            sb = self.full_text_batch(batch)
            nll, valid = self.decoder.token_nll(sb)
        loss = nll.sum() / valid.sum().clamp_min(1)
        stats: dict[str, float] = {
            "loss/ce": float(loss.item()),
            "ctx_mean": float(batch.attention_mask.sum(-1).float().mean().item()),
        }
        for t, idx in task_indices(batch).items():
            stats[f"loss/{t}"] = float((nll[idx].sum() / valid[idx].sum().clamp_min(1)).item())
        return float(self.cfg2.kd.ce_weight) * loss, stats

    # ------------------------------------------------------------------
    # replay stream: plain conversations, self-distilled to the base
    # ------------------------------------------------------------------
    def replay_rows(self, batch: Batch) -> list[int]:
        fam = str(self.cfg2.kd.replay_family)
        return [i for i, t in enumerate(batch.tasks) if t == fam]

    def replay_loss(self, sub: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """kd.replay_*: no encoder, no slot. One student forward over
        [prefix][target]; the teacher is the same decoder with its adapter
        disabled (``token_nll_and_kd`` with ``teacher_model=None``), so the
        KL term is "do not drift from the base" on real chat / agentic turns
        and the CE term is the stored turn."""
        kd = self.cfg2.kd
        with self.autocast():
            sb = self.splice(sub, None, None)
            nll, valid, kl = self.decoder.token_nll_and_kd(
                sb, sb, temperature=float(kd.temperature), teacher_model=None
            )
        ce = self.pooled_ce(sub, nll, valid)
        loss = float(kd.replay_kl_weight) * kl + float(kd.replay_ce_weight) * ce
        stats = {
            f"loss/{kd.replay_family}": float(ce.item()),
            f"loss/{kd.replay_family}_kl": float(kl.item()),
            "replay_rows": float(len(sub.tasks)),
        }
        return loss, stats

    @torch.no_grad()
    def replay_eval_metrics(self, batches: list[Batch]) -> dict[str, float]:
        """The capability panel on a replay store's eval split: CE of the
        adapted decoder vs the base on the stored assistant turns, and the
        per-token KL(base || student) — the drift the replay stream is meant
        to hold at zero."""
        tot_s = tot_b = tot_kl = cnt = 0.0
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                sb = self.splice(batch, None, None)
                nll, valid, kl = self.decoder.token_nll_and_kd(
                    sb, sb, temperature=1.0, teacher_model=None
                )
                with self.decoder.disable_adapter():
                    nll_b, valid_b = self.decoder.token_nll(sb)
            n = float(valid.sum().item())
            tot_s += float((nll * valid).sum().item())
            tot_b += float((nll_b * valid_b).sum().item())
            tot_kl += float(kl.item()) * n
            cnt += n
        cnt = max(cnt, 1.0)
        return {
            "ce": tot_s / cnt,
            "ce_base": tot_b / cnt,
            "ce_delta": (tot_s - tot_b) / cnt,
            "kl_to_base": tot_kl / cnt,
        }

    def tree_bridge(self):
        """None = encoder.reproduce (the trained head); identity = no bridge."""
        b = str(self.cfg2.multilevel.bridge)
        if b == "identity":
            return lambda x: x
        if b == "repro":
            return None
        raise ValueError(f"multilevel.bridge must be 'repro' or 'identity', got {b!r}")

    def load_tree_root_cache(self, path: str) -> None:
        """Node reps per repo from scripts/bgkit2_cache_repo_trees.py. The cache
        is the BASE encoder's output (post-projection, decoder space), so it is
        valid exactly while nothing the base role reads trains: encoder_mode
        ``frozen`` (nothing trains) or ``adapter`` (the adapter is OFF on the
        cached-tree path and the base weights are out of the optimizer --
        ``_adapter_param_groups`` enforces it). ``backbone`` trains the base
        encoder itself, so the cache would go stale under it.

        Why not cache PRE-projection survivors instead and train the base
        projection: the head and survive embedding sit upstream of every tree
        level (they choose each node's survivors), so a trained head changes
        every cached node, not only its projection; a pre-projection cache
        would also cost a full re-encode (~18 h for the all-repos store). The
        slot path gets its own copies instead (``inject_slot_adapter``)."""
        mode = encoder_mode(self.cfg2)
        if mode == "backbone" and not self.cfg2.multilevel.prompted_readout:
            # (prompted read-out: the cache holds the FROZEN tree below the node
            # and the read-out pass is what trains -- by design)
            raise ValueError(
                "data.repo_tree_root_cache needs the base encoder frozen: "
                "optim.encoder_mode frozen or adapter (got backbone)")
        from ..data.node_cache import NodeCache

        # single file (root-only / all-nodes) or a directory of shards read lazily
        self.tree_node_cache = NodeCache(path)
        self.tree_root_cache = self.tree_node_cache  # roots are node ""
        logger.info("tree_root_cache_loaded", path=path, repos=len(self.tree_node_cache.repos),
                    nodes=len(self.tree_node_cache))

    # "question" (default) | "none" | "wrong": eval-time control of the read-out prompt
    readout_prompt_mode: str = "question"

    def prompted_node(self, repo: str, node: str, question: str):
        """QUESTION-CONDITIONED read-out: one encoder pass over the node's cached
        INPUT with the question as the encoder prompt. (reps (K, D), mask (K,)).
        A graph is built iff the encoder trains and grad is enabled."""
        inp = self.tree_node_cache.get_input(repo, node).to(self.device)
        mode = self.readout_prompt_mode
        text = {"question": question, "none": "",
                "wrong": "Which file configures the release pipeline?"}[mode]
        p_ids = p_mask = None
        if text:
            ids = self.enc_tok(text, add_special_tokens=False)["input_ids"][:256]
            p_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            p_mask = torch.ones_like(p_ids)
        msk = torch.ones(1, inp.shape[0], dtype=torch.bool, device=self.device)
        with self.autocast():
            out = self.encoder(None, msk, ratio=float(self.cfg2.multilevel.ratio_node),
                               doc_embeds=inp[None].to(self.amp_dtype),
                               prompt_ids=p_ids, prompt_mask=p_mask)
        k = out.rep_mask[0]
        return out.reps[0][k], torch.ones(int(k.sum()), dtype=torch.bool, device=self.device)

    def tree_is_live(self, n_files: int) -> bool:
        """A tree small enough to encode LIVE through the encoder adapter
        (``multilevel.live_tree_max_files``, adapter mode only); larger trees
        take the frozen base path / the node cache."""
        lim = int(self.cfg2.multilevel.live_tree_max_files)
        return encoder_mode(self.cfg2) in ADAPTER_MODES and lim > 0 and 0 < int(n_files) <= lim

    def tree_role(self, n_files: int) -> str:
        """The encoder role of a tree of ``n_files`` files -- ONE rule for the
        trainer, the tree eval and the student server: ``slots`` (the trained
        adapter) for live trees, ``base`` (the frozen tree encoder whose output
        the node cache holds) for everything else."""
        return "slots" if self.tree_is_live(n_files) else "base"

    def tree_grad_levels(self, live: bool) -> int | None:
        """grad_levels for ``encode_tree``: none with a frozen encoder or on the
        frozen base path; ``multilevel.grad_levels`` (-1 = all) otherwise."""
        mode = encoder_mode(self.cfg2)
        g = int(self.cfg2.multilevel.grad_levels)
        if mode in ("frozen", "frozen_adapter") or (mode == "adapter" and not live):
            return 0
        return None if g < 0 else g

    def tree_root(self, t) -> tuple[torch.Tensor, torch.Tensor]:
        """(reps (K, D), mask (K,)) of the repo's ROOT (or ``t.node``) encoding.
        A tree small enough for ``tree_is_live`` is encoded LIVE through the
        encoder adapter (role slots, ``multilevel.grad_levels`` bounds the
        graph); any other tree comes from the node cache when one is loaded,
        else is encoded live on the frozen base path (role base, no graph)."""
        node = getattr(t, "node", None) or ""
        cache = getattr(self, "tree_node_cache", None)
        files = list(getattr(t, "file_ids", None) or [])
        paths = list(getattr(t, "paths", None) or [])
        live = bool(paths) and self.tree_is_live(len(paths)) and len(files) == len(paths)
        if cache is not None and not live and self.cfg2.multilevel.prompted_readout:
            return self.prompted_node(t.repo, node, getattr(t, "question", ""))
        if cache is not None and not live:
            r, mk = cache.get(t.repo, node)
            return r.to(self.device), mk.to(self.device)
        from ..models.multilevel import encode_tree

        m = self.cfg2.multilevel
        tok_encode = lambda s: self.enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        gl = self.tree_grad_levels(live)
        # Without gradient (eval) a live tree is a fixed function of the repo
        # for the current weights: every walk step of an eval reads the SAME
        # encoding, so encode it once per (step, repo, file list).
        memo_key = None
        if live and not torch.is_grad_enabled():
            memo_key = (self.current_step, t.repo, tuple(t.paths))
            memo = self.__dict__.setdefault("_live_tree_memo", {})
            if memo_key not in memo:
                memo.clear()  # one tree at a time: eval walks a repo's steps in order
        out = self.__dict__.get("_live_tree_memo", {}).get(memo_key) if memo_key else None
        files = [f.to(self.device) for f in files]
        if out is None:
            with (torch.no_grad() if gl == 0 else contextlib.nullcontext()), \
                    self.encoder.role(self.tree_role(len(t.paths))), self.autocast():
                out = self._encode_tree_live(encode_tree, files, t, tok_encode, m, gl)
            if memo_key is not None:
                self._live_tree_memo[memo_key] = out
        if node:
            if not out.nodes or node not in out.nodes:
                raise KeyError(f"node {node!r} is not a directory of the encoded tree")
            return out.nodes[node]
        return out.root.reps[0], out.root.rep_mask[0]

    def _encode_tree_live(self, encode_tree, files, t, tok_encode, m, gl):
        return encode_tree(self.encoder, files, t.paths, tok_encode=tok_encode,
                           **self.tree_encode_kwargs(grad_levels=gl))

    def tree_file_tokens(self) -> int:
        """Per-file token cap of a LIVE tree: what the trainer's tree dataset
        cuts each file to (``data.repo_tree_file_tokens``; the S1 runs set
        131072 = whole files, the windowing below does the rest)."""
        return int(self.cfg2.data.repo_tree_file_tokens)

    def tree_encode_kwargs(self, grad_levels: int | None = 0) -> dict:
        """``encode_tree`` settings of this run -- ONE source for the trainer's
        live trees and the student server's tree slots (2026-09-22 audit: the
        student cut files to 512 tokens and ran without the leaf window / leaf
        cap / chunk budgets the trainer used, so a whole-file checkpoint was
        evaluated on a different tree than it trained on)."""
        m = self.cfg2.multilevel
        return dict(
            ratio_leaf=m.ratio_leaf, ratio_node=m.ratio_node,
            max_depth=m.max_depth, device=self.device, bridge=self.tree_bridge(),
            leaf_chunk=int(m.leaf_chunk), grad_levels=grad_levels,
            leaf_chunk_tokens=int(m.leaf_chunk_tokens),
            node_max_tokens=int(m.node_max_tokens),
            leaf_window=int(m.leaf_window),
            leaf_max_reps=int(m.leaf_max_reps),
        )

    @staticmethod
    def tree_prompt(t) -> tuple[str, str]:
        """(decoder prompt text, target text) of a tree item. Whole-path task:
        the question -> the path. PATH WALK step: the question plus the current
        directory -> the next component ("<dir>/" = keep walking, else the file
        relative to the directory), newline-terminated so generation stops."""
        if getattr(t, "node", None) is None:
            return f"{t.question}\nWhich file? ", t.answer
        return f"{t.question}\nCurrent directory: {t.node or '.'}\nNext: ", f"{t.answer}\n"

    def tree_training_loss(self, trees: list) -> tuple[torch.Tensor, dict[str, float]]:
        """Stage C: encode each repo as a TREE, the decoder reads the ROOT and
        must produce the answer PATH given the question. The decoder is frozen
        by config (model.lora.enabled false); what trains is the encoder side."""
        prefixes, suffixes, reps, masks = [], [], [], []
        for t in trees:
            r, mk = self.tree_root(t)
            reps.append(r)
            masks.append(mk)
            q_text, a_text = self.tree_prompt(t)
            q = self.tok(q_text, add_special_tokens=False)["input_ids"]
            a = self.tok(a_text, add_special_tokens=False)["input_ids"]
            prefixes.append(torch.tensor(q, dtype=torch.long, device=self.device))
            suffixes.append(torch.tensor(a, dtype=torch.long, device=self.device))
        k = max(int(r.shape[0]) for r in reps)
        root_reps = torch.zeros(
            len(reps), k, reps[0].shape[-1], dtype=reps[0].dtype, device=self.device
        )
        root_mask = torch.zeros(len(reps), k, dtype=torch.bool, device=self.device)
        for i, (r, mk) in enumerate(zip(reps, masks, strict=True)):
            root_reps[i, : r.shape[0]] = r
            root_mask[i, : mk.shape[0]] = mk
        with self.autocast():
            sb = self.decoder.build_batch(prefixes, root_reps, root_mask, suffixes)
            nll, valid = self.decoder.token_nll(sb)
        ce = nll.sum() / valid.sum().clamp_min(1)
        total = ce
        stats = {"loss": float(ce.item()), "loss/tree_nav": float(ce.item()),
                 "tree/root_reps": float(root_mask.sum().item() / len(reps)),
                 "tree/files": float(sum(len(t.paths) for t in trees) / len(trees))}
        return total, stats

    def training_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """One microbatch: task CE (averaged over ``ratios_per_step`` sampled
        ratios) + anchor; returns the total loss and unscaled metrics (incl.
        ``enc_grad/*``: slot tokens encoded / encoded with gradient)."""
        self._enc_grad_reset()
        self.__dict__["_slot_price_acc"] = [0, 0]
        total, stats = self._training_loss_inner(batch)
        stats.update(self.enc_grad_stats())
        priced, n = self.__dict__.get("_slot_price_acc") or [0, 0]
        if n and self.cfg2.ratio.slot_reps_mode != "row":
            stats["ratio/priced_slots"] = float(priced)
            stats["ratio/priced_slot_frac"] = float(priced) / float(n)
        return total, stats

    def _training_loss_inner(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(batch, list):  # Stage C repo trees (data.repo_tree_store)
            return self.tree_training_loss(batch)
        if self.cfg2.kd.full_text_student:
            return self.full_text_training_loss(batch)
        # Replay rows (kd.replay_family) carry no context: split them off
        # before anything touches the encoder; the rep rows continue below.
        rp = self.replay_rows(batch)
        if rp:
            batch = batch.to(self.device)
            r_loss, r_stats = self.replay_loss(batch.subset(rp))
            rest = [i for i in range(len(batch.tasks)) if i not in set(rp)]
            if not rest:
                r_stats["loss"] = r_stats[f"loss/{self.cfg2.kd.replay_family}"]
                r_stats["ratio"] = float("nan")
                self._prompt_contrast_capture = None
                self._utility_capture = None
                return r_loss, r_stats
            batch = batch.subset(rest)
            total, stats = self._training_loss_slotted(batch)
            stats.update(r_stats)
            return total + r_loss, stats
        return self._training_loss_slotted(batch)

    # ------------------------------------------------------------------
    # always-on repo overview: rows whose PRIMARY slot is a cached root
    # ------------------------------------------------------------------
    @staticmethod
    def cached_rows(batch: Batch) -> list[int]:
        cs = getattr(batch, "cached_slot", None)
        return [i for i, r in enumerate(cs) if r] if cs else []

    def cached_primary(self, sub: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, R, D) root reps + (B, R) mask for rows whose primary slot is a
        repo overview, from the per-repo root cache."""
        cache = getattr(self, "tree_root_cache", None)
        if cache is None:
            raise ValueError(
                "rows with a repo-overview slot need data.repo_tree_root_cache "
                "(scripts/bgkit2_cache_repo_trees.py)"
            )
        parts = []
        for repo in sub.cached_slot or []:
            parts.append(cache.get(repo, "")[0])
        width = max(int(p.shape[0]) for p in parts)
        reps = torch.zeros(len(parts), width, parts[0].shape[-1], dtype=parts[0].dtype,
                           device=self.device)
        mask = torch.zeros(len(parts), width, dtype=torch.bool, device=self.device)
        for i, p in enumerate(parts):
            reps[i, : p.shape[0]] = p.to(self.device)
            mask[i, : p.shape[0]] = True
        return reps, mask

    def overview_loss(self, sub: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """CE on rows whose primary slot is the cached repo overview. The
        overview is never encoded; the rows' OTHER slots (tool results) go
        through the encoder as usual. The target is the teacher's turn, so the
        CE is the distillation signal; the encoder-side losses (selection,
        anchor) have nothing to say about a cached slot and stay with the
        text-slot rows."""
        ratio = self.ratio_schedule.sample(self.current_step)
        with self.autocast():
            fam_ratio, extra_ratio = self.slot_ratios(sub, ratio)
            reps, mask = self.cached_primary(sub)
            extra = self.encode_extra(sub, fam_ratio, None, extra_ratio=extra_ratio)
            sb = (
                self.splice(sub, reps, mask)
                if extra is None
                else self.splice(sub, reps, mask, extra=extra)
            )
            nll, valid = self.decoder.token_nll(sb)
        ce = self.pooled_ce(sub, nll, valid)
        fams = sorted(set(sub.tasks))
        stats = {f"loss/{f}_overview": float(ce.item()) for f in fams}
        if encoder_mode(self.cfg2) != "frozen":
            # the overview itself is the frozen cache; its tool-result slots
            # are what the (trained) encoder emits
            st = self.stack_slots(extra)
            if st is not None:
                self.guard_reps("rep_extra", st[0], st[1], stats)
        stats["overview_rows"] = float(len(sub.tasks))
        stats["overview_reps"] = float(mask.sum().item() / max(len(sub.tasks), 1))
        return ce, stats

    def _training_loss_slotted(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        sr = self.spine_rows(batch)
        if sr:
            batch = batch.to(self.device)
            s_loss, s_stats = self.spine_loss(batch.subset(sr))
            rest = [i for i in range(len(batch.tasks)) if i not in set(sr)]
            if not rest:
                s_stats["loss"] = float(s_loss.item())
                s_stats["ratio"] = float("nan")
                self._prompt_contrast_capture = None
                self._utility_capture = None
                return s_loss, s_stats
            total, stats = self._training_loss_slotted_nospine(batch.subset(rest))
            stats.update(s_stats)
            return total + s_loss, stats
        return self._training_loss_slotted_nospine(batch)

    @staticmethod
    def noprimary_rows(batch: Batch) -> list[int]:
        """Rows whose PRIMARY slot is empty and not filled from a cache: an
        agent's first move (framing "chat", 11% of agent_tool_slots_v8) has no
        tool result yet. A microbatch made only of them hands the encoder a
        0-width document (S2 died at step ~830, 2026-09-23)."""
        if batch.attention_mask is None:
            return []
        cs = getattr(batch, "cached_slot", None) or [None] * len(batch.tasks)
        n = batch.attention_mask.sum(-1)
        return [i for i in range(len(batch.tasks)) if int(n[i]) == 0 and not cs[i]]

    def noprimary_loss(self, sub: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """CE on rows with no primary slot: no primary reps (their only arm is
        the floor), any extra slots encoded as usual. Like the overview rows,
        the encoder-side losses have nothing to say about an absent slot."""
        ratio = self.ratio_schedule.sample(self.current_step)
        d = self.rep_dim
        reps = torch.zeros(len(sub.tasks), 0, d, device=self.device,
                           dtype=self.decoder.embed_tokens.weight.dtype)
        mask = torch.zeros(len(sub.tasks), 0, dtype=torch.bool, device=self.device)
        with self.autocast():
            fam_ratio, extra_ratio = self.slot_ratios(sub, ratio)
            extra = self.encode_extra(sub, fam_ratio, None, extra_ratio=extra_ratio)
            sb = (
                self.splice(sub, reps, mask)
                if extra is None
                else self.splice(sub, reps, mask, extra=extra)
            )
            nll, valid = self.decoder.token_nll(sb)
        ce = self.pooled_ce(sub, nll, valid)
        stats = {f"loss/{f}_noslot": float(ce.item()) for f in sorted(set(sub.tasks))}
        stats["noslot_rows"] = float(len(sub.tasks))
        return ce, stats

    def _training_loss_slotted_nospine(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch = batch.to(self.device)
        nr = self.noprimary_rows(batch)
        if nr:
            n_loss, n_stats = self.noprimary_loss(batch.subset(nr))
            rest = [i for i in range(len(batch.tasks)) if i not in set(nr)]
            if not rest:
                n_stats["loss"] = float(n_loss.item())
                n_stats["ratio"] = float("nan")
                self._prompt_contrast_capture = None
                self._utility_capture = None
                return n_loss, n_stats
            total, stats = self._training_loss_primary(batch.subset(rest))
            stats.update(n_stats)
            return total + n_loss, stats
        return self._training_loss_primary(batch)

    def _training_loss_primary(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        cr = self.cached_rows(batch)
        if not cr:
            return self._training_loss_reps(batch)
        batch = batch.to(self.device)
        o_loss, o_stats = self.overview_loss(batch.subset(cr))
        rest = [i for i in range(len(batch.tasks)) if i not in set(cr)]
        if not rest:
            o_stats["loss"] = float(o_loss.item())
            o_stats["ratio"] = float("nan")
            self._prompt_contrast_capture = None
            self._utility_capture = None
            return o_loss, o_stats
        total, stats = self._training_loss_reps(batch.subset(rest))
        stats.update(o_stats)
        return total + o_loss, stats

    def _training_loss_reps(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        step = self.current_step
        n_ratios = max(1, int(self.cfg2.ratio.ratios_per_step))
        selection = (
            "random" if self.select_rng.random() < self.cfg2.ratio.p_random_selection else None
        )
        batch = batch.to(self.device)
        total_ce = None
        sel_loss = None
        stats: dict[str, float] = {}
        fam_sums: dict[str, list[float]] = {}
        ratios = [self.ratio_schedule.sample(step) for _ in range(n_ratios)]
        kd_w = float(self.cfg2.kd.weight)
        if step % max(1, int(self.cfg2.kd.kd_every)) != 0:
            kd_w = 0.0  # CE-only step (kd.kd_every cadence)
        kl_total = None
        for j, ratio in enumerate(ratios):
            arm_utility = (
                j == 0
                and float(self.cfg2.kd.selection_weight) > 0
                and step % max(1, int(self.cfg2.kd.selection_every)) == 0
                and self.cfg2.kd.selection_signal == "utility_grad"
                and selection != "random"
            )
            self._utility_capture = {} if arm_utility else None
            if arm_utility:
                # The encoder hooks dL/dh at the hook layer over ALL positions
                # during the main backward (no extra pass).
                self.encoder.utility_capture = self._utility_capture
            with self.autocast():
                fam_ratio, extra_ratio = self.slot_ratios(batch, ratio)
                out = self.encode(batch, fam_ratio, selection)
                self.encoder.utility_capture = None
                extra = self.encode_extra(batch, fam_ratio, selection, extra_ratio=extra_ratio)
                sb = (
                    self.splice(batch, out.reps, out.rep_mask)
                    if extra is None
                    else self.splice(batch, out.reps, out.rep_mask, extra=extra)
                )
                sel_due = (
                    j == 0
                    and float(self.cfg2.kd.selection_weight) > 0
                    and step % max(1, int(self.cfg2.kd.selection_every)) == 0
                )
                want_sel = sel_due and self.cfg2.kd.selection_signal == "attention"
                if self._utility_capture is not None:
                    # (random-selection steps arm nothing: no head to teach)
                    self._utility_capture["out"] = out
                tb = (
                    self.full_text_batch(batch, teacher_window=self.cfg2.kd.teacher_gold_window)
                    if (kd_w > 0 or want_sel)
                    else None
                )
                if kd_w > 0:
                    nll, valid, kl = self.decoder.token_nll_and_kd(
                        sb,
                        tb,
                        temperature=self.cfg2.kd.temperature,
                        teacher_model=getattr(self, "teacher", None),
                    )
                    kl_total = kl if kl_total is None else kl_total + kl
                else:
                    nll, valid = self.decoder.token_nll(sb)
                self._sel_due_this_step = want_sel
                if want_sel:
                    # Independent of KD; without a separate teacher the (frozen)
                    # decoder's own answer-to-context attention is the target.
                    rows = self.selection_rows(batch)
                    if rows is None:
                        sel_loss = self.selection_loss(batch, out, tb)
                    elif rows:
                        sub = batch.subset(rows)
                        sel_loss = self.selection_loss(
                            batch, out, self.full_text_batch(sub), rows=rows
                        )
            loss = self.pooled_ce(batch, nll, valid)
            total_ce = loss if total_ce is None else total_ce + loss
            for t, idx in task_indices(batch).items():
                v = float((nll[idx].sum() / valid[idx].sum().clamp_min(1)).item())
                fam_sums.setdefault(t, []).append(v)
            if j == 0:
                first_out = out
                self._prompt_contrast_capture = None
                if (
                    float(self.cfg2.kd.prompt_contrast_weight) > 0
                    and batch.alt_present is not None
                    and bool(batch.alt_present.any())
                ):
                    per = (nll * valid).sum(-1) / valid.sum(-1).clamp_min(1)
                    self._prompt_contrast_capture = {
                        "batch": batch,
                        "ratio": fam_ratio,
                        "selection": selection,
                        "nll_right": per.detach(),
                    }
                stats.update(
                    {
                        "k_mean": float(out.k.float().mean().item()),
                        "ctx_mean": float(out.n_valid.float().mean().item()),
                    }
                )
                self.guard_reps("rep", out.reps, out.rep_mask, stats)
        ce = total_ce / n_ratios
        hs_loss, head_raw_std = self.head_scale_terms(first_out)
        if head_raw_std is not None:
            stats["head_raw_std"] = head_raw_std
        aw = self.anchor_weight(step)
        if getattr(self, "ctx_tok", None) is not None:
            # two tokenizers: an encoder position has no decoder token to anchor to (refused at
            # setup when weighted; not computed for logging either)
            a_loss, cos_to_token = torch.zeros((), device=self.device), float("nan")
        else:
            a_loss, cos_to_token = self.anchor_loss(batch, first_out)
        total = float(self.cfg2.kd.ce_weight) * ce + aw * a_loss
        if kl_total is not None:
            kl_mean = kl_total / n_ratios
            total = total + kd_w * kl_mean
            stats["loss/kd"] = float(kl_mean.item())
        if sel_loss is not None:
            total = total + float(self.cfg2.kd.selection_weight) * sel_loss
            stats["loss/selection"] = float(sel_loss.item())
        if hs_loss is not None:
            total = total + hs_loss
            stats["loss/head_scale"] = float(hs_loss.item())
        if self.cfg2.multilevel.enabled:
            d = self.multilevel_distill(batch, step)
            if d is not None:
                total = total + d[0]
                stats["loss/distill"] = float(d[0].item())
                stats.update({f"distill/{k}": v for k, v in d[1].items()})
        rw = float(self.cfg2.anchor.repro_weight)
        if rw > 0:
            r = self.repro_loss(batch, first_out)
            if r is not None:
                total = total + rw * r[0]
                stats["loss/repro"] = float(r[0].item())
                stats["repro/cos_to_token"] = r[1]
        dw = float(self.cfg2.anchor.decorrelation_weight)
        if dw > 0:
            d_loss = self.decorrelation_loss(first_out.reps, first_out.rep_mask)
            total = total + dw * d_loss
            stats["loss/decorrelation"] = float(d_loss.item())
        vw = float(self.cfg2.anchor.within_doc_diversity_weight)
        if vw > 0:
            v_loss = self.within_doc_diversity_loss(first_out.reps, first_out.rep_mask)
            total = total + vw * v_loss
            stats["loss/within_doc_diversity"] = float(v_loss.item())
        stats.update(
            {
                "loss": float(ce.item()),
                "loss/anchor": float(a_loss.item()),
                "anchor_weight": aw,
                "ratio": float(sum(ratios) / len(ratios)),
                "ratio_floor": self.ratio_schedule.floor(step),
                "ratio_ceiling": self.ratio_schedule.ceiling(step),
                "ratio_mastery": float(self.ratio_schedule.mastery),
                "ratio_steps_per_doubling": float(
                    min(1e9, self.ratio_schedule.steps_per_doubling(step))
                ),
                "ratios_per_step": float(n_ratios),
                "selection_random": float(selection == "random"),
                "rep/cos_to_token": cos_to_token,
            }
        )
        for t, vals in fam_sums.items():
            stats[f"loss/{t}"] = float(sum(vals) / len(vals))
        return total, stats

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def eval_batches(self, ds, batch_size: int, rows: list[int] | None = None) -> list[Batch]:
        """``rows`` restricts the batches to those dataset indices (default:
        every row) -- the task eval uses it to keep rows WITHOUT a context
        slot out of the rep arms."""
        cap = int(getattr(self.cfg2.data, "eval_batch_max_tokens", 0) or 0)
        hint = getattr(ds, "length_hint", None)
        order_src = list(rows) if rows is not None else list(range(len(ds)))
        if cap > 0 and hint is not None:
            # Length-packed eval batches: rows sorted by their length hint,
            # packed until rows x longest > cap. Teacher forcing is
            # deterministic, so the batching changes nothing but the time.
            order = sorted(order_src, key=lambda i: int(hint(i)))
            groups: list[list[int]] = []
            cur: list[int] = []
            cur_max = 0
            for i in order:
                h = max(1, int(hint(i)))
                if cur and (len(cur) + 1) * max(cur_max, h) > cap:
                    groups.append(cur)
                    cur, cur_max = [], 0
                cur.append(i)
                cur_max = max(cur_max, h)
            if cur:
                groups.append(cur)
            return [self.collate([ds[i] for i in g]) for g in groups]
        if rows is not None:
            ds = torch.utils.data.Subset(ds, order_src)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate, num_workers=0
        )
        return list(loader)

    @staticmethod
    def overview_rows(ds) -> list[int]:
        """Indices of rows whose FIRST slot is a repo overview (cached reps)."""
        row = getattr(ds, "row", None)
        if row is None:
            return []
        return [
            i for i in range(len(ds))
            if str(row(i).get("context") or "").startswith('["{\\"repo_overview')
        ]

    @torch.no_grad()
    def overview_eval_metrics(self, batches: list[Batch]) -> dict[str, float]:
        """Teacher-forced CE on overview rows: with the cached overview, with
        the overview ZEROED (other slots intact -- isolates the overview), and
        with no slot at all (the floor). Extra slots at x8."""
        tot = {"ov": 0.0, "zero": 0.0, "noslot": 0.0}
        n_tok = 0.0
        n_rows = 0
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                reps, mask = self.cached_primary(batch)
                extra = self.encode_extra(batch, 1.0 / 8.0, self.eval_selection)
                arms = {
                    "ov": self.splice(batch, reps, mask, extra=extra)
                    if extra is not None else self.splice(batch, reps, mask),
                    "zero": self.splice(batch, torch.zeros_like(reps), mask, extra=extra)
                    if extra is not None else self.splice(batch, torch.zeros_like(reps), mask),
                    "noslot": self.splice(batch, None, None),
                }
                for k, sb in arms.items():
                    nll, valid = self.decoder.token_nll(sb)
                    tot[k] += float(nll.sum().item())
                    if k == "ov":
                        n_tok += float(valid.sum().item())
            n_rows += len(batch.tasks)
        n_tok = max(n_tok, 1.0)
        return {
            "overview/loss": tot["ov"] / n_tok,
            "overview/zeroed_overview_loss": tot["zero"] / n_tok,
            "overview/noslot_loss": tot["noslot"] / n_tok,
            "overview/gain_vs_zeroed_nats": (tot["zero"] - tot["ov"]) / n_tok,
            "overview/rows": float(n_rows),
        }

    @staticmethod
    def noslot_rows(ds) -> list[int]:
        """Indices of rows rendered WITHOUT a context slot (framing "chat":
        a plain conversation -- e.g. an agent's FIRST move, before any tool
        result exists). Such rows have nothing to encode: a batch of them
        alone hands the encoder a 0-length document (P28 died at its
        step-12000 eval on agent_tool_slots_v4, which mixes first-move rows
        with slotted ones under one family), and in a mixed batch they carry
        the same loss into every arm and only dilute the rep gain."""
        row = getattr(ds, "row", None)
        if row is None:
            return []
        return [i for i in range(len(ds)) if row(i).get("framing") in ("chat", "chat_think")]

    def _flush(self) -> None:
        """Release the allocator pool between eval phases. On unified memory
        the RESERVED pool is host RAM; an eval walks a 7-ratio grid plus the
        task stores and, with LoRA active, grew the pool to 78 GB (P3) even
        though allocated memory peaked at 17 GB."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate_all(self) -> dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()
        d = self.cfg2.data
        batches = self.eval_batches(self.eval_ds, d.eval_batch_size)
        m: dict[str, float] = {}
        floor, floor_vec = self._arm(batches, ratio=None)
        for t, (nll, acc) in floor.items():
            m[f"eval/noctx/{t}/loss"] = nll
            m[f"eval/noctx/{t}/token_acc"] = acc
        full, _ = self._arm(batches, ratio="full")
        for t, (nll, _acc) in full.items():
            m[f"eval/full_text/{t}/loss"] = nll
            m[f"eval/full_text/{t}/ceiling_gain_nats"] = floor[t][0] - nll
        self._flush()
        # The zeroed arm at 4x: slot present, zeros — the decoder's reaction
        # to the positions alone. Reps must beat THIS, not just no-context.
        for t, (nll, _acc) in self.teacher_forced(batches, ratio=0.25, zero_reps=True).items():
            m[f"eval/zeroed_x4/{t}/loss"] = nll
        self._flush()
        for r in EVAL_RATIOS:
            tag = f"x{round(1 / r)}"
            tf, vec = self._arm(batches, ratio=r)
            for t, (nll, acc) in tf.items():
                m[f"eval/{tag}/{t}/loss"] = nll
                m[f"eval/{tag}/{t}/token_acc"] = acc
                m[f"eval/{tag}/{t}/rep_gain_nats"] = floor[t][0] - nll
                if t in full:
                    m[f"eval/{tag}/{t}/captured"] = self.captured_fraction(
                        floor[t][0], nll, full[t][0]
                    )
                if t in floor_vec and t in vec:
                    for k, v in self.gap_quantiles(floor_vec[t], vec[t]).items():
                        m[f"eval/{tag}/{t}/{k}"] = v
            if d.gen_samples > 0:
                m.update({f"eval/{tag}/gen/{k}": v for k, v in self.generative(batches, r).items()})
            self._flush()
        sched = getattr(self, "ratio_schedule", None)
        if sched is not None and sched.adaptive:
            # The arm at the CURRENT FLOOR ratio: the mastery signal of the
            # adaptive descent (gap to full text at the hardest live ratio).
            rf = float(sched.floor(self.current_step))
            m["eval/xfloor/ratio"] = rf
            m["eval/xfloor/factor"] = 1.0 / rf
            for t, (nll, acc) in self.teacher_forced(batches, ratio=rf).items():
                m[f"eval/xfloor/{t}/loss"] = nll
                m[f"eval/xfloor/{t}/token_acc"] = acc
                m[f"eval/xfloor/{t}/rep_gain_nats"] = floor[t][0] - nll
            self._flush()
        skip = set(d.train_eval_skip_stores)
        for name, ds in self.task_eval.items():
            if name in skip:
                continue
            m.update({f"task/{name}/{k}": v for k, v in self.task_eval_metrics(ds).items()})
            self._flush()
        m.update({f"probe/{k}": v for k, v in self.probe(batches, d.probe_ratio).items()})
        m["probe/nn_token_acc_x1"] = self.nn_token_accuracy(batches[: max(1, len(batches) // 4)])
        m.update(self.lm_health(batches))
        self._flush()
        self.encoder.train()
        self.decoder.train()
        return m

    # ---- eval-time ablation switches (scripts/bgkit2_eval_ablations.py) ----
    # eval_selection: None (learned head) | "random" (random positions at the
    # same budget). eval_no_prompt: hand the encoder no compression prompt.
    # The token-pruning arm toggles cfg2.model.rep_token_residual_only.
    eval_selection: str | None = None
    eval_no_prompt: bool = False
    # eval_wrong_prompt: every sample gets the NEXT sample's compression
    # prompt (eval batches are consecutive rows, so on a multi-question
    # store that is another question about the same document). Sharper than
    # no-prompt: a model that USES the prompt must get worse here.
    eval_wrong_prompt: bool = False

    def _ablate_batch(self, batch: Batch) -> Batch:
        if self.eval_no_prompt and batch.prompt_ids is not None:
            batch = dataclasses.replace(batch, prompt_ids=None, prompt_mask=None)
        elif self.eval_wrong_prompt and batch.prompt_ids is not None:
            batch = dataclasses.replace(
                batch,
                prompt_ids=torch.roll(batch.prompt_ids, shifts=1, dims=0),
                prompt_mask=torch.roll(batch.prompt_mask, shifts=1, dims=0),
            )
        return batch

    def teacher_forced(
        self, batches: list[Batch], ratio: float | None, zero_reps: bool = False
    ) -> dict[str, tuple[float, float]]:
        """``ratio`` None = no-context floor; "full" = full-text ceiling;
        ``zero_reps`` = the slot is present at ``ratio`` but holds zeros (the
        old lineage's ``zeroed`` arm: separates "the decoder ignores the reps"
        from "the slot positions themselves helped")."""
        summary, _ = self._arm(batches, ratio, zero_reps=zero_reps)
        return summary

    def _arm(
        self, batches: list[Batch], ratio: float | None, zero_reps: bool = False
    ) -> tuple[dict[str, tuple[float, float]], dict[str, torch.Tensor]]:
        """One evaluation arm: per-family (mean nll, token acc) AND the
        per-token nll vectors per family in batch order (the labelled suffix
        is identical across arms, so vectors from two arms align token by
        token — the pooled mean hides rep-dependence, see
        bgkit/eval/rep_dependence.py)."""
        sums: dict[str, list[float]] = {}  # family -> [nll, correct, count]
        vecs: dict[str, list[torch.Tensor]] = {}
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                if ratio is None:
                    sb = self.splice(batch, None, None)
                elif ratio == "full":
                    sb = self.full_text_batch(batch)
                else:
                    ab = self._ablate_batch(batch)
                    out = self.encode(ab, ratio, self.eval_selection)
                    extra = self.encode_extra(ab, ratio, self.eval_selection)
                    reps = torch.zeros_like(out.reps) if zero_reps else out.reps
                    if zero_reps and extra is not None:
                        extra = [[(torch.zeros_like(r), m) for r, m in ex] for ex in extra]
                    sb = (
                        self.splice(batch, reps, out.rep_mask)
                        if extra is None
                        else self.splice(batch, reps, out.rep_mask, extra=extra)
                    )
                nll, valid = self.decoder.per_token_nll(sb)
                pred, _ = self.decoder.argmax_at_labels(sb)
            targets = sb.labels[:, 1:]
            correct = torch.zeros_like(valid)
            correct[valid] = pred == targets[valid]
            for t, idx in task_indices(batch).items():
                sums.setdefault(t, [0.0, 0.0, 0.0])
                sums[t][0] += float((nll[idx] * valid[idx]).sum())
                sums[t][1] += float(correct[idx].sum())
                sums[t][2] += float(valid[idx].sum())
                vecs.setdefault(t, []).append(nll[idx][valid[idx]].float().cpu())
        summary = {
            t: (s[0] / max(s[2], 1), s[1] / max(s[2], 1)) for t, s in sums.items() if s[2] > 0
        }
        return summary, {t: torch.cat(v) for t, v in vecs.items() if v}

    @staticmethod
    def captured_fraction(noctx: float, arm: float, full: float, min_span: float = 0.05) -> float:
        """(noctx - arm) / (noctx - full); NaN when the ceiling arm buys less
        than ``min_span`` nats over no-context (the ratio is then noise)."""
        span = noctx - full
        if span < min_span:
            return float("nan")
        return (noctx - arm) / span

    @staticmethod
    def gap_quantiles(noctx_vec: torch.Tensor, arm_vec: torch.Tensor) -> dict[str, float]:
        """Distribution of the per-token rep gain (noctx nll - arm nll)."""
        n = min(noctx_vec.numel(), arm_vec.numel())
        if n == 0:
            return {}
        g = (noctx_vec[:n] - arm_vec[:n]).float()
        return {
            "gap_p50": float(g.quantile(0.5)),
            "gap_p90": float(g.quantile(0.9)),
            "gap_frac_gt_0.5": float((g > 0.5).float().mean()),
            "gap_frac_lt_0": float((g < 0).float().mean()),
        }

    def generative(
        self,
        batches: list[Batch],
        ratio: float,
        want: int | None = None,
        task_filter: str | None = "reconstruct",
        max_new: int | None = None,
    ) -> dict[str, float]:
        want = self.cfg2.data.gen_samples if want is None else want
        preds, golds = [], []
        for batch in batches:
            idx = [i for i, t in enumerate(batch.tasks) if task_filter is None or t == task_filter]
            if not idx:
                continue
            idx = idx[: max(0, want - len(preds))]
            if not idx:
                break
            sub = batch.subset(idx).to(self.device)
            with self.autocast():
                out = self.encode(self._ablate_batch(sub), ratio, self.eval_selection)
                prompt = self.decoder.build_batch(
                    sub.prefix_ids,
                    out.reps,
                    out.rep_mask,
                    sub.instr_ids,
                    pad_side="left",
                    loss_on_suffix=False,
                )
                cap = self.cfg2.data.gen_max_new_tokens if max_new is None else max_new
                n_new = min(cap, max(t.shape[0] for t in sub.target_ids) + 8)
                gen = self.decoder.generate(
                    prompt, max_new_tokens=n_new, eos_token_id=self.templates.eos_id
                )
            for i in range(len(idx)):
                g = gen[i].tolist()
                if self.templates.eos_id in g:
                    g = g[: g.index(self.templates.eos_id)]
                g = [t for t in g if t != self.templates.pad_id]
                preds.append(g)
                golds.append(sub.target_ids[i].tolist())
            if len(preds) >= want:
                break
        if not preds:
            return {}
        pairs = list(zip(preds, golds, strict=True))
        # The last few raw generations, for reading (bgkit2_eval_ablations
        # writes them next to the metrics): a 0.000 EM must be inspectable.
        dec = getattr(self.tok, "decode", None)
        show = (lambda ids: dec(ids, skip_special_tokens=True)) if dec else (lambda ids: list(ids))
        self.last_gen_examples = [{"pred": show(p), "gold": show(g)} for p, g in pairs[:8]]
        out = {
            "n": float(len(preds)),
            "bleu": corpus_bleu(preds, golds),
            "exact_match": float(np.mean([exact_match(p, g) for p, g in pairs])),
            "token_f1": float(np.mean([token_f1(p, g) for p, g in pairs])),
            "prefix_acc": float(np.mean([prefix_accuracy(p, g) for p, g in pairs])),
        }
        if dec is not None:
            # SQuAD-style normalised text metrics (case, punctuation, articles,
            # whitespace): the token-id EM scores "garden." vs "garden" as 0.
            from bgkit.eval.metrics.qa_metrics import exact_match as text_em
            from bgkit.eval.metrics.qa_metrics import token_f1 as text_f1

            texts = [(show(p), show(g)) for p, g in pairs]
            out["exact_match_norm"] = float(np.mean([text_em(p, [g]) for p, g in texts]))
            out["token_f1_norm"] = float(np.mean([text_f1(p, [g]) for p, g in texts]))
        return out

    def task_eval_metrics(self, ds: TaskDataset) -> dict[str, float]:
        d = self.cfg2.data
        out: dict[str, float] = {}
        if list(getattr(ds, "families", [])) == [str(self.cfg2.kd.replay_family)]:
            return self.replay_eval_metrics(self.eval_batches(ds, d.eval_batch_size))
        noslot = self.noslot_rows(ds)
        ov = self.overview_rows(ds)
        if ov:
            out.update(
                self.overview_eval_metrics(self.eval_batches(ds, d.eval_batch_size, rows=ov))
            )
        spr = self.spine_row_indices(ds)
        if spr:
            out.update(self.spine_eval_metrics(self.eval_batches(ds, d.eval_batch_size, rows=spr)))
        slotted = [i for i in range(len(ds))
                   if i not in set(noslot) and i not in set(ov) and i not in set(spr)]
        single = len(getattr(ds, "families", [])) <= 1

        def key(fam: str, rest: str) -> str:
            return rest if single else f"{fam}/{rest}"

        if noslot:
            # Floor arm only: no slot, so no rep arm exists for these rows.
            for fam, (nll, acc) in self.teacher_forced(
                self.eval_batches(ds, d.eval_batch_size, rows=noslot), ratio=None
            ).items():
                out[key(fam, "noslot/loss")] = nll
                out[key(fam, "noslot/token_acc")] = acc
            out[key(getattr(ds, "families", ["task"])[0], "noslot/rows")] = float(len(noslot))
        if not slotted:
            return out
        batches = self.eval_batches(ds, d.eval_batch_size, rows=slotted)
        floor = self.teacher_forced(batches, ratio=None)

        for fam, (nll, acc) in floor.items():
            out[key(fam, "noctx/loss")] = nll
            out[key(fam, "noctx/token_acc")] = acc
        for fam, (nll, _acc) in self.teacher_forced(batches, ratio="full").items():
            out[key(fam, "full_text/loss")] = nll
            out[key(fam, "full_text/ceiling_gain_nats")] = floor[fam][0] - nll
        for f in d.task_eval_factors:
            r = 1.0 / float(f)
            tag = f"x{int(f)}"
            tf = self.teacher_forced(batches, ratio=r)
            for fam, (nll, acc) in tf.items():
                out[key(fam, f"{tag}/loss")] = nll
                out[key(fam, f"{tag}/token_acc")] = acc
                out[key(fam, f"{tag}/rep_gain_nats")] = floor[fam][0] - nll
        gen_ok = not d.task_gen_families or any(
            f in set(d.task_gen_families) for f in getattr(ds, "families", [])
        )
        if d.task_gen_samples > 0 and gen_ok:
            for f in d.task_gen_factors:
                r = 1.0 / float(f)
                g = self.generative(
                    batches,
                    r,
                    want=d.task_gen_samples,
                    task_filter=None,
                    max_new=d.task_gen_max_new_tokens,
                )
                out.update({f"gen_x{round(1 / r)}/{k}": v for k, v in g.items()})
                # raw generations in the log: an EM below chance (8B-A QuALITY 0.03 at step
                # 12000, full-text arm as bad) is a format question only the text answers
                logger.info(
                    "task_gen_examples",
                    families=list(getattr(ds, "families", [])),
                    factor=round(1 / r),
                    examples=[
                        {k: str(v)[:200] for k, v in e.items()}
                        for e in getattr(self, "last_gen_examples", [])[:3]
                    ],
                )
        return out

    def probe(self, batches: list[Batch], ratio: float) -> dict[str, float]:
        per_doc, stats_acc, n = [], {}, 0
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                out = self.encode(self._ablate_batch(batch), ratio, self.eval_selection)
            for i in range(out.reps.shape[0]):
                per_doc.append(out.reps[i][out.rep_mask[i]].float().cpu())
            for k, v in rep_stats(out.reps, out.rep_mask, self.rep_row_norm).items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v
            n += 1
        res = rep_distinguishability(per_doc)
        res.update({f"rep_{k}": v / max(n, 1) for k, v in stats_acc.items()})
        return res

    @torch.no_grad()
    def nn_token_accuracy(self, batches: list[Batch]) -> float:
        """At ratio 1.0, the fraction of reps whose nearest decoder token
        embedding (cosine) is their own token — the standing form of the
        2026-09-07 CPU probe; the identity the interface must not lose.
        NaN on two-tokenizer runs: an encoder position has no decoder token of its own."""
        if getattr(self, "ctx_tok", None) is not None:
            return float("nan")
        table = self.decoder.embed_tokens.weight.detach().float()
        table = table / (table.norm(dim=-1, keepdim=True) + 1e-6)
        hit = total = 0.0
        for batch in batches:
            batch = batch.to(self.device)
            with self.autocast():
                out = self.encode(batch, 1.0)
            reps = out.reps.float()
            reps = reps / (reps.norm(dim=-1, keepdim=True) + 1e-6)
            ids = batch.input_ids.gather(1, out.positions.clamp_min(0))
            for i in range(reps.shape[0]):
                r = reps[i][out.rep_mask[i]]
                if r.numel() == 0:
                    continue
                nn_id = (r @ table.T).argmax(-1)
                hit += float((nn_id == ids[i][out.rep_mask[i]]).sum())
                total += float(r.shape[0])
        return hit / max(total, 1.0)

    def lm_health(self, batches: list[Batch]) -> dict[str, float]:
        tot_a = tot_b = cnt = 0.0
        for batch in batches[: max(1, len(batches) // 4)]:
            batch = batch.to(self.device)
            ids, am = batch.input_ids, batch.attention_mask
            if batch.dec_ctx is not None:  # two tokenizers: the same text in decoder ids
                ids, am = self._pad_ids(batch.dec_ctx)
            valid = float(am[:, 1:].sum().item())
            with self.autocast():
                a = self.decoder.plain_text_nll(ids, am)
                with self.decoder.disable_adapter():
                    b = self.decoder.plain_text_nll(ids, am)
            tot_a += a * valid
            tot_b += b * valid
            cnt += valid
        return {
            "lm_health/nll": tot_a / max(cnt, 1),
            "lm_health/nll_base": tot_b / max(cnt, 1),
            "lm_health/delta": (tot_a - tot_b) / max(cnt, 1),
        }
