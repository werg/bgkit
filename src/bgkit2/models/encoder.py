"""BgKIT2Encoder: document tokens -> k representation vectors in decoder space.

    tokens -> embed -> layers[0:h) -> head scores -> exact top-k
           -> + survive_embedding at survivors -> layers[h:) -> final norm
           -> gather survivors -> projection -> DecoderInterfaceNorm -> reps

Batched with right padding; the backbone masks padded positions in both the
attention and the convolutions, so a document encodes identically alone or in
a batch (tested).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from .interface_norm import DecoderInterfaceNorm
from .lfm2_bidirectional import Lfm2BidirectionalBackbone, _bidirectional_additive_mask
from .survivorship import (
    SurvivorshipHead,
    exact_topk_select,
    gather_survivors,
    random_scores,
    segment_zscore,
    straight_through_gate,
)


@dataclass
class EncoderOutput:
    reps: torch.Tensor  # (B, K, D) in decoder space, zero where padded
    rep_mask: torch.Tensor  # (B, K) bool
    positions: torch.Tensor  # (B, K) original token positions of the survivors
    keep: torch.Tensor  # (B, L) bool
    scores: torch.Tensor  # (B, L) z-scored head scores (0 where invalid)
    k: torch.Tensor  # (B,) survivor counts
    n_valid: torch.Tensor  # (B,) document lengths
    slot_reps: torch.Tensor | None = None  # (B, M, D) summary slots pooled over dropped tokens
    slot_mask: torch.Tensor | None = None  # (B, M) bool
    gate: torch.Tensor | None = None  # (B, K) straight-through gate (forward 1) — head gradient
    # For utility-gradient selection: the head's input at the hook layer
    # (detached, full sequence incl. prompt), the eligibility mask over that
    # sequence, and the prompt offset (positions + offset index into them).
    head_input: torch.Tensor | None = None
    eligible: torch.Tensor | None = None
    offset: int = 0
    # The head's RAW scores (B, L_full, with grad; None under random
    # selection): the z-score the operator uses is scale-invariant, so the
    # raw scale is a gradient-free direction that needs its own anchor and
    # its own logging (old lineage: raw std 129 -> 8431 in 30 steps).
    raw: torch.Tensor | None = None
    # Gathered survivors BEFORE the decoder-facing projection, in the ENCODER's
    # hidden space (B, K, D). This is what a multi-level tree feeds upward:
    # `reproduce(survivors)` maps them into the encoder's input space and the
    # next level consumes them via `forward(doc_embeds=...)`. `reps` is already
    # in DECODER space and is only correct at the top of the tree.
    survivors: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class BgKIT2Encoder(nn.Module):
    def __init__(
        self,
        backbone: Lfm2BidirectionalBackbone,
        head_layer: int,
        head_inner_dim: int = 256,
        projection_mult: int = 4,
        interface_momentum: float = 0.1,
        target_row_norm: float = 1.0,
        selection: str = "learned",
        min_survivors: int = 1,
        gate_tau: float = 1.0,
        interface_row_norm: bool = False,
        interface_prenorm: bool = False,
        interface_match_moments: bool = False,
        summary_slot_frac: float = 0.0,
        projection_block: bool = False,
        projection_input_skip: bool = False,
        interface_rms_only: bool = False,
        prompt_separator: bool = False,
        prompt_conditioning: bool = False,
        repro_head: bool = False,
    ) -> None:
        super().__init__()
        if selection not in {"learned", "random"}:
            raise ValueError(f"selection must be 'learned' or 'random'; got {selection!r}")
        if not 0 <= head_layer <= backbone.num_layers:
            raise ValueError(f"head_layer must be in [0, {backbone.num_layers}]")
        d = backbone.hidden_size
        self.backbone = backbone
        self.head_layer = int(head_layer)
        self.selection = selection
        self.min_survivors = int(min_survivors)
        self.gate_tau = float(gate_tau)
        # Summary slots (2026-09-06): besides the k survivors, emit m =
        # ceil(frac * k) extra reps, each the mean of the backbone's final
        # hidden states over the DROPPED tokens of one contiguous chunk of the
        # document, through the same projection + interface. The token budget
        # is kept: the survivor ratio is scaled to ratio / (1 + frac) so that
        # survivors + slots ~= ratio * n. Slots cannot perturb the survivors
        # (which may be verbatim token embeddings) and can be ignored by the
        # decoder if useless -- the clean test of what an encoder adds beyond
        # selection.
        self.summary_slot_frac = float(summary_slot_frac)
        self.head = SurvivorshipHead(d, head_inner_dim)
        self.survive_embedding = nn.Parameter(torch.randn(d) * 0.02)
        self.projection = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, d * projection_mult),
            nn.GELU(),
            nn.Linear(d * projection_mult, d),
        )
        # The REPRO HEAD: encoder output -> the encoder's OWN INPUT space,
        # deliberately distinct from `projection` (which targets the DECODER's
        # input space). It is what lets a level of the encoder consume another
        # level's survivors: the backbone was trained on token embeddings, so
        # raw survivors are off-distribution to it. This is bgkit1's
        # `auto_repro_head` (see src/bgkit/training/joint_block_trainer.py,
        # which pre-warms it against embed_tokens with MSE + cosine) ported to
        # the single-backbone bgkit2 encoder. Off by default: it costs nothing
        # until a multi-level run enables it.
        self.repro: nn.Sequential | None = None
        if repro_head:
            self.repro = nn.Sequential(
                RMSNorm(d),
                nn.Linear(d, d * projection_mult),
                nn.GELU(),
                nn.Linear(d * projection_mult, d),
            )
        # Optional CONTEXTUAL projection (the old lineage's projection block):
        # a trainable copy of the backbone's last attention layer run
        # bidirectionally over the gathered survivors, with rotary positions =
        # the survivors' original document positions. Survivors can exchange
        # information AFTER selection, which the position-local MLP cannot.
        # Optional INPUT-EMBEDDING SKIP: each survivor's own input embedding
        # (the encoder's, not the decoder's) enters the projection through a
        # zero-initialised linear map. A decoder-initialised encoder (old
        # Phase 1) had token identity in its residual stream by construction;
        # the MLM backbone discards it, so the skip restores it and leaves the
        # backbone to carry context.
        self.input_skip: nn.Linear | None = None
        if projection_input_skip:
            self.input_skip = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.input_skip.weight)
        # A learned separator between the compression prompt and the document
        # (user 2026-09-09: the prompt should be SEPARATED): one extra
        # position, attended by everything, never selectable. Initialised at
        # the mean token embedding so it starts at token scale.
        self.prompt_separator: nn.Parameter | None = None
        if prompt_separator:
            emb = self.backbone.embed_tokens.weight.detach()
            self.prompt_separator = nn.Parameter(emb.float().mean(0).to(emb.dtype).clone())
        # Prompt conditioning (P19o, 2026-09-10): an EXPLICIT route from the
        # compression prompt into every document position. In-context alone
        # the prompt barely reaches the document — two different questions
        # about the same document gave reps with cosine 0.997 and a 3-6%
        # relative difference (scripts/bgkit2_probe_prompt_sensitivity.py on
        # P19j/P19k/P19n), and three contrast objectives could not open a
        # gap through that route. The pooled prompt (masked mean) is
        # projected and ADDED to every document position at the input and,
        # pooled again from the hook-layer hidden states, before the head at
        # the hook layer (so selection and consolidation both see it). Both
        # maps start at zero: the model begins exactly as its checkpoint.
        self.prompt_cond_in: nn.Linear | None = None
        self.prompt_cond_hook: nn.Linear | None = None
        if prompt_conditioning:
            self.prompt_cond_in = nn.Linear(d, d, bias=False)
            self.prompt_cond_hook = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.prompt_cond_in.weight)
            nn.init.zeros_(self.prompt_cond_hook.weight)
        # Utility-gradient capture: when the trainer sets this dict before a
        # forward, the hook-layer hidden state (EVERY position, not just the
        # survivors) is made to require grad and its dL/dh is stored in it
        # during the main backward — the old lineage's capture site
        # (level_compressor.py), so dropped positions get a target too.
        self.utility_capture: dict | None = None
        self.projection_block: nn.Module | None = None
        self._projection_block_src: int | None = None
        if projection_block:
            self._projection_block_src = self._last_attention_layer_index()
            self.reset_projection_block_from_backbone()
        self.interface_norm = DecoderInterfaceNorm(
            d,
            target_row_norm=target_row_norm,
            momentum=interface_momentum,
            row_norm=interface_row_norm,
            prenorm=interface_prenorm,
            match_moments=interface_match_moments,
            rms_only=interface_rms_only,
        )

    # ------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size

    def set_backbone_trainable(self, layers: list[int] | None) -> list[int]:
        """Freeze the backbone except the listed layer indices.

        ``None`` or ``[]`` freezes it entirely (the hunting-phase default:
        only head, survive embedding and projection learn). Embeddings and
        the final norm stay frozen in every case. Returns the trainable set.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        wanted = sorted(set(int(i) for i in (layers or [])))
        n = self.backbone.num_layers
        bad = [i for i in wanted if not 0 <= i < n]
        if bad:
            raise ValueError(f"train_backbone_layers out of range [0, {n}): {bad}")
        for i in wanted:
            for p in self.backbone.model.layers[i].parameters():
                p.requires_grad_(True)
        return wanted

    def inject_lora(
        self, rank: int, alpha: int, dropout: float = 0.0, layers: list[int] | None = None
    ) -> int:
        """Inject LoRA adapters into the backbone in place (after any warm start
        so the base weights load with their original keys). Returns the number
        of adapter parameters; only those become trainable in the backbone."""
        from peft import LoraConfig, inject_adapter_in_model

        cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
            layers_to_transform=list(layers) if layers else None,
            bias="none",
        )
        inject_adapter_in_model(cfg, self.backbone.model)
        n = 0
        for name, p in self.backbone.named_parameters():
            is_lora = "lora_" in name
            p.requires_grad_(is_lora)
            n += p.numel() if is_lora else 0
        return n

    # ------------------------------------------------------------------
    # SCOPED ADAPTERS and ROLES (2026-09-22; generalises the compaction-spine
    # adapter of bd09be1). A scoped adapter is a named peft LoRA on the
    # backbone, switched on and off per ROLE:
    #
    #   base   no scoped LoRA, the base head / projection / interface. The
    #          frozen TREE encoder: cached tree nodes (bgkit2_cache_repo_trees.py)
    #          are this role's output, and stay valid because in adapter mode
    #          nothing this role reads is ever trained.
    #   slots  the "slots" LoRA plus trainable COPIES (``slot_parts``) of every
    #          non-backbone part the forward reads (head, survive embedding,
    #          prompt separator / conditioning, input skip, projection block,
    #          projection, interface norm). Ordinary slot encoding: tool results,
    #          pages, QA documents, memory, prompted read-outs, small live trees.
    #   spine  compaction-spine nodes: the "spine" LoRA when one is installed
    #          (frozen-encoder runs, bd09be1), otherwise the slot adapter; the
    #          bridge is ``spine_repro`` either way.
    #
    # The DEFAULT role is "slots" when the slot adapter is installed, else
    # "base" -- so every slot call site trains the adapter without having to
    # say so, and an encoder without adapters is exactly the old encoder.
    # ``encode_tree`` refuses to run on an adapter encoder unless the caller
    # names the role explicitly (``role("base")`` for cached / large trees), so
    # a tree path cannot silently pick up the adapter and drift from its cache.
    # ------------------------------------------------------------------
    SPINE_ADAPTER = "spine"
    SLOT_ADAPTER = "slots"
    # the non-backbone parts ``forward`` reads; ``slot_parts`` holds a copy of
    # each that exists on this encoder
    ADAPTER_PARTS: tuple[str, ...] = (
        "head", "survive_embedding", "prompt_separator", "prompt_cond_in", "prompt_cond_hook",
        "input_skip", "projection_block", "projection", "interface_norm",
    )
    ROLES: tuple[str, ...] = ("base", "slots", "spine")

    def _scoped_layers(self) -> dict[str, list]:
        d = self.__dict__.get("_scoped_lora_layers")
        if d is None:
            d = self.__dict__["_scoped_lora_layers"] = {}
        return d

    @property
    def has_spine_adapter(self) -> bool:
        """A dedicated spine LoRA is installed (frozen-encoder spine runs)."""
        return bool(self._scoped_layers().get(self.SPINE_ADAPTER))

    @property
    def has_slot_adapter(self) -> bool:
        return bool(self._scoped_layers().get(self.SLOT_ADAPTER))

    def _inject_scoped_lora(
        self,
        name: str,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        layers: list[int] | None = None,
        targets: list[str] | None = None,
    ) -> None:
        """The same peft machinery as ``inject_lora`` under adapter ``name``.
        Only this adapter's tensors are made trainable; nothing else's
        ``requires_grad`` changes (peft freezes what it injects into)."""
        from peft import LoraConfig, inject_adapter_in_model
        from peft.tuners.lora.layer import LoraLayer

        cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=list(targets or ["q_proj", "k_proj", "v_proj", "out_proj",
                                            "in_proj", "w1", "w2", "w3"]),
            layers_to_transform=list(layers) if layers else None,
            bias="none",
        )
        grads = {n: p.requires_grad for n, p in self.backbone.named_parameters()}
        # into the WRAPPER: peft's ``layers_to_transform`` matches ".layers.<i>."
        # and so needs a prefix before "layers" (injected into the LFM2 model the
        # keys start with "layers." and no layer matches); the parameter names
        # (backbone.model.layers.<i>...) are the same either way
        inject_adapter_in_model(cfg, self.backbone, adapter_name=name)
        self._scoped_layers()[name] = [
            m for m in self.backbone.modules() if isinstance(m, LoraLayer) and name in m.lora_A
        ]
        for n, p in self.backbone.named_parameters():
            if f".{name}." in n or n.endswith(f".{name}"):
                p.requires_grad_(True)
            else:
                p.requires_grad_(grads.get(n.replace(".base_layer", ""), grads.get(n, False)))
        self._applied_loras = None  # force the next sync
        self._sync_role()

    def _lora_parameters(self, name: str) -> list[nn.Parameter]:
        return [p for n, p in self.backbone.named_parameters() if f".{name}." in n]

    # ---- the compaction-spine adapter (frozen-encoder runs) ----
    def inject_spine_adapter(
        self,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        layers: list[int] | None = None,
        targets: list[str] | None = None,
        train_bridge: bool = True,
    ) -> int:
        """A dedicated "spine" LoRA plus the spine's bridge copy (see
        ``add_spine_bridge``). Returns the number of spine parameters."""
        if not self.has_spine_adapter:
            self._inject_scoped_lora(self.SPINE_ADAPTER, rank, alpha, dropout, layers, targets)
            self.add_spine_bridge(train_bridge)
        return sum(p.numel() for p in self.spine_parameters())

    def add_spine_bridge(self, trainable: bool = True) -> None:
        """``spine_repro``: the spine's own copy of the repro bridge, so the
        shared ``repro`` head (tree / overview paths) is never touched."""
        if getattr(self, "spine_repro", None) is None and self.repro is not None:
            self.spine_repro = copy.deepcopy(self.repro)
        if getattr(self, "spine_repro", None) is not None:
            for p in self.spine_repro.parameters():
                p.requires_grad_(bool(trainable))

    def spine_parameters(self) -> list[nn.Parameter]:
        ps = [p for p in self._lora_parameters(self.SPINE_ADAPTER) if p.requires_grad]
        if getattr(self, "spine_repro", None) is not None:
            ps += [p for p in self.spine_repro.parameters() if p.requires_grad]
        return ps

    # ---- the slot adapter (optim.encoder_mode: adapter) ----
    def inject_slot_adapter(
        self,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        layers: list[int] | None = None,
        targets: list[str] | None = None,
        train_parts: bool = True,
    ) -> int:
        """The "slots" LoRA plus trainable copies of the non-backbone parts
        (``slot_parts``), taken from the parts AS THEY ARE NOW -- inject after
        any warm start so the copies start at the loaded weights (a later load
        of a checkpoint without them seeds them the same way, see
        ``BgKIT2Core._remap_adapter_keys``). The base parts keep their
        ``requires_grad``; the caller's optimizer policy freezes them.
        Idempotent. Returns the number of slot-adapter parameters."""
        if not self.has_slot_adapter:
            self._inject_scoped_lora(self.SLOT_ADAPTER, rank, alpha, dropout, layers, targets)
            parts = nn.Module()
            for name in self.ADAPTER_PARTS:
                src = getattr(self, name, None)
                if src is None:
                    continue
                dup = copy.deepcopy(src)
                if isinstance(dup, nn.Parameter):
                    dup.requires_grad_(bool(train_parts))
                else:
                    for p in dup.parameters():
                        p.requires_grad_(bool(train_parts))
                setattr(parts, name, dup)
            self.slot_parts = parts
            self._sync_role()
        return sum(p.numel() for p in self.slot_adapter_parameters())

    def slot_adapter_parameters(self) -> list[nn.Parameter]:
        ps = [p for p in self._lora_parameters(self.SLOT_ADAPTER) if p.requires_grad]
        sp = getattr(self, "slot_parts", None)
        if sp is not None:
            ps += [p for p in sp.parameters() if p.requires_grad]
        return ps

    def slot_part_groups(self) -> dict[str, list[nn.Parameter]]:
        """Trainable slot-part copies by optimizer group."""
        sp = getattr(self, "slot_parts", None)

        def ps(*names: str) -> list[nn.Parameter]:
            out: list[nn.Parameter] = []
            for n in names:
                m = getattr(sp, n, None) if sp is not None else None
                if m is None:
                    continue
                out += [m] if isinstance(m, nn.Parameter) else list(m.parameters())
            return [p for p in out if p.requires_grad]

        return {
            "head": ps("head", "survive_embedding", "prompt_separator"),
            "prompt_cond": ps("prompt_cond_in", "prompt_cond_hook"),
            "projection": ps("projection", "input_skip"),
            "projection_block": ps("projection_block"),
        }

    def resync_slot_parts(self, names: tuple[str, ...] | None = None) -> None:
        """Re-copy base parts into ``slot_parts`` (after a base part was
        re-initialised, e.g. ``reset_projection_block_from_backbone``)."""
        sp = getattr(self, "slot_parts", None)
        if sp is None:
            return
        for name in names or self.ADAPTER_PARTS:
            src = getattr(self, name, None)
            dst = getattr(sp, name, None)
            if src is None or dst is None:
                continue
            with torch.no_grad():
                if isinstance(src, nn.Parameter):
                    dst.copy_(src)
                else:
                    dst.load_state_dict(src.state_dict())

    def interface_norms(self) -> list[DecoderInterfaceNorm]:
        """Every interface contract this encoder emits through (base + slot copy)."""
        out = [self.interface_norm]
        sp = getattr(self, "slot_parts", None)
        if sp is not None and getattr(sp, "interface_norm", None) is not None:
            out.append(sp.interface_norm)
        return out

    # ---- roles ----
    @property
    def default_role(self) -> str:
        return "slots" if self.has_slot_adapter else "base"

    @property
    def active_role(self) -> str:
        stack = self.__dict__.get("_role_stack") or []
        return stack[-1] if stack else self.default_role

    @property
    def role_explicit(self) -> bool:
        return bool(self.__dict__.get("_role_stack"))

    def _loras_for(self, role: str) -> tuple[set[str], bool]:
        """(scoped LoRA names active, whether the slot-part copies are read)."""
        if role == "slots" and self.has_slot_adapter:
            return {self.SLOT_ADAPTER}, True
        if role == "spine":
            if self.has_spine_adapter:
                return {self.SPINE_ADAPTER}, False
            if self.has_slot_adapter:
                return {self.SLOT_ADAPTER}, True
        return set(), False

    def _sync_role(self) -> None:
        """Make the peft layers match the active role (cheap when unchanged)."""
        want, parts = self._loras_for(self.active_role)
        self._use_slot_parts = parts
        key = frozenset(want)
        if self.__dict__.get("_applied_loras") == key:
            return
        scoped = self._scoped_layers()
        names = set(scoped)
        layers = {id(m): m for ms in scoped.values() for m in ms}
        for m in layers.values():
            keep = [a for a in m.lora_A if a not in names]  # unscoped (encoder_lora): always on
            on = [a for a in sorted(want) if a in m.lora_A]
            if keep or on:
                m._active_adapter = [*keep, *on]
                m._disable_adapters = False
            else:
                # nothing active: toggle the layer's flag directly (peft's
                # enable_adapters() would also flip requires_grad mid-graph)
                m._active_adapter = [next(iter(a for a in m.lora_A if a in names))]
                m._disable_adapters = True
        self._applied_loras = key

    def role(self, name: str):
        """Context manager: encode as ``name`` (base | slots | spine) inside.

        Under backbone gradient checkpointing a role switch with gradients on
        would make the backward's recompute run under a different adapter set
        than the forward did; refused rather than silently wrong."""
        import contextlib

        if name not in self.ROLES:
            raise ValueError(f"role must be one of {self.ROLES}; got {name!r}")

        @contextlib.contextmanager
        def scope():
            if (
                torch.is_grad_enabled()
                and self._loras_for(name)[0] != self._loras_for(self.active_role)[0]
                and bool(getattr(self.backbone.model, "is_gradient_checkpointing", False))
                and any(p.requires_grad for p in self.parameters())
            ):
                raise RuntimeError(
                    f"encoder role switch {self.active_role!r} -> {name!r} with gradients on "
                    "under backbone gradient checkpointing: the recompute would not see the "
                    "same adapters (turn gradient_checkpointing_encoder off)")
            stack = self.__dict__.setdefault("_role_stack", [])
            stack.append(name)
            self._sync_role()
            try:
                yield
            finally:
                stack.pop()
                self._sync_role()

        return scope()

    def part(self, name: str):
        """The part ``forward`` reads for the active role: the slot copy under
        an adapter role, else the base part."""
        if self.__dict__.get("_use_slot_parts"):
            sp = getattr(self, "slot_parts", None)
            if sp is not None and getattr(sp, name, None) is not None:
                return getattr(sp, name)
        return getattr(self, name)

    def spine_scope(self):
        """Context manager: the spine role (its adapter, if any) inside."""
        return self.role("spine")

    def spine_reproduce(self, survivors: torch.Tensor) -> torch.Tensor:
        """The spine's bridge: its own copy when the adapter is installed."""
        sr = getattr(self, "spine_repro", None)
        return sr(survivors) if sr is not None else self.reproduce(survivors)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.has_slot_adapter:
            # the base interface belongs to the frozen tree encoder: its
            # running statistics must not move with training traffic, or the
            # cached tree nodes stop being its output
            self.interface_norm.train(False)
        return self

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        block = self.projection_block.parameters() if self.projection_block is not None else []
        cond = [self.prompt_cond_in, self.prompt_cond_hook]
        return {
            "backbone": [p for p in self.backbone.parameters() if p.requires_grad],
            "head": [p for p in self.head.parameters() if p.requires_grad],
            "survive_embedding": [self.survive_embedding]
            + ([self.prompt_separator] if self.prompt_separator is not None else []),
            "prompt_cond": [p for m in cond if m is not None for p in m.parameters()],
            "projection": [p for p in self.projection.parameters() if p.requires_grad]
            + ([p for p in self.input_skip.parameters()] if self.input_skip is not None else []),
            "projection_block": [p for p in block if p.requires_grad],
            # The repro head trains on its OWN group: Stage A pre-warms it
            # alone, with everything else frozen.
            "repro": ([p for p in self.repro.parameters() if p.requires_grad]
                      if self.repro is not None else []),
        }

    # ------------------------------------------------------------------
    def _last_attention_layer_index(self) -> int:
        layers = self.backbone.model.layers
        for i in range(len(layers) - 1, -1, -1):
            if layers[i].is_attention_layer:
                return i
        raise ValueError("projection_block needs an attention layer in the backbone")

    def reset_projection_block_from_backbone(self) -> None:
        """(Re)copy the projection block from the backbone's last attention
        layer — at construction, and after a warm start whose checkpoint
        predates the block (so it starts from the LOADED backbone)."""
        if self._projection_block_src is None:
            return
        block = copy.deepcopy(self.backbone.model.layers[self._projection_block_src])
        for p in block.parameters():
            p.requires_grad_(True)
        # Identity at init: zero the residual branches' output projections so
        # inserting the block does not disturb an aligned projection (P19a3
        # lost ~500 steps re-learning the mapping from a live copy).
        with torch.no_grad():
            block.self_attn.out_proj.weight.zero_()
            block.feed_forward.w2.weight.zero_()
        self.projection_block = block

    def _apply_projection_block(
        self, survivors: torch.Tensor, rep_mask: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run the contextual block over the survivor set (B, K, D)."""
        block = self.part("projection_block")
        assert block is not None
        pe = self.backbone.model.rotary_emb(survivors, position_ids=positions)
        mask4d = _bidirectional_additive_mask(rep_mask.to(torch.long), survivors.dtype)
        out = block(
            survivors,
            attention_mask=mask4d,
            position_embeddings=pe,
            position_ids=positions,
            past_key_values=None,
        )
        return out * rep_mask.to(out.dtype)[:, :, None]

    # ------------------------------------------------------------------
    def reproduce(self, survivors: torch.Tensor) -> torch.Tensor:
        """Survivors -> the encoder's OWN input space, for the next level up.

        Distinct from ``projection``, which targets the DECODER's input space.
        Pre-warm it against ``backbone.embed_tokens`` before using it to feed a
        level (bgkit1's joint-block pretrain), otherwise the next backbone reads
        an input distribution it was never trained on.
        """
        if self.repro is None:
            raise RuntimeError("encoder was built without repro_head=True")
        return self.repro(survivors)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        ratio: float | torch.Tensor,
        selection: str | None = None,
        generator: torch.Generator | None = None,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        doc_embeds: torch.Tensor | None = None,
        forced_keep: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """``prompt_ids``/``prompt_mask``: an optional LEFT-padded compression
        prompt (B, P). It is prepended to the document so every document
        position attends to it (the backbone is bidirectional), it is never
        eligible for selection, and reported ``positions``/``keep`` are
        document-relative.

        ``forced_keep`` (B, L_full) bool overrides selection entirely: those
        positions survive (intersected with eligibility) and the head's scores
        are ignored for the mask. Distillation needs it so a student and a
        frozen teacher select the SAME positions and the loss measures the
        bridge, not a disagreement about which tokens to keep.

        ``doc_embeds`` (B, L, D) feeds the document as EMBEDDINGS instead of
        ``input_ids``, skipping the token lookup. That is how a level of a
        multi-level tree consumes the level below: pass
        ``reproduce(survivors)``. The backbone already accepts ``inputs_embeds``,
        so no other plumbing is involved."""
        src = input_ids if input_ids is not None else doc_embeds
        if src is not None and src.shape[1] == 0:
            # A batch of 0-length documents: LFM2's short conv (kernel 3)
            # cannot run and the caller has sent rows that have NO context
            # slot (framing "chat"). Those rows must not reach the encoder;
            # see TrainerCore.noslot_rows. Fail with the reason, not a conv
            # shape error (P28 step-12000 eval, 2026-09-17).
            raise ValueError("encoder received a batch of empty documents (no-slot rows?)")

        self._sync_role()  # the peft layers follow the active role (see ``role``)
        part = self.part
        head, survive = part("head"), part("survive_embedding")
        separator, projection = part("prompt_separator"), part("projection")
        cond_in, cond_hook = part("prompt_cond_in"), part("prompt_cond_hook")
        input_skip, inorm = part("input_skip"), part("interface_norm")
        mode = selection or self.selection
        token_ratio = ratio
        if self.summary_slot_frac > 0:
            token_ratio = ratio / (1.0 + self.summary_slot_frac)
        doc_mask = attention_mask.bool()
        if doc_embeds is None:
            if input_ids is None:
                raise ValueError("pass input_ids or doc_embeds")
            doc_embeds = self.backbone.embed_tokens(input_ids)
        elif doc_embeds.shape[:2] != attention_mask.shape:
            raise ValueError(
                f"doc_embeds {tuple(doc_embeds.shape[:2])} does not match "
                f"attention_mask {tuple(attention_mask.shape)}"
            )
        if prompt_ids is not None:
            if prompt_mask is None or prompt_ids.shape != prompt_mask.shape:
                raise ValueError("prompt_ids needs a prompt_mask of the same shape")
            offset = int(prompt_ids.shape[1])
            parts = [self.backbone.embed_tokens(prompt_ids)]
            masks = [prompt_mask.bool()]
            if separator is not None:
                bsz = prompt_ids.shape[0]
                sep = separator.to(doc_embeds.dtype)[None, None, :].expand(bsz, 1, -1)
                parts.append(sep)
                masks.append(torch.ones(bsz, 1, dtype=torch.bool, device=prompt_ids.device))
                offset += 1
            if cond_in is not None:
                pm = prompt_mask.to(doc_embeds.dtype)[:, :, None]
                pooled = (parts[0] * pm).sum(1) / pm.sum(1).clamp_min(1.0)
                cond = cond_in(pooled.float()).to(doc_embeds.dtype)
                doc_embeds = (
                    doc_embeds + cond[:, None, :] * doc_mask.to(doc_embeds.dtype)[:, :, None]
                )
            inputs_embeds = torch.cat([*parts, doc_embeds], dim=1)
            full_mask = torch.cat([*masks, doc_mask], dim=1)
            eligible = torch.cat(
                [
                    torch.zeros(
                        full_mask.shape[0], offset, dtype=torch.bool, device=doc_mask.device
                    ),
                    doc_mask,
                ],
                dim=1,
            )
        else:
            offset = 0
            inputs_embeds, full_mask, eligible = doc_embeds, doc_mask, doc_mask
        state: dict[str, torch.Tensor] = {}

        def hook(hidden: torch.Tensor) -> torch.Tensor:
            cap = self.utility_capture
            if cap is not None and torch.is_grad_enabled():
                if not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                # The closure references ``cap``; ``cap`` will hold the encoder
                # output whose graph holds THIS tensor -> a reference cycle
                # that refcounting cannot free (P19b leaked 0.5 GB/step).
                # The consumer removes the handle and clears the dict.
                cap["handle"] = hidden.register_hook(
                    lambda g, cap=cap: cap.__setitem__("grad", g.detach())
                )
                self.utility_capture = None  # one capture per forward
            if cond_hook is not None and prompt_ids is not None:
                n_p = int(prompt_ids.shape[1])
                pm = prompt_mask.to(hidden.dtype)[:, :, None]
                pooled = (hidden[:, :n_p] * pm).sum(1) / pm.sum(1).clamp_min(1.0)
                cond = cond_hook(pooled.float()).to(hidden.dtype)
                hidden = hidden + cond[:, None, :] * eligible.to(hidden.dtype)[:, :, None]
            state["head_input"] = hidden.detach()
            raw = (
                head(hidden).float()
                if mode == "learned"
                else random_scores(eligible, generator)
            )
            if mode == "learned":
                state["raw"] = raw
            z = segment_zscore(raw, eligible)
            if forced_keep is not None:
                # Distillation forces the student to select exactly what the
                # teacher selected, so the two projections are comparable
                # POSITION BY POSITION and the loss measures the bridge rather
                # than a difference in which tokens survived (bgkit1 Step 4.7,
                # src/bgkit/training/phase1/bridge_distill.py).
                keep = forced_keep & eligible
                k = keep.sum(dim=-1)
                threshold = torch.zeros_like(k, dtype=z.dtype)
            else:
                keep, k, threshold = exact_topk_select(
                    z, eligible, token_ratio, self.min_survivors
                )
            state.update(z=z, keep=keep, k=k, threshold=threshold)
            add = keep.to(hidden.dtype)[:, :, None] * survive.to(hidden.dtype)
            return hidden + add

        final = self.backbone(
            inputs_embeds, attention_mask=full_mask, hook_layer=self.head_layer, hook=hook
        )
        keep, k, z, threshold = state["keep"], state["k"], state["z"], state["threshold"]
        survivors, rep_mask, positions = gather_survivors(final, keep, k)
        gate = None
        if mode == "learned" and torch.is_grad_enabled():
            z_sel = z.gather(1, positions)
            gate = straight_through_gate(z_sel, threshold, self.gate_tau)
            survivors = survivors * gate.to(survivors.dtype)[:, :, None]
        # Projection + interface contract in fp32 regardless of autocast: they
        # are tiny next to the backbone and the row normalisation's 1/|x|
        # gradient is where bf16 overflowed.
        if input_skip is not None:
            own = inputs_embeds.gather(
                1, positions[..., None].expand(-1, -1, inputs_embeds.shape[-1])
            )
            survivors = (
                survivors
                + input_skip(own.to(survivors.dtype))
                * rep_mask.to(survivors.dtype)[:, :, None]
            )
        if part("projection_block") is not None:
            survivors = self._apply_projection_block(survivors, rep_mask, positions)
        pre_projection = survivors
        with torch.autocast(device_type=survivors.device.type, enabled=False):
            projected = projection(survivors.float())
            flat = inorm(projected[rep_mask])
        reps = torch.zeros_like(projected)
        reps[rep_mask] = flat.to(reps.dtype)
        slot_reps = slot_mask = None
        if self.summary_slot_frac > 0:
            pooled, slot_mask = self._summary_slots(final, keep, eligible, k)
            with torch.autocast(device_type=pooled.device.type, enabled=False):
                sflat = inorm(projection(pooled.float())[slot_mask])
            slot_reps = torch.zeros_like(pooled)
            slot_reps[slot_mask] = sflat.to(slot_reps.dtype)
        if offset:
            positions = (positions - offset).clamp_min(0) * rep_mask
            keep = keep[:, offset:]
            z = z[:, offset:]
        return EncoderOutput(
            reps=reps,
            rep_mask=rep_mask,
            positions=positions,
            keep=keep,
            scores=z,
            k=k,
            n_valid=doc_mask.sum(dim=-1),
            slot_reps=slot_reps,
            slot_mask=slot_mask,
            gate=gate,
            head_input=state.get("head_input"),
            eligible=eligible,
            offset=offset,
            raw=state.get("raw"),
            survivors=pre_projection,
        )

    def _summary_slots(
        self, final: torch.Tensor, keep: torch.Tensor, eligible: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, M, D) pooled hidden states + (B, M) mask. Document ``b`` gets
        m_b = ceil(frac * k_b) contiguous chunks of its eligible positions;
        slot c is the mean over the DROPPED positions of chunk c (over all its
        positions if none were dropped); a chunk with no positions is masked."""
        bsz, _seq, dim = final.shape
        ms = torch.ceil(k.float() * self.summary_slot_frac).long().clamp_min(1)
        mmax = int(ms.max().item()) if bsz > 0 else 0
        pooled = final.new_zeros(bsz, mmax, dim)
        smask = torch.zeros(bsz, mmax, dtype=torch.bool, device=final.device)
        for b in range(bsz):
            pos = eligible[b].nonzero(as_tuple=False).squeeze(-1)
            n, m = int(pos.numel()), int(ms[b].item())
            if n == 0:
                continue
            chunk = (torch.arange(n, device=final.device) * m) // n
            h = final[b, pos]
            w = (~keep[b, pos]).to(h.dtype)
            sums = h.new_zeros(m, dim).index_add_(0, chunk, h * w[:, None])
            cnt = h.new_zeros(m).index_add_(0, chunk, w)
            all_sums = h.new_zeros(m, dim).index_add_(0, chunk, h)
            all_cnt = h.new_zeros(m).index_add_(0, chunk, torch.ones_like(w))
            use_all = cnt == 0
            sums = torch.where(use_all[:, None], all_sums, sums)
            cnt = torch.where(use_all, all_cnt, cnt)
            pooled[b, :m] = sums / cnt.clamp_min(1.0)[:, None]
            smask[b, :m] = all_cnt > 0
        return pooled, smask
