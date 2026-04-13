"""Per-level LoRA adapters for the BgKIT encoder.

The Phase 2 KB-scale pipeline freezes the encoder base weights (from Phase 1)
and trains small LoRA adapters at each compression level:

- **L0 LoRA** — trained in Stage A, then frozen. Shapes within-document
  salience on top of the code-trained Phase 1 encoder.
- **L1 LoRA** — trained throughout Stages A/B/C. Shapes query-conditioned
  cross-document fusion.

A single base encoder carries both adapters. At each forward pass the trainer
activates whichever level applies for that call via ``lora_level="l0"`` or
``"l1"``. Internally the encoder resolves ``LoRARouter.get()`` and enters a
context manager that flips the active adapter on every wrapped Linear.

This is a lightweight LoRA implementation rather than a peft dependency so
that the router can switch adapters at Python call granularity without the
state-dict churn of peft's ``set_adapter``. Adapters are registered as
regular ``nn.Module`` children of the ``LoRARouter`` object so that they
appear in ``state_dict()`` and ``optimizer.add_param_group`` calls cleanly.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRALinearAdapter(nn.Module):
    """A*B low-rank delta for a single ``nn.Linear`` target.

    Forward: ``y = base(x) + scaling * dropout(x) @ A^T @ B^T``

    A: (rank, in_features), B: (out_features, rank)

    Initialized so the delta is zero at creation (A ~ Kaiming, B = 0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # B stays zero so the adapter is an identity at init.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        h = self.dropout(x)
        h = torch.nn.functional.linear(h, self.lora_A)  # (..., rank)
        h = torch.nn.functional.linear(h, self.lora_B)  # (..., out_features)
        return h * self.scaling


class LoRALinearWrapper(nn.Module):
    """Wrap an ``nn.Linear`` and add one or more named LoRA adapters.

    The base Linear is stored as a submodule and is *always* kept frozen by
    callers (Phase 1 weights). A dict of :class:`LoRALinearAdapter` children
    provides per-level deltas. The active adapter name is resolved from the
    thread-local :class:`LoRARouter` at every forward call.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base_layer = base
        self.adapters = nn.ModuleDict()

    def add_adapter(
        self,
        name: str,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> LoRALinearAdapter:
        if name in self.adapters:
            raise ValueError(f"LoRA adapter {name!r} already exists on this layer")
        adapter = LoRALinearAdapter(
            in_features=self.base_layer.in_features,
            out_features=self.base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        adapter.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
        self.adapters[name] = adapter
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_layer(x)
        active = LoRARouter.active_level_static()
        if active is not None and active in self.adapters:
            delta = self.adapters[active](x)
            y = y + delta.to(y.dtype)
        return y


class LoRARouter(nn.Module):
    """Thread-local active-level dispatcher for LoRA adapters.

    Holds references to all :class:`LoRALinearWrapper` instances installed on
    an encoder so the trainer can freeze/unfreeze a whole level at once. Also
    owns the thread-local state that tells wrappers which adapter (if any) is
    active for the current forward pass.

    Usage::

        router = LoRARouter.install(encoder, target_names=("q_proj", ...),
                                    levels={"l0": 32, "l1": 32})
        LoRARouter.bind(router)  # makes get() return this router

        with router.active("l1"):
            out = encoder(...)   # L1 adapter applied on every wrapped Linear
    """

    _current: LoRARouter | None = None
    _tls = threading.local()

    def __init__(self) -> None:
        super().__init__()
        self._wrappers: list[LoRALinearWrapper] = []
        self._levels: set[str] = set()

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    @classmethod
    def install(
        cls,
        module: nn.Module,
        target_names: Iterable[str],
        levels: dict[str, int],
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> LoRARouter:
        """Wrap every ``nn.Linear`` under ``module`` whose local name is in
        ``target_names`` with a :class:`LoRALinearWrapper` that carries one
        adapter per entry in ``levels``.

        Args:
            module: Root module to scan (e.g. the encoder).
            target_names: Linear submodule names to wrap (``"q_proj"``, ...).
            levels: Map level name → LoRA rank (e.g. ``{"l0": 32, "l1": 32}``).
            alpha: LoRA alpha. Defaults to 2 * rank per level.
            dropout: LoRA dropout.
        """
        router = cls()
        targets = set(target_names)

        # Walk the module tree and replace matching Linear children in-place.
        for parent in list(module.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name in targets and isinstance(child, nn.Linear):
                    wrapper = LoRALinearWrapper(child)
                    wrapper.to(device=child.weight.device, dtype=child.weight.dtype)
                    setattr(parent, child_name, wrapper)
                    for level_name, rank in levels.items():
                        wrapper.add_adapter(
                            level_name,
                            rank=int(rank),
                            alpha=(alpha if alpha is not None else 2.0 * rank),
                            dropout=dropout,
                        )
                    router._wrappers.append(wrapper)
        router._levels.update(levels.keys())
        return router

    # ------------------------------------------------------------------
    # Thread-local active-level stack
    # ------------------------------------------------------------------

    @classmethod
    def _stack(cls) -> list[str | None]:
        stack = getattr(cls._tls, "stack", None)
        if stack is None:
            stack = []
            cls._tls.stack = stack
        return stack

    @classmethod
    def active_level_static(cls) -> str | None:
        stack = cls._stack()
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def active(self, level: str | None):
        stack = self._stack()
        stack.append(level)
        try:
            yield
        finally:
            stack.pop()

    # ------------------------------------------------------------------
    # Binding: make one router globally visible to the encoder's forward
    # ------------------------------------------------------------------

    @classmethod
    def bind(cls, router: LoRARouter | None) -> None:
        cls._current = router

    @classmethod
    def get(cls) -> LoRARouter | None:
        return cls._current

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    @property
    def levels(self) -> set[str]:
        return set(self._levels)

    def adapter_parameters(self, level: str) -> list[nn.Parameter]:
        """All parameters belonging to ``level`` across all wrapped layers."""
        params: list[nn.Parameter] = []
        for wrapper in self._wrappers:
            if level in wrapper.adapters:
                params.extend(wrapper.adapters[level].parameters())
        return params

    def set_level_trainable(self, level: str, trainable: bool) -> None:
        """Freeze or unfreeze every adapter parameter at ``level``."""
        for p in self.adapter_parameters(level):
            p.requires_grad = trainable

    def freeze_base(self, module: nn.Module) -> None:
        """Freeze every parameter of ``module`` that is not a LoRA adapter.

        This is the canonical way to set up "Phase 1 base frozen, adapters
        trainable": freeze everything, then call ``set_level_trainable`` on
        the levels you want to train.
        """
        for name, p in module.named_parameters():
            if any(
                name.endswith(f"adapters.{lv}.lora_A") or name.endswith(f"adapters.{lv}.lora_B")
                for lv in self._levels
            ):
                continue
            # Also skip param children of any wrapper's adapter ModuleDict
            # (named_parameters should already have picked them up above).
            p.requires_grad = False
        # Now re-enable any existing adapters that the caller wants trainable;
        # start with everything trainable, trainers will disable levels selectively.
        for lv in self._levels:
            self.set_level_trainable(lv, True)


# Default target set for Qwen3.5 attention + MLP layers.
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# ---------------------------------------------------------------------------
# State-dict remapping for resume paths
# ---------------------------------------------------------------------------


def _split_module_and_leaf(key: str) -> tuple[str, str]:
    if "." not in key:
        return "", key
    head, _, tail = key.rpartition(".")
    return head, tail


def _rewrap_base_key(key: str, target_names: Iterable[str]) -> str:
    """Rewrite ``*.q_proj.weight`` → ``*.q_proj.base_layer.weight`` for every
    target name in ``target_names``. Non-target keys are returned unchanged.

    Used to load a pre-LoRA state dict into an encoder that already has the
    LoRA router installed.
    """
    targets = set(target_names)
    parts = key.split(".")
    # Walk through looking for a match like ``...{target}.{attr}``
    for i in range(len(parts) - 1):
        if parts[i] in targets and parts[i + 1] in {"weight", "bias"}:
            # Only wrap if not already under base_layer.
            if i + 1 < len(parts) and parts[i + 1] != "base_layer":
                new_parts = [*parts[:i + 1], "base_layer", *parts[i + 1:]]
                return ".".join(new_parts)
            break
    return key


def _unwrap_base_key(key: str, target_names: Iterable[str]) -> str:
    """Reverse of :func:`_rewrap_base_key`: strip ``.base_layer.`` from a
    target module's weight/bias path.

    Used when exporting a LoRA-wrapped encoder back to a pre-LoRA state dict
    (e.g. for merging adapters and saving a clean checkpoint).
    """
    targets = set(target_names)
    for t in targets:
        key = key.replace(f"{t}.base_layer.weight", f"{t}.weight")
        key = key.replace(f"{t}.base_layer.bias", f"{t}.bias")
    return key


def remap_base_keys_to_lora(
    state_dict: dict[str, torch.Tensor],
    target_names: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> dict[str, torch.Tensor]:
    """Rewrite every target-module weight key to the LoRA-wrapped form.

    Input: a pre-LoRA state dict (e.g. loaded from a Phase 1 checkpoint).
    Output: a new dict with the same tensors but keys renamed so they match
    an encoder that has had :meth:`LoRARouter.install` called on it.
    LoRA adapter parameters themselves (``adapters.{level}.lora_A/B``) are
    NOT in the input and not invented — a subsequent ``load_state_dict``
    against a LoRA-wrapped encoder will leave the adapters at their
    zero-initialized values, which is the correct bootstrap state for a
    fresh Stage A run that starts from a Phase 1 base.
    """
    targets = tuple(target_names)
    return {_rewrap_base_key(k, targets): v for k, v in state_dict.items()}


def remap_lora_keys_to_base(
    state_dict: dict[str, torch.Tensor],
    target_names: Iterable[str] = DEFAULT_LORA_TARGETS,
    drop_adapters: bool = True,
) -> dict[str, torch.Tensor]:
    """Reverse remap — ``*.q_proj.base_layer.weight`` → ``*.q_proj.weight``.

    If ``drop_adapters`` is True, any ``*.adapters.{level}.lora_A/B`` keys
    are stripped (use this when exporting a base-only checkpoint). If False,
    they're preserved under their original keys so you can save a hybrid
    "base + adapters" state dict with pre-LoRA-shaped base weights.
    """
    targets = tuple(target_names)
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if drop_adapters and ".adapters." in k and (".lora_A" in k or ".lora_B" in k):
            continue
        out[_unwrap_base_key(k, targets)] = v
    return out
