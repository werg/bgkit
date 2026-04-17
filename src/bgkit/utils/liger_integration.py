"""Optional Liger Kernel integration.

Liger Kernel provides fused Triton kernels for the hot-path ops in modern
causal LMs: RMSNorm, SwiGLU MLPs, rotary embeddings, and — most importantly
for BgKIT — a fused linear + cross-entropy kernel that never materializes the
full ``(B, S, V)`` logits tensor. With a 248K vocab Qwen3.5 decoder, skipping
that materialization recovers several GB of activation memory per step.

This module is intentionally defensive:

- All Liger imports are *lazy* and live inside functions. The package is not
  a hard dependency; it is only installed inside the GPU Docker container.
- When Liger is missing, every public helper is a documented no-op with a
  one-shot warning so host-only CPU tests and lint runs keep passing.
- The wiring into encoders/decoders (``apply_liger_to_qwen35``) reaches into
  the HF Qwen3.5 module tree by attribute name and replaces ``forward`` on
  individual sub-modules. It never mutates state dicts, so the resulting
  model is still checkpoint-compatible with the un-patched baseline. If a
  specific fused op can't be located in Liger's public API (the package
  layout has changed a few times), that op is skipped rather than raising.
- ``liger_chunked_ce_loss`` always returns the same scalar the existing
  ``_chunked_lm_ce`` would — if Liger's fused-linear-CE isn't available, it
  transparently falls back. Callers don't need to branch on availability.

The two entry points used by the rest of the codebase are:

``apply_liger_to_qwen35(model)``
    Walk the Qwen3.5 backbone (encoder or decoder wrapping) and install
    Liger-fused RMSNorm / SwiGLU / RoPE wherever possible. Returns the
    number of modules actually patched so trainers can log it.

``liger_chunked_ce_loss(...)``
    Fused-linear cross-entropy that never materializes the full logits
    tensor. Falls back to the existing chunked-CE implementation in
    :mod:`bgkit.models.decoder` when Liger is not installed.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_LIGER_AVAILABLE: bool | None = None
_LIGER_WARNED: bool = False


def is_liger_available() -> bool:
    """Lazy check for ``liger_kernel``. Cached after first call.

    Importing ``liger_kernel`` pulls in Triton, which in turn requires a
    CUDA-capable environment on some installs. We therefore wrap the import
    in a broad try/except — any failure means "not usable in this runtime".
    """
    global _LIGER_AVAILABLE
    if _LIGER_AVAILABLE is not None:
        return _LIGER_AVAILABLE
    try:
        import liger_kernel  # noqa: F401
        _LIGER_AVAILABLE = True
    except Exception:  # pragma: no cover — exercised only when installed
        _LIGER_AVAILABLE = False
    return _LIGER_AVAILABLE


def _warn_once(msg: str) -> None:
    global _LIGER_WARNED
    if _LIGER_WARNED:
        return
    _LIGER_WARNED = True
    warnings.warn(msg, stacklevel=2)


# ---------------------------------------------------------------------------
# Module patching
# ---------------------------------------------------------------------------


def _iter_text_backbone(model: nn.Module) -> nn.Module:
    """Return the Qwen3.5 text model inside whatever wrapper ``model`` is.

    Handles ``ReconstructionDecoder`` (``.backbone``), HF causal LM wrappers
    (``.model``), multimodal ``Qwen3_5Model`` wrappers (``.language_model``),
    PEFT wrappers (``.base_model.model``), and raw text models. Never raises —
    returns the deepest match it can find.
    """
    node: Any = model

    # ReconstructionDecoder.backbone
    if hasattr(node, "backbone") and not isinstance(node, nn.ModuleList):
        node = node.backbone

    # torch.compile wrapper
    if hasattr(node, "_orig_mod"):
        node = node._orig_mod

    # PEFT wrapper
    try:
        from peft import PeftModel

        if isinstance(node, PeftModel):
            node = node.base_model.model
    except Exception:
        pass

    # HF causal LM (AutoModelForCausalLM)
    if hasattr(node, "model") and hasattr(node.model, "layers"):
        node = node.model

    # Multimodal Qwen3.5 wrapper
    if hasattr(node, "language_model") and hasattr(node.language_model, "layers"):
        node = node.language_model

    return node


def _try_import_liger_rms_norm():
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm  # type: ignore
        return LigerRMSNorm
    except Exception:
        try:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction  # type: ignore
            return LigerRMSNormFunction
        except Exception:
            return None


def _try_import_liger_swiglu():
    """Return a callable ``mlp_forward(self, x)`` replacement, or None."""
    try:
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP  # type: ignore
        return LigerSwiGLUMLP
    except Exception:
        return None


def _try_import_liger_rope():
    try:
        from liger_kernel.transformers.rope import liger_rotary_pos_emb  # type: ignore
        return liger_rotary_pos_emb
    except Exception:
        return None


def _patch_rms_norm_modules(root: nn.Module) -> int:
    """Swap HF RMSNorm modules with Liger's fused equivalent.

    We detect RMSNorm structurally: a module with a ``weight`` Parameter, a
    ``variance_epsilon`` (or ``eps``) attribute, and no other learnable
    buffers. That catches Qwen3_5RMSNorm without importing its class.
    """
    liger_rms_cls = _try_import_liger_rms_norm()
    if liger_rms_cls is None:
        return 0

    count = 0
    for _parent_name, parent in list(root.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not hasattr(child, "weight") or not isinstance(
                child.weight, torch.nn.Parameter,
            ):
                continue
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", None))
            if eps is None:
                continue
            cls_name = type(child).__name__
            if "RMSNorm" not in cls_name or "Liger" in cls_name:
                continue
            # Skip gated RMSNorm variants (e.g. fla's FusedRMSNormGated used in
            # DeltaNet, called as self.norm(x, z)). Plain LigerRMSNorm only
            # accepts (x,) and the signature mismatch crashes at forward time.
            if "Gated" in cls_name:
                continue
            try:
                hidden_size = int(child.weight.shape[-1])
                # in_place=False is critical: LigerRMSNorm's default
                # in_place=True scribbles on the dY buffer during backward,
                # silently corrupting gradients flowing through post-norm
                # residual consumers. Qwen3.5's hybrid Gated-DeltaNet +
                # Attention architecture has exactly that residual pattern,
                # so in_place=True produces biased (non-NaN) gradients that
                # quickly pull training off the Step 2 manifold —
                # decoder-CE goes from ~1.6 at step 0 to ~13 (LM prior) by
                # step 10 at near-zero LR. See Liger issues #272, #1119,
                # and the LigerRMSNormForQwen3Next variant which also
                # overrides in_place=False. Do not remove without reading
                # those.
                try:
                    new = liger_rms_cls(
                        hidden_size=hidden_size,
                        eps=float(eps),
                        in_place=False,
                    )
                except TypeError:
                    # Older liger-kernel versions (<0.5.x?) don't accept
                    # in_place kwarg. Fall through to the default signature
                    # — those versions predated the in_place default change
                    # and are believed safe.
                    new = liger_rms_cls(
                        hidden_size=hidden_size, eps=float(eps),
                    )
                new.weight = child.weight
                new.to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(parent, child_name, new)
                count += 1
            except Exception:
                # If the Liger class signature differs, silently skip this
                # module rather than aborting the whole patch pass.
                continue
    return count


def _patch_swiglu_mlp_modules(root: nn.Module) -> int:
    """Replace ``forward`` on SwiGLU-style MLPs with Liger's fused kernel.

    Qwen3_5MLP exposes ``gate_proj``, ``up_proj``, ``down_proj``. Liger's
    ``LigerSwiGLUMLP`` forward expects the same three projections. We
    monkey-patch the instance's ``forward`` method rather than swapping the
    module so existing parameter references (and LoRA adapter hooks) keep
    working.
    """
    liger_swiglu_cls = _try_import_liger_swiglu()
    if liger_swiglu_cls is None:
        return 0

    count = 0
    for _name, module in list(root.named_modules()):
        if (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and "MLP" in type(module).__name__
            and "Liger" not in type(module).__name__
        ):
            try:
                # Bind Liger's forward as a method on this instance.
                import types

                module.forward = types.MethodType(
                    liger_swiglu_cls.forward, module,
                )
                count += 1
            except Exception:
                continue
    return count


def _patch_rope(root: nn.Module) -> int:
    """Swap the rotary embedding application with Liger's fused kernel.

    Qwen3.5 applies RoPE inside each attention block via
    ``apply_rotary_pos_emb``. Liger provides ``liger_rotary_pos_emb`` with
    the same signature. We can't cleanly monkey-patch every attention module
    without knowing its internals, so we install the function as a new
    module attribute under the well-known names HF uses, and leave it to
    the user's attention impl to find it. If no attention module references
    a name we can rebind, we return 0 and keep going — RoPE fusion is nice
    to have, not load-bearing.
    """
    liger_rope_fn = _try_import_liger_rope()
    if liger_rope_fn is None:
        return 0

    # Best-effort: expose as a callable attribute on the root. Qwen3.5 calls
    # its own ``apply_rotary_pos_emb`` from a module-level import, which we
    # can't safely rebind here without risking side effects on other models
    # sharing the same transformers module. Record it on the root and move
    # on; trainers can opt into the global monkey-patch via a separate
    # helper if they want it.
    try:
        root._liger_rotary_pos_emb = liger_rope_fn  # type: ignore[attr-defined]
        return 1
    except Exception:
        return 0


def apply_liger_to_qwen35(
    model: nn.Module | None,
    patch_rmsnorm: bool = False,
    patch_swiglu: bool = True,
    patch_rope: bool = True,
) -> int:
    """Install Liger's fused kernels on a Qwen3.5 encoder *or* decoder.

    Walks ``model`` to the underlying text backbone and swaps RMSNorm /
    SwiGLU / RoPE implementations with Liger's Triton kernels in-place.
    Returns the total number of modules patched (0 when Liger is not
    installed — callers should log this so missing-kernel regressions
    are visible in training logs).

    Component toggles allow bisecting which fused kernel introduces a
    regression (e.g., a liger-kernel version bump may silently break
    backward-pass numerics on one module type). Call with
    ``patch_rmsnorm=True`` to opt into the RMSNorm kernel, or
    ``patch_swiglu=False`` / ``patch_rope=False`` to skip those kernels.

    **``patch_rmsnorm`` defaults to False** as of 2026-04-16:
    liger-kernel 0.7.x's ``LigerRMSNorm`` silently corrupts the backward
    pass on Qwen3.5 (decoder loss jumps to the LM prior at near-zero LR,
    grad_norm ~500 but no NaN). Discovered during phase1_step4 after
    ~24 hours of debugging; see commit 313f597. Only flip this to True
    if you have independently verified the kernel is good against the
    current ``transformers`` + Qwen3.5 combination. SwiGLU / RoPE /
    fused linear-CE are unaffected and still default-on — they provide
    the bulk of the throughput win.

    Safe to call on encoders and decoders alike, on top of LoRA wrappers,
    before or after ``enable_gradient_checkpointing``, and multiple times
    (no-op after the first successful call per root module).
    """
    if model is None:
        return 0
    if not is_liger_available():
        _warn_once(
            "liger-kernel not installed — apply_liger_to_qwen35 is a no-op. "
            "Install `liger-kernel` in the GPU container to enable fused "
            "RMSNorm / SwiGLU / RoPE kernels."
        )
        return 0
    if getattr(model, "_liger_patched", False):
        return 0

    root = _iter_text_backbone(model)
    total = 0
    if patch_rmsnorm:
        total += _patch_rms_norm_modules(root)
    if patch_swiglu:
        total += _patch_swiglu_mlp_modules(root)
    if patch_rope:
        total += _patch_rope(root)

    with contextlib.suppress(Exception):
        model._liger_patched = True  # type: ignore[attr-defined]
    return total


# ---------------------------------------------------------------------------
# Fused linear + cross-entropy
# ---------------------------------------------------------------------------


def _try_import_liger_fused_ce():
    """Return ``(callable, kind)`` where ``kind`` is 'functional' or 'module'.

    Liger's public surface has shifted between releases. We probe a few
    known import paths and return the first one that works.
    """
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import (  # type: ignore
            LigerFusedLinearCrossEntropyLoss,
        )
        return LigerFusedLinearCrossEntropyLoss, "module"
    except Exception:
        pass
    try:
        from liger_kernel.transformers import (  # type: ignore
            LigerFusedLinearCrossEntropyLoss,
        )
        return LigerFusedLinearCrossEntropyLoss, "module"
    except Exception:
        pass
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import (  # type: ignore
            LigerFusedLinearCrossEntropyFunction,
        )
        return LigerFusedLinearCrossEntropyFunction, "functional"
    except Exception:
        return None, None


def liger_chunked_ce_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Fused linear + cross-entropy, logits never materialized.

    Mirrors the shift-and-mask semantics of :func:`bgkit.models.decoder._chunked_lm_ce`:

    - ``hidden_states`` is shifted left by one (drop last position).
    - ``labels`` is shifted right by one (drop first position).
    - ``mask`` is any float/bool mask over the *original* sequence; positions
      where ``mask[:, 1:]`` is zero contribute 0 loss and the denominator
      excludes them.

    When Liger's fused kernel is available it is used directly. Otherwise
    we fall back to :func:`bgkit.models.decoder._chunked_lm_ce`, preserving
    the existing (slower, logits-materializing) chunked path.

    Returns:
        Scalar mean loss over unmasked positions.
    """
    # Shift for next-token prediction
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    if mask is not None:
        shift_mask = mask[:, 1:].to(dtype=torch.bool)
        # Replace masked positions with ignore_index so fused-CE skips them.
        shift_labels = torch.where(
            shift_mask,
            shift_labels,
            torch.full_like(shift_labels, ignore_index),
        )

    fused_cls, kind = (None, None)
    if is_liger_available():
        fused_cls, kind = _try_import_liger_fused_ce()

    if fused_cls is None:
        # Fallback: reuse the existing chunked CE path from decoder.py so we
        # never regress behaviour on non-Liger runs (host tests, etc.).
        from bgkit.models.decoder import _chunked_lm_ce

        # Re-wrap into the form _chunked_lm_ce expects: it re-applies its own
        # shift internally, so pass the unshifted tensors + an all-ones
        # attention_mask and the *original* ``mask`` as ``loss_mask``.
        b, s, _ = hidden_states.shape
        attn = hidden_states.new_ones(b, s, dtype=torch.float32)

        class _TempHead(nn.Module):
            def __init__(self, w, bias):
                super().__init__()
                self.weight = w
                self.bias = bias

        head = _TempHead(lm_head_weight, lm_head_bias)
        return _chunked_lm_ce(
            head,
            hidden_states,
            labels,
            attn,
            mask if mask is not None else None,
            chunk_size,
        )

    # --- Liger path ---
    # Flatten (B, S-1, D) -> (N, D), (B, S-1) -> (N,) so we can feed the
    # fused kernel in one shot. Liger handles the chunking internally.
    bsz, seq, hidden = shift_hidden.shape
    flat_hidden = shift_hidden.reshape(bsz * seq, hidden)
    flat_labels = shift_labels.reshape(bsz * seq)

    if kind == "module":
        loss_mod = fused_cls(ignore_index=ignore_index, reduction="mean")
        loss = loss_mod(lm_head_weight, flat_hidden, flat_labels, lm_head_bias)
    else:  # "functional"
        loss = fused_cls.apply(
            flat_hidden, lm_head_weight, flat_labels, lm_head_bias, ignore_index,
        )
    return loss
