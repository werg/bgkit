"""Shared recursive-L1 tree-node encode primitive (child-ID injection).

Single source of truth for encoding ONE tree node from its children WITH
child-ID injection. EVERY tree-encode path funnels through
:func:`encode_tree_node`:

- live full-backprop interior/leaf nodes
  (``KRKBTrainer._encode_subtree`` / ``_encode_leaf_subtree``),
- cached-QA drill-down
  (``KRKBTrainer._cached_tree_node_survivor`` / ``_shared_tree_head_survivor``),
- the offline cache builder (``scripts/precompute_l1_tree.py``).

Because there is no other way to encode a tree node, it is IMPOSSIBLE for any
current or future tree path to encode a node without injecting its children's
IDs into the node's rep — the decoder can therefore read a child's ID out of a
parent's rep and choose where to drill.

Contract
--------
``children_survivors_l1in[i]`` is child *i*'s survivors ALREADY bridged into
L1-INPUT space **by the caller** (interior: ``encoder.l1_auto_reproduce``;
leaf: ``encoder.l0.auto_reproduce``). This primitive NEVER bridges — bridging
stays caller-side (it was never the bug).

For each child the primitive builds the child's ID embedding in L1-INPUT space
directly from the shared token embeddings (``encoder.l0.backbone
.get_input_embeddings()``), UN-bridged (an ID token embedding is already in
input space; passing it through ``auto_reproduce`` would mangle it). The ID
embeddings carry gradient into ``embed_tokens`` — desired, the model learns the
ID token embeddings.

Per child the layout is ``[id_emb | survivors]``; a ``pinned`` bool mask marks
the ``id_emb`` rows True (so the survivorship head always keeps them, planting
each child's ID in the node output rep) and the survivor rows False. The whole
node is ONE L1 segment; ``query_emb`` (or ``None``) is the L1 compression
prompt.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from bgkit.utils.packing import position_ids_from_cu

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bgkit.models.encoder import BgKITEncoder


def _child_id_embedding(
    encoder: BgKITEncoder,
    tokenizer,
    child_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return child ``child_id``'s ID token embeddings in L1-INPUT space.

    Mirrors the proven L0-leaf pinning in ``KRKBTrainer._prepare_l1_turn``:
    ``embed_tokens(tokenizer.encode(f" {child_id}"))`` where ``embed_tokens =
    encoder.l0.backbone.get_input_embeddings()``. UN-bridged (already
    input-space) and gradient-carrying (do NOT detach).
    """
    embed_tokens = encoder.l0.backbone.get_input_embeddings()
    ids = tokenizer.encode(f" {child_id}", add_special_tokens=False) or [0]
    id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    return embed_tokens(id_tensor).to(dtype)


def encode_tree_node(
    encoder: BgKITEncoder,
    tokenizer,
    children_ids: list[str],
    children_survivors_l1in: list[torch.Tensor],
    query_emb: torch.Tensor | None,
    ratio: float,
    *,
    selection_mode: str = "threshold",
    project: bool = True,
    adapter_context=None,
):
    """Encode ONE tree node from its children WITH child-ID injection.

    Args:
        encoder: the ``BgKITEncoder`` (provides ``l0.backbone`` embeddings and
            :meth:`run_l1_and_project`).
        tokenizer: encoder-side tokenizer (same vocab as ``embed_tokens``).
        children_ids: per-child ID strings (interior node ids / leaf file
            names). Length must equal ``children_survivors_l1in``.
        children_survivors_l1in: per-child survivors ALREADY bridged into
            L1-INPUT space by the caller (this primitive does NOT bridge).
        query_emb: ``(L_query, D)`` L1 compression prompt in input space, or
            ``None`` (query-agnostic offline cache).
        ratio: L1 compression ratio for this node/depth.
        selection_mode: passed through to ``run_l1_and_project``.
        project: when False, skip returning the projected decoder embeddings
            (offline cache stores L1-output only) — still returns ``l1_out``.
        adapter_context: optional context manager wrapping the L1 forward
            (e.g. ``KRKBTrainer._l1_adapter_context()`` to activate the L1
            LoRA). ``None`` → no-op.

    Returns:
        ``(proj_or_None, l1_out)`` — ``proj_or_None`` is
        ``proj_out.projected_embeddings`` (or ``None`` when ``project=False``);
        ``l1_out`` is the L1 ``LevelOutput`` (``.survivor_embeddings`` are the
        node's rep in L1-OUTPUT / pre-norm space, plus keep-rate counts).
    """
    if len(children_ids) != len(children_survivors_l1in):
        raise ValueError(
            "encode_tree_node: children_ids / children_survivors_l1in length "
            f"mismatch ({len(children_ids)} != {len(children_survivors_l1in)})."
        )
    if not children_survivors_l1in:
        raise ValueError("encode_tree_node: no children to encode.")

    device = children_survivors_l1in[0].device
    dtype = children_survivors_l1in[0].dtype

    pieces: list[torch.Tensor] = []
    pinned_pieces: list[torch.Tensor] = []
    for cid, surv in zip(children_ids, children_survivors_l1in, strict=True):
        # Child ID (input-space, un-bridged, gradient-carrying) — pinned.
        id_emb = _child_id_embedding(
            encoder, tokenizer, str(cid), dtype=dtype, device=device,
        )
        pieces.append(id_emb)
        pinned_pieces.append(
            torch.ones(int(id_emb.shape[0]), dtype=torch.bool, device=device)
        )
        # Child survivors (already L1-input space) — NOT pinned.
        surv = surv.to(dtype)
        pieces.append(surv)
        pinned_pieces.append(
            torch.zeros(int(surv.shape[0]), dtype=torch.bool, device=device)
        )

    content = torch.cat(pieces, dim=0)
    pinned = torch.cat(pinned_pieces, dim=0)
    n = int(content.shape[0])
    content_cu = torch.tensor([0, n], dtype=torch.int32, device=device)

    prompt_emb = prompt_cu = prompt_pos = None
    if query_emb is not None and int(query_emb.shape[0]) > 0:
        prompt_emb = query_emb.to(dtype)
        prompt_cu = torch.tensor(
            [0, int(prompt_emb.shape[0])], dtype=torch.int32, device=device,
        )
        prompt_pos = position_ids_from_cu(prompt_cu, int(prompt_emb.shape[0]))

    ctx = adapter_context if adapter_context is not None else contextlib.nullcontext()
    with ctx:
        l1_out, proj_out, _proj_cu = encoder.run_l1_and_project(
            l1_input_embeddings=content,
            l1_input_cu_seqlens=content_cu,
            target_ratio_l1=float(ratio),
            content_group_cu_seqlens=None,
            prompt_embeddings_l1=prompt_emb,
            prompt_cu_seqlens_l1=prompt_cu,
            prompt_position_ids_l1=prompt_pos,
            pinned_positions_l1=pinned,
            selection_mode_l1=selection_mode,
        )

    proj = proj_out.projected_embeddings if project else None
    return proj, l1_out
