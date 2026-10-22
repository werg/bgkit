"""Two-level (repo-tree) encoding: files -> bridge -> directory/repo.

See ``plans/multilevel_encoder_2026_09_16.md``. A level's survivors are mapped
back into the encoder's OWN input space by ``encoder.reproduce`` and fed to the
next level through ``forward(doc_embeds=...)``; the decoder-facing projection is
only correct at the TOP of the tree.

Why this needs its own module rather than living in the trainer: Stage B
(distillation against a frozen single-level teacher) and Stage C (retrieval /
repo understanding) both encode the same way, and the 2026-09-16 audit found a
train/eval divergence that arose precisely because two call sites grew their own
copy of one operation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .encoder import BgKIT2Encoder, EncoderOutput


@dataclass
class TwoLevelOutput:
    top: EncoderOutput  # the level whose `reps` go to the decoder
    lower: EncoderOutput  # the per-item level below it
    lower_positions: torch.Tensor  # (B_low, K0) token positions of lower survivors


def teacher_forced_lower_keep(
    teacher_keep: torch.Tensor,
    valid: torch.Tensor,
    frac_extras: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Lower-level mask that keeps every teacher survivor PLUS some extras.

    The upper level then drops the extras, so the student's FINAL survivor set
    equals the teacher's and the distillation loss measures the bridge rather
    than a disagreement about which tokens to keep (bgkit1 Step 4.7). With
    ``frac_extras`` = 0 the lower level keeps exactly the teacher's set, which
    makes the upper level's job trivial; the curriculum raises it so the upper
    level has to do real compression.
    """
    if frac_extras <= 0:
        return teacher_keep.clone()
    doomed = valid & ~teacher_keep
    u = torch.rand(doomed.shape, device=doomed.device, generator=generator)
    return teacher_keep | (doomed & (u < float(frac_extras)))


def encode_two_level(
    encoder: BgKIT2Encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    ratio_lower: float,
    ratio_upper: float,
    group_sizes: list[int] | None = None,
    forced_lower_keep: torch.Tensor | None = None,
    forced_upper_from: torch.Tensor | None = None,
) -> TwoLevelOutput:
    """Encode ``input_ids`` (B, L) as items, then group them into one document.

    ``group_sizes`` says how many consecutive items form each upper-level
    document (default: all of them, i.e. one repo). ``forced_upper_from`` is a
    token-level mask (B, L); the upper level is forced to keep exactly those
    lower survivors whose token position is set in it.
    """
    lower = encoder(
        input_ids,
        attention_mask,
        ratio=ratio_lower,
        forced_keep=forced_lower_keep,
    )
    bridged = encoder.reproduce(lower.survivors)
    sizes = group_sizes or [int(input_ids.shape[0])]
    if sum(sizes) != int(input_ids.shape[0]):
        raise ValueError(f"group_sizes sum to {sum(sizes)}, expected {input_ids.shape[0]}")

    k = int(bridged.shape[1])
    chunks, masks, forced = [], [], []
    start = 0
    for n in sizes:
        sl = slice(start, start + n)
        chunks.append(bridged[sl].reshape(1, n * k, bridged.shape[-1]))
        masks.append(lower.rep_mask[sl].reshape(1, n * k))
        if forced_upper_from is not None:
            keep_tok = forced_upper_from[sl].gather(1, lower.positions[sl])
            forced.append((keep_tok & lower.rep_mask[sl]).reshape(1, n * k))
        start += n
    width = max(c.shape[1] for c in chunks)

    def _pad(t: torch.Tensor, fill) -> torch.Tensor:
        if t.shape[1] == width:
            return t
        pad = torch.full(
            (1, width - t.shape[1], *t.shape[2:]), fill, dtype=t.dtype, device=t.device
        )
        return torch.cat([t, pad], dim=1)

    upper = encoder(
        None,
        torch.cat([_pad(m, False) for m in masks], dim=0),
        ratio=ratio_upper,
        doc_embeds=torch.cat([_pad(c, 0.0) for c in chunks], dim=0),
        forced_keep=(torch.cat([_pad(f, False) for f in forced], dim=0) if forced else None),
    )
    return TwoLevelOutput(top=upper, lower=lower, lower_positions=lower.positions)


def distill_loss(
    student_reps: torch.Tensor,
    teacher_reps: torch.Tensor,
    rep_mask: torch.Tensor,
    *,
    mse_weight: float = 1.0,
    cos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Student-vs-teacher agreement on the projection output.

    Ported from bgkit1's Step 4.7 ``_distill_loss``: MSE plus ``1 - cosine``,
    both on the DECODER-facing projection, over the valid survivor rows only.
    It is only meaningful when the two selected the SAME positions — see
    ``teacher_forced_lower_keep`` and ``forced_keep`` — otherwise it also
    measures a disagreement about which tokens survived.
    """
    if student_reps.shape != teacher_reps.shape:
        raise ValueError(
            f"student {tuple(student_reps.shape)} vs teacher {tuple(teacher_reps.shape)}"
        )
    s = student_reps[rep_mask].float()
    t = teacher_reps[rep_mask].float()
    if s.numel() == 0:
        return student_reps.sum() * 0.0, {"mse": 0.0, "cos": 0.0}
    mse = torch.nn.functional.mse_loss(s, t)
    cos = torch.nn.functional.cosine_similarity(s, t, dim=-1).mean()
    total = mse_weight * mse + cos_weight * (1.0 - cos)
    return total, {"mse": float(mse.item()), "cos": float(cos.item())}


# ----------------------------------------------------------------------------
# Recursive tree encoding over a directory structure
# ----------------------------------------------------------------------------


@dataclass
class TreeNode:
    path: str  # "" for the root
    files: list[int]  # indices into the file list
    children: dict[str, TreeNode]


def build_tree(paths: list[str], max_depth: int) -> TreeNode:
    """Directory tree of ``paths``, with directories deeper than ``max_depth``
    folded into their depth-cap ancestor.

    ``max_depth`` counts levels of ENCODING below the root: 1 = every file is a
    direct child of the root (flat), 2 = root -> top-level dir -> file, 3 adds
    one more directory level. The user's constraint (2026-09-16): a few levels
    suffice for basic capability, and each level is another encoder pass
    holding activations.
    """
    root = TreeNode("", [], {})
    for i, p in enumerate(paths):
        parts = p.split("/")
        dirs = parts[:-1][: max(0, max_depth - 1)]
        node = root
        for d in dirs:
            full = f"{node.path}/{d}" if node.path else d
            node = node.children.setdefault(d, TreeNode(full, [], {}))
        node.files.append(i)
    return root


def walk_steps(
    paths: list[str], answer: str, max_depth: int, tree: TreeNode | None = None
) -> list[tuple[str, str]]:
    """The PATH WALK to ``answer`` through the ENCODED tree: a list of
    (node path, target) steps. At a node whose child directory leads to the
    answer the target is that child's name plus "/" (keep walking); at the node
    that holds the file -- directly, or folded in from deeper directories by
    the depth cap -- the target is the answer's path relative to the node.
    Follows ``build_tree`` exactly, so the steps name nodes that encode_tree
    really produces."""
    node = tree if tree is not None else build_tree(paths, max_depth)  # reuse per repo
    steps: list[tuple[str, str]] = []
    while True:
        rel = answer[len(node.path) + 1 :] if node.path else answer
        head = rel.split("/", 1)[0]
        child = node.children.get(head) if "/" in rel else None
        if child is None:
            steps.append((node.path, rel))
            return steps
        steps.append((node.path, head + "/"))
        node = child


def node_children(paths: list[str], max_depth: int) -> dict[str, set[str]]:
    """{node path: the valid targets at that node} (child dirs with "/", files
    relative to the node) -- what a greedy walk may legally emit."""
    out: dict[str, set[str]] = {}

    def rec(n: TreeNode) -> None:
        pre = len(n.path) + 1 if n.path else 0
        out[n.path] = {name + "/" for name in n.children} | {paths[i][pre:] for i in n.files}
        for c in n.children.values():
            rec(c)

    rec(build_tree(paths, max_depth))
    return out


@dataclass
class TreeOutput:
    root: EncoderOutput  # `reps` here go to the decoder
    leaf_reps: torch.Tensor  # (F, K, D) bridged file survivors (encoder-input space)
    leaf_mask: torch.Tensor  # (F, K)
    n_nodes: int  # directory nodes encoded (excluding leaves)
    depth: int  # realised depth
    n_virtual: int = 0  # virtual sub-directory nodes added to keep node inputs bounded
    # Decoder-space reps of EVERY real directory node, keyed by its path ("" =
    # root): {path: (reps (K, D), mask (K,))}. The root alone keeps the top of
    # the hierarchy and loses the files (2026-09-19: generated path exact 2.3%
    # on held-out repos); a PATH WALK reads these level by level.
    nodes: dict | None = None
    # INPUT of every real directory node's encoder pass ([header | bridged
    # survivors of its children], valid rows only), when ``keep_inputs``: what a
    # QUESTION-CONDITIONED read-out re-encodes at browse time with the question
    # as the encoder prompt, the tree below the node staying cached.
    node_inputs: dict | None = None


def _header_ids(tok_encode, path: str) -> list[int]:
    return tok_encode(f"# {path}\n" if path else "# /\n")


def encode_tree(
    encoder: BgKIT2Encoder,
    file_ids: list[torch.Tensor],
    paths: list[str],
    *,
    tok_encode,
    ratio_leaf: float,
    ratio_node: float,
    max_depth: int = 3,
    device=None,
    bridge=None,
    leaf_chunk: int = 32,
    grad_levels: int | None = None,
    leaf_chunk_tokens: int = 32768,
    node_max_tokens: int = 8192,
    leaf_window: int = 0,
    leaf_max_reps: int = 0,
    keep_inputs: bool = False,
    leaves: tuple[torch.Tensor, torch.Tensor] | None = None,
    prompt_ids: torch.Tensor | None = None,
) -> TreeOutput:
    """Encode a repository as a tree that mirrors its directories.

    ``leaf_chunk`` / ``leaf_chunk_tokens``: files are encoded in chunks of at
    most ``leaf_chunk`` files AND ``leaf_chunk_tokens`` padded tokens (files x
    longest file), packed in length order so short files do not pay for a long
    one -- a whole repository (hundreds of files, thousands of tokens each)
    cannot go through the encoder as one batch, and a fixed FILE count let 16
    x 4096-token calls through (the 2026-09-17 host OOM). ``grad_levels``: how
    many levels from the ROOT keep a gradient (None = all). With a frozen encoder pass 0; for a very
    large repo pass 1-2 so the leaves (the bulk of the compute) run under
    ``no_grad`` and only the directory/root passes build a graph -- the
    user's "partial unfreezing for very large repos" (2026-09-17).

    ``node_max_tokens``: a directory's input is the concatenation of its
    children's survivors, UNBOUNDED in the data (a flat directory of 337 files
    is one 54k-token sequence = a 39 GiB attention allocation, the 2026-09-17
    host OOM). Above the budget the children are grouped, in order, into
    VIRTUAL sub-directories (same path header, same ratio), recursively.

    ``leaves``: precomputed (leaf_reps (F, K, D), leaf_mask (F, K)) for exactly
    these files -- a VARIANT of a repo (a random subset of its files) re-runs
    only the cheap directory levels on the leaves of the full pass, so a repo
    can be shown in several forms without re-encoding a single file.

    Every node -- file or directory -- is encoded with its PATH prepended as a
    header, so the path survives into the encoding and the decoder can answer
    navigation questions with a path. Files are encoded from tokens (level 0);
    a directory is encoded from ``[header | bridged survivors of its children]``
    where children are its files and its subdirectories, bottom-up; the root's
    ``reps`` are what the decoder reads.

    Siblings at one depth are batched. Nothing here is specific to two levels:
    the recursion runs to ``max_depth`` and the same weights serve every level.

    ``prompt_ids``: an optional compression prompt (1-D token ids) handed to
    EVERY encoder call of the tree -- leaves, windows, directory and virtual
    nodes -- so a tree built for one question (the BrowseComp-Plus candidate
    tree, ``bgkit2.agent.candtree``) is encoded for that question at every
    level. None = the unprompted tree, bit-identical to before.
    """
    if len(file_ids) != len(paths):
        raise ValueError("file_ids and paths must align")
    if getattr(encoder, "has_slot_adapter", False) and not encoder.role_explicit:
        # An adapter encoder has two tree encoders in it: the frozen BASE one
        # (whose output the node caches hold) and the trained SLOT one (small
        # live trees). Silently picking the default would let a cache builder
        # or an eval server drift from the other side -- the caller must say.
        raise RuntimeError(
            "encode_tree on an encoder with a slot adapter needs an explicit role: "
            "`with encoder.role('base')` (cached / large trees) or role('slots') "
            "(small live trees); see BgKIT2Core.tree_role")
    device = device or next(encoder.parameters()).device
    # `bridge` overrides encoder.reproduce so a probe can run a NO-BRIDGE
    # control (identity: survivors fed straight to the next level) or a
    # properly scaled random projection, alongside the trained head.
    br = bridge if bridge is not None else encoder.reproduce
    tree = build_tree(paths, max_depth)
    p_row = None
    if prompt_ids is not None and int(prompt_ids.numel()) > 0:
        p_row = prompt_ids.long().reshape(1, -1).to(device)

    def prompt_kw(rows: int) -> dict:
        if p_row is None:
            return {}
        return {"prompt_ids": p_row.expand(rows, -1),
                "prompt_mask": torch.ones(rows, p_row.shape[1], dtype=torch.long, device=device)}

    # ---- leaves: every file, batched, with its path as a header ----
    # WHOLE FILES (2026-09-20): with ``leaf_window`` a file longer than the
    # window is encoded in consecutive windows, each under the file's `# path`
    # header, and the windows' survivors concatenate into ONE leaf -- no tail is
    # dropped. ``leaf_max_reps`` bounds a leaf: a long file is compressed HARDER
    # (ratio = min(ratio_leaf, cap / tokens)) instead of cut. Both 0 = the
    # original single-window fixed-ratio leaf, bit-identical.
    n_files = len(file_ids)
    win = int(leaf_window)
    cap = int(leaf_max_reps)
    seqs: list[torch.Tensor] = []
    owner: list[int] = []  # window -> file
    win_ratio: list[float] = []
    for f, (ids, p) in enumerate(zip(file_ids, paths, strict=True)):
        hdr = torch.tensor(_header_ids(tok_encode, p), dtype=torch.long, device=device)
        body = ids.long().to(device)
        n_tok = int(body.shape[0])
        chunks = [body] if win <= 0 or n_tok <= win else list(torch.split(body, win))
        r = float(ratio_leaf)
        if cap > 0 and n_tok > 0:
            # the ratio applies to what the encoder SEES: every window repeats the header
            r = min(r, cap / (n_tok + len(chunks) * int(hdr.shape[0])))
        for c in chunks:
            seqs.append(torch.cat([hdr, c]))
            owner.append(f)
            win_ratio.append(r)
    windowed = win > 0 or cap > 0
    def depth_of(n: TreeNode) -> int:
        return 1 + max((depth_of(c) for c in n.children.values()), default=0)

    realised = depth_of(tree)  # directory levels; leaves sit one below the deepest
    # Level d (0 = root) keeps a gradient iff d < grad_levels; leaves are
    # level ``realised``.
    def grad_ctx(level: int):
        if grad_levels is None or level < grad_levels:
            return contextlib.nullcontext()
        return torch.no_grad()

    # ---- leaves: every file, in chunks, with its path as a header ----
    max_files = max(1, int(leaf_chunk))
    budget = max(1, int(leaf_chunk_tokens))
    order = sorted(range(len(seqs)), key=lambda i: int(seqs[i].shape[0]))
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in order:  # ascending length: the newest file is the group's longest
        w = int(seqs[i].shape[0])
        if cur and (len(cur) + 1 > max_files or (len(cur) + 1) * w > budget):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    parts: list[tuple[list[int], torch.Tensor, torch.Tensor]] = []
    if leaves is not None:
        if leaves[0].shape[0] != n_files:
            raise ValueError("leaves must align with file_ids/paths")
        groups = []
        parts = [(list(range(n_files)), leaves[0].to(device), leaves[1].to(device))]
    with grad_ctx(realised):
        for g in groups:
            width = max(int(seqs[i].shape[0]) for i in g)
            ids = torch.zeros(len(g), width, dtype=torch.long, device=device)
            mask = torch.zeros(len(g), width, dtype=torch.bool, device=device)
            for r, i in enumerate(g):
                ids[r, : seqs[i].shape[0]] = seqs[i]
                mask[r, : seqs[i].shape[0]] = True
            rr = (
                torch.tensor([win_ratio[i] for i in g], dtype=torch.float32, device=device)
                if windowed
                else ratio_leaf
            )
            out = encoder(ids, mask, ratio=rr, **prompt_kw(len(g)))
            parts.append((g, br(out.survivors), out.rep_mask))  # encoder-input space
    if not windowed or leaves is not None:
        kmax = max(int(m.shape[1]) for _, _, m in parts)
        leaf_bridged = torch.zeros(
            n_files, kmax, parts[0][1].shape[-1], dtype=parts[0][1].dtype, device=device
        )
        leaf_mask = torch.zeros(n_files, kmax, dtype=torch.bool, device=device)
        for g, sp, mp in parts:
            idx = torch.tensor(g, device=device)
            leaf_bridged[idx, : sp.shape[1]] = sp
            leaf_mask[idx, : mp.shape[1]] = mp
    else:
        # windows -> files: the valid survivors of a file's windows, in order
        per_win: dict[int, torch.Tensor] = {}
        for g, sp, mp in parts:
            for r, i in enumerate(g):
                per_win[i] = sp[r][mp[r]]
        per_file: list[list[torch.Tensor]] = [[] for _ in range(n_files)]
        for i in range(len(seqs)):
            per_file[owner[i]].append(per_win[i])
        cat = [torch.cat(x) for x in per_file]
        kmax = max(1, max(int(c.shape[0]) for c in cat))
        leaf_bridged = torch.zeros(
            n_files, kmax, cat[0].shape[-1], dtype=cat[0].dtype, device=device
        )
        leaf_mask = torch.zeros(n_files, kmax, dtype=torch.bool, device=device)
        for f, c in enumerate(cat):
            leaf_bridged[f, : c.shape[0]] = c
            leaf_mask[f, : c.shape[0]] = True

    # ---- directory nodes, bottom-up, siblings batched per depth ----
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}  # path -> (bridged, mask)
    n_nodes = 0

    def nodes_at(n: TreeNode, d: int, target: int, out: list[TreeNode]) -> None:
        if d == target:
            out.append(n)
            return
        for c in n.children.values():
            nodes_at(c, d + 1, target, out)

    node_budget = max(64, int(node_max_tokens))
    n_virtual = 0

    def with_header(path: str, body: torch.Tensor, bmask: torch.Tensor):
        hdr = torch.tensor(_header_ids(tok_encode, path), dtype=torch.long, device=device)
        hdr_emb = encoder.backbone.embed_tokens(hdr).to(body.dtype)
        return (
            torch.cat([hdr_emb, body], dim=0),
            torch.cat([torch.ones(hdr.shape[0], dtype=torch.bool, device=device), bmask]),
        )

    def encode_payloads(payloads, level_idx: int) -> EncoderOutput:
        """One batched encoder call over node inputs, in row chunks that keep
        rows x width under the leaf token budget as well."""
        outs, row_groups = [], []
        order_p = sorted(range(len(payloads)), key=lambda i: int(payloads[i][0].shape[0]))
        cur_p: list[int] = []
        for i in order_p:
            w_i = int(payloads[i][0].shape[0])
            # rows x width under the token budget AND rows x width^2 under what
            # that budget costs at a 4096-wide row: the encoder is bidirectional,
            # so a chunk of 8192-wide directory inputs costs 4x a leaf chunk of
            # the same token count (2026-09-20: a 2,000-file repo's level asked
            # for 13 GiB of attention scores in one call).
            rows = len(cur_p) + 1
            if cur_p and (
                rows * w_i > max(budget, node_budget)
                or rows * w_i * w_i > max(budget, node_budget) * 4096
            ):
                row_groups.append(cur_p)
                cur_p = []
            cur_p.append(i)
        if cur_p:
            row_groups.append(cur_p)
        surv = [None] * len(payloads)
        rmask = [None] * len(payloads)
        reps = [None] * len(payloads)
        for g in row_groups:
            w = max(int(payloads[i][0].shape[0]) for i in g)
            emb = torch.zeros(len(g), w, payloads[g[0]][0].shape[-1],
                              dtype=payloads[g[0]][0].dtype, device=device)
            msk = torch.zeros(len(g), w, dtype=torch.bool, device=device)
            for r, i in enumerate(g):
                p_, m_ = payloads[i]
                emb[r, : p_.shape[0]] = p_
                msk[r, : m_.shape[0]] = m_
            with grad_ctx(level_idx):
                o = encoder(None, msk, ratio=ratio_node, doc_embeds=emb,
                            **prompt_kw(len(g)))
            for r, i in enumerate(g):
                surv[i], rmask[i], reps[i] = o.survivors[r], o.rep_mask[r], o.reps[r]
            outs.append(o)
        return surv, rmask, reps, outs

    def bounded_parts(n: TreeNode, level_idx: int):
        """The node's children as a list of (bridged, mask) parts, grouped into
        virtual sub-directories until their total length fits the budget."""
        nonlocal n_virtual
        parts = [(leaf_bridged[fi], leaf_mask[fi]) for fi in n.files]
        parts += [encoded[c.path] for c in n.children.values()]
        while sum(int(m.sum()) for _, m in parts) > node_budget and len(parts) > 1:
            groups_v: list[list[int]] = []
            cur_v: list[int] = []
            tot = 0
            for i, (_, m) in enumerate(parts):
                k_i = int(m.sum())
                if cur_v and tot + k_i > node_budget:
                    groups_v.append(cur_v)
                    cur_v, tot = [], 0
                cur_v.append(i)
                tot += k_i
            if cur_v:
                groups_v.append(cur_v)
            if len(groups_v) >= len(parts):  # cannot shrink further (single huge parts)
                break
            payloads_v = [
                with_header(n.path, torch.cat([parts[i][0][parts[i][1]] for i in g], dim=0),
                            torch.ones(sum(int(parts[i][1].sum()) for i in g),
                                       dtype=torch.bool, device=device))
                for g in groups_v
            ]
            surv_v, rmask_v, _reps_v, _ = encode_payloads(payloads_v, level_idx + 1)
            parts = [(br(sv[None])[0], rm) for sv, rm in zip(surv_v, rmask_v, strict=True)]
            n_virtual += len(groups_v)
        return parts

    root_out: EncoderOutput | None = None
    node_reps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    node_inputs: dict[str, torch.Tensor] = {}
    for d in range(realised - 1, -1, -1):  # deepest directories first, root last
        level: list[TreeNode] = []
        nodes_at(tree, 0, d, level)
        level = [n for n in level if n.files or n.children]
        if not level:
            continue
        payloads = []
        for n in level:
            parts = bounded_parts(n, d)
            body = torch.cat([b[m] for b, m in parts], dim=0)
            bmask = torch.ones(body.shape[0], dtype=torch.bool, device=device)
            payloads.append(with_header(n.path, body, bmask))
        # a LEVEL can hold hundreds of directories (large repos): encode it in
        # budgeted chunks like the leaves, never as one batch padded to the widest
        surv_l, rmask_l, reps_l, outs_l = encode_payloads(payloads, d)
        n_nodes += len(level)
        for i, n in enumerate(level):
            node_reps[n.path] = (reps_l[i], rmask_l[i])
            if keep_inputs:
                node_inputs[n.path] = payloads[i][0][payloads[i][1]].detach()
        if d == 0:
            root_out = outs_l[0]  # the root level is one node = one call
        else:
            for i, n in enumerate(level):
                encoded[n.path] = (br(surv_l[i][None])[0], rmask_l[i])
    assert root_out is not None
    return TreeOutput(root=root_out, leaf_reps=leaf_bridged, leaf_mask=leaf_mask,
                      n_nodes=n_nodes, depth=realised, n_virtual=n_virtual, nodes=node_reps,
                      node_inputs=node_inputs if keep_inputs else None)


# ----------------------------------------------------------------------------
# Compaction SPINE (cons-mode compaction, 2026-09-22): wide cells
# ----------------------------------------------------------------------------
# A cons compaction slot is a list of LEAVES (folded rounds, oldest first),
# grouped into at most ``max_depth`` consecutive CELLS (``bgkit2.data.spine``):
#
#   leaf i  = encode([header "# folded i" | leaf tokens])           at leaf_ratio (1/16)
#   cell c  = encode([header "# compaction c" | bridged survivors of cell c-1 |
#                     bridged survivors of each of its leaves])      at cell_ratio (1/4)
#
# under the slot's compression prompt; the decoder reads the TOP cell's reps.
# Raw text is compressed hard, what passes from cell to cell gently, and the
# chain stays shallow (<= 4 cells) however long the history, so the gradient
# (top ``grad_levels`` cells and all their leaves) covers most of it. The node
# encoding is the tree's (doc_embeds through the repro bridge) but the layout is
# explicit -- no ``build_tree`` paths. Shared by the trainer
# (``BgKIT2Core.spine_output``) and the student server (which passes the leaves
# and cells it has cached).

LEAF_HEADER = "# folded {i}\n"
CELL_HEADER = "# compaction {c}\n"


@dataclass
class SpineOutput:
    reps: list[torch.Tensor]  # per spine: (K, D) decoder-space reps of the TOP cell (valid rows)
    cells: list[list[list[int]]]  # per spine: leaf indices per cell
    # per spine: {leaf index: bridged survivors} / {cell index: bridged survivors} of what was
    # ENCODED in this call (the student caches them)
    leaf_bridged: list[dict[int, torch.Tensor]]
    cell_bridged: list[dict[int, torch.Tensor]]


def encode_spines(
    encoder: BgKIT2Encoder,
    spines: list[list[torch.Tensor]],
    *,
    tok_encode,
    leaf_ratio: float,
    cell_ratio: float,
    prompts: list[torch.Tensor | None] | None = None,
    grad_levels: int | None = 2,
    max_depth: int = 4,
    node_max_tokens: int = 8192,
    device=None,
    bridge=None,
    token_sq_budget: int = 8192 * 8192,
    leaf_given: list[dict[int, torch.Tensor]] | None = None,
    cell_given: list[dict[int, torch.Tensor]] | None = None,
) -> SpineOutput:
    """Encode compaction spines (``spines[i]`` = leaf token ids, oldest first):
    every leaf batched in one level, then the cells level by level across
    spines. ``grad_levels``: the top ``grad_levels`` cells of each spine and
    their leaves keep a gradient, the rest run under ``no_grad`` (None = all).
    ``leaf_given`` / ``cell_given``: already-encoded bridged survivors per
    spine (the student's cache); a given cell must come with every cell below
    it. The spine role (``encoder.spine_scope``: the spine LoRA, or the slot
    adapter on an adapter-mode encoder) is active throughout and the bridge is
    ``encoder.spine_reproduce`` unless ``bridge`` is given."""
    from ..data.spine import spine_cells

    device = device or next(encoder.parameters()).device
    br = bridge if bridge is not None else encoder.spine_reproduce
    n = len(spines)
    prompts = prompts or [None] * n
    leaf_given = leaf_given or [{} for _ in range(n)]
    cell_given = cell_given or [{} for _ in range(n)]
    emb_w = encoder.backbone.embed_tokens.weight
    cells = [
        spine_cells([int(x.shape[0]) for x in sp], max_depth, node_max_tokens, leaf_ratio,
                    cell_ratio)
        for sp in spines
    ]
    if any(not c for c in cells):
        raise ValueError("every spine needs at least one leaf")

    def grad_cell(i: int, c: int) -> bool:
        return grad_levels is None or c >= len(cells[i]) - int(grad_levels)

    def run(items: list[tuple], ratio: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """items = (spine, embeds (L, D)); one or more encoder calls under the
        token budget -> [(bridged valid survivors, decoder reps valid)] in order."""
        out: list = [None] * len(items)
        order = sorted(range(len(items)), key=lambda k: int(items[k][1].shape[0]))
        chunks: list[list[int]] = []
        cur: list[int] = []
        for k in order:
            w = int(items[k][1].shape[0])
            if cur and (len(cur) + 1) * w * w > token_sq_budget:
                chunks.append(cur)
                cur = []
            cur.append(k)
        if cur:
            chunks.append(cur)
        for g in chunks:
            width = max(int(items[k][1].shape[0]) for k in g)
            emb = torch.zeros(len(g), width, emb_w.shape[-1], dtype=emb_w.dtype, device=device)
            msk = torch.zeros(len(g), width, dtype=torch.bool, device=device)
            for r, k in enumerate(g):
                e = items[k][1]
                emb[r, : e.shape[0]] = e
                msk[r, : e.shape[0]] = True
            p_ids = p_mask = None
            ps = [prompts[items[k][0]] for k in g]
            if any(p is not None and p.shape[0] for p in ps):
                pmax = max(int(p.shape[0]) for p in ps if p is not None)
                p_ids = torch.zeros(len(g), pmax, dtype=torch.long, device=device)
                p_mask = torch.zeros(len(g), pmax, dtype=torch.long, device=device)
                for r, p in enumerate(ps):
                    if p is not None and p.shape[0]:
                        p_ids[r, pmax - p.shape[0] :] = p.to(device)
                        p_mask[r, pmax - p.shape[0] :] = 1
            o = encoder(None, msk, ratio=float(ratio), doc_embeds=emb, prompt_ids=p_ids,
                        prompt_mask=p_mask)
            b_all = br(o.survivors)
            for r, k in enumerate(g):
                keep = o.rep_mask[r]
                out[k] = (b_all[r][keep], o.reps[r][keep])
        return out

    def header(text: str) -> torch.Tensor:
        ids = torch.tensor(tok_encode(text), dtype=torch.long, device=device)
        return encoder.backbone.embed_tokens(ids)

    leaf_b: list[dict[int, torch.Tensor]] = [dict(g) for g in leaf_given]
    new_leaf: list[dict[int, torch.Tensor]] = [{} for _ in range(n)]
    cell_b: list[dict[int, torch.Tensor]] = [dict(g) for g in cell_given]
    new_cell: list[dict[int, torch.Tensor]] = [{} for _ in range(n)]
    top: list[torch.Tensor | None] = [None] * n
    with encoder.spine_scope():
        # ---- leaves: everything not given, in one level (grad / no-grad groups) ----
        for want in (True, False):
            todo = [
                (i, j)
                for i in range(n)
                for c, cell in enumerate(cells[i])
                if c not in cell_given[i] and grad_cell(i, c) == want
                for j in cell
                if j not in leaf_b[i]
            ]
            if not todo:
                continue
            with contextlib.nullcontext() if want else torch.no_grad():
                items = [(i, torch.cat([header(LEAF_HEADER.format(i=j + 1)),
                                        encoder.backbone.embed_tokens(
                                            spines[i][j].long().to(device))]))
                         for i, j in todo]
                for (i, j), (b, _r) in zip(todo, run(items, leaf_ratio), strict=True):
                    b = b if want else b.detach()
                    leaf_b[i][j] = new_leaf[i][j] = b
        # ---- cells, bottom-up, batched across spines per depth ----
        depth = max(len(c) for c in cells)
        for c in range(depth):
            for want in (True, False):
                sel = [i for i in range(n) if c < len(cells[i]) and c not in cell_given[i]
                       and grad_cell(i, c) == want]
                if not sel:
                    continue
                with contextlib.nullcontext() if want else torch.no_grad():
                    items = []
                    for i in sel:
                        parts = [header(CELL_HEADER.format(c=c + 1))]
                        if c > 0:
                            parts.append(cell_b[i][c - 1].to(emb_w.dtype))
                        parts += [leaf_b[i][j].to(emb_w.dtype) for j in cells[i][c]]
                        items.append((i, torch.cat(parts)))
                    for i, (b, r) in zip(sel, run(items, cell_ratio), strict=True):
                        b = b if want else b.detach()
                        cell_b[i][c] = new_cell[i][c] = b
                        if c == len(cells[i]) - 1:
                            top[i] = r
    for i in range(n):
        if top[i] is None:
            raise ValueError("the top cell of a spine cannot be given: it must be encoded")
    return SpineOutput(reps=top, cells=cells, leaf_bridged=new_leaf,  # type: ignore[arg-type]
                       cell_bridged=new_cell)
