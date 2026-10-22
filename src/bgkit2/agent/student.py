"""The STUDENT behind the agent harness: a bgkit2 checkpoint (encoder + head +
projection + decoder adapter) that answers a multi-slot conversation.

``StudentModel.generate_slots`` is the GPU half of ``runner.CompressedModel``
(see ``SlotRendering``): the conversation is rendered with one sentinel per
compressed tool result, every slot's raw text is encoded under its
compression prompt at the rep count the harness's ``RatioPolicy`` granted
(``ratio = n_reps / n_tokens``, the budget regime training used for
``tool_slots``), and the decoder generates from
``[prefix][reps_0][mid_0][reps_1]...[suffix]`` exactly as ``TaskDataset``
lays out a ``tool_slots`` row. ``generate_text`` is the same decoder on a
plain rendered prompt (the full-text arm of the SAME weights).

Two-tokenizer checkpoints (``model.ctx_tokenizer_id``; the LFM2.5-8B-A1B decoder
of bgkit2_8b_a) are served by the same code: slot text, compression prompts and
spine leaves are ENCODER ids (``ChatRenderer.encode_ctx``), the rendering, the
sentinel and generation the decoder's; every rep (text slots, spines, cached tree
nodes -- all in the encoder's rep space) reaches the decoder through its space
adapter inside ``SpliceDecoder.build_batch``, as in training. Clients price slots
in encoder tokens (their ``TemplateRenderer`` keeps the LFM2.5-350M tokenizer).

Served by scripts/bgkit2_student_server.py; consumed by
scripts/bgkit2_eval_agent.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bgkit2.data.templates import SPLICE_SENTINEL

MIN_RATIO, MAX_RATIO = 1.0 / 1024, 0.5


def tree_payload(text: str) -> dict | None:
    """The repo_tree tool's payload, or None for an ordinary text slot."""
    if not text.startswith('{"repo_tree"'):
        return None
    try:
        d = json.loads(text)
    except ValueError:
        return None
    return d if isinstance(d, dict) and d.get("repo_tree") and d.get("files") else None


def slot_ratio(n_reps: int, n_tokens: int) -> float:
    """The encoder's keep-ratio for a slot granted ``n_reps`` reps (the
    trainer's family_rep_budget rule: budget / length, clamped)."""
    if n_tokens <= 0:
        return MAX_RATIO
    return float(min(MAX_RATIO, max(MIN_RATIO, n_reps / n_tokens)))


class StudentModel:
    def __init__(self, cfg, checkpoint: Path, device: torch.device) -> None:
        from bgkit.training.checkpointing import load_checkpoint
        from bgkit2.training.bgkit2_trainer import bgkit2_config_from_hydra
        from bgkit2.training.core import BgKIT2Core

        self.cfg2 = bgkit2_config_from_hydra(cfg)
        self.device = device
        self.checkpoint = str(checkpoint)
        core = BgKIT2Core()
        core.cfg2, core.device = self.cfg2, device
        core.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        # the trainer's tokenizers: ``tok`` = the decoder's, ``core.ctx_tok`` = the encoder's on
        # two-tokenizer runs (LFM2.5-8B-A1B decoder); built before the decoder, whose
        # decoder-space adapter (model.decoder_space_adapter) is initialised over both
        core.load_tokenizers()
        self.tok = core.tok
        self.encoder = core.build_encoder()
        self.decoder = core.build_decoder()
        core.encoder, core.decoder = self.encoder, self.decoder
        _meta, state = load_checkpoint(checkpoint)
        # compaction spines: auto = on when the checkpoint carries the spine
        # adapter; a checkpoint without one still serves spine slots (base encoder)
        if self.cfg2.compaction_spine.enabled is None:
            has = core.resolve_spine_enabled(checkpoint_state=state["encoder"])
            if not has:
                self.cfg2.compaction_spine.enabled = True
                self.cfg2.compaction_spine.trainable = "frozen"
        # the encoder adapter (optim.encoder_mode adapter): served exactly as
        # trained -- ON for text slots and spines (the default role), OFF for
        # cached tree nodes and large trees (``_encode_tree_slot`` asks
        # ``core.tree_role``, the trainer's rule)
        core.resolve_encoder_adapter(state["encoder"])
        core.finish_models()
        core.load_encoder_state(self.encoder, state["encoder"])
        if "decoder" in state:
            # strict; trainable-only checkpoints (model.decoder_save_trainable_only) restore the
            # LoRA + the decoder-space adapter over the base re-read from decoder_id
            core.load_decoder_state(state["decoder"], source=str(checkpoint))
        self.encoder.eval()
        self.decoder.eval()
        # decoder-side text (rendering, generation, the sentinel) with the decoder's tokenizer;
        # slot text, compression prompts and spine leaves with the encoder's (``encode_ctx``)
        self.renderer = core.chat_renderer()
        self.core = core
        self.max_ctx_tokens = int(self.cfg2.data.task_max_ctx_tokens)

    @classmethod
    def from_core(cls, core, checkpoint: str | Path = "") -> StudentModel:
        """A student over models ALREADY built and loaded by
        ``bgkit2.training.standalone.load_models`` (the in-process eval
        runners: one process, no HTTP server). Same slot / tree / spine code
        as the server's student."""
        self = cls.__new__(cls)
        self.cfg2, self.device, self.core = core.cfg2, core.device, core
        self.checkpoint = str(checkpoint)
        core.load_tokenizers()  # kept when load_models already set them
        self.tok = core.tok
        self.encoder, self.decoder = core.encoder, core.decoder
        self.renderer = core.chat_renderer()
        self.max_ctx_tokens = int(core.cfg2.data.task_max_ctx_tokens)
        return self

    # -- helpers -------------------------------------------------------------
    # Always-thinking serving (--thinking, 2026-09-26): the decoder renders its own
    # generation prompt (no empty think block) and reasons before it answers; the
    # generation budget is raised to at least ``think_min_new_tokens``.
    thinking: bool = False
    think_min_new_tokens: int = 4096

    def _decode(self, gen_row: list[int]) -> str:
        if self.renderer.eos_id in gen_row:
            gen_row = gen_row[: gen_row.index(self.renderer.eos_id)]
        gen_row = [t for t in gen_row if t != self.renderer.pad_id]
        return self.tok.decode(gen_row, skip_special_tokens=True).strip()

    def _split_think(self, gen_row: list[int]) -> tuple[str | None, str]:
        """(reasoning, answer) of a generation: the text before the decoder's ``</think>``
        token and the text after it (special tokens are dropped when decoding, so the
        split is made on ids). No ``</think>`` -> (None, everything)."""
        end = self.tok.convert_tokens_to_ids("</think>")
        if end is None or end == self.tok.unk_token_id or end not in gen_row:
            return None, self._decode(gen_row)
        k = gen_row.index(end)
        return self._decode(gen_row[:k]), self._decode(gen_row[k + 1 :])

    def _decode_turn(self, gen_row: list[int]) -> str:
        """The generated turn as text; a thinking turn keeps its reasoning as a
        ``<think>...</think>`` block (``runner.split_think`` separates it again)."""
        if not self.thinking:
            return self._decode(gen_row)
        reasoning, answer = self._split_think(gen_row)
        return answer if reasoning is None else f"<think>\n{reasoning}\n</think>\n{answer}"

    def decode_answer(self, gen_row: list[int]) -> str:
        """The answer only (a single-shot benchmark's prediction)."""
        return self._split_think(gen_row)[1]

    def _think_arg(self) -> bool | None:
        return True if self.thinking else None

    def _budget(self, max_new_tokens: int) -> int:
        return max(int(max_new_tokens), self.think_min_new_tokens) if self.thinking else int(
            max_new_tokens)

    @torch.no_grad()
    def _encode_tree_slot(self, payload: dict, n_reps: int):
        """A repo_tree slot: the whole repository encoded as a recursive tree
        (file -> directory -> root); the slot holds the ROOT reps."""
        from transformers import AutoTokenizer

        from ..models.multilevel import encode_tree

        if getattr(self.encoder, "repro", None) is None:
            raise RuntimeError("repo_tree slots need an encoder with a repro head (M3+)")
        if not hasattr(self, "_enc_tok"):
            self._enc_tok = AutoTokenizer.from_pretrained(self.core.cfg2.model.encoder_id)
        tok_encode = lambda s: self._enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        files = payload["files"]
        paths = [f["path"] for f in files]
        # the trainer's per-file cap and encode_tree settings, from the run's
        # own config (``core.tree_file_tokens`` / ``tree_encode_kwargs``); a
        # hard-coded 512 cut here silently served whole-file checkpoints a
        # different tree than they trained on (2026-09-22 audit)
        cap = self.core.tree_file_tokens()
        ids = [torch.tensor(tok_encode(f["source"])[:cap], dtype=torch.long) for f in files]
        # the trainer's rule: small trees through the encoder adapter, the rest
        # on the frozen base path (what the node caches hold)
        with self.encoder.role(self.core.tree_role(len(paths))), self.core.autocast():
            out = encode_tree(self.encoder, ids, paths, tok_encode=tok_encode,
                              **self.core.tree_encode_kwargs(grad_levels=0))
        n_tok = sum(int(x.shape[0]) for x in ids)
        return out.root.reps[0], out.root.rep_mask[0], n_tok

    # No SILENT truncation (2026-09-22): the harness splits over-long results
    # into parts (Conversation.slot_max_tokens); a slot still longer than
    # max_ctx is counted here and, with ``strict_slot_len``, refused.
    strict_slot_len: bool = False
    over_long_slots: int = 0

    def _encode_text_slot(self, text: str) -> torch.Tensor:
        ids = self.renderer.encode_ctx(text)
        if ids.shape[0] > self.max_ctx_tokens:
            self.over_long_slots += 1
            if self.strict_slot_len:
                raise ValueError(f"slot of {ids.shape[0]} tokens exceeds max_ctx "
                                 f"{self.max_ctx_tokens}")
        return ids[: self.max_ctx_tokens]

    SPINE_CACHE_MAX = 8192

    def spine_keys(self, leaves: list[str], prompt: str, cells: list[list[int]]):
        """Content addresses: a leaf's key covers the prompt, the leaf ratio, its
        index (the header) and its text; a cell's key its prompt, ratio, index
        and the keys of EVERY leaf up to it (the child chain). A fold therefore
        hits the cache for every unchanged leaf, and for every cell whose
        grouping is unchanged."""
        import hashlib

        c = self.core.cfg2.compaction_spine
        lk = [hashlib.sha1(f"L{c.leaf_ratio}\x00{prompt}\x00{i}\x00{x}".encode()).hexdigest()
              for i, x in enumerate(leaves)]
        ck, h = [], hashlib.sha1(f"C{c.cell_ratio}\x00{prompt}".encode())
        for idx, cell in enumerate(cells):
            h = h.copy()
            h.update(f"|{idx}:{','.join(lk[j] for j in cell)}".encode())
            ck.append(h.hexdigest())
        return lk, ck

    @torch.no_grad()
    def _encode_spine_slot(self, payload: dict, prompt: str):
        """A cons compaction slot: the TOP cell's reps through the same
        ``BgKIT2Core.spine_output`` the trainer uses. Leaves and cells already
        encoded for this conversation come from the node cache, so a fold
        encodes only its new leaves and the cells above the last unchanged one."""
        from collections import OrderedDict

        from ..data.spine import spine_cells
        from .context import compaction_leaves

        c = self.core.cfg2.compaction_spine
        cache = getattr(self, "_spine_cache", None)
        if cache is None:
            cache = self._spine_cache = OrderedDict()
        leaves = compaction_leaves(payload)
        ids = [self.renderer.encode_ctx(x) for x in leaves]
        cells = spine_cells([int(x.shape[0]) for x in ids], int(c.max_depth),
                            int(c.node_max_tokens), float(c.leaf_ratio), float(c.cell_ratio))
        lk, ck = self.spine_keys(leaves, prompt, cells)
        leaf_given = {j: cache[k] for j, k in enumerate(lk) if k in cache}
        cell_given: dict[int, torch.Tensor] = {}
        for idx in range(len(cells) - 1):  # a prefix of cached cells; never the top
            if ck[idx] not in cache:
                break
            cell_given[idx] = cache[ck[idx]]
        p_ids = self.renderer.encode_ctx(prompt) if prompt else None
        with self.core.autocast():
            out = self.core.spine_output([ids], [p_ids], leaf_given=[leaf_given],
                                         cell_given=[cell_given])
        for j, b in out.leaf_bridged[0].items():
            cache[lk[j]] = b
        for idx, b in out.cell_bridged[0].items():
            cache[ck[idx]] = b
        for k in [*lk, *ck]:
            if k in cache:
                cache.move_to_end(k)
        while len(cache) > self.SPINE_CACHE_MAX:
            cache.popitem(last=False)
        self.last_spine_encoded = (len(out.leaf_bridged[0]), len(out.cell_bridged[0]))
        r = out.reps[0]
        n_tok = sum(int(x.shape[0]) for x in ids)
        return r, torch.ones(r.shape[0], dtype=torch.bool, device=r.device), n_tok

    def load_tree_root_cache(self, path: str) -> int:
        """Tree-node reps from scripts/bgkit2_cache_repo_trees.py (the same
        cache the trainer reads: a roots file, an all-nodes file or a shard
        dir). Must come from the encoder this student's decoder was trained
        against."""
        from ..data.node_cache import NodeCache

        self.tree_root_cache = NodeCache(path)
        return len(self.tree_root_cache.repos)

    def _overview_slot(self, repo: str, node: str = ""):
        cache = getattr(self, "tree_root_cache", None)
        if cache is None:
            raise RuntimeError("an overview slot arrived but no --tree-root-cache is loaded")
        r = cache.get(repo, node)[0].to(self.device)  # KeyError names the missing node
        return r, torch.ones(r.shape[0], dtype=torch.bool, device=self.device), 0

    def _encode_slots(self, contexts: list[str], prompts: list[str], n_reps: list[int]):
        """Encode K slots (rows = slots, per-row ratio). Text slots go through
        one encoder call; repo_tree slots through encode_tree. Returns
        (reps (K, R, D), rep_mask (K, R), n_tokens per slot)."""
        from ..data.overview import parse_node
        from .context import compaction_payload

        trees = {j: tree_payload(c) for j, c in enumerate(contexts)}
        trees = {j: d for j, d in trees.items() if d is not None}
        # cons-mode compaction: the slot is a SPINE, encoded node by node
        spines = {j: compaction_payload(c) for j, c in enumerate(contexts)}
        spines = {j: d for j, d in spines.items() if d is not None}
        trees.update({j: ("spine", d) for j, d in spines.items()})
        # ALWAYS-ON overview slots: the context only NAMES the repo; the slot
        # is filled with the repo's cached whole-tree ROOT reps.
        # (root = the opening overview, a directory node = a ``browse`` result)
        cached = {j: parse_node(c) for j, c in enumerate(contexts)}
        cached = {j: r for j, r in cached.items() if r is not None}
        if trees or cached:
            text_idx = [j for j in range(len(contexts)) if j not in trees and j not in cached]
            parts: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
            if text_idx:
                r, mk, nt = self._encode_slots(
                    [contexts[j] for j in text_idx], [prompts[j] for j in text_idx],
                    [n_reps[j] for j in text_idx],
                )
                for i, j in enumerate(text_idx):
                    parts[j] = (r[i], mk[i], nt[i])
            for j, d in trees.items():
                if isinstance(d, tuple) and d[0] == "spine":
                    parts[j] = self._encode_spine_slot(d[1], prompts[j])
                else:
                    parts[j] = self._encode_tree_slot(d, int(n_reps[j]))
            for j, (repo, node, _n) in cached.items():
                parts[j] = self._overview_slot(repo, node)
            width = max(int(parts[j][0].shape[0]) for j in range(len(contexts)))
            dim = parts[0][0].shape[-1]
            reps = torch.zeros(len(contexts), width, dim, dtype=parts[0][0].dtype,
                               device=self.device)
            mask = torch.zeros(len(contexts), width, dtype=torch.bool, device=self.device)
            n_tok = []
            for j in range(len(contexts)):
                r, mk, nt = parts[j]
                reps[j, : r.shape[0]] = r
                mask[j, : mk.shape[0]] = mk
                n_tok.append(nt)
            return reps, mask, n_tok
        return self._encode_text_slots_cached(contexts, prompts, n_reps)

    # -- content-keyed rep cache for TEXT slots (2026-09-22) ------------------
    # The memory index (~50 slots under one generic prompt) and every earlier tool result are
    # re-sent on every turn and, for the index, every question over the same memory: without a
    # cache the student re-encoded all of them per /generate. A text slot's reps are a pure
    # function of (weights, text, prompt, rep count), so the key is sha1 of the checkpoint
    # identity + those three; entries are the trimmed reps (real reps only), LRU-bounded by the
    # total number of cached reps (``SLOT_CACHE_MAX_REPS``; ~2 KB per rep in bf16 at D=1024).
    SLOT_CACHE_MAX_REPS = 400_000

    def _cache_identity(self) -> str:
        return f"{getattr(self, 'checkpoint', '')}\x00{getattr(self, 'cache_tag', '')}"

    def slot_key(self, text: str, prompt: str, n_reps: int) -> str:
        import hashlib

        return hashlib.sha1(
            f"{self._cache_identity()}\x00{int(n_reps)}\x00{prompt}\x00{text}".encode()
        ).hexdigest()

    def clear_slot_cache(self) -> None:
        self.__dict__["_slot_cache"] = None
        self.slot_cache_reps = 0

    def slot_cache_stats(self) -> dict:
        c = self.__dict__.get("_slot_cache") or {}
        return {"entries": len(c), "reps": int(getattr(self, "slot_cache_reps", 0)),
                "hits": int(getattr(self, "slot_cache_hits", 0)),
                "misses": int(getattr(self, "slot_cache_misses", 0))}

    def _encode_text_slots_cached(self, contexts: list[str], prompts: list[str],
                                  n_reps: list[int]):
        from collections import OrderedDict

        cache = self.__dict__.get("_slot_cache")
        if cache is None:
            cache = self.__dict__["_slot_cache"] = OrderedDict()
            self.slot_cache_reps = 0
        keys = [self.slot_key(c, p or "", int(r))
                for c, p, r in zip(contexts, prompts, n_reps, strict=True)]
        todo = [j for j, k in enumerate(keys) if k not in cache]
        self.slot_cache_hits = getattr(self, "slot_cache_hits", 0) + len(keys) - len(todo)
        self.slot_cache_misses = getattr(self, "slot_cache_misses", 0) + len(todo)
        if todo:
            r, mk, nt = self._encode_text_slots_uncached(
                [contexts[j] for j in todo], [prompts[j] for j in todo],
                [n_reps[j] for j in todo])
            for q, j in enumerate(todo):
                if keys[j] in cache:  # the same slot twice in one call
                    continue
                rep = r[q][mk[q]].detach()
                cache[keys[j]] = (rep, int(nt[q]))
                self.slot_cache_reps += int(rep.shape[0])
        got = [cache[k] for k in keys]
        for k in keys:
            cache.move_to_end(k)
        keep = set(keys)
        while self.slot_cache_reps > self.SLOT_CACHE_MAX_REPS and len(cache) > len(keep):
            k, (rep, _n) = next(iter(cache.items()))
            if k in keep:  # only this call's slots are left at the old end
                break
            cache.popitem(last=False)
            self.slot_cache_reps -= int(rep.shape[0])
        width = max(1, max(int(rep.shape[0]) for rep, _ in got))
        dim = got[0][0].shape[-1]
        reps = torch.zeros(len(got), width, dim, dtype=got[0][0].dtype, device=got[0][0].device)
        mask = torch.zeros(len(got), width, dtype=torch.bool, device=got[0][0].device)
        for j, (rep, _n) in enumerate(got):
            reps[j, : rep.shape[0]] = rep
            mask[j, : rep.shape[0]] = True
        return reps, mask, [n for _, n in got]

    def _encode_text_slots_uncached(self, contexts: list[str], prompts: list[str],
                                    n_reps: list[int]):
        """One encoder call over K text slots (rows = slots, per-row ratio)."""
        ctx_ids = [self._encode_text_slot(c) for c in contexts]
        p_ids = [self.renderer.encode_ctx(p or "Compress the tool result.\n") for p in prompts]
        k = len(ctx_ids)
        lmax = max(1, max(c.shape[0] for c in ctx_ids))
        pmax = max(1, max(p.shape[0] for p in p_ids))
        ids = torch.zeros(k, lmax, dtype=torch.long)
        mask = torch.zeros(k, lmax, dtype=torch.long)
        pid = torch.zeros(k, pmax, dtype=torch.long)
        pmask = torch.zeros(k, pmax, dtype=torch.long)
        ratios = torch.zeros(k, dtype=torch.float32)
        n_tok = []
        for j, (c, p) in enumerate(zip(ctx_ids, p_ids, strict=True)):
            ids[j, : c.shape[0]] = c
            mask[j, : c.shape[0]] = 1
            pid[j, pmax - p.shape[0] :] = p
            pmask[j, pmax - p.shape[0] :] = 1
            ratios[j] = slot_ratio(int(n_reps[j]), int(c.shape[0]))
            n_tok.append(int(c.shape[0]))
        with self.core.autocast():
            out = self.encoder(
                ids.to(self.device),
                mask.to(self.device),
                ratio=ratios.to(self.device),
                prompt_ids=pid.to(self.device),
                prompt_mask=pmask.to(self.device),
            )
            out = self.core.finish_reps(ids.to(self.device), out)
        return out.reps, out.rep_mask, n_tok

    # -- public --------------------------------------------------------------
    @torch.no_grad()
    def generate_slots(
        self,
        messages: list[dict],
        contexts: list[str],
        prompts: list[str],
        tools: list[dict] | None,
        n_reps: list[int],
        max_new_tokens: int = 512,
    ) -> dict:
        """Answer a slotted conversation (one sentinel per compressed tool
        result, in order). Returns {text, prefix_tokens, n_reps, n_slots}."""
        if not contexts:
            # No compressed result yet (the first turn): a plain rendering.
            ids = torch.tensor(
                self.renderer._apply(messages, tools or None, self._think_arg()), dtype=torch.long
            )
            out = self._generate_ids(ids, max_new_tokens)
            return {
                "text": out["text"],
                "prefix_tokens": out["prefix_tokens"],
                "n_reps": 0,
                "n_slots": 0,
            }
        segs = self.renderer.render_slots(json.dumps(messages), tools or None, self._think_arg())
        k = len(segs) - 1
        if k != len(contexts) or k != len(prompts) or k != len(n_reps):
            raise ValueError(
                f"{k} sentinels in the rendering but {len(contexts)} contexts / "
                f"{len(prompts)} prompts / {len(n_reps)} rep counts"
            )
        prefix, suffix = segs[0], segs[-1]
        reps = rep_mask = extra = mid = None
        total_reps = 0
        if k > 0:
            all_reps, all_mask, _ = self._encode_slots(contexts, prompts, n_reps)
            reps, rep_mask = all_reps[0:1], all_mask[0:1]
            extra = [[(all_reps[j], all_mask[j]) for j in range(1, k)]]
            mid = [list(segs[1:-1])]
            total_reps = int(all_mask.sum().item())
        with self.core.autocast():
            batch = self.decoder.build_batch(
                [prefix],
                reps,
                rep_mask,
                [suffix],
                pad_side="left",
                loss_on_suffix=False,
                extra=extra,
                mid_ids=mid,
            )
            gen = self.decoder.generate(
                batch, max_new_tokens=self._budget(max_new_tokens),
                eos_token_id=self.renderer.eos_id,
            )
        return {
            "text": self._decode_turn(gen[0].tolist()),
            "prefix_tokens": int(batch.inputs_embeds.shape[1]),
            "n_reps": total_reps,
            "n_slots": k,
        }

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 512) -> dict:
        """The same weights on a plain rendered prompt (no slots): the
        full-text arm. ``prompt`` is the template rendering with the
        generation prompt (``Conversation.render_full()``)."""
        if SPLICE_SENTINEL in prompt:
            raise ValueError("generate_text got a slotted prompt; use generate_slots")
        ids = torch.tensor(
            self.tok(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long
        )
        return self._generate_ids(ids, max_new_tokens)

    @torch.no_grad()
    def _generate_ids(self, ids: torch.Tensor, max_new_tokens: int) -> dict:
        empty = torch.zeros(0, dtype=torch.long)
        with self.core.autocast():
            batch = self.decoder.build_batch(
                [ids], None, None, [empty], pad_side="left", loss_on_suffix=False
            )
            gen = self.decoder.generate(
                batch, max_new_tokens=self._budget(max_new_tokens),
                eos_token_id=self.renderer.eos_id,
            )
        return {"text": self._decode_turn(gen[0].tolist()), "prefix_tokens": int(ids.shape[0])}


def handle(student, path: str, body: dict) -> tuple[int, dict]:
    """The server's request logic, separated from HTTP so it is testable with
    a stub student. Returns (status, json)."""
    try:
        if path == "/health":
            stats = student.slot_cache_stats() if hasattr(student, "slot_cache_stats") else {}
            return 200, {"ok": True, "checkpoint": getattr(student, "checkpoint", ""),
                         "slot_cache": stats}
        if path == "/generate":
            for k in ("messages", "contexts", "prompts", "n_reps"):
                if k not in body:
                    return 400, {"error": f"missing {k}"}
            return 200, student.generate_slots(
                body["messages"],
                list(body["contexts"]),
                list(body["prompts"]),
                body.get("tools") or None,
                [int(x) for x in body["n_reps"]],
                int(body.get("max_new_tokens", 512)),
            )
        if path == "/generate_full":
            if "prompt" not in body:
                return 400, {"error": "missing prompt"}
            return 200, student.generate_text(body["prompt"], int(body.get("max_new_tokens", 512)))
        return 404, {"error": f"unknown path {path}"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
