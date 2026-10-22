"""The M0 objective's data: (context span, target) pairs cut from documents.

reconstruct  target == context span (autoencoding: the reps must carry content)
continue     target == the span that FOLLOWS the context (the LM use of reps)

Neither task can be solved from the prompt: the context reaches the decoder
only as reps. Spans are cut at random per (seed, index) so an epoch never
repeats a cut, and the context length is log-uniform in ``[ctx_min, ctx_max]``
so every ratio sees short and long documents.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .templates import Templates
from .token_store import TokenStore


@dataclass
class Sample:
    ctx_ids: torch.Tensor
    target_ids: torch.Tensor
    task: str
    store: int
    doc: int
    prompt_ids: torch.Tensor | None = None  # compression prompt for the ENCODER
    # Pre-rendered decoder framing (task stores). When absent the collator
    # falls back to the per-task Templates.
    prefix_ids: torch.Tensor | None = None
    instr_ids: torch.Tensor | None = None
    # Loss mask over target_ids (bool, same length); None = every target token.
    # Trajectory chunks mask the environment's tool results out of the loss.
    target_loss_mask: torch.Tensor | None = None
    # The target was cut at max_target_tokens: the appended end-of-turn token
    # must carry NO loss (it would teach stopping mid-sentence).
    target_truncated: bool = False
    # A SIBLING compression prompt: another question about the SAME document
    # (multi-question stores). The prompt-contrast loss encodes the document
    # under it and requires the true answer to be served WORSE — the pressure
    # that makes the reps prompt-conditioned.
    alt_prompt_ids: torch.Tensor | None = None
    # Multi-slot samples (agent turns with several compressed tool results):
    # further contexts, their compression prompts, and the decoder token
    # segment that FOLLOWS each extra slot (between it and the next slot /
    # the suffix). ctx_ids stays the primary (first) slot; len(mid_ids) ==
    # len(extra_ctx) == len(extra_prompt).
    extra_ctx: list[torch.Tensor] | None = None
    extra_prompt: list[torch.Tensor | None] | None = None
    mid_ids: list[torch.Tensor] | None = None
    # The sibling as a full sample on the same context (prompt, framing,
    # target) for the PAIRED contrast; None where there is none.
    sibling: Sample | None = None
    # Oracle window for the teacher (meta.gold_window), tokenised; None = the
    # teacher reads the full context.
    teacher_ctx_ids: torch.Tensor | None = None
    # Bool mask over ctx_ids: the supporting facts' tokens (selection target).
    gold_span_mask: torch.Tensor | None = None
    # The PRIMARY slot is an always-on repo overview: (repo, n root reps). Its
    # reps come from the per-repo root cache, ctx_ids is empty and nothing is
    # encoded for it (bgkit2/data/overview.py).
    cached_slot: tuple[str, int] | None = None
    # Per EXTRA slot: (repo, node) when the slot is a cached tree node (the
    # agent's ``browse`` result) -- its ``extra_ctx`` entry is then empty and
    # the trainer fills the slot from the node cache -- else None.
    extra_cached: list[tuple[str, str] | None] | None = None
    # A cons-mode COMPACTION SPINE in the primary slot: its leaves' token ids,
    # oldest first (ctx_ids is then empty; the trainer encodes the spine node by
    # node). ``extra_spine`` is the same per extra slot (None = a text slot).
    spine_slot: list[torch.Tensor] | None = None
    extra_spine: list[list[torch.Tensor] | None] | None = None
    # Per slot (primary first, then extra_ctx order): the rep count the eval HARNESS prices it
    # at (bgkit2.data.slot_pricing; -1 = unpriced) and where the prices came from ("meta" =
    # the row's meta.slot_reps, "prompt" = inferred from the compression prompts). None for
    # rows that are not tool_slots. Read by ``ratio.slot_reps_mode``.
    slot_reps: list[int] | None = None
    slot_reps_src: str | None = None
    # Two-tokenizer runs (encoder and decoder vocabularies differ, e.g. the LFM2.5-8B-A1B decoder):
    # the SAME context text in the DECODER's tokenization, for every path that shows a context to
    # the decoder as text (full-text arm, KD teacher, teacher-window fallback). None = one
    # tokenizer serves both sides and ctx_ids already are decoder ids.
    dec_ctx_ids: torch.Tensor | None = None
    dec_extra_ctx: list[torch.Tensor] | None = None


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, L) right-padded context for the encoder
    attention_mask: torch.Tensor  # (B, L)
    prefix_ids: list[torch.Tensor]
    suffix_ids: list[torch.Tensor]
    suffix_label_start: list[int]
    target_ids: list[torch.Tensor]
    instr_ids: list[torch.Tensor]
    tasks: list[str]
    doc_keys: list[tuple[int, int]]
    prompt_ids: torch.Tensor | None = None  # (B, P) LEFT-padded compression prompts
    prompt_mask: torch.Tensor | None = None
    # Per-sample bool mask over suffix_ids (None = label everything after
    # suffix_label_start); from Sample.target_loss_mask.
    suffix_loss_mask: list[torch.Tensor | None] | None = None
    # (B, P') LEFT-padded sibling prompts (rows without one are all-masked)
    # and the (B,) bool mask of rows that carry one.
    alt_prompt_ids: torch.Tensor | None = None
    alt_prompt_mask: torch.Tensor | None = None
    alt_present: torch.Tensor | None = None
    # The sibling rows as their own Batch, in the order of the rows where
    # ``alt_present`` is True (same contexts, sibling prompts and targets).
    sibling: Batch | None = None
    # (B, L') right-padded teacher contexts: the gold window where a row has
    # one, its full context otherwise. None when no row carries a window.
    teacher_input_ids: torch.Tensor | None = None
    teacher_attention_mask: torch.Tensor | None = None
    # (B, L) bool: gold-span tokens (False everywhere for rows without one).
    gold_span: torch.Tensor | None = None
    # Extra slots of multi-slot samples, stacked over ALL samples' extra slots
    # in (sample, slot) order: (E, L2) contexts, (E,) owner sample index, the
    # LEFT-padded prompts, and per sample the token segments after each of
    # its extra slots. None = single-slot batch.
    extra_input_ids: torch.Tensor | None = None
    extra_attention_mask: torch.Tensor | None = None
    extra_owner: torch.Tensor | None = None
    extra_prompt_ids: torch.Tensor | None = None
    extra_prompt_mask: torch.Tensor | None = None
    mid_ids: list[list[torch.Tensor]] | None = None
    # Per sample: the repo whose cached root fills the PRIMARY slot, or None
    # (an ordinary text slot). None = no such row in the batch.
    cached_slot: list[str | None] | None = None
    # aligned with the rows of ``extra_input_ids``: (repo, node) or None
    extra_cached: list[tuple[str, str] | None] | None = None
    # Per sample: the primary slot's compaction spine (leaf ids, oldest first)
    # or None; None = no such row. ``extra_spine`` is aligned with the rows of
    # ``extra_input_ids`` (see Sample.spine_slot).
    spine_slot: list[list[torch.Tensor] | None] | None = None
    extra_spine: list[list[torch.Tensor] | None] | None = None
    # Harness slot prices (Sample.slot_reps): (B,) for the primary slots and (E,) aligned with
    # ``extra_input_ids``, float, -1 = unpriced; ``slot_reps_src`` per sample (None = none).
    # None when no sample in the batch carries prices.
    slot_reps: torch.Tensor | None = None
    extra_slot_reps: torch.Tensor | None = None
    slot_reps_src: list[str | None] | None = None
    # Sample.dec_ctx_ids per sample and dec_extra_ctx aligned with the rows of extra_input_ids
    # (unpadded, on CPU; the full-text paths pad them). None on single-tokenizer runs.
    dec_ctx: list[torch.Tensor] | None = None
    dec_extra_ctx: list[torch.Tensor] | None = None

    @property
    def n_extra(self) -> list[int]:
        """Number of extra slots per sample (all zero for a single-slot batch)."""
        if self.mid_ids is None:
            return [0] * len(self.tasks)
        return [len(m) for m in self.mid_ids]

    @property
    def n_target_tokens(self) -> int:
        return int(sum(t.shape[0] for t in self.target_ids))

    def to(self, device) -> Batch:
        self.input_ids = self.input_ids.to(device)
        if self.slot_reps is not None:
            self.slot_reps = self.slot_reps.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.prompt_ids is not None:
            self.prompt_ids = self.prompt_ids.to(device)
            self.prompt_mask = self.prompt_mask.to(device)
        if self.alt_prompt_ids is not None:
            self.alt_prompt_ids = self.alt_prompt_ids.to(device)
            self.alt_prompt_mask = self.alt_prompt_mask.to(device)
            self.alt_present = self.alt_present.to(device)
        if self.sibling is not None:
            self.sibling.to(device)
        if self.teacher_input_ids is not None:
            self.teacher_input_ids = self.teacher_input_ids.to(device)
            self.teacher_attention_mask = self.teacher_attention_mask.to(device)
        if self.gold_span is not None:
            self.gold_span = self.gold_span.to(device)
        if self.extra_input_ids is not None:
            self.extra_input_ids = self.extra_input_ids.to(device)
            self.extra_attention_mask = self.extra_attention_mask.to(device)
            self.extra_owner = self.extra_owner.to(device)
            if self.extra_slot_reps is not None:
                self.extra_slot_reps = self.extra_slot_reps.to(device)
            if self.extra_prompt_ids is not None:
                self.extra_prompt_ids = self.extra_prompt_ids.to(device)
                self.extra_prompt_mask = self.extra_prompt_mask.to(device)
        return self

    def with_alt_prompts(self) -> Batch:
        """The same batch with the SIBLING prompts in the prompt slot."""
        if self.alt_prompt_ids is None:
            raise ValueError("batch carries no sibling prompts")
        return dataclasses.replace(
            self, prompt_ids=self.alt_prompt_ids, prompt_mask=self.alt_prompt_mask
        )

    def subset(self, idx: list[int]) -> Batch:
        lmax = int(self.attention_mask[idx].sum(-1).max())
        p_ids = p_mask = None
        if self.prompt_ids is not None:
            p_ids, p_mask = self.prompt_ids[idx].clone(), self.prompt_mask[idx].clone()
        a_ids = a_mask = a_present = None
        if self.alt_prompt_ids is not None:
            a_ids, a_mask = self.alt_prompt_ids[idx].clone(), self.alt_prompt_mask[idx].clone()
            a_present = self.alt_present[idx].clone()
        t_ids = t_mask = None
        if self.teacher_input_ids is not None:
            tmax = int(self.teacher_attention_mask[idx].sum(-1).max())
            t_ids = self.teacher_input_ids[idx, :tmax].clone()
            t_mask = self.teacher_attention_mask[idx, :tmax].clone()
        # The sibling batch is in present-row order: keep the siblings of the
        # kept rows that carry one (the replay split leaves the paired
        # contrast intact for the rows that remain).
        sib = None
        if self.sibling is not None and self.alt_present is not None:
            pos = torch.cumsum(self.alt_present.long(), 0) - 1
            keep = [int(pos[i]) for i in idx if bool(self.alt_present[i])]
            sib = self.sibling.subset(keep) if keep else None
        e_ids = e_mask = e_owner = e_p = e_pm = None
        e_cached = None
        e_spine = None
        e_sr = None
        e_dec = None
        mids = None
        if self.extra_input_ids is not None:
            remap = {old: new for new, old in enumerate(idx)}
            owners = [int(o) for o in self.extra_owner.tolist()]
            rows = [r for r, o in enumerate(owners) if o in remap]
            if rows:
                e_ids = self.extra_input_ids[rows]
                e_mask = self.extra_attention_mask[rows]
                l2 = max(1, int(e_mask.sum(-1).max()))
                e_ids, e_mask = e_ids[:, :l2].clone(), e_mask[:, :l2].clone()
                if self.extra_cached is not None:
                    e_cached = [self.extra_cached[r] for r in rows]
                if self.extra_spine is not None:
                    e_spine = [self.extra_spine[r] for r in rows]
                if self.dec_extra_ctx is not None:
                    e_dec = [self.dec_extra_ctx[r] for r in rows]
                if self.extra_slot_reps is not None:
                    e_sr = self.extra_slot_reps[
                        torch.tensor(rows, dtype=torch.long, device=self.extra_slot_reps.device)
                    ].clone()
                e_owner = torch.tensor([remap[owners[r]] for r in rows], dtype=torch.long).to(
                    self.extra_owner.device
                )
                if self.extra_prompt_ids is not None:
                    e_p = self.extra_prompt_ids[rows].clone()
                    e_pm = self.extra_prompt_mask[rows].clone()
                mids = [self.mid_ids[i] for i in idx]
        return Batch(
            input_ids=self.input_ids[idx, :lmax].clone(),
            attention_mask=self.attention_mask[idx, :lmax].clone(),
            prefix_ids=[self.prefix_ids[i] for i in idx],
            suffix_ids=[self.suffix_ids[i] for i in idx],
            suffix_label_start=[self.suffix_label_start[i] for i in idx],
            target_ids=[self.target_ids[i] for i in idx],
            instr_ids=[self.instr_ids[i] for i in idx],
            tasks=[self.tasks[i] for i in idx],
            doc_keys=[self.doc_keys[i] for i in idx],
            prompt_ids=p_ids,
            prompt_mask=p_mask,
            suffix_loss_mask=(
                [self.suffix_loss_mask[i] for i in idx] if self.suffix_loss_mask else None
            ),
            alt_prompt_ids=a_ids,
            alt_prompt_mask=a_mask,
            alt_present=a_present,
            sibling=sib,
            teacher_input_ids=t_ids,
            teacher_attention_mask=t_mask,
            gold_span=None if self.gold_span is None else self.gold_span[idx, :lmax].clone(),
            extra_input_ids=e_ids,
            extra_attention_mask=e_mask,
            extra_owner=e_owner,
            extra_prompt_ids=e_p,
            extra_prompt_mask=e_pm,
            mid_ids=mids,
            cached_slot=(
                [self.cached_slot[i] for i in idx] if self.cached_slot is not None else None
            ),
            extra_cached=e_cached,
            spine_slot=(
                [self.spine_slot[i] for i in idx] if self.spine_slot is not None else None
            ),
            extra_spine=e_spine,
            slot_reps=None if self.slot_reps is None else self.slot_reps[
                torch.tensor(idx, dtype=torch.long, device=self.slot_reps.device)].clone(),
            extra_slot_reps=e_sr,
            slot_reps_src=(
                [self.slot_reps_src[i] for i in idx] if self.slot_reps_src is not None else None
            ),
            dec_ctx=None if self.dec_ctx is None else [self.dec_ctx[i] for i in idx],
            dec_extra_ctx=e_dec,
        )


class AutoencodeDataset(Dataset):
    def __init__(
        self,
        stores: list[TokenStore],
        weights: list[float] | None = None,
        ctx_min: int = 64,
        ctx_max: int = 1024,
        cont_min: int = 32,
        cont_max: int = 256,
        p_continue: float = 0.5,
        seed: int = 0,
        epoch_size: int | None = None,
        min_doc_tokens: int = 64,
        task_prompt_ids: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not stores:
            raise ValueError("need at least one token store")
        self.stores = stores
        w = np.asarray(weights if weights is not None else [1.0] * len(stores), dtype=np.float64)
        if w.shape[0] != len(stores) or (w <= 0).any():
            raise ValueError("weights must be positive, one per store")
        self.weights = w / w.sum()
        self.ctx_min, self.ctx_max = int(ctx_min), int(ctx_max)
        self.cont_min, self.cont_max = int(cont_min), int(cont_max)
        self.p_continue = float(p_continue)
        self.seed = int(seed)
        self.min_doc_tokens = int(min_doc_tokens)
        self.task_prompt_ids = task_prompt_ids
        self._eligible = [np.nonzero(s.lengths >= self.min_doc_tokens)[0] for s in stores]
        if any(e.size == 0 for e in self._eligible):
            raise ValueError(f"a store has no documents >= {self.min_doc_tokens} tokens")
        self.epoch_size = (
            int(epoch_size) if epoch_size else int(sum(e.size for e in self._eligible))
        )

    # Two-tokenizer runs: ChatRenderer.dec_view (encoder ids -> the same text in decoder ids);
    # set by the trainer. None = one tokenizer, the corpus ids are decoder ids.
    dec_view = None

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(index)])

    def length_hint(self, index: int) -> int:
        """Decoder-sequence length of sample ``index`` WITHOUT loading tokens:
        the same seeded draws as ``__getitem__`` (store, doc, task, ctx_len,
        cont_len). Used by the length-bucketing batch sampler."""
        rng = self._rng(index)
        si = int(rng.choice(len(self.stores), p=self.weights))
        store = self.stores[si]
        di = int(self._eligible[si][rng.integers(self._eligible[si].size)])
        n = store.length(di)
        task = "continue" if rng.random() < self.p_continue else "reconstruct"
        lo, hi = math.log(self.ctx_min), math.log(min(self.ctx_max, n))
        ctx_len = round(math.exp(rng.uniform(lo, max(lo, hi))))
        ctx_len = max(self.ctx_min, min(ctx_len, n))
        cont_len = ctx_len
        if task == "continue":
            cont_len = int(rng.integers(self.cont_min, self.cont_max + 1))
            cont_len = ctx_len if ctx_len + self.cont_min > n else min(cont_len, n - ctx_len)
        return ctx_len + cont_len

    def __getitem__(self, index: int) -> Sample:
        rng = self._rng(index)
        si = int(rng.choice(len(self.stores), p=self.weights))
        store = self.stores[si]
        di = int(self._eligible[si][rng.integers(self._eligible[si].size)])
        n = store.length(di)
        task = "continue" if rng.random() < self.p_continue else "reconstruct"
        # log-uniform context length, clipped to the document
        lo, hi = math.log(self.ctx_min), math.log(min(self.ctx_max, n))
        ctx_len = round(math.exp(rng.uniform(lo, max(lo, hi))))
        ctx_len = max(self.ctx_min, min(ctx_len, n))
        if task == "continue":
            cont_len = int(rng.integers(self.cont_min, self.cont_max + 1))
            if ctx_len + self.cont_min > n:
                task = "reconstruct"
            else:
                cont_len = min(cont_len, n - ctx_len)
        if task == "continue":
            start = int(rng.integers(0, n - ctx_len - cont_len + 1))
            ctx = store.slice(di, start, start + ctx_len)
            tgt = store.slice(di, start + ctx_len, start + ctx_len + cont_len)
        else:
            start = int(rng.integers(0, n - ctx_len + 1))
            ctx = store.slice(di, start, start + ctx_len)
            tgt = ctx.copy()
        ctx_t, tgt_t = torch.from_numpy(ctx), torch.from_numpy(tgt)
        dec_ctx = None
        if self.dec_view is not None:
            # Two tokenizers: the corpus holds ENCODER ids; the decoder's target (and its text
            # view of the context) is the same text in the decoder's tokenization.
            dec_ctx = self.dec_view(ctx_t.long())
            tgt_t = self.dec_view(tgt_t.long())
        return Sample(
            ctx_ids=ctx_t,
            target_ids=tgt_t,
            task=task,
            store=si,
            doc=di,
            prompt_ids=self.task_prompt_ids[task] if self.task_prompt_ids else None,
            dec_ctx_ids=dec_ctx,
        )


class Collator:
    def __init__(self, templates: Templates, enc_pad_id: int | None = None) -> None:
        self.t = templates
        # ENCODER-side tensors (contexts, compression prompts) are padded with the encoder's pad
        # id: a decoder pad id can lie outside the encoder's vocabulary (LFM2.5-8B-A1B: 124893).
        self.enc_pad = int(templates.pad_id if enc_pad_id is None else enc_pad_id)

    def __call__(self, samples: list[Sample]) -> Batch:
        lmax = max(s.ctx_ids.shape[0] for s in samples)
        ids = torch.full((len(samples), lmax), self.enc_pad, dtype=torch.long)
        mask = torch.zeros((len(samples), lmax), dtype=torch.long)
        prefix, suffix, starts, targets, instrs, tasks, keys = [], [], [], [], [], [], []
        loss_masks: list[torch.Tensor | None] = []
        for i, s in enumerate(samples):
            n = s.ctx_ids.shape[0]
            ids[i, :n] = s.ctx_ids
            mask[i, :n] = 1
            if s.prefix_ids is not None and s.instr_ids is not None:
                ins = s.instr_ids
                suf = torch.cat([ins, s.target_ids.to(torch.long), torch.tensor([self.t.eos_id])])
                st = int(ins.shape[0])
                prefix.append(s.prefix_ids)
            else:
                suf, st = self.t.suffix(s.task, s.target_ids)
                ins = self.t.instr[s.task]
                prefix.append(self.t.prefix)
            if s.target_loss_mask is not None or s.target_truncated:
                if (
                    s.target_loss_mask is not None
                    and s.target_loss_mask.shape[0] != s.target_ids.shape[0]
                ):
                    raise ValueError("target_loss_mask must match target_ids in length")
                lm = torch.zeros(suf.shape[0], dtype=torch.bool)
                n_t = s.target_ids.shape[0]
                lm[st : st + n_t] = (
                    s.target_loss_mask.bool() if s.target_loss_mask is not None else True
                )
                # the end-of-turn token: loss-bearing only when the target is complete
                lm[st + n_t :] = not s.target_truncated
                loss_masks.append(lm)
            else:
                loss_masks.append(None)
            suffix.append(suf)
            starts.append(st)
            targets.append(s.target_ids)
            instrs.append(ins)
            tasks.append(s.task)
            keys.append((s.store, s.doc))
        p_ids = p_mask = None
        if any(s.prompt_ids is not None for s in samples):
            # A sample without a prompt (the framing bank's no-prompt draws)
            # gets an all-masked prompt row: the encoder attends to nothing
            # there, so it is the unprompted forward for that sample.
            pmax = max(s.prompt_ids.shape[0] for s in samples if s.prompt_ids is not None)
            p_ids = torch.full((len(samples), pmax), self.enc_pad, dtype=torch.long)
            p_mask = torch.zeros((len(samples), pmax), dtype=torch.long)
            for i, s in enumerate(samples):
                if s.prompt_ids is None:
                    continue
                n = s.prompt_ids.shape[0]
                p_ids[i, pmax - n :] = s.prompt_ids  # LEFT-padded
                p_mask[i, pmax - n :] = 1
        a_ids = a_mask = a_present = None
        if any(s.alt_prompt_ids is not None for s in samples):
            amax = max(s.alt_prompt_ids.shape[0] for s in samples if s.alt_prompt_ids is not None)
            a_ids = torch.full((len(samples), amax), self.enc_pad, dtype=torch.long)
            a_mask = torch.zeros((len(samples), amax), dtype=torch.long)
            a_present = torch.zeros(len(samples), dtype=torch.bool)
            for i, s in enumerate(samples):
                if s.alt_prompt_ids is None:
                    continue
                n = s.alt_prompt_ids.shape[0]
                a_ids[i, amax - n :] = s.alt_prompt_ids
                a_mask[i, amax - n :] = 1
                a_present[i] = True
        t_ids = t_mask = None
        if any(s.teacher_ctx_ids is not None for s in samples):
            # teacher windows are DECODER text: fall back to the decoder tokenization of the context
            tctx = [
                s.teacher_ctx_ids if s.teacher_ctx_ids is not None
                else (s.dec_ctx_ids if s.dec_ctx_ids is not None else s.ctx_ids)
                for s in samples
            ]
            tmax = max(c.shape[0] for c in tctx)
            t_ids = torch.full((len(samples), tmax), self.t.pad_id, dtype=torch.long)
            t_mask = torch.zeros((len(samples), tmax), dtype=torch.long)
            for i, c in enumerate(tctx):
                t_ids[i, : c.shape[0]] = c
                t_mask[i, : c.shape[0]] = 1
        g_span = None
        if any(s.gold_span_mask is not None for s in samples):
            g_span = torch.zeros((len(samples), lmax), dtype=torch.bool)
            for i, s in enumerate(samples):
                if s.gold_span_mask is not None:
                    m_ = s.gold_span_mask[:lmax]
                    g_span[i, : m_.shape[0]] = m_
        sibling = None
        if a_present is not None and all(
            s.sibling is not None for s in samples if s.alt_prompt_ids is not None
        ):
            sibling = self([s.sibling for s in samples if s.alt_prompt_ids is not None])
        e_ids = e_mask = e_owner = e_p = e_pm = None
        e_cached = None
        e_spine = None
        mids = None
        if any(s.extra_ctx for s in samples):
            flat: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
            flat_cached: list[tuple[str, str] | None] = []
            flat_spine: list[list[torch.Tensor] | None] = []
            mids = []
            for i, s in enumerate(samples):
                ec = s.extra_ctx or []
                ep = s.extra_prompt or [None] * len(ec)
                em = s.mid_ids or []
                if not (len(ec) == len(ep) == len(em)):
                    raise ValueError("extra_ctx, extra_prompt and mid_ids must align")
                for c, pr in zip(ec, ep, strict=True):
                    flat.append((i, c, pr))
                flat_cached.extend(s.extra_cached or [None] * len(ec))
                flat_spine.extend(s.extra_spine or [None] * len(ec))
                mids.append(list(em))
            if len(flat_cached) != len(flat):
                raise ValueError("extra_cached must align with extra_ctx")
            if any(flat_cached):
                e_cached = flat_cached
            if any(x is not None for x in flat_spine):
                e_spine = flat_spine
            l2 = max(1, max(c.shape[0] for _, c, _ in flat))
            e_ids = torch.full((len(flat), l2), self.enc_pad, dtype=torch.long)
            e_mask = torch.zeros((len(flat), l2), dtype=torch.long)
            e_owner = torch.tensor([i for i, _, _ in flat], dtype=torch.long)
            for r, (_, c, _) in enumerate(flat):
                e_ids[r, : c.shape[0]] = c
                e_mask[r, : c.shape[0]] = 1
            if any(pr is not None for _, _, pr in flat):
                pmax = max(pr.shape[0] for _, _, pr in flat if pr is not None)
                e_p = torch.full((len(flat), pmax), self.enc_pad, dtype=torch.long)
                e_pm = torch.zeros((len(flat), pmax), dtype=torch.long)
                for r, (_, _, pr) in enumerate(flat):
                    if pr is None:
                        continue
                    e_p[r, pmax - pr.shape[0] :] = pr
                    e_pm[r, pmax - pr.shape[0] :] = 1
        return Batch(
            ids,
            mask,
            prefix,
            suffix,
            starts,
            targets,
            instrs,
            tasks,
            keys,
            p_ids,
            p_mask,
            suffix_loss_mask=loss_masks if any(m is not None for m in loss_masks) else None,
            alt_prompt_ids=a_ids,
            alt_prompt_mask=a_mask,
            alt_present=a_present,
            sibling=sibling,
            extra_input_ids=e_ids,
            extra_attention_mask=e_mask,
            extra_owner=e_owner,
            extra_prompt_ids=e_p,
            extra_prompt_mask=e_pm,
            mid_ids=mids,
            teacher_input_ids=t_ids,
            teacher_attention_mask=t_mask,
            gold_span=g_span,
            cached_slot=(
                [x.cached_slot[0] if x.cached_slot else None for x in samples]
                if any(x.cached_slot for x in samples)
                else None
            ),
            extra_cached=e_cached,
            spine_slot=(
                [x.spine_slot for x in samples]
                if any(x.spine_slot is not None for x in samples)
                else None
            ),
            extra_spine=e_spine,
            **_collate_slot_reps(samples),
            **_collate_dec_ctx(samples),
        )


def _collate_dec_ctx(samples: list[Sample]) -> dict:
    """Batch.dec_ctx / dec_extra_ctx (extra slots in the collator's (sample, slot) order); {} on
    single-tokenizer runs. Every sample of a two-tokenizer batch carries them."""
    if all(s.dec_ctx_ids is None for s in samples):
        return {}
    if any(s.dec_ctx_ids is None for s in samples):
        raise ValueError("dec_ctx_ids must be set on every sample of a two-tokenizer batch")
    extra: list[torch.Tensor] = []
    for s in samples:
        n = len(s.extra_ctx or [])
        de = list(s.dec_extra_ctx or [])
        if len(de) != n:
            raise ValueError(f"dec_extra_ctx has {len(de)} entries for {n} extra slots")
        extra.extend(de)
    return {"dec_ctx": [s.dec_ctx_ids for s in samples], "dec_extra_ctx": extra or None}


def _collate_slot_reps(samples: list[Sample]) -> dict:
    """Batch.slot_reps / extra_slot_reps / slot_reps_src from the samples' harness prices
    (extra slots in the collator's (sample, slot) order); {} when no sample carries any."""
    if not any(s.slot_reps for s in samples):
        return {}
    prim, extra, src = [], [], []
    for s in samples:
        n_extra = len(s.extra_ctx or [])
        sr = list(s.slot_reps or [])
        if sr and len(sr) != 1 + n_extra:
            raise ValueError(f"slot_reps has {len(sr)} entries for {1 + n_extra} slots")
        sr = sr or [-1] * (1 + n_extra)
        prim.append(float(sr[0]))
        extra.extend(float(x) for x in sr[1:])
        src.append(s.slot_reps_src if s.slot_reps else None)
    return {"slot_reps": torch.tensor(prim, dtype=torch.float32),
            "extra_slot_reps": torch.tensor(extra, dtype=torch.float32) if extra else None,
            "slot_reps_src": src}


def task_indices(batch: Batch) -> dict[str, list[int]]:
    """Family name -> sample indices (any family, not only the stage-0 TASKS)."""
    out: dict[str, list[int]] = {}
    for i, t in enumerate(batch.tasks):
        out.setdefault(t, []).append(i)
    return out
