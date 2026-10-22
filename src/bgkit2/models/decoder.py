"""SpliceDecoder: a frozen causal LM that reads dense vectors spliced into its
embedding sequence.

    [prefix tokens] [k rep vectors] [suffix tokens]

Loss is computed on the suffix tokens only (the decoder READS reps; it never
emits anything about them). The base weights are frozen at construction and
never unfrozen. Optional LoRA (rung 1) is restricted to the lower layers;
embeddings and the top layers are never adapted.
"""

from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

try:
    from peft.tuners.lora.layer import Linear as LoraLinear
except ImportError:  # pragma: no cover - peft is a gpu extra

    class LoraLinear:  # type: ignore[no-redef]
        pass


@dataclass
class LoraSpec:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    # Apply the adapter delta at REP positions only: without reps (the
    # no-context and full-text arms, plain text) the decoder is bit-identical
    # to the base by construction, so the adapter can only change how reps
    # are read, never the decoder's priors (P19d absorbed the answer format
    # through the answer positions: noctx 4.41 -> 2.52).
    rep_gated: bool = False
    # Layers that receive adapters (the LOWER part of the stack).
    layers: list[int] = field(default_factory=lambda: list(range(8)))
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ]
    )


class RepGate:
    """Shared holder for the rep-position mask a rep-gated LoRA multiplies its
    delta by. ``mask`` is (B, T, 1) for the forward in flight, or None (no
    reps in the sequence -> the adapter contributes nothing)."""

    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None


def _rep_gated_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """peft ``lora.Linear.forward`` with the delta gated by the rep mask.
    Any forward whose sequence length differs from the mask's (KV-cache
    decode steps, plain-text probes) gets the pristine base layer."""
    gate = self._rep_gate
    mask = gate.mask
    if self.disable_adapters or self.merged:
        return LoraLinear.forward(self, x, *args, **kwargs)
    result = self.base_layer(x, *args, **kwargs)
    if mask is None or x.dim() != 3 or mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
        return result
    out_dtype = result.dtype
    m = mask.to(result.dtype)
    for name in self.active_adapters:
        if name not in self.lora_A:
            continue
        xa = self._cast_input_dtype(x, self.lora_A[name].weight.dtype)
        delta = (
            self.lora_B[name](self.lora_A[name](self.lora_dropout[name](xa))) * self.scaling[name]
        )
        result = result + delta * m
    return result.to(out_dtype)


@dataclass
class SpliceBatch:
    inputs_embeds: torch.Tensor  # (B, T, D)
    attention_mask: torch.Tensor  # (B, T)
    labels: torch.Tensor  # (B, T), -100 outside the suffix
    n_target_tokens: torch.Tensor  # (B,)
    slot_start: torch.Tensor | None = None  # (B,) first position of the rep slot
    slot_len: torch.Tensor | None = None  # (B,) number of reps in the slot
    # Every slot per sample as (start, len), in sequence order — the primary
    # slot first, then the extra slots (multi-slot agent turns: several
    # compressed tool results in one context). None = single-slot batch.
    slots: list[list[tuple[int, int]]] | None = None


class SpliceDecoder(nn.Module):
    def __init__(self, lm: Lfm2ForCausalLM, lora: LoraSpec | None = None) -> None:
        super().__init__()
        for p in lm.parameters():
            p.requires_grad_(False)
        self.lora = lora
        self._has_adapter = False
        if lora is not None:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=list(lora.target_modules),
                layers_to_transform=list(lora.layers),
                bias="none",
                task_type="CAUSAL_LM",
            )
            lm = get_peft_model(lm, cfg)
            self._has_adapter = True
            if lora.rep_gated:
                self.rep_gate = RepGate()
                n = 0
                for mod in lm.modules():
                    if isinstance(mod, LoraLinear):
                        mod._rep_gate = self.rep_gate
                        mod.forward = types.MethodType(_rep_gated_lora_forward, mod)
                        n += 1
                if n == 0:
                    raise RuntimeError("rep_gated LoRA: no LoRA Linear layers found to gate")
        self.lm = lm
        self.config = lm.config if not self._has_adapter else lm.base_model.model.config
        emb = self.embed_tokens
        emb.weight.requires_grad_(False)
        self._embed_row_norm = float(emb.weight.detach().float().norm(dim=-1).mean().item())
        # Rows of the vocabulary projection materialised at once (4096 x 65536
        # fp32 = 1 GB); everything else in the loss path is O(tokens x hidden).
        self.ce_chunk_rows = 4096

    # ------------------------------------------------------------------
    @property
    def base_lm(self) -> Lfm2ForCausalLM:
        return self.lm.base_model.model if self._has_adapter else self.lm

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.base_lm.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def embed_row_norm(self) -> float:
        return self._embed_row_norm

    # P8 (models.space_adapter.DecoderSpaceAdapter), registered as ``self.space_adapter`` when
    # the run has one: maps encoder reps into THIS decoder's embedding space at splice time.
    # (No class-level default: it would shadow the registered submodule.)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The decoder adapter's parameters (LoRA). P8 has its own optimizer group:
        ``space_adapter_parameters``."""
        return [p for p in self.lm.parameters() if p.requires_grad]

    def space_adapter_parameters(self) -> list[nn.Parameter]:
        ad = self._modules.get("space_adapter")
        return [] if ad is None else [p for p in ad.parameters() if p.requires_grad]

    def adapt_reps(self, reps: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encoder reps -> this decoder's embedding space (identity without a space adapter)."""
        ad = self._modules.get("space_adapter")
        if ad is None:
            return reps
        return ad(reps, mask)

    def gradient_checkpointing_enable(self) -> None:
        self.base_lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @contextlib.contextmanager
    def rep_gated(self, batch: SpliceBatch):
        """Arm the rep gate with ``batch``'s slot for the forwards inside;
        a no-op without a rep-gated adapter."""
        gate = getattr(self, "rep_gate", None)
        if gate is None:
            yield
            return
        mask = None
        if (
            batch.slot_start is not None
            and batch.slot_len is not None
            and bool((batch.slot_len > 0).any())
        ):
            _bsz, t = batch.attention_mask.shape
            pos = torch.arange(t, device=batch.attention_mask.device)[None, :]
            st = batch.slot_start[:, None]
            mask = ((pos >= st) & (pos < st + batch.slot_len[:, None])).unsqueeze(-1)
        gate.mask = mask
        try:
            yield
        finally:
            gate.mask = None

    def disable_adapter(self):
        """Context manager: run the pristine base (no-op without LoRA)."""
        if self._has_adapter:
            return self.lm.disable_adapter()
        import contextlib

        return contextlib.nullcontext()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(ids)

    # ------------------------------------------------------------------
    def build_batch(
        self,
        prefix_ids: list[torch.Tensor],
        reps: torch.Tensor | None,
        rep_mask: torch.Tensor | None,
        suffix_ids: list[torch.Tensor],
        pad_side: str = "right",
        loss_on_suffix: bool = True,
        suffix_label_start: list[int] | None = None,
        suffix_loss_mask: list[torch.Tensor | None] | None = None,
        extra: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        mid_ids: list[list[torch.Tensor]] | None = None,
        decoder_space: bool = False,
    ) -> SpliceBatch:
        """Assemble per-sample ``[prefix][reps][suffix]`` and pad to a batch.

        ``reps`` may be None (the no-context arm) or zeros (the zeroed arm);
        a zeroed arm keeps the slot so sequence positions match the reps arm.
        ``suffix_label_start[i]`` masks the first n suffix tokens (an
        instruction that precedes the target) out of the loss;
        ``suffix_loss_mask[i]`` (bool over the suffix, or None) additionally
        keeps only the True positions (tool results inside a trajectory
        chunk are the environment's, not the model's).

        Multi-slot samples (agent turns with several compressed tool results):
        ``extra[i]`` is the list of further slots of sample i as
        ``(reps_j (K_j, D), mask_j (K_j,))`` and ``mid_ids[i][j]`` the token
        segment that follows slot j — the layout becomes
        ``[prefix][reps][mid_0][extra_0][mid_1][extra_1]...[suffix]``. An
        entry of ``(None, None)`` in ``extra[i]`` drops that slot (the
        no-context arm keeps the segments).
        """
        bsz = len(prefix_ids)
        if len(suffix_ids) != bsz:
            raise ValueError("prefix_ids and suffix_ids must have the same length")
        if not decoder_space and self._modules.get("space_adapter") is not None:
            # encoder reps -> this decoder's space (P8); ``decoder_space`` = they already are
            # decoder embeddings (the full-text arm)
            if reps is not None:
                reps = self.adapt_reps(reps, rep_mask)
            if extra is not None:
                extra = [[(None, None) if r is None else (self.adapt_reps(r, mk), mk)
                          for r, mk in slots] for slots in extra]
        if reps is not None and (rep_mask is None or reps.shape[0] != bsz):
            raise ValueError("reps needs a rep_mask of matching batch size")
        if (extra is None) != (mid_ids is None):
            raise ValueError("extra and mid_ids go together")
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        pieces, labels, lengths, slot_starts, slot_lens = [], [], [], [], []
        all_slots: list[list[tuple[int, int]]] = []
        for i in range(bsz):
            p = prefix_ids[i].to(device)
            s = suffix_ids[i].to(device)
            segs = [self.embed(p)]
            lab = [torch.full((p.shape[0],), -100, dtype=torch.long, device=device)]
            slot_starts.append(int(p.shape[0]))
            slot_lens.append(0)
            pos = int(p.shape[0])
            if reps is not None:
                r = reps[i][rep_mask[i].bool()].to(dtype)
                segs.append(r)
                lab.append(torch.full((r.shape[0],), -100, dtype=torch.long, device=device))
                slot_lens[-1] = int(r.shape[0])
            sample_slots = [(slot_starts[-1], slot_lens[-1])]
            pos += slot_lens[-1]
            if extra is not None and extra[i]:
                if len(mid_ids[i]) != len(extra[i]):
                    raise ValueError("mid_ids[i] must have one segment per extra slot")
                for (er, em), mid in zip(extra[i], mid_ids[i], strict=True):
                    m_ids = mid.to(device)
                    segs.append(self.embed(m_ids))
                    lab.append(torch.full((m_ids.shape[0],), -100, dtype=torch.long, device=device))
                    pos += int(m_ids.shape[0])
                    n_e = 0
                    if er is not None:
                        e = er[em.bool()].to(dtype)
                        segs.append(e)
                        lab.append(torch.full((e.shape[0],), -100, dtype=torch.long, device=device))
                        n_e = int(e.shape[0])
                    sample_slots.append((pos, n_e))
                    pos += n_e
            all_slots.append(sample_slots)
            segs.append(self.embed(s))
            s_lab = s.clone() if loss_on_suffix else torch.full_like(s, -100)
            if suffix_label_start is not None and suffix_label_start[i] > 0:
                s_lab[: suffix_label_start[i]] = -100
            if suffix_loss_mask is not None and suffix_loss_mask[i] is not None:
                m = suffix_loss_mask[i].to(device)
                if m.shape[0] != s.shape[0]:
                    raise ValueError("suffix_loss_mask must match suffix_ids in length")
                s_lab = torch.where(m.bool(), s_lab, torch.full_like(s_lab, -100))
            lab.append(s_lab)
            pieces.append(torch.cat(segs, dim=0))
            labels.append(torch.cat(lab, dim=0))
            lengths.append(pieces[-1].shape[0])
        tmax = max(lengths)
        d = pieces[0].shape[-1]
        inputs = torch.zeros(bsz, tmax, d, dtype=dtype, device=device)
        attn = torch.zeros(bsz, tmax, dtype=torch.long, device=device)
        lab_out = torch.full((bsz, tmax), -100, dtype=torch.long, device=device)
        starts = torch.zeros(bsz, dtype=torch.long, device=device)
        for i in range(bsz):
            n = lengths[i]
            if pad_side == "right":
                sl = slice(0, n)
                starts[i] = slot_starts[i]
                off = 0
            elif pad_side == "left":
                sl = slice(tmax - n, tmax)
                starts[i] = tmax - n + slot_starts[i]
                off = tmax - n
            else:
                raise ValueError("pad_side must be 'right' or 'left'")
            inputs[i, sl] = pieces[i]
            attn[i, sl] = 1
            lab_out[i, sl] = labels[i]
            all_slots[i] = [(st + off, ln) for st, ln in all_slots[i]]
        n_target = (lab_out != -100).sum(dim=-1)
        multi = extra is not None and any(len(x) > 0 for x in extra)
        return SpliceBatch(
            inputs,
            attn,
            lab_out,
            n_target,
            slot_start=starts,
            slot_len=torch.tensor(slot_lens, dtype=torch.long, device=device),
            slots=all_slots if multi else None,
        )

    # ------------------------------------------------------------------
    def _hidden(self, inputs_embeds=None, attention_mask=None, input_ids=None) -> torch.Tensor:
        """Final hidden states from the (possibly LoRA-wrapped) base model.

        The LoRA layers live inside the modules, so calling the inner
        ``Lfm2Model`` directly still runs the adapter."""
        inner = self.base_lm.model
        out = inner(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state

    # Row budget of a no-grad forward (the KD teacher): its full-text batch can be ~16x the
    # student's compressed one, and one padded forward over all rows OOMed S2 twice at step
    # ~17630 (2026-09-25, 49 GB). Chunking rows is exact (rows are independent) and bounds the peak.
    nograd_chunk_tokens = 32768

    def _hidden_nograd_chunked(self, inputs_embeds, attention_mask, model=None) -> torch.Tensor:
        """``_hidden`` under no_grad, over row chunks of at most ``nograd_chunk_tokens`` padded
        positions (at least one row per chunk); same result as one call."""
        model = self if model is None else model
        b, t = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        rows = max(1, int(self.nograd_chunk_tokens) // max(t, 1))
        if rows >= b:
            return model._hidden(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        parts = [
            model._hidden(inputs_embeds=inputs_embeds[a : a + rows],
                          attention_mask=attention_mask[a : a + rows])
            for a in range(0, b, rows)
        ]
        return torch.cat(parts, dim=0)

    def _nll_rows(self, hidden: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor):
        """CE only at the labelled positions, in chunks, never materialising
        full-vocabulary logits for a whole batch (that transient reached tens
        of GB at ratio 1.0 and nearly froze the host, 2026-09-03)."""
        head = self.base_lm.lm_head
        rows = hidden[:, :-1, :][valid]  # (N, D)
        tgt = targets[valid]  # (N,)
        out = torch.zeros(tgt.shape[0], dtype=torch.float32, device=hidden.device)
        chunk = self.ce_chunk_rows
        for a in range(0, rows.shape[0], chunk):
            logits = head(rows[a : a + chunk]).float()
            out[a : a + chunk] = F.cross_entropy(logits, tgt[a : a + chunk], reduction="none")
        return out

    def token_nll(self, batch: SpliceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Grad-enabled (B, T-1) NLL per position and the label mask."""
        with self.rep_gated(batch):
            hidden = self._hidden(
                inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask
            )
        targets = batch.labels[:, 1:]
        valid = targets != -100
        nll = torch.zeros(targets.shape, dtype=torch.float32, device=hidden.device)
        if bool(valid.any()):
            nll = nll.masked_scatter(valid, self._nll_rows(hidden, targets, valid))
        return nll, valid

    def forward(self, batch: SpliceBatch) -> torch.Tensor:
        """Token-mean cross-entropy over the labelled suffix tokens."""
        nll, valid = self.token_nll(batch)
        return nll.sum() / valid.sum().clamp_min(1)

    def token_nll_and_kd(
        self,
        batch: SpliceBatch,
        teacher: SpliceBatch,
        temperature: float = 1.0,
        teacher_model: SpliceDecoder | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CE per position (B, T-1) + label mask, and the mean KL(teacher ||
        student) over the labelled rows, from ONE student forward.

        ``teacher`` is a batch over the same samples (same suffixes, so the
        labelled rows align in order) with a different context slot, e.g. the
        context's own token embeddings; it is run with the adapter disabled
        and no grad — or through ``teacher_model`` (a separate, frozen
        decoder, e.g. one carrying a full-text adapter) when given. Chunked
        over rows so no full (N, V) logits exist."""
        with self.rep_gated(batch):
            hidden = self._hidden(
                inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask
            )
        targets = batch.labels[:, 1:]
        valid = targets != -100
        rows_s = hidden[:, :-1, :][valid]
        tgt = targets[valid]
        if teacher_model is None:
            with torch.no_grad(), self.disable_adapter():
                h_t = self._hidden_nograd_chunked(teacher.inputs_embeds, teacher.attention_mask)
            head_t = self.base_lm.lm_head
        else:
            # A rep-gated teacher needs its gate armed for the slot it reads
            # (token embeddings); without it the adapter would be silent.
            with torch.no_grad(), teacher_model.rep_gated(teacher):
                h_t = self._hidden_nograd_chunked(
                    teacher.inputs_embeds, teacher.attention_mask, model=teacher_model
                )
            head_t = teacher_model.base_lm.lm_head
        valid_t = teacher.labels[:, 1:] != -100
        rows_t = h_t[:, :-1, :][valid_t]
        if rows_t.shape[0] != rows_s.shape[0]:
            raise ValueError(
                f"teacher/student labelled rows differ: {rows_t.shape[0]} vs {rows_s.shape[0]}"
            )
        head = self.base_lm.lm_head
        n = rows_s.shape[0]
        nll_flat = torch.zeros(n, dtype=torch.float32, device=hidden.device)
        kl_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        chunk = self.ce_chunk_rows
        t = float(temperature)
        for a in range(0, n, chunk):
            ls = head(rows_s[a : a + chunk]).float()
            nll_flat[a : a + chunk] = F.cross_entropy(ls, tgt[a : a + chunk], reduction="none")
            with torch.no_grad():
                lt = head_t(rows_t[a : a + chunk]).float() / t
                pt = torch.softmax(lt, dim=-1)
                logpt = torch.log_softmax(lt, dim=-1)
            logps = torch.log_softmax(ls / t, dim=-1)
            kl_sum = kl_sum + (pt * (logpt - logps)).sum()
        nll = torch.zeros(targets.shape, dtype=torch.float32, device=hidden.device)
        if n:
            nll = nll.masked_scatter(valid, nll_flat)
        kl = kl_sum / max(n, 1) * (t * t)
        return nll, valid, kl

    attention_importance_chunk = 4  # samples per teacher pass: the eager (B, H, T, T)
    # transient at B=12, T~2.3k reached 29 GB allocated / 70 GB reserved on unified memory

    @torch.no_grad()
    def attention_importance(self, batch: SpliceBatch) -> torch.Tensor:
        """Chunked over samples; see ``_attention_importance``."""
        bsz = batch.labels.shape[0]
        c = max(1, int(self.attention_importance_chunk))
        if bsz <= c:
            return self._attention_importance(batch)
        parts = []
        for a in range(0, bsz, c):
            sub = SpliceBatch(
                batch.inputs_embeds[a : a + c],
                batch.attention_mask[a : a + c],
                batch.labels[a : a + c],
                batch.n_target_tokens[a : a + c],
                slot_start=None if batch.slot_start is None else batch.slot_start[a : a + c],
                slot_len=None if batch.slot_len is None else batch.slot_len[a : a + c],
            )
            parts.append(self._attention_importance(sub))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _attention_importance(self, batch: SpliceBatch) -> torch.Tensor:
        """(B, T) importance of every key position = the attention mass it
        receives from the LABELLED (answer) query positions, averaged over
        heads and over this model's attention layers. Needs eager attention
        (the weights are reduced inside a forward hook, one layer at a time,
        so no (B, H, T, T) tensor outlives its layer)."""
        inner = self.base_lm.model
        # The attention interface is looked up per forward from config, so
        # eager is switched on only for this pass; the KD pass stays on sdpa.
        prev_impl = getattr(inner.config, "_attn_implementation", "sdpa")
        inner.config._attn_implementation = "eager"
        q_rows = (batch.labels != -100).to(torch.float32)  # (B, T) answer positions
        acc = torch.zeros(batch.labels.shape, dtype=torch.float32, device=batch.labels.device)
        n_layers = 0

        def hook(_module, _inputs, output):
            nonlocal n_layers
            weights = output[1] if isinstance(output, tuple) else None
            if weights is None:
                return
            w = weights.float().mean(dim=1)  # (B, Tq, Tk) over heads
            acc.add_(torch.einsum("bq,bqk->bk", q_rows, w))
            n_layers += 1

        handles = [
            layer.self_attn.register_forward_hook(hook)
            for layer in inner.layers
            if getattr(layer, "is_attention_layer", False)
        ]
        try:
            with self.rep_gated(batch):
                inner(
                    inputs_embeds=batch.inputs_embeds,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()
            inner.config._attn_implementation = prev_impl
        if n_layers == 0:
            raise RuntimeError("no attention weights were captured")
        denom = q_rows.sum(dim=-1, keepdim=True).clamp_min(1.0) * n_layers
        return acc / denom * batch.attention_mask.to(acc.dtype)

    @torch.no_grad()
    def argmax_at_labels(self, batch: SpliceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) predicted ids at labelled positions and the (B, T-1) label mask."""
        with self.rep_gated(batch):
            hidden = self._hidden(
                inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask
            )
        targets = batch.labels[:, 1:]
        valid = targets != -100
        rows = hidden[:, :-1, :][valid]
        head = self.base_lm.lm_head
        preds = []
        for a in range(0, rows.shape[0], self.ce_chunk_rows):
            preds.append(head(rows[a : a + self.ce_chunk_rows]).argmax(-1))
        pred = torch.cat(preds) if preds else torch.zeros(0, dtype=torch.long, device=hidden.device)
        return pred, valid

    @torch.no_grad()
    def per_token_nll(self, batch: SpliceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """No-grad (B, T-1) NLL per position and the matching label mask."""
        return self.token_nll(batch)

    @torch.no_grad()
    def generate(
        self,
        batch: SpliceBatch,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy decode from a LEFT-padded prompt batch (labels ignored)."""
        was_training = self.lm.training
        self.lm.eval()
        try:
            with self.rep_gated(batch):
                out = self.lm.generate(
                    inputs_embeds=batch.inputs_embeds,
                    attention_mask=batch.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.config.pad_token_id
                    if self.config.pad_token_id is not None
                    else (eos_token_id if eos_token_id is not None else 0),
                    use_cache=True,
                )
        finally:
            self.lm.train(was_training)
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def plain_text_nll(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Mean next-token NLL on plain token ids (the language-health probe).

        The corpus token stores carry no BOS; without it the LFM2.5 decoder is
        ~3.5 nats worse on the same text (measured 2026-09-14: 6.86 vs 3.38
        with BOS), so ``lm_health/nll`` read 9-10 nats and was meaningless in
        absolute terms. A BOS is prepended when the rows do not start with one."""
        bos = self.config.bos_token_id
        needs_bos = bos is not None and input_ids.shape[1] > 0
        if needs_bos and bool((input_ids[:, 0] != bos).any()):
            col = torch.full(
                (input_ids.shape[0], 1), int(bos), dtype=input_ids.dtype, device=input_ids.device
            )
            input_ids = torch.cat([col, input_ids], dim=1)
            attention_mask = torch.cat([torch.ones_like(col), attention_mask], dim=1)
        hidden = self._hidden(input_ids=input_ids, attention_mask=attention_mask)
        targets = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        nll = self._nll_rows(hidden, targets, valid)
        return float(nll.sum().item() / max(int(valid.sum()), 1))


class TrainableDecoderState(nn.Module):
    """Checkpoint face of a decoder whose BASE is frozen and re-read from ``decoder_id``
    (``model.decoder_save_trainable_only``; an 8B base is 16 GB per save): ``state_dict`` holds
    the trainable tensors (LoRA, P8) plus P8's buffers (its interface statistics), and
    ``load_state_dict`` restores exactly those -- strict means every one of them must be there."""

    def __init__(self, decoder: SpliceDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def _keys(self) -> list[str]:
        full = self.decoder.state_dict()
        train = {n for n, p in self.decoder.named_parameters() if p.requires_grad}
        return [k for k in full if k in train or k.startswith("space_adapter.")]

    def state_dict(self, *args, **kwargs):
        import collections

        full = self.decoder.state_dict()
        return collections.OrderedDict((k, full[k]) for k in self._keys())

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        import collections

        want, known = set(self._keys()), set(self.decoder.state_dict())
        missing = sorted(want - set(state))
        unexpected = sorted(set(state) - known)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"trainable decoder state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.decoder.load_state_dict({k: v for k, v in state.items() if k in known}, strict=False)
        Result = collections.namedtuple("Result", ["missing_keys", "unexpected_keys"])
        return Result(missing, unexpected)

