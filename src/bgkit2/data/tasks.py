"""Task stores: text rows rendered into (encoder prompt, context, decoder framing).

A task store is a parquet file of TEXT rows (tokenised on the fly by the
dataloader workers), one row per training sample:

    family        e.g. qa_squad, file_overview, tool_read, compaction
    framing       chat_context | tool_response
    context       the text the ENCODER compresses
    prompt        the compression prompt handed to the encoder
    instruction   what the decoder is asked (user question / "give an overview")
    tool_name     tool_response only: the tool the assistant "called"
    tool_args     tool_response only: JSON object of arguments
    target        the assistant's answer (loss-bearing) — text
    split         train | eval
    meta          free JSON (source ids, repo, path, ...)

Framings are rendered with the decoder tokenizer's OWN chat template by
placing a reserved single-token sentinel where the reps go and splitting the
rendered ids at it:

    chat_context   user: "Context:\\n<sentinel>\\n{instruction}"
    tool_response  user: {instruction}; assistant: tool call; tool: <sentinel>
    messages       ``instruction`` is a JSON list of chat messages, exactly one
                   of which contains the sentinel (compaction: the history
                   message); rendered verbatim through the template
    trajectory     like ``messages``, but ``target`` is ALSO a JSON list of
                   messages (the next chunk of the trajectory: assistant
                   turns with tool calls + the tool results they caused),
                   rendered through the template after the prefix; loss on
                   the assistant messages only (tool results are masked)

    chat           ``instruction`` is a JSON list of chat messages with NO
                   sentinel (the replay stream); ``target`` is the next
                   assistant turn — plain text, or a JSON list holding that
                   one assistant message when the turn carries tool calls
                   (rendered like a trajectory chunk: content + call + end)

The prefix (before the sentinel) and the instruction tail (after it, through
"<|im_start|>assistant\\n") are per-sample; loss covers target + <|im_end|>.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .autoencode import Sample
from .framing_bank import FramingBank
from .templates import SPLICE_SENTINEL, decoder_sentinel, with_sentinel

# How many rows __getitem__ may skip past before it gives up on a draw. Over-long
# rows cluster by source (a run of 872 in tool_slots_merged), so the budget is
# paired with a random jump in _hop_over_long — 8 sequential steps could not
# clear a bad region and turned a partly-usable store into a dead dataloader.
_LONG_PREFIX_ATTEMPTS = 32

FRAMINGS = (
    "chat_context", "tool_response", "messages", "trajectory", "chat", "chat_think", "tool_slots"
)
COLUMNS = [
    "family",
    "framing",
    "context",
    "prompt",
    "instruction",
    "tool_name",
    "tool_args",
    "target",
    "split",
    "meta",
]


_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def _reasoning_of(m) -> str | None:
    """An assistant message's reasoning text (first non-empty reasoning field), else None."""
    if not isinstance(m, dict) or m.get("role") != "assistant":
        return None
    for k in _REASONING_KEYS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def has_reasoning(msgs: list[dict]) -> bool:
    """Whether any assistant message carries reasoning (a ``reasoning_content`` /
    ``reasoning`` / ``thinking`` string): such rows render in the decoder's own
    thinking format (``ChatRenderer._apply``)."""
    return any(_reasoning_of(m) is not None for m in msgs)


def row_has_reasoning(r: dict) -> bool:
    """``has_reasoning`` over a store row's instruction and target message lists."""
    if "reasoning" not in r["instruction"] and "reasoning" not in r["target"] \
            and "thinking" not in r["instruction"] and "thinking" not in r["target"]:
        return False
    msgs: list[dict] = []
    for col in ("instruction", "target"):
        v = json_message_list(r[col])
        if v:
            msgs += v
    return has_reasoning(msgs)


def _native_reasoning(msgs: list[dict]) -> list[dict]:
    """Reasoning-bearing assistant messages in the spacing the 8B itself emits,
    ``<think>\\n{reasoning}\\n</think>\\n{content}`` (the template concatenates
    ``"<think>" + reasoning + "</think>" + content``); idempotent."""
    out = []
    for m in msgs:
        r = _reasoning_of(m)
        if r is not None:
            m = {k: v for k, v in m.items() if k not in _REASONING_KEYS}
            m["reasoning_content"] = "\n" + r.strip() + "\n"
            m["content"] = "\n" + str(m.get("content") or "").strip()
        out.append(m)
    return out


def json_message_list(target: str) -> list[dict] | None:
    """The message list a ``target`` column holds, or None when the target is
    plain text — a text answer may itself start with ``[`` ("[Note] ..."), so
    the shape is checked, not the first character."""
    if not target.lstrip().startswith("["):
        return None
    try:
        v = json.loads(target)
    except json.JSONDecodeError:
        return None
    if isinstance(v, list) and v and all(isinstance(m, dict) and "role" in m for m in v):
        return v
    return None


SPINE_CONTEXT_PREFIX = '["{\\"compaction'  # a row whose FIRST slot is a compaction spine
_SPINE_ANY = '{\\"compaction\\": 1'  # a spine payload in any slot (JSON-escaped)


def store_has_spines(path: str, max_rows: int | None = None) -> bool:
    """Whether any row of a task store carries a cons-mode compaction spine
    (``bgkit2.agent.context.compaction_payload`` in its context column). Streams
    the context column (memory-mapped for .arrow) and stops at the first hit."""
    seen = 0
    if str(path).endswith(".arrow"):
        batches = pa.ipc.open_file(pa.memory_map(str(path))).read_all().select(
            ["context"]).to_batches(max_chunksize=4096)
    else:
        batches = pq.ParquetFile(str(path)).iter_batches(batch_size=4096, columns=["context"])
    for b in batches:
        col = b.column(0)
        if pc.any(pc.match_substring(col, _SPINE_ANY)).as_py():
            return True
        seen += b.num_rows
        if max_rows is not None and seen >= max_rows:
            return False
    return False


def write_task_store_arrow(path: str, table: pa.Table) -> None:
    """Write a task-store table as an uncompressed Arrow IPC file (see
    ``TaskDataset``: memory-mapped, zero-copy)."""
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as w:
        for batch in table.to_batches(max_chunksize=4096):
            w.write_batch(batch)


def write_task_store(path: str, rows: list[dict]) -> int:
    """Write rows (dicts with COLUMNS; missing -> "") to parquet. Returns count."""
    cols = {c: [] for c in COLUMNS}
    for r in rows:
        if r.get("framing", "chat_context") not in FRAMINGS:
            raise ValueError(f"bad framing {r.get('framing')!r}")
        for c in COLUMNS:
            v = r.get(c, "")
            if c in ("tool_args", "meta") and not isinstance(v, str):
                v = json.dumps(v)
            cols[c].append("" if v is None else str(v))
    table = pa.table({c: pa.array(v, type=pa.string()) for c, v in cols.items()})
    pq.write_table(table, path, compression="zstd")
    return len(rows)


@dataclass
class ChatRenderer:
    """Renders a framing into (prefix_ids, instr_ids) with the real template."""

    tok: object  # the DECODER's tokenizer: prompts, targets, anything the decoder reads as text
    sentinel_id: int
    eos_id: int
    bos_id: int
    pad_id: int
    last_chunk_truncated: bool = False  # set by render_chunk
    # The ENCODER's tokenizer (context slots, compression prompts, spine leaves, gold spans).
    # None = one tokenizer serves both sides (the LFM2.5-350M decoder); set when the decoder's
    # vocabulary differs (LFM2.5-8B-A1B, 2026-09-25: no token ids in common).
    enc_tok: object = None
    sentinel_str: str = SPLICE_SENTINEL  # the decoder's sentinel token (templates.decoder_sentinel)

    @classmethod
    def from_tokenizer(cls, tok, enc_tok=None) -> ChatRenderer:
        sent_str, sid = decoder_sentinel(tok)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        return cls(
            tok=tok,
            sentinel_id=int(sid),
            eos_id=int(im_end),
            bos_id=int(tok.bos_token_id),
            pad_id=int(tok.pad_token_id if tok.pad_token_id is not None else 0),
            enc_tok=enc_tok,
            sentinel_str=sent_str,
        )

    @property
    def ctx_tok(self):
        """The tokenizer of everything the ENCODER reads."""
        return self.enc_tok if self.enc_tok is not None else self.tok

    @property
    def split_tokenizers(self) -> bool:
        return self.enc_tok is not None

    @property
    def enc_pad_id(self) -> int:
        t = self.ctx_tok
        return int(t.pad_token_id if t.pad_token_id is not None else 0)

    def dec_view(self, enc_ids: torch.Tensor) -> torch.Tensor | None:
        """The DECODER's tokenization of exactly the text ``enc_ids`` covers (decode, re-encode:
        the same span, never a re-cut), for the paths that show a context to the decoder as
        text. None on single-tokenizer runs (the encoder ids ARE decoder ids)."""
        if not self.split_tokenizers:
            return None
        if enc_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        text = self.ctx_tok.decode(enc_ids.tolist())
        return torch.tensor(self.tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)

    def encode_ctx(self, text: str, max_tokens: int | None = None) -> torch.Tensor:
        """Encoder-side ids of ``text`` (context slots, compression prompts, spine leaves)."""
        ids = self.ctx_tok(text, add_special_tokens=False)["input_ids"]
        if max_tokens is not None:
            ids = ids[:max_tokens]
        return torch.tensor(ids, dtype=torch.long)

    def split_segments(self, ids: list[int]) -> list[torch.Tensor]:
        """The rendered prompt cut at EVERY sentinel: K sentinels -> K+1
        segments (prefix, K-1 middles, the instruction tail)."""
        segs: list[list[int]] = [[]]
        for x in ids:
            if x == self.sentinel_id:
                segs.append([])
            else:
                segs[-1].append(x)
        if len(segs) < 2:
            raise ValueError("rendered prompt must contain at least one sentinel")
        return [torch.tensor(g, dtype=torch.long) for g in segs]

    def render_slots(
        self, instruction: str, tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> list[torch.Tensor]:
        """framing "tool_slots": ``instruction`` is a JSON message list with K
        sentinels (each a compressed tool result); returns the K+1 segments.
        ``tools`` (the row's meta.tools) renders the tool list into the system
        message exactly as the agent harness does at inference."""
        msgs = json.loads(instruction)
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("tool_slots framing needs a non-empty JSON list")
        return self.split_segments(self._apply(msgs, tools, enable_thinking))

    def _split(self, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        hits = [i for i, x in enumerate(ids) if x == self.sentinel_id]
        if len(hits) != 1:
            raise ValueError(f"rendered prompt must contain the sentinel once; got {len(hits)}")
        cut = hits[0]
        return (
            torch.tensor(ids[:cut], dtype=torch.long),
            torch.tensor(ids[cut + 1 :], dtype=torch.long),
        )

    def _apply(
        self, msgs: list[dict], tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        """The template rendering with the generation prompt. ``enable_thinking``
        (None = decided from ``msgs``: any assistant reasoning -> True) selects the
        decoder's own thinking format over a non-thinking template
        (``templates.set_no_think_template``); a template without that switch
        ignores it."""
        if enable_thinking is None:
            enable_thinking = has_reasoning(msgs)
        kw = {"tools": tools} if tools else {}
        if enable_thinking:
            # every turn keeps its reasoning: a later user message must not drop an
            # earlier turn's (the stock template's default), or chunks go prefix-inconsistent
            kw["enable_thinking"] = kw["preserve_thinking"] = True
        msgs = with_sentinel(_native_reasoning(msgs), self.sentinel_str)
        ids = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return [int(i) for i in ids]

    def render(
        self,
        framing: str,
        instruction: str,
        tool_name: str = "",
        tool_args: dict | None = None,
        bank: FramingBank | None = None,
        rng: np.random.Generator | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(prefix_ids, instr_ids). With ``bank`` + ``rng`` the chat and tool
        framings are phrased from the bank; without, the stored phrasing.
        ``enable_thinking``: see ``_apply`` (the caller decides from the whole
        row, target included; "chat_think" rows always think)."""
        if framing == "chat_context":
            if bank is not None:
                msgs = bank.chat_messages(rng, instruction)
            else:
                msgs = [{"role": "user", "content": f"Context:\n{SPLICE_SENTINEL}\n{instruction}"}]
        elif framing == "tool_response":
            if bank is not None:
                msgs = bank.tool_messages(rng, instruction, tool_name, tool_args)
            else:
                msgs = [
                    {"role": "user", "content": instruction},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": tool_name or "read",
                                    "arguments": tool_args or {},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": SPLICE_SENTINEL},
                ]
        elif framing in ("chat", "chat_think"):
            # A plain conversation with NO context slot (the replay stream):
            # the whole rendered prompt is the prefix, the instruction part is
            # empty, and the target is the assistant turn that follows.
            # "chat_think": a reasoning-replay turn, rendered in the decoder's
            # own thinking format (templates.set_no_think_template).
            msgs = json.loads(instruction)
            if not isinstance(msgs, list) or not msgs:
                raise ValueError("chat framing needs a non-empty JSON list")
            if any(SPLICE_SENTINEL in str(m.get("content", "")) for m in msgs):
                raise ValueError("chat framing must not contain the sentinel")
            ids = self._apply(msgs, enable_thinking=(
                True if framing == "chat_think" else enable_thinking))
            return torch.tensor(ids, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        elif framing in ("messages", "trajectory"):
            msgs = json.loads(instruction)
            if not isinstance(msgs, list) or not msgs:
                raise ValueError("messages framing needs a non-empty JSON list")
            if sum(SPLICE_SENTINEL in str(m.get("content", "")) for m in msgs) != 1:
                raise ValueError("messages framing needs the sentinel in exactly one message")
        else:
            raise ValueError(f"unknown framing {framing!r}")
        return self._split(self._apply(msgs, enable_thinking=enable_thinking))

    def render_chunk(
        self,
        prefix_msgs: list[dict],
        chunk: list[dict],
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(target_ids, loss_mask) for a chunk of messages continuing
        ``prefix_msgs``: the template rendering of prefix + chunk, minus the
        prefix rendering (with its generation prompt) — so the target starts
        with the first assistant message's content and includes every
        later turn header. loss_mask is True on assistant messages (content,
        tool calls, end-of-turn) and False on tool / user messages. The
        final end-of-turn token is stripped (the collator appends it)."""
        if not chunk or chunk[0].get("role") != "assistant":
            raise ValueError("a trajectory chunk must start with an assistant message")
        if enable_thinking is None:  # one decision for the whole row
            enable_thinking = has_reasoning(prefix_msgs + chunk)
        prev = self._apply(prefix_msgs, tools, enable_thinking)  # ends with the generation prompt
        kw = {"tools": tools} if tools else {}
        if enable_thinking:
            # every turn keeps its reasoning: a later user message must not drop an
            # earlier turn's (the stock template's default), or chunks go prefix-inconsistent
            kw["enable_thinking"] = kw["preserve_thinking"] = True
        ids: list[int] = []
        mask: list[bool] = []
        for j, m in enumerate(chunk):
            full = self.tok.apply_chat_template(
                with_sentinel(_native_reasoning(prefix_msgs + chunk[: j + 1]), self.sentinel_str),
                tokenize=True, add_generation_prompt=False, **kw,
            )
            if not isinstance(full, list):
                full = full["input_ids"]
            full = [int(i) for i in full]
            if full[: len(prev)] != prev:
                raise ValueError("chat template is not prefix-consistent for this chunk")
            span = full[len(prev) :]
            ids += span
            mask += [m.get("role") == "assistant"] * len(span)
            prev = full
        # Strip trailing whitespace tokens and the closing end-of-turn.
        while ids and self.tok.decode(ids[-1:]).strip() == "" and ids[-1] != self.eos_id:
            ids.pop()
            mask.pop()
        if ids and ids[-1] == self.eos_id:
            ids.pop()
            mask.pop()
        if max_tokens is not None:
            self.last_chunk_truncated = len(ids) > max_tokens
            ids, mask = ids[:max_tokens], mask[:max_tokens]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def encode(self, text: str, max_tokens: int | None = None) -> torch.Tensor:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if max_tokens is not None:
            ids = ids[:max_tokens]
        return torch.tensor(ids, dtype=torch.long)

    def encode_target(self, text: str, max_tokens: int | None) -> tuple[torch.Tensor, bool]:
        """``encode`` plus whether the cap cut the text. A truncated target must
        NOT teach the end-of-turn token at the cut (2026-09-14 audit: 9-13% of
        tool_slots targets were cut and then ended with a loss-bearing
        <|im_end|>, i.e. "stop mid-sentence")."""
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        truncated = max_tokens is not None and len(ids) > max_tokens
        if truncated:
            ids = ids[:max_tokens]
        return torch.tensor(ids, dtype=torch.long), bool(truncated)

    def span_mask(self, text: str, pieces: list[str], max_tokens: int | None = None):
        """Bool mask over ``encode(text)`` marking the tokens that overlap any
        of ``pieces`` found verbatim in ``text`` (character offsets from the
        tokenizer — never re-encode substrings, BPE is not concat-distributive).
        None when the tokenizer has no offset mapping or nothing matches."""
        try:
            enc = self.ctx_tok(text, add_special_tokens=False, return_offsets_mapping=True)
        except (TypeError, NotImplementedError):
            return None
        offs = enc.get("offset_mapping")
        if not offs:
            return None
        if max_tokens is not None:
            offs = offs[:max_tokens]
        spans = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            start = text.find(piece)
            if start >= 0:
                spans.append((start, start + len(piece)))
        if not spans:
            return None
        mask = torch.zeros(len(offs), dtype=torch.bool)
        for i, (a, b) in enumerate(offs):
            if b <= a:
                continue
            for s0, s1 in spans:
                if a < s1 and b > s0:
                    mask[i] = True
                    break
        return mask if bool(mask.any()) else None


_TABLE_CACHE: dict[str, pa.Table] = {}


def _load_store_table(path: str) -> pa.Table:
    """The store's Arrow table, loaded once per process and shared by every
    TaskDataset over the same file (train / eval / held-out arms used to
    decode the same parquet three times: 8.4 GB of anonymous memory for the
    five parquet stores on 2026-09-08).

    ``.arrow`` (IPC) files are memory-mapped and zero-copy — page-cache
    backed, no anonymous memory, shared by every process mapping them.
    Parquet is decoded into anonymous buffers."""
    table = _TABLE_CACHE.get(path)
    if table is None:
        if path.endswith(".arrow"):
            table = pa.ipc.open_file(pa.memory_map(path)).read_all().select(COLUMNS)
        else:
            table = pq.read_table(path, columns=COLUMNS, memory_map=True)
        _TABLE_CACHE[path] = table
    return table


class TaskDataset(Dataset):
    """One task store (optionally filtered by split) -> Samples with rendered framings."""

    def __init__(
        self,
        path: str,
        renderer: ChatRenderer,
        split: str = "train",
        max_ctx_tokens: int = 2048,
        max_target_tokens: int = 512,
        seed: int = 0,
        epoch_size: int | None = None,
        max_prefix_tokens: int | None = 1536,
        bank: FramingBank | None = None,
        sibling_prompts: bool = False,
    ) -> None:
        # Rows stay in the Arrow table (memory-mapped, shared copy-on-write
        # by forked dataloader workers) and are decoded per access; the
        # earlier to_pydict() held every row as Python strings in EVERY
        # worker (a 460k-row store would have cost ~15 GB per process).
        table = _load_store_table(str(path))
        if split:
            keep = pc.equal(table.column("split"), split).to_numpy(zero_copy_only=False)
            self.index = np.flatnonzero(keep).astype(np.int64)
        else:
            self.index = np.arange(table.num_rows, dtype=np.int64)
        if self.index.shape[0] == 0:
            raise ValueError(f"{path}: no rows for split {split!r}")
        self.table = table
        self.n = int(self.index.shape[0])
        # Character lengths for length_hint (vectorised once, small).
        self._len = {
            c: pc.utf8_length(table.column(c)).to_numpy(zero_copy_only=False)[self.index]
            for c in ("context", "target", "instruction")
        }
        fam = table.column("family").take(pa.array(self.index)).unique().to_pylist()
        self.renderer = renderer
        self.max_ctx = int(max_ctx_tokens)
        self.max_target = int(max_target_tokens)
        self.max_prefix = int(max_prefix_tokens) if max_prefix_tokens else None
        self.skipped_long_prefix = 0
        self.skipped_unrenderable = 0
        self.seed = int(seed)
        self.epoch_size = int(epoch_size) if epoch_size else self.n
        self.families = sorted(fam)
        # Phrasing bank (framing_bank.py); None = the stored phrasing.
        self.bank = bank
        # Sibling prompts (the prompt-contrast loss): rows of the SAME document
        # (meta.doc) grouped once; a row draws another question about its
        # document as Sample.alt_prompt_ids. None where the store carries no
        # meta.doc or a document has a single row.
        self._doc_group: np.ndarray | None = None
        self._group_rows: list[np.ndarray] = []
        if sibling_prompts:
            # Document key: meta.doc where the builder wrote one, else the
            # context text itself (SQuAD-style stores repeat a paragraph
            # under several questions).
            docs = table.column("meta").take(pa.array(self.index)).to_pylist()
            keys = [json.loads(m).get("doc") if m else None for m in docs]
            if all(k is None for k in keys):
                keys = table.column("context").take(pa.array(self.index)).to_pylist()
            by: dict[str, list[int]] = {}
            for pos, k in enumerate(keys):
                if k is not None:
                    by.setdefault(str(k), []).append(pos)
            self._doc_group = np.full(self.n, -1, dtype=np.int64)
            for rows in by.values():
                if len(rows) < 2:
                    continue
                self._doc_group[rows] = len(self._group_rows)
                self._group_rows.append(np.asarray(rows, dtype=np.int64))
            if not self._group_rows:
                raise ValueError(f"{path}: sibling_prompts requested but no multi-row meta.doc")

    def sibling(self, i: int, rng: np.random.Generator) -> int | None:
        """Another row of row ``i``'s document, or None."""
        if self._doc_group is None or self._doc_group[i] < 0:
            return None
        rows = self._group_rows[int(self._doc_group[i])]
        j = int(rows[int(rng.integers(0, rows.shape[0] - 1))])
        return j if j != i else int(rows[-1])

    def __len__(self) -> int:
        return self.epoch_size

    def row(self, i: int) -> dict:
        """Row ``i`` of this split as a dict of Python values."""
        j = int(self.index[int(i) % self.n])
        return {c: self.table.column(c)[j].as_py() for c in COLUMNS}

    def length_hint(self, index: int) -> int:
        """Approximate decoder-sequence length (chars / 4, clipped like the
        real encode) — good enough to bucket rows by length without
        tokenising."""
        i = int(index) % self.n
        ctx = min(int(self._len["context"][i]) // 4, self.max_ctx)
        tgt = min(int(self._len["target"][i]) // 4, self.max_target or 1 << 30)
        pre = int(self._len["instruction"][i]) // 4 + 48
        if self.max_prefix is not None:
            pre = min(pre, self.max_prefix)
        return int(ctx + tgt + pre)

    def _slots_item(self, i: int, r: dict) -> Sample | None:
        """framing "tool_slots": context / prompt are JSON lists (one per
        slot), instruction a message list with one sentinel per slot, target
        a JSON message list (an assistant chunk, rendered with the loss on
        the assistant tokens) or plain text."""
        from .overview import parse_node, parse_overview

        ctxs = json.loads(r["context"])
        prompts = json.loads(r["prompt"]) if r["prompt"] else [""] * len(ctxs)
        # An always-on repo overview is the FIRST slot of the conversation; its
        # reps come from the per-repo root cache, so nothing is tokenised for it.
        cached = parse_overview(ctxs[0]) if ctxs else None
        if any(parse_overview(c) for c in ctxs[1:]):
            raise ValueError(f"tool_slots row {i}: an overview slot must be the first slot")
        # ``browse`` results: directory nodes of the same cached tree, anywhere
        # after the overview. Filled from the node cache like the overview.
        nodes = [parse_node(c) if j else None for j, c in enumerate(ctxs)]
        if any(nodes) and not cached:
            raise ValueError(f"tool_slots row {i}: a browse slot without the repo overview")
        meta = json.loads(r["meta"]) if r["meta"] else {}
        tools = meta.get("tools") or None
        segs = self.renderer.render_slots(r["instruction"], tools, row_has_reasoning(r))
        if len(segs) != len(ctxs) + 1 or len(prompts) != len(ctxs):
            raise ValueError(
                f"tool_slots row {i}: {len(ctxs)} contexts, {len(prompts)} prompts, "
                f"{len(segs) - 1} sentinels"
            )
        n_pre = sum(int(g.shape[0]) for g in segs)
        if self.max_prefix is not None and n_pre > self.max_prefix:
            # Enforced like the other framings (2026-09-14 audit: it was only
            # counted): the caller hops to the next row.
            self.skipped_long_prefix += 1
            return None
        # Cons-mode compaction slots: a SPINE of leaves (oldest first), encoded
        # node by node by the trainer (BgKIT2Core.encode_spine_slots), never as text.
        from ..agent.context import compaction_leaves, compaction_payload

        spines: list[list[torch.Tensor] | None] = []
        for c in ctxs:
            sp = compaction_payload(c)
            if sp is None:
                spines.append(None)
                continue
            leaves = [self.renderer.encode_ctx(x) for x in compaction_leaves(sp)]
            for leaf in leaves:
                self._check_slot_len(int(leaf.shape[0]))
            spines.append(leaves)
        tgt_mask = None
        tgt_truncated = False
        self.renderer.last_chunk_truncated = False
        if r["target"].lstrip().startswith("["):
            try:
                tgt, tgt_mask = self.renderer.render_chunk(
                    json.loads(r["instruction"]), json.loads(r["target"]), self.max_target, tools
                )
            except ValueError:
                # A chunk the template cannot render prefix-consistently (rare;
                # P26 died on one at step ~17000). Hop like an over-long row
                # instead of killing the dataloader worker and the whole run —
                # one unusable row must never cost hours of training.
                self.skipped_unrenderable += 1
                return None
            tgt_truncated = bool(self.renderer.last_chunk_truncated)
        else:
            tgt, tgt_truncated = self.renderer.encode_target(r["target"], self.max_target)
        enc = [
            torch.zeros(0, dtype=torch.long) if ((j == 0 and cached) or nodes[j] or spines[j])
            else self._encode_slot(c)
            for j, c in enumerate(ctxs)
        ]
        pr = [self.renderer.encode_ctx(p) if p else None for p in prompts]
        # the eval harness's price of every slot (ratio.slot_reps_mode; slot_pricing)
        from .slot_pricing import row_slot_reps

        unpriced = [bool((j == 0 and cached) or nodes[j] or spines[j]) for j in range(len(ctxs))]
        slot_reps, slot_src = row_slot_reps(meta, prompts, [int(e.shape[0]) for e in enc],
                                            unpriced)
        return Sample(
            slot_reps=slot_reps,
            slot_reps_src=slot_src,
            cached_slot=cached,
            ctx_ids=enc[0],
            target_ids=tgt,
            task=r["family"],
            store=-1,
            doc=i,
            target_truncated=tgt_truncated,
            prompt_ids=pr[0],
            prefix_ids=segs[0],
            instr_ids=segs[-1],
            target_loss_mask=tgt_mask,
            extra_ctx=enc[1:],
            extra_cached=[(n[0], n[1]) if n else None for n in nodes[1:]] if any(nodes) else None,
            extra_prompt=pr[1:],
            mid_ids=segs[1:-1],
            spine_slot=spines[0] if spines else None,
            extra_spine=spines[1:] if any(spines[1:]) else None,
            dec_ctx_ids=self.renderer.dec_view(enc[0]),
            dec_extra_ctx=(
                [self.renderer.dec_view(e) for e in enc[1:]] if self.renderer.split_tokenizers else None
            ),
        )

    # No SILENT truncation (2026-09-22): a slot longer than max_ctx is counted
    # (``over_long_slots``, read by the store validator); with ``strict_slot_len``
    # it raises instead of being cut. Default off: existing stores still carry
    # over-long slots that the encoder has always cut at max_ctx.
    strict_slot_len: bool = False
    over_long_slots: int = 0

    def _check_slot_len(self, n: int) -> None:
        if n > self.max_ctx:
            self.over_long_slots += 1
            if self.strict_slot_len:
                raise ValueError(f"slot of {n} tokens exceeds max_ctx {self.max_ctx}")

    def _encode_slot(self, text: str) -> torch.Tensor:
        ids = self.renderer.encode_ctx(text)
        self._check_slot_len(int(ids.shape[0]))
        return ids[: self.max_ctx]

    def _hop_over_long(self, index: int, i: int, attempt: int) -> int:
        """Next row to try after an over-long prefix.

        Over-long rows CLUSTER by source in a concatenated store — the
        2026-09-16 pre-launch scan found a run of 872 consecutive over-long
        rows in ``tool_slots_merged`` — so stepping by one walks the whole
        region and exhausts the attempt budget. Step by one first (the
        neighbour is usually fine) and then jump out of the region, the same
        shape the unrenderable-chunk hop below already uses.
        """
        hop = (
            1
            if attempt < 4
            else int(
                np.random.default_rng([self.seed, 4243, int(index), attempt]).integers(1, self.n)
            )
        )
        return (i + hop) % self.n

    def __getitem__(self, index: int) -> Sample:
        i = int(index) % self.n
        # One draw stream per (dataset, index): the same row renders the same
        # phrasing every time within a dataset.
        rng = np.random.default_rng([self.seed, 977, int(index)]) if self.bank else None
        for _attempt in range(_LONG_PREFIX_ATTEMPTS):
            r = self.row(i)
            if r["framing"] == "tool_slots":
                item = self._slots_item(i, r)
                if item is not None:
                    return item
                # over-long prefix (already counted by _slots_item): hop away
                i = self._hop_over_long(index, i, _attempt)
                continue
            args = json.loads(r["tool_args"]) if r["tool_args"] else None
            prefix, instr = self.renderer.render(
                r["framing"], r["instruction"], r["tool_name"], args, self.bank, rng,
                enable_thinking=row_has_reasoning(r),
            )
            if self.max_prefix is None or prefix.shape[0] + instr.shape[0] <= self.max_prefix:
                break
            # Over-long decoder prefix (e.g. a compaction anchor): take
            # another row instead so the batch shape stays bounded.
            self.skipped_long_prefix += 1
            i = self._hop_over_long(index, i, _attempt)
        else:
            raise ValueError(
                f"{self.n} rows: {_LONG_PREFIX_ATTEMPTS} over-long prefixes in a row "
                f"from index {index} (framing {r['framing']})"
            )
        ctx = self.renderer.encode_ctx(r["context"], self.max_ctx)
        tgt_mask = None
        tgt_truncated = False
        self.renderer.last_chunk_truncated = False
        if r["framing"] == "trajectory":
            for _attempt in range(64):
                try:
                    tgt, tgt_mask = self.renderer.render_chunk(
                        json.loads(r["instruction"]), json.loads(r["target"]), self.max_target
                    )
                    tgt_truncated = bool(self.renderer.last_chunk_truncated)
                    break
                except ValueError:
                    # A chunk the template cannot render as an extension of
                    # its prefix (BPE merge at a message boundary): step to
                    # the next row, then jump elsewhere (bad rows cluster by
                    # source in a concatenated store).
                    self.skipped_unrenderable += 1
                    hop = (
                        1
                        if _attempt < 4
                        else int(
                            np.random.default_rng([self.seed, 4242, int(index), _attempt]).integers(
                                1, self.n
                            )
                        )
                    )
                    i = (i + hop) % self.n
                    r = self.row(i)
                    args = json.loads(r["tool_args"]) if r["tool_args"] else None
                    prefix, instr = self.renderer.render(
                        r["framing"], r["instruction"], r["tool_name"], args, self.bank, rng,
                        enable_thinking=row_has_reasoning(r),
                    )
                    ctx = self.renderer.encode_ctx(r["context"], self.max_ctx)
            else:
                raise ValueError(f"{self.n} rows: 64 unrenderable trajectory chunks in a row")
        elif r["framing"] in ("chat", "chat_think") and (
            turn := json_message_list(r["target"])
        ) is not None:
            # A replay turn stored as a JSON message list (an assistant turn
            # that carries tool calls): rendered through the template like a
            # trajectory chunk, loss on the content, the rendered call and
            # the end-of-turn — never a text preamble cut before its call.
            tgt, tgt_mask = self.renderer.render_chunk(
                json.loads(r["instruction"]), turn, self.max_target,
                # same decision as the prompt's (render: row_has_reasoning); None = from the row
                enable_thinking=True if r["framing"] == "chat_think" else None,
            )
        else:
            tgt, tgt_truncated = self.renderer.encode_target(r["target"], self.max_target)
        ptxt = (
            self.bank.prompt(rng, r["family"], r["prompt"], r["instruction"])
            if self.bank
            else r["prompt"]
        )
        prompt = self.renderer.encode_ctx(ptxt) if ptxt else None
        teacher_ctx = None
        gold_span = None
        if r["meta"]:
            gw = json.loads(r["meta"]).get("gold_window")
            if gw:
                teacher_ctx = self.renderer.encode(gw, self.max_ctx)
                # the supporting facts as token spans inside the document
                pieces = [x + "." if not x.endswith(".") else x for x in gw.split(". ") if x]
                gold_span = self.renderer.span_mask(r["context"], pieces, self.max_ctx)
        alt = None
        sibling = None
        if self._doc_group is not None:
            srng = np.random.default_rng([self.seed, 7331, int(index)])
            j = self.sibling(i, srng)
            if j is not None:
                sib = self.row(j)
                atxt = (
                    self.bank.prompt(srng, sib["family"], sib["prompt"], sib["instruction"])
                    if self.bank
                    else sib["prompt"]
                )
                # A no-prompt draw carries no question to contrast against.
                alt = self.renderer.encode_ctx(atxt) if atxt else None
                if alt is not None and sib["framing"] != "trajectory":
                    # The sibling as a FULL sample on the same context (its
                    # own framing, prompt and target): the paired contrast
                    # needs its answer, not only its prompt.
                    s_args = json.loads(sib["tool_args"]) if sib["tool_args"] else None
                    s_prefix, s_instr = self.renderer.render(
                        sib["framing"],
                        sib["instruction"],
                        sib["tool_name"],
                        s_args,
                        self.bank,
                        srng,
                    )
                    if self.max_prefix is None or (
                        s_prefix.shape[0] + s_instr.shape[0] <= self.max_prefix
                    ):
                        sibling = Sample(
                            dec_ctx_ids=self.renderer.dec_view(ctx),
                            ctx_ids=ctx,
                            target_ids=self.renderer.encode(sib["target"], self.max_target),
                            task=sib["family"],
                            store=-1,
                            doc=j,
                            prompt_ids=alt,
                            prefix_ids=s_prefix,
                            instr_ids=s_instr,
                        )
        return Sample(
            ctx_ids=ctx,
            target_ids=tgt,
            task=r["family"],
            store=-1,
            doc=i,
            target_truncated=tgt_truncated,
            prompt_ids=prompt,
            prefix_ids=prefix,
            instr_ids=instr,
            target_loss_mask=tgt_mask,
            alt_prompt_ids=alt,
            sibling=sibling,
            teacher_ctx_ids=teacher_ctx,
            gold_span_mask=gold_span,
            dec_ctx_ids=self.renderer.dec_view(ctx),
        )
