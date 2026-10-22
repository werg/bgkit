"""bgkit: read compressed context through a language model's embedding layer.

Standalone inference for the bgkit release (needs only ``torch`` and ``transformers``):

    from modeling_bgkit import Bgkit

    bg = Bgkit.from_pretrained("gabrielpickard/bgkit-350m", device="cuda")
    print(bg.answer(context=long_text, question="Who founded the company?", budget=256))

The repository holds three parts:

* ``encoder/``  -- LFM2.5-Encoder-350M with the bgkit encoder adapter merged (Liquid's layout).
* ``decoder/``  -- LFM2.5-350M with the bgkit decoder adapter merged: a plain ``Lfm2ForCausalLM``.
* ``compressor.safetensors`` + ``bgkit_config.json`` -- the scoring head, the prompt separator,
  the survive embedding, the projection block, the projection MLP and the interface constant.

A slot is a compressed text: the encoder reads ``[prompt][separator][text]`` bidirectionally,
scores the text positions after layer ``head_layer``, keeps the top ``k = ceil(ratio * n)``,
marks them with the survive embedding, runs the remaining layers, gathers the kept positions,
and maps them into the decoder's embedding space. The decoder reads the vectors in place of a
``<|reserved_6|>`` placeholder in its chat-templated input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2Model

SYSTEM_QA = "You answer questions about the provided context. Be brief: give only the answer."


def _resolve(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo))


class _NonCausalShortConv(nn.Module):
    """LFM2 short convolution with symmetric padding (each position sees both neighbours)."""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.in_proj, self.conv, self.out_proj = conv.in_proj, conv.conv, conv.out_proj

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                **_) -> torch.Tensor:
        x = hidden_states
        if attention_mask is not None:
            x = x * attention_mask.to(x.dtype)[:, :, None]
        n = x.shape[1]
        b, c, h = self.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
        w = self.conv.weight
        out = F.conv1d(b * h, w, self.conv.bias, padding=w.shape[-1] // 2, groups=h.shape[1])
        return self.out_proj((c * out[..., :n]).transpose(-1, -2).contiguous())


def _pad_mask(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """(B, 1, L, L) additive mask hiding padded keys; None without padding."""
    if mask is None or bool(mask.all()):
        return None
    m = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    m = m.masked_fill(mask == 0, torch.finfo(dtype).min)
    return m[:, None, None, :].expand(-1, 1, mask.shape[1], -1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class BgkitEncoder(nn.Module):
    """Prompt-conditioned compressor: text -> (k, hidden) vectors in the decoder's embedding space."""

    def __init__(self, backbone: Lfm2Model, cfg: dict) -> None:
        super().__init__()
        d = backbone.config.hidden_size
        self.cfg = cfg
        self.backbone = backbone
        for layer in backbone.layers:  # run the backbone bidirectionally, this instance only
            if layer.is_attention_layer:
                layer.self_attn.is_causal = False
            else:
                layer.conv = _NonCausalShortConv(layer.conv)
        self.head = nn.Sequential(nn.Linear(d, cfg["head_inner_dim"]), nn.GELU(), nn.Linear(cfg["head_inner_dim"], 1))
        self.survive_embedding = nn.Parameter(torch.zeros(d))
        self.prompt_separator = nn.Parameter(torch.zeros(d))
        self.projection_block = Lfm2DecoderLayer(backbone.config, layer_idx=cfg["projection_block_layer"])
        self.projection_block.self_attn.is_causal = False
        m = cfg["projection_mult"]
        self.projection = nn.Sequential(_RMSNorm(d), nn.Linear(d, d * m), nn.GELU(), nn.Linear(d * m, d))

    @classmethod
    def from_pretrained(cls, root: str | Path, device: str | torch.device = "cpu") -> BgkitEncoder:
        root = _resolve(str(root))
        cfg = json.loads((root / "bgkit_config.json").read_text())
        ecfg = Lfm2Config.from_pretrained(root / "encoder")
        ecfg._attn_implementation = "sdpa"
        backbone = Lfm2Model(ecfg)
        state = load_file(root / "encoder" / "model.safetensors")
        backbone.load_state_dict({k.removeprefix("lfm2."): v for k, v in state.items() if k.startswith("lfm2.")})
        enc = cls(backbone, cfg)
        missing, unexpected = enc.load_state_dict(load_file(root / "compressor.safetensors"), strict=False)
        if unexpected or any(not k.startswith("backbone.") for k in missing):
            raise RuntimeError(f"compressor.safetensors mismatch: {missing[:5]} {unexpected[:5]}")
        return enc.to(device).eval()

    def ratio_for(self, n_tokens: int, n_vectors: int) -> float:
        lo, hi = self.cfg["min_ratio"], self.cfg["max_ratio"]
        return float(min(hi, max(lo, n_vectors / max(n_tokens, 1))))

    @torch.no_grad()
    def forward(self, doc_ids: torch.Tensor, prompt_ids: torch.Tensor, ratio: float) -> torch.Tensor:
        """One document (n,) under one prompt (p,) -> (k, hidden) vectors, k = ceil(ratio * n)."""
        emb = self.backbone.embed_tokens
        dev = emb.weight.device
        doc_ids, prompt_ids = doc_ids.to(dev), prompt_ids.to(dev)
        doc = emb(doc_ids)[None]
        sep = self.prompt_separator.to(doc.dtype)[None, None, :]
        x = torch.cat([emb(prompt_ids)[None], sep, doc], dim=1)
        offset, n = prompt_ids.shape[0] + 1, doc_ids.shape[0]
        seq = x.shape[1]
        pos = torch.arange(seq, device=dev)[None]
        pe = self.backbone.rotary_emb(x, position_ids=pos)
        r = float(torch.tensor(ratio, dtype=torch.float32))  # the ratio as trained: float32
        k = min(max(math.ceil(n * r - 1e-9), self.cfg["min_survivors"]), n)
        keep = None
        for i, layer in enumerate(self.backbone.layers):
            if i == self.cfg["head_layer"]:
                raw = self.head(x).squeeze(-1).float()[0, offset:]
                z = (raw - raw.mean()) / (raw.var(unbiased=False) + 1e-6).sqrt()
                top = z.argsort(descending=True)[:k]
                keep = torch.zeros(seq, dtype=torch.bool, device=dev)
                keep[offset + top] = True
                x = x + keep[None, :, None].to(x.dtype) * self.survive_embedding.to(x.dtype)
            x = layer(x, attention_mask=None, position_embeddings=pe, position_ids=pos, past_key_values=None)
        x = self.backbone.embedding_norm(x)
        positions = keep.nonzero().squeeze(-1)[None]  # document order
        surv = x[:, positions[0]]
        surv = self.projection_block(
            surv, attention_mask=None, position_ids=positions, past_key_values=None,
            position_embeddings=self.backbone.rotary_emb(surv, position_ids=positions))
        with torch.autocast(device_type=dev.type, enabled=False):
            y = self.projection(surv.float())[0]
            t = self.cfg["interface_row_norm"]
            y = y / (y.norm(dim=-1, keepdim=True) + max(1e-5, 0.05 * t)) * t
        return y


class Bgkit:
    """Encoder + decoder + tokenizer, with helpers to compress text and generate from slots."""

    def __init__(self, encoder: BgkitEncoder, decoder, tokenizer, cfg: dict) -> None:
        self.encoder, self.decoder, self.tok, self.cfg = encoder, decoder, tokenizer, cfg
        self.sentinel = cfg["slot_token"]
        self.sentinel_id = tokenizer.convert_tokens_to_ids(self.sentinel)

    @classmethod
    def from_pretrained(cls, repo: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Bgkit:
        root = _resolve(repo)
        cfg = json.loads((root / "bgkit_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(root / "decoder")
        dec = AutoModelForCausalLM.from_pretrained(root / "decoder", dtype=dtype).to(device).eval()
        enc = BgkitEncoder.from_pretrained(root, device=device)
        return cls(enc, dec, tok, cfg)

    @property
    def device(self) -> torch.device:
        return self.decoder.get_input_embeddings().weight.device

    def _ids(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)

    @torch.no_grad()
    def compress(self, text: str, prompt: str | None = None, n_vectors: int | None = None,
                 ratio: float | None = None) -> torch.Tensor:
        """Compress ``text`` under ``prompt`` (what the reader needs from it) into vectors.
        Give the number of vectors or a keep ratio (1/16 = 16x compression); default 16x."""
        ids = self._ids(text)[: self.cfg["max_slot_tokens"]]
        if ratio is None:
            ratio = self.encoder.ratio_for(ids.shape[0], n_vectors or max(4, ids.shape[0] // 16))
        return self._encode(ids, prompt or self.cfg["default_prompt"], ratio)

    def _encode(self, ids: torch.Tensor, prompt: str, ratio: float) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            return self.encoder(ids, self._ids(prompt), ratio)

    @torch.no_grad()
    def generate(self, messages: list[dict], slots: list[torch.Tensor], max_new_tokens: int = 64) -> str:
        """``messages`` (chat format) contain one ``<|reserved_6|>`` per entry of ``slots``, in order."""
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        cuts = [i for i, t in enumerate(ids) if t == self.sentinel_id]
        if len(cuts) != len(slots):
            raise ValueError(f"{len(cuts)} slot tokens in the messages but {len(slots)} slots")
        emb = self.decoder.get_input_embeddings()
        parts, start = [], 0
        for c, s in zip(cuts, slots, strict=True):
            parts += [emb(torch.tensor(ids[start:c], device=self.device)), s.to(self.device, emb.weight.dtype)]
            start = c + 1
        parts.append(emb(torch.tensor(ids[start:], device=self.device)))
        x = torch.cat(parts)[None]
        out = self.decoder.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=self.device, dtype=torch.long),
                                    max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                    eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def answer(self, context: str | list[str], question: str, budget: int = 512,
               max_new_tokens: int = 32) -> str:
        """Answer ``question`` from ``context`` (a text or a list of documents) with at most about
        ``budget`` decoder positions for the context. Long contexts are cut into chunks of up to
        ``max_slot_tokens`` tokens, each compressed under the question into its own slot."""
        docs = [context] if isinstance(context, str) else list(context)
        chunks = self._chunks(docs)
        total = sum(len(c) for c in chunks)
        ratio = min(max(total / budget, 1.0), self.cfg["max_compression"])
        slots = [self._encode(c, question, self.encoder.ratio_for(len(c), max(4, round(len(c) / ratio))))
                 for c in chunks]
        messages = [{"role": "system", "content": SYSTEM_QA},
                    {"role": "user", "content": "Context:\n" + "\n".join([self.sentinel] * len(slots))
                     + f"\n\nQuestion: {question}"}]
        return self.generate(messages, slots, max_new_tokens=max_new_tokens)

    def _chunks(self, docs: list[str]) -> list[torch.Tensor]:
        """Documents -> token chunks of <= max_slot_tokens: short neighbours packed together
        (joined by a blank line), long documents split."""
        size, sep = self.cfg["chunk_tokens"], self._ids("\n\n").tolist()
        out: list[list[int]] = []
        cur: list[int] = []
        for d in docs:
            ids = self._ids(d).tolist()
            if len(ids) > size:
                if cur:
                    out.append(cur)
                    cur = []
                out += [ids[i : i + size] for i in range(0, len(ids), size)]
                continue
            if cur and len(cur) + len(sep) + len(ids) > size:
                out.append(cur)
                cur = []
            cur = [*cur, *(sep if cur else []), *ids]
        if cur:
            out.append(cur)
        # re-tokenise each chunk from its text, as the evaluation harness did
        return [self._ids(self.tok.decode(c)) for c in out]
