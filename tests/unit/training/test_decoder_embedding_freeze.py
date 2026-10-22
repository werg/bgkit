"""The tied-embedding freeze that stops the Phase-2 language collapse.

Wide-net training took the decoder from PPL 33 to 671 to 2585 on held-out
plain text. Qwen3.5 ties lm_head to embed_tokens, so every softmax step
rewrote the input embedding of all 248,320 tokens to fit ~107 loss-bearing
tool-call tokens per sample. See KRKBTrainer._freeze_decoder_embeddings.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _Backbone(torch.nn.Module):
    def __init__(self, tied: bool = True):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        if tied:
            self.head.weight = self.embed.weight
        self.mlp = torch.nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


class _Dec:
    def __init__(self, tied: bool = True):
        self.backbone = _Backbone(tied)


def _trainer(dec, **cfg):
    cfg.setdefault("freeze_decoder_embeddings", True)
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._decoders_by_family = {"qwen35": dec} if not isinstance(dec, list) else {
        f"d{i}": d for i, d in enumerate(dec)
    }
    t.step_cfg = cfg
    return t


def test_freeze_stops_embedding_and_tied_head_but_not_the_backbone():
    dec = _Dec(tied=True)
    _trainer(dec)._freeze_decoder_embeddings()
    assert not dec.backbone.embed.weight.requires_grad
    assert not dec.backbone.head.weight.requires_grad
    # The rest of the decoder must still train — it has to adapt to the splice.
    assert dec.backbone.mlp.weight.requires_grad


def test_untied_head_is_frozen_too():
    """An untied lm_head is still a 248k-row output layer fitted to a narrow
    task distribution; freeze it on the same grounds."""
    dec = _Dec(tied=False)
    _trainer(dec)._freeze_decoder_embeddings()
    assert not dec.backbone.embed.weight.requires_grad
    assert not dec.backbone.head.weight.requires_grad


def test_every_round_robin_family_is_covered():
    decs = [_Dec(), _Dec()]
    _trainer(decs)._freeze_decoder_embeddings()
    assert all(not d.backbone.embed.weight.requires_grad for d in decs)


def test_default_is_off_because_freezing_measured_worse():
    """Matched 700-step runs: freeze OFF holds PPL 31.2 (base 30.4), freeze ON
    sits at 41.1. The per-group-LR fix alone prevents the collapse, so the
    freeze costs ~10 perplexity for no protection."""
    dec = _Dec()
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._decoders_by_family = {"qwen35": dec}
    t.step_cfg = {}  # no key set -> default
    t._freeze_decoder_embeddings()
    assert dec.backbone.embed.weight.requires_grad
