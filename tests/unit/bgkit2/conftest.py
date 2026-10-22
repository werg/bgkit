"""Tiny random LFM2 models so every bgkit2 test runs on CPU in seconds."""

from __future__ import annotations

import pytest
import torch
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

VOCAB = 257
HIDDEN = 32
LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "full_attention", "conv"]


def tiny_config(**overrides) -> Lfm2Config:
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=list(LAYER_TYPES),
        conv_L_cache=3,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        use_cache=False,
    )
    kwargs.update(overrides)
    cfg = Lfm2Config(**kwargs)
    cfg._attn_implementation = "eager"
    return cfg


@pytest.fixture
def tiny_lfm2_model() -> Lfm2Model:
    torch.manual_seed(0)
    return Lfm2Model(tiny_config())


@pytest.fixture
def tiny_causal_lm() -> Lfm2ForCausalLM:
    torch.manual_seed(1)
    lm = Lfm2ForCausalLM(tiny_config())
    lm.config.use_cache = False
    return lm


@pytest.fixture
def small_ids():
    torch.manual_seed(2)
    return torch.randint(3, VOCAB, (2, 12))
