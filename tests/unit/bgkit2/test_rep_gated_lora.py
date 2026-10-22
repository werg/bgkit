"""Rep-gated LoRA: without reps the adapted decoder equals the base exactly;
with reps, positions before the slot are untouched and later ones change."""

import torch
from transformers import Lfm2ForCausalLM

from bgkit2.models.decoder import LoraSpec, SpliceBatch, SpliceDecoder

from .conftest import tiny_config


def _decoder(rep_gated: bool) -> SpliceDecoder:
    torch.manual_seed(0)
    dec = SpliceDecoder(
        Lfm2ForCausalLM(tiny_config()),
        lora=LoraSpec(rank=4, alpha=8, layers=[0, 1], rep_gated=rep_gated),
    )
    # peft zero-inits lora_B; give the adapter a real delta.
    for n, p in dec.lm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    dec.eval()
    return dec


def _hidden(dec: SpliceDecoder, batch: SpliceBatch) -> torch.Tensor:
    with dec.rep_gated(batch):
        return dec._hidden(inputs_embeds=batch.inputs_embeds, attention_mask=batch.attention_mask)


def test_no_reps_is_bit_identical_to_base():
    dec = _decoder(rep_gated=True)
    prefix = [torch.tensor([3, 4, 5, 6])]
    suffix = [torch.tensor([7, 8, 9])]
    b = dec.build_batch(prefix, None, None, suffix)
    with torch.no_grad():
        h_gated = _hidden(dec, b)
        with dec.disable_adapter():
            h_base = _hidden(dec, b)
    assert torch.equal(h_gated, h_base)
    # An ungated adapter with the same weights does change the output.
    dec2 = _decoder(rep_gated=False)
    with torch.no_grad():
        h_full = dec2._hidden(inputs_embeds=b.inputs_embeds, attention_mask=b.attention_mask)
        with dec2.disable_adapter():
            h_base2 = dec2._hidden(inputs_embeds=b.inputs_embeds, attention_mask=b.attention_mask)
    assert not torch.allclose(h_full, h_base2)


def test_reps_change_only_positions_from_the_slot_on():
    dec = _decoder(rep_gated=True)
    d = dec.embed_tokens.weight.shape[1]
    prefix = [torch.tensor([3, 4, 5, 6]), torch.tensor([3, 4])]
    suffix = [torch.tensor([7, 8, 9]), torch.tensor([7, 8, 9, 10])]
    reps = torch.randn(2, 3, d) * dec.embed_row_norm
    mask = torch.tensor([[True, True, True], [True, True, False]])
    b = dec.build_batch(prefix, reps, mask, suffix)
    with torch.no_grad():
        h_gated = _hidden(dec, b)
        with dec.disable_adapter():
            h_base = _hidden(dec, b)
    for i in range(2):
        st = int(b.slot_start[i])
        assert torch.allclose(h_gated[i, :st], h_base[i, :st], atol=1e-5)
        assert not torch.allclose(h_gated[i, st:], h_base[i, st:])
    # The gate is cleared after the context manager.
    assert dec.rep_gate.mask is None


def test_plain_text_probe_is_base_under_gating():
    dec = _decoder(rep_gated=True)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    am = torch.ones_like(ids)
    a = dec.plain_text_nll(ids, am)
    with dec.disable_adapter():
        b = dec.plain_text_nll(ids, am)
    assert abs(a - b) < 1e-6


def test_generate_runs_with_gate():
    dec = _decoder(rep_gated=True)
    d = dec.embed_tokens.weight.shape[1]
    b = dec.build_batch(
        [torch.tensor([3, 4, 5])],
        torch.randn(1, 2, d),
        torch.ones(1, 2, dtype=torch.bool),
        [torch.tensor([7])],
        pad_side="left",
        loss_on_suffix=False,
    )
    out = dec.generate(b, max_new_tokens=3)
    assert out.shape[0] == 1 and out.shape[1] >= 3
    assert dec.rep_gate.mask is None


def test_plain_text_nll_prepends_bos_when_absent():
    """lm_health read 9-10 nats because corpus rows carry no BOS (2026-09-14)."""
    dec = _decoder(rep_gated=False)
    bos = dec.config.bos_token_id
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    am = torch.ones_like(ids)
    with_bos = torch.cat([torch.tensor([[bos]]), ids], dim=1)
    a = dec.plain_text_nll(ids, am)
    b = dec.plain_text_nll(with_bos, torch.ones_like(with_bos))
    assert abs(a - b) < 1e-6
