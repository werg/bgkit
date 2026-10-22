"""P8, the decoder-space adapter of the decoder scale-up (src/bgkit2/models/space_adapter.py)."""

from __future__ import annotations

import torch

from bgkit2.models.space_adapter import DecoderSpaceAdapter

from .test_data_and_training import _tiny_trainer


def test_closed_form_init_recovers_a_linear_map():
    torch.manual_seed(0)
    src = torch.randn(500, 8)
    W, b = torch.randn(8, 12), torch.randn(12)
    dst = torch.zeros(700, 12)
    pairs = [(i, i + 100) for i in range(500)]
    dst[100:600] = src @ W + b
    ad = DecoderSpaceAdapter(8, 12)
    fit = ad.init_from_tables(src, dst, pairs, ridge=1e-9)
    assert fit["rel_residual"] < 1e-4 and fit["mean_cos"] > 0.9999
    assert torch.allclose(ad.linear(src[:5]), (src @ W + b)[:5], atol=1e-3)


def test_zero_and_masked_rows_stay_zero():
    ad = DecoderSpaceAdapter(8, 12)
    reps = torch.randn(2, 4, 8)
    reps[1, 2] = 0.0  # the zeroed arm's rows carry nothing
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 3] = False
    out = ad(reps, mask)
    assert out.shape == (2, 4, 12)
    assert float(out[1, 2].abs().max()) == 0.0 and float(out[0, 3].abs().max()) == 0.0
    assert float(out[0, 0].abs().max()) > 0.0


def test_build_batch_adapts_encoder_reps_but_not_decoder_embeddings(tmp_path):
    tr = _tiny_trainer(tmp_path)
    dec = tr.decoder
    d = dec.embed_tokens.weight.shape[1]
    dec.space_adapter = DecoderSpaceAdapter(5, d)
    reps = torch.randn(1, 3, 5)  # encoder rep space (dim 5) != decoder (dim d)
    mask = torch.ones(1, 3, dtype=torch.bool)
    sb = dec.build_batch([torch.tensor([3, 4])], reps, mask, [torch.tensor([6, 7])])
    assert sb.inputs_embeds.shape[-1] == d
    emb = dec.embed(torch.tensor([[5, 6, 7]]))
    sb2 = dec.build_batch([torch.tensor([3, 4])], emb, mask, [torch.tensor([6, 7])],
                          decoder_space=True)
    assert torch.allclose(sb2.inputs_embeds[0, 2:5].float(), emb[0].float(), atol=1e-4)


def test_trainable_decoder_state_saves_only_trainables_and_p8(tmp_path):
    from bgkit2.models.decoder import TrainableDecoderState

    tr = _tiny_trainer(tmp_path)
    dec = tr.decoder
    dec.space_adapter = DecoderSpaceAdapter(5, dec.embed_tokens.weight.shape[1])
    view = TrainableDecoderState(dec)
    sd = {k: v.clone() for k, v in view.state_dict().items()}
    assert sd and all(k.startswith("space_adapter.") or "lora" in k for k in sd)
    assert any(k.endswith("running_mean") for k in sd)  # P8's interface statistics ride along
    with torch.no_grad():
        dec.space_adapter.linear.weight.add_(1.0)
    view.load_state_dict(sd)
    assert torch.equal(dec.space_adapter.linear.weight, sd["space_adapter.linear.weight"])
    import pytest

    with pytest.raises(RuntimeError, match="missing"):
        view.load_state_dict({k: v for k, v in sd.items() if "linear" not in k})
