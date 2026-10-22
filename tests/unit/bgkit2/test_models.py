from __future__ import annotations

import torch

from bgkit2.models.decoder import LoraSpec, SpliceDecoder
from bgkit2.models.encoder import BgKIT2Encoder
from bgkit2.models.interface_norm import DecoderInterfaceNorm, rep_stats
from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone
from bgkit2.models.survivorship import (
    exact_topk_select,
    gather_survivors,
    segment_zscore,
    survivor_counts,
)

from .conftest import HIDDEN, VOCAB, tiny_config


def _embeds(backbone, ids):
    return backbone.embed_tokens(ids)


# ----------------------------------------------------------------------------
# Bidirectional backbone
# ----------------------------------------------------------------------------


def test_backbone_is_bidirectional(tiny_lfm2_model, small_ids):
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model).eval()
    ids = small_ids.clone()
    mask = torch.ones_like(ids)
    out_a = bb(_embeds(bb, ids), mask)
    ids_b = ids.clone()
    ids_b[:, -1] = (ids_b[:, -1] + 5) % VOCAB  # change the LAST token only
    out_b = bb(_embeds(bb, ids_b), mask)
    # position 0 must see the change at the last position
    assert not torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-6)


def test_building_the_encoder_leaves_a_decoder_causal(tiny_lfm2_model, tiny_causal_lm, small_ids):
    """The remote-code module patches transformers globally; ours must not."""
    _ = Lfm2BidirectionalBackbone(tiny_lfm2_model)
    lm = tiny_causal_lm.eval()
    ids = small_ids.clone()
    out_a = lm(input_ids=ids, use_cache=False).logits
    ids_b = ids.clone()
    ids_b[:, -1] = (ids_b[:, -1] + 5) % VOCAB
    out_b = lm(input_ids=ids_b, use_cache=False).logits
    # Every position before the last must be unchanged.
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)
    # ... and the decoder's own conv layers are still the causal originals.
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    convs = [layer.conv for layer in lm.model.layers if not layer.is_attention_layer]
    assert all(type(c) is Lfm2ShortConv for c in convs)


def test_padding_invariance(tiny_lfm2_model, small_ids):
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model).eval()
    ids = small_ids[:1]  # (1, 12)
    alone = bb(_embeds(bb, ids), torch.ones_like(ids))
    padded = torch.cat([ids, torch.zeros(1, 5, dtype=ids.dtype)], dim=1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros(1, 5, dtype=ids.dtype)], dim=1)
    batched = bb(_embeds(bb, padded), mask)
    assert torch.allclose(alone[0], batched[0, :12], atol=1e-5)


def test_hook_fires_once_at_the_requested_layer(tiny_lfm2_model, small_ids):
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model).eval()
    calls = []

    def hook(h):
        calls.append(h.shape)
        return h

    bb(_embeds(bb, small_ids), torch.ones_like(small_ids), hook_layer=2, hook=hook)
    assert calls == [(2, 12, HIDDEN)]


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------


def test_survivor_counts_and_topk():
    mask = torch.tensor([[1] * 10, [1] * 4 + [0] * 6]).bool()
    k = survivor_counts(mask, 0.25)
    assert k.tolist() == [3, 1]
    scores = torch.arange(10, dtype=torch.float32).repeat(2, 1)
    keep, k2, thr = exact_topk_select(scores, mask, 0.25)
    assert keep[0].nonzero().flatten().tolist() == [7, 8, 9]
    assert keep[1].nonzero().flatten().tolist() == [3]
    assert thr.tolist() == [7.0, 3.0]
    assert k2.tolist() == [3, 1]


def test_zscore_ignores_padding():
    scores = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    mask = torch.tensor([[1, 1, 1, 0]]).bool()
    z = segment_zscore(scores, mask)
    assert abs(float(z[0, :3].mean())) < 1e-5
    assert float(z[0, 3]) == 0.0


def test_gather_keeps_document_order():
    hidden = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    keep = torch.tensor([[0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 0, 1]]).bool()
    k = keep.sum(-1)
    reps, rep_mask, pos = gather_survivors(hidden, keep, k)
    assert reps.shape == (2, 3, 2)
    assert pos[0].tolist() == [1, 3, 4]
    assert pos[1].tolist() == [0, 5, 0]
    assert rep_mask.tolist() == [[True, True, True], [True, True, False]]
    assert torch.equal(reps[0, 0], hidden[0, 1])
    assert float(reps[1, 2].abs().sum()) == 0.0


# ----------------------------------------------------------------------------
# Interface norm
# ----------------------------------------------------------------------------


def test_interface_norm_lands_on_target_norm_and_removes_constant():
    torch.manual_seed(0)
    norm = DecoderInterfaceNorm(16, target_row_norm=0.5, momentum=1.0)
    x = torch.randn(256, 16) * 3.0 + 40.0  # a big shared offset
    y = norm(x)
    assert abs(float(y.norm(dim=-1).mean()) - 0.5) < 0.05
    assert float(y.mean(dim=0).abs().max()) < 0.05
    stats = rep_stats(y[None], torch.ones(1, 256, dtype=torch.bool), embed_row_norm=0.5)
    assert abs(stats["norm_ratio"] - 1.0) < 0.1
    assert stats["shared_frac"] < 0.2


def test_interface_norm_calibrates_once_in_eval_mode():
    norm = DecoderInterfaceNorm(4, momentum=0.5).eval()
    x = torch.tensor([[10.0, 0, 0, 0], [12.0, 0, 0, 0], [8.0, 0, 0, 0]])
    norm(x)
    assert int(norm.num_updates) == 1
    norm(x + 100)  # eval: reference must NOT move
    assert int(norm.num_updates) == 1


# ----------------------------------------------------------------------------
# Encoder + decoder end to end (tiny)
# ----------------------------------------------------------------------------


def _encoder(tiny_lfm2_model, **kw):
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model)
    return BgKIT2Encoder(bb, head_layer=2, head_inner_dim=8, **kw)


def test_encoder_realises_ratio_and_shapes(tiny_lfm2_model, small_ids):
    enc = _encoder(tiny_lfm2_model).eval()
    mask = torch.ones_like(small_ids)
    mask[1, 8:] = 0
    out = enc(small_ids, mask, ratio=0.25)
    assert out.k.tolist() == [3, 2]
    assert out.reps.shape == (2, 3, HIDDEN)
    assert out.rep_mask.tolist() == [[True] * 3, [True, True, False]]
    assert float(out.reps[1, 2].abs().sum()) == 0.0
    assert bool((out.positions[1, :2] < 8).all())


def test_encoder_random_selection_matches_ratio(tiny_lfm2_model, small_ids):
    enc = _encoder(tiny_lfm2_model, selection="random").eval()
    g = torch.Generator().manual_seed(0)
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5, generator=g)
    assert out.k.tolist() == [6, 6]


def test_head_receives_gradient_through_the_gate(tiny_lfm2_model, small_ids):
    enc = _encoder(tiny_lfm2_model).train()
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5)
    loss = out.reps.pow(2).sum()
    loss.backward()
    grads = [p.grad for p in enc.head.parameters()]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_decoder_is_frozen_and_lora_targets_lower_layers(tiny_causal_lm):
    dec = SpliceDecoder(tiny_causal_lm, lora=LoraSpec(rank=2, alpha=4, layers=[0, 1]))
    names = [n for n, p in dec.lm.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert all(".layers.0." in n or ".layers.1." in n for n in names)
    assert not dec.embed_tokens.weight.requires_grad


def test_splice_loss_only_on_suffix_and_reps_receive_gradient(
    tiny_lfm2_model, tiny_causal_lm, small_ids
):
    enc = _encoder(tiny_lfm2_model).train()
    dec = SpliceDecoder(tiny_causal_lm)
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    batch = dec.build_batch(prefix, out.reps, out.rep_mask, suffix)
    # prefix + reps carry no labels; suffix does
    assert batch.n_target_tokens.tolist() == [4, 3]
    assert int((batch.labels[0, : 3 + 6] != -100).sum()) == 0
    loss = dec(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert enc.projection[1].weight.grad is not None
    assert all(p.grad is None for p in dec.lm.parameters())


def test_generate_from_left_padded_prompt(tiny_lfm2_model, tiny_causal_lm, small_ids):
    enc = _encoder(tiny_lfm2_model).eval()
    dec = SpliceDecoder(tiny_causal_lm).eval()
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7]), torch.tensor([7, 8])]
    batch = dec.build_batch(prefix, out.reps, out.rep_mask, suffix, pad_side="left")
    gen = dec.generate(batch, max_new_tokens=4)
    assert gen.shape[0] == 2 and 1 <= gen.shape[1] <= 4


def test_plain_text_nll_is_finite(tiny_causal_lm, small_ids):
    dec = SpliceDecoder(tiny_causal_lm).eval()
    nll = dec.plain_text_nll(small_ids, torch.ones_like(small_ids))
    assert nll > 0 and nll < 20


def test_tiny_config_smoke():
    cfg = tiny_config()
    assert cfg.num_hidden_layers == 6


def test_backbone_freeze_policy(tiny_lfm2_model):
    enc = _encoder(tiny_lfm2_model)
    assert enc.set_backbone_trainable([]) == []
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    groups = enc.trainable_groups()
    assert groups["backbone"] == []
    assert groups["head"] and groups["projection"] and groups["survive_embedding"]
    assert enc.set_backbone_trainable([4, 5]) == [4, 5]
    trainable = {n for n, p in enc.backbone.named_parameters() if p.requires_grad}
    assert trainable and all(".layers.4." in n or ".layers.5." in n for n in trainable)
    assert not enc.backbone.embed_tokens.weight.requires_grad
    import pytest

    with pytest.raises(ValueError):
        enc.set_backbone_trainable([99])


def test_compression_prompt_conditions_but_never_survives(tiny_lfm2_model, small_ids):
    enc = _encoder(tiny_lfm2_model).eval()
    mask = torch.ones_like(small_ids)
    # left-padded prompts of different lengths
    p_ids = torch.tensor([[0, 0, 9, 10, 11], [0, 12, 13, 14, 15]])
    p_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    out = enc(small_ids, mask, ratio=0.25, prompt_ids=p_ids, prompt_mask=p_mask)
    assert out.k.tolist() == [3, 3]
    assert out.keep.shape == small_ids.shape and out.scores.shape == small_ids.shape
    assert bool((out.positions[out.rep_mask] < small_ids.shape[1]).all())
    # a different prompt changes the reps (the prompt conditions the encoding)
    p2 = torch.tensor([[0, 0, 20, 21, 22], [0, 23, 24, 25, 26]])
    out2 = enc(small_ids, mask, ratio=0.25, prompt_ids=p2, prompt_mask=p_mask)
    assert not torch.allclose(out.reps, out2.reps, atol=1e-5)
    # and the reps match a prompt-free encode only when the prompt is empty-equivalent? no:
    # without a prompt the output differs too, which is the point of conditioning
    out0 = enc(small_ids, mask, ratio=0.25)
    assert out0.reps.shape == out.reps.shape


def test_kd_is_zero_against_itself_and_positive_against_full_text(
    tiny_lfm2_model, tiny_causal_lm, small_ids
):
    enc = _encoder(tiny_lfm2_model).train()
    dec = SpliceDecoder(tiny_causal_lm)
    mask = torch.ones_like(small_ids)
    out = enc(small_ids, mask, ratio=0.5)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    sb = dec.build_batch(prefix, out.reps, out.rep_mask, suffix)
    # teacher == student batch -> KL exactly 0, CE matches token_nll
    nll, valid, kl = dec.token_nll_and_kd(sb, sb)
    nll_ref, valid_ref = dec.token_nll(sb)
    assert torch.allclose(nll, nll_ref, atol=1e-5) and torch.equal(valid, valid_ref)
    assert abs(float(kl)) < 1e-5
    # full-text teacher -> KL > 0 and gradient reaches the encoder
    tb = dec.build_batch(prefix, dec.embed(small_ids), mask.bool(), suffix)
    nll2, valid2, kl2 = dec.token_nll_and_kd(sb, tb, temperature=2.0)
    assert float(kl2) > 0
    (nll2.sum() / valid2.sum() + kl2).backward()
    assert enc.projection[1].weight.grad is not None


def test_encoder_lora_injection_trains_only_adapters(tiny_lfm2_model, small_ids):
    enc = _encoder(tiny_lfm2_model)
    enc.set_backbone_trainable([])
    n = enc.inject_lora(rank=2, alpha=4, layers=None)
    assert n > 0
    names = [nm for nm, p in enc.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in nm for nm in names)
    assert not enc.backbone.embed_tokens.weight.requires_grad
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5)
    out.reps.pow(2).sum().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for nm, p in enc.backbone.named_parameters()
        if "lora_B" in nm or "lora_A" in nm
    )
    groups = enc.trainable_groups()
    assert groups["backbone"] and all(
        "lora_" in nm for nm, p in enc.backbone.named_parameters() if p.requires_grad
    )


def test_interface_row_norm_is_a_hard_constraint():
    norm = DecoderInterfaceNorm(16, target_row_norm=0.5, momentum=1.0, row_norm=True)
    x = torch.randn(64, 16) * 3.0 + 40.0
    y = norm(x)
    # eps is 5% of the target, so rows land within ~1% of it (never above)
    assert torch.allclose(y.norm(dim=-1), torch.full((64,), 0.5), atol=5e-3)
    assert bool((y.norm(dim=-1) <= 0.5 + 1e-6).all())
    y2 = norm.eval()(x * 100.0)  # a 100x upstream runaway cannot pass through
    assert torch.allclose(y2.norm(dim=-1), torch.full((64,), 0.5), atol=5e-3)


def test_interface_prenorm_bounds_the_reference_under_a_runaway():
    torch.manual_seed(0)
    norm = DecoderInterfaceNorm(16, target_row_norm=0.5, momentum=0.1, prenorm=True)
    x = torch.randn(64, 16) * 3.0 + 40.0
    y0 = norm(x)
    # a 1000x upstream runaway: the reference must stay in a bounded space
    for _ in range(30):
        norm(x * 1000.0)
    assert float(norm.running_var.mean()) <= 1.0 + 1e-4  # unit-RMS rows: E[var] <= 1
    assert float(norm.running_mean.abs().max()) <= 16**0.5  # |mean_c| <= row RMS * sqrt(dim)
    # ... and the emitted reps are the same function of the direction
    y1 = norm.eval()(x * 1000.0)
    assert torch.allclose(y1, norm(x), atol=1e-4)
    assert abs(float(y1.norm(dim=-1).mean()) - 0.5) < 0.05
    del y0


def test_interface_prenorm_reseeds_a_reference_from_unnormalised_inputs():
    norm = DecoderInterfaceNorm(8, momentum=0.1, prenorm=True)
    sd = norm.state_dict()
    sd["running_mean"] = torch.full((8,), 400.0)
    sd["running_var"] = torch.full((8,), 9.0e4)
    sd["num_updates"] = torch.tensor(5001)
    norm.load_state_dict(sd)
    assert norm.reseed_if_reference_out_of_contract() is True
    assert int(norm.num_updates) == 0
    norm.eval()(torch.randn(16, 8))  # first call re-seeds from the batch
    assert int(norm.num_updates) == 1
    assert float(norm.running_var.mean()) <= 1.0 + 1e-4
    assert norm.reseed_if_reference_out_of_contract() is False
    plain = DecoderInterfaceNorm(8, momentum=0.1)
    plain.load_state_dict(sd)
    assert plain.reseed_if_reference_out_of_contract() is False  # only under prenorm


def test_kd_against_a_separate_teacher_model(tiny_lfm2_model, tiny_causal_lm, small_ids):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model).train()
    spec = LoraSpec(rank=2, alpha=4, layers=[0, 1])
    student = SpliceDecoder(tiny_causal_lm, lora=spec)
    teacher = SpliceDecoder(Lfm2ForCausalLM(tiny_config()), lora=spec)
    teacher.load_state_dict(student.state_dict())  # identical weights, incl. adapter
    with torch.no_grad():  # a distinct (fine-tuned) teacher adapter
        for prm in teacher.trainable_parameters():
            prm.add_(torch.randn_like(prm) * 0.5)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    teacher.eval()
    mask = torch.ones_like(small_ids)
    out = enc(small_ids, mask, ratio=0.5)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    sb = student.build_batch(prefix, out.reps, out.rep_mask, suffix)
    tb = student.build_batch(prefix, student.embed(small_ids), mask.bool(), suffix)
    _nll_base, _v, kl_base = student.token_nll_and_kd(sb, tb)
    nll, valid, kl = student.token_nll_and_kd(sb, tb, teacher_model=teacher)
    assert torch.isfinite(kl) and float(kl) > 0
    assert abs(float(kl) - float(kl_base)) > 1e-4  # a different teacher, a different target
    (nll.sum() / valid.sum() + kl).backward()
    assert enc.projection[1].weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())


def test_interface_match_moments_has_the_embedding_table_as_a_fixed_point():
    torch.manual_seed(0)
    dim = 16
    # an anisotropic "embedding table": a shared mean + very unequal channel scales
    table = torch.randn(512, dim) * torch.linspace(0.03, 0.2, dim) + torch.randn(dim) * 0.3
    norm = DecoderInterfaceNorm(dim, target_row_norm=0.5, momentum=1.0, match_moments=True)
    norm.set_target_moments(table)
    y = norm(table)  # the table through the contract...
    assert torch.allclose(y, table, atol=5e-3, rtol=0.02)  # ...comes out unchanged (eps aside)
    # whereas the unit-variance contract cannot: it whitens the anisotropy away
    plain = DecoderInterfaceNorm(dim, target_row_norm=0.5, momentum=1.0)
    yp = plain(table)
    assert not torch.allclose(yp, table, atol=1e-1)
    assert float(yp.std(dim=0).max() / yp.std(dim=0).min()) < 1.5  # isotropic
    assert float(y.std(dim=0).max() / y.std(dim=0).min()) > 5  # anisotropy kept
    # an arbitrary rep distribution is mapped onto the table's moments
    x = torch.randn(256, dim) * 40.0 + 100.0
    norm2 = DecoderInterfaceNorm(dim, momentum=1.0, match_moments=True)
    norm2.set_target_moments(table)
    z2 = norm2.train()(x)
    assert torch.allclose(z2.mean(dim=0), table.mean(dim=0), atol=1e-3)
    assert torch.allclose(
        z2.std(dim=0, unbiased=False), table.std(dim=0, unbiased=False), rtol=0.05
    )


def test_load_encoder_state_tolerates_only_the_new_interface_buffers(tiny_lfm2_model):
    import pytest

    from bgkit2.training.core import BgKIT2Core

    enc = _encoder(tiny_lfm2_model)
    state = {k: v.clone() for k, v in enc.state_dict().items()}
    old = {k: v for k, v in state.items() if not k.startswith("interface_norm.target_")}
    assert set(BgKIT2Core.load_encoder_state(enc, old)) == {
        "interface_norm.target_mean",
        "interface_norm.target_std",
    }
    assert BgKIT2Core.load_encoder_state(enc, state) == []
    broken = {k: v for k, v in state.items() if k != "survive_embedding"}
    with pytest.raises(RuntimeError):
        BgKIT2Core.load_encoder_state(enc, broken)


def test_projection_block_mixes_survivors_and_respects_padding(tiny_lfm2_model):
    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model, projection_block=True)
    assert enc.projection_block is not None
    assert enc.trainable_groups()["projection_block"]
    ids = torch.randint(5, 50, (2, 24))
    mask = torch.ones_like(ids)
    mask[1, 16:] = 0
    out = enc(ids, mask, ratio=0.5)
    assert out.reps.shape[-1] == enc.hidden_size
    # gradient reaches the block
    out.reps.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.projection_block.parameters()
    )
    # padded survivors (row 1 has fewer valid tokens) do not change row 0's
    # reps — compared in eval mode so the interface statistics stay fixed
    enc.zero_grad()
    enc.eval()
    with torch.no_grad():
        both = enc(ids, mask, ratio=0.5)
        alone = enc(ids[:1], mask[:1], ratio=0.5)
    k0 = int(both.k[0])
    assert torch.allclose(both.reps[0, :k0], alone.reps[0, :k0], atol=1e-4)


def test_projection_input_skip_is_identity_at_init_and_trains(tiny_lfm2_model):
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    base = _encoder(tiny_lfm2_model)
    skip = _encoder(tiny_lfm2_model, projection_input_skip=True)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    tolerated = BgKIT2Core.load_encoder_state(skip, state)
    assert tolerated == ["input_skip.weight"]
    ids = torch.randint(5, 50, (2, 20))
    mask = torch.ones_like(ids)
    base.eval()
    skip.eval()
    with torch.no_grad():
        a = base(ids, mask, ratio=0.5)
        b = skip(ids, mask, ratio=0.5)
    assert torch.allclose(a.reps, b.reps, atol=1e-5)  # zero-initialised skip
    skip.train()
    out = skip(ids, mask, ratio=0.5)
    out.reps.sum().backward()
    assert skip.input_skip.weight.grad is not None and skip.input_skip.weight.grad.abs().sum() > 0
    assert any(p is skip.input_skip.weight for p in skip.trainable_groups()["projection"])


def test_interface_rms_only_passes_true_embeddings_exactly():
    torch.manual_seed(0)
    dim = 32
    norm = DecoderInterfaceNorm(
        dim, target_row_norm=0.8, prenorm=True, match_moments=True, rms_only=True
    )
    norm.set_target_moments(torch.randn(100, dim))
    x = torch.randn(64, dim) * 5.0 + 2.0
    y = norm(x)
    cos = torch.nn.functional.cosine_similarity(y, x, dim=-1)
    assert cos.min() > 0.999
    assert torch.allclose(y.norm(dim=-1), torch.full((64,), 0.8), atol=1e-2)
    assert not norm.reseed_if_reference_out_of_contract()


def test_load_encoder_state_drop_prefixes_leaves_fresh_modules(tiny_lfm2_model):
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    src = _encoder(tiny_lfm2_model)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.3)
    dst = _encoder(tiny_lfm2_model)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    tolerated = BgKIT2Core.load_encoder_state(
        dst, src.state_dict(), drop_prefixes=("projection.", "interface_norm.")
    )
    assert all(k.startswith(("projection.", "interface_norm.")) for k in tolerated)
    after = dst.state_dict()
    for k, v in after.items():
        if k.startswith(("projection.", "interface_norm.")):
            assert torch.equal(v, before[k]), k
        else:
            assert torch.equal(v, src.state_dict()[k]), k


def test_captured_fraction_and_gap_quantiles():
    from bgkit2.training.core import BgKIT2Core

    assert abs(BgKIT2Core.captured_fraction(3.0, 2.0, 1.0) - 0.5) < 1e-9
    assert BgKIT2Core.captured_fraction(3.0, 2.0, 2.99) != BgKIT2Core.captured_fraction(
        3.0, 2.0, 2.99
    )  # NaN
    q = BgKIT2Core.gap_quantiles(
        torch.tensor([3.0, 3.0, 3.0, 3.0]), torch.tensor([2.0, 3.5, 1.0, 3.0])
    )
    assert q["gap_frac_gt_0.5"] == 0.5 and q["gap_frac_lt_0"] == 0.25
    assert abs(q["gap_p50"] - 0.5) < 1e-6


def test_projection_block_is_identity_at_init(tiny_lfm2_model):
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    base = _encoder(tiny_lfm2_model)
    withblock = _encoder(tiny_lfm2_model, projection_block=True)
    BgKIT2Core.load_encoder_state(withblock, {k: v.clone() for k, v in base.state_dict().items()})
    ids = torch.randint(5, 50, (2, 20))
    mask = torch.ones_like(ids)
    base.eval()
    withblock.eval()
    with torch.no_grad():
        a = base(ids, mask, ratio=0.5)
        b = withblock(ids, mask, ratio=0.5)
    assert torch.allclose(a.reps, b.reps, atol=1e-5)


def test_load_encoder_state_tolerates_a_missing_projection_block(tiny_lfm2_model):
    from bgkit2.training.core import BgKIT2Core

    old = _encoder(tiny_lfm2_model)  # no block
    state = {k: v.clone() for k, v in old.state_dict().items()}
    with torch.no_grad():
        for p in old.backbone.parameters():
            p.add_(0.5)
        state = {k: v.clone() for k, v in old.state_dict().items()}
    new = _encoder(tiny_lfm2_model, projection_block=True)
    tolerated = BgKIT2Core.load_encoder_state(new, state)
    assert any(k.startswith("projection_block.") for k in tolerated)
    src = new.backbone.model.layers[new._projection_block_src]
    for (n1, p1), (n2, p2) in zip(
        src.named_parameters(), new.projection_block.named_parameters(), strict=True
    ):
        assert n1 == n2
        if n1 in ("self_attn.out_proj.weight", "feed_forward.w2.weight"):
            assert float(p2.abs().sum()) == 0.0
        else:
            assert torch.equal(p1, p2)


def test_load_head_state_copies_only_the_selection_head(tiny_lfm2_model):
    import pytest

    from bgkit2.training.core import BgKIT2Core

    src = _encoder(tiny_lfm2_model)
    dst = _encoder(tiny_lfm2_model)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    n = BgKIT2Core.load_head_state(dst, src.state_dict())
    assert n >= 2
    after = dst.state_dict()
    for k, v in after.items():
        if k.startswith("head.") or k == "survive_embedding":
            assert torch.equal(v, src.state_dict()[k])
        else:
            assert torch.equal(v, before[k]), k
    with pytest.raises(ValueError):
        BgKIT2Core.load_head_state(dst, {"projection.x": torch.zeros(1)})


def test_interface_variance_floor_bounds_low_variance_channel_amplification():
    torch.manual_seed(0)
    dim = 16
    norm = DecoderInterfaceNorm(dim, target_row_norm=1.0, momentum=1.0, var_floor_frac=0.01)
    x = torch.randn(512, dim)
    x[:, 3] = 5.0 + 1e-4 * torch.randn(512)  # a channel the projection holds ~constant
    y = norm(x)
    # channel 3's deviations (1e-4) are amplified at most 1/sqrt(0.01) = 10x, not ~1e4x
    gain = float(y[:, 3].std() / x[:, 3].std())
    ref = float(y[:, 0].std() / x[:, 0].std())
    assert gain <= 10.5 * ref
    unfloored = DecoderInterfaceNorm(dim, target_row_norm=1.0, momentum=1.0, var_floor_frac=0.0)
    y0 = unfloored(x)
    assert float(y0[:, 3].std() / x[:, 3].std()) > 100 * ref  # the defect, for contrast


def test_summary_slots_keep_the_budget_and_carry_gradient(tiny_lfm2_model, small_ids):
    torch.manual_seed(0)
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(tiny_lfm2_model),
        head_layer=2,
        head_inner_dim=8,
        projection_mult=1,
        summary_slot_frac=0.25,
    ).train()
    mask = torch.ones_like(small_ids)
    n = small_ids.shape[1]
    out = enc(small_ids, mask, ratio=0.5)
    k = out.k
    # survivors at ratio / (1 + frac), slots = ceil(frac * k): total ~= ratio * n
    assert int(k[0]) == max(1, int(torch.ceil(torch.tensor(0.5 / 1.25 * n)).item()))
    m = int(torch.ceil(k.float() * 0.25)[0])
    assert out.slot_reps is not None and out.slot_reps.shape[1] == m
    assert bool(out.slot_mask.all())
    assert torch.isfinite(out.slot_reps).all()
    assert abs(int(k[0]) + m - 0.5 * n) <= 2
    # slots are a function of the backbone: gradient reaches it without the gate
    out.slot_reps.sum().backward()
    assert any(p.grad is not None for p in enc.backbone.parameters() if p.requires_grad)


def test_finish_reps_appends_slots_after_verbatim_tokens(
    tiny_lfm2_model, tiny_causal_lm, small_ids
):
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(tiny_lfm2_model),
        head_layer=2,
        head_inner_dim=8,
        projection_mult=1,
        summary_slot_frac=0.5,
    ).eval()
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.model.rep_token_residual_only = True
    core.decoder = SpliceDecoder(tiny_causal_lm)
    mask = torch.ones_like(small_ids)
    with torch.no_grad():
        out = enc(small_ids, mask, ratio=0.5)
        fin = core.finish_reps(small_ids, out)
    k, m = int(out.k[0]), out.slot_reps.shape[1]
    assert fin.reps.shape[1] == out.reps.shape[1] + m and fin.rep_mask.shape[1] == fin.reps.shape[1]
    ids = small_ids.gather(1, out.positions.clamp_min(0))
    emb = core.decoder.embed(ids) * out.rep_mask.unsqueeze(-1).float()
    assert torch.allclose(fin.reps[:, : out.reps.shape[1]], emb, atol=1e-6)  # verbatim tokens
    assert torch.equal(fin.reps[:, out.reps.shape[1] :], out.slot_reps)  # then the slots
    assert int(fin.rep_mask[0].sum()) == k + m


def test_token_pruning_trains_the_head_through_the_gate(tiny_lfm2_model, tiny_causal_lm, small_ids):
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model).train()
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.model.rep_token_residual_only = True
    core.cfg2.model.rep_token_gate = True
    core.decoder = SpliceDecoder(tiny_causal_lm)
    mask = torch.ones_like(small_ids)
    out = enc(small_ids, mask, ratio=0.5)
    assert out.gate is not None and torch.allclose(out.gate, torch.ones_like(out.gate))
    fin = core.finish_reps(small_ids, out)
    ids = small_ids.gather(1, out.positions.clamp_min(0))
    emb = core.decoder.embed(ids) * out.rep_mask.unsqueeze(-1).float()
    assert torch.allclose(fin.reps, emb, atol=1e-6)  # forward: verbatim tokens
    (fin.reps * torch.randn_like(fin.reps)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.head.parameters())
    # and in eval mode there is no gate (plain tokens, no grad path needed)
    with torch.no_grad():
        assert enc.eval()(small_ids, mask, ratio=0.5).gate is None


def test_build_batch_records_the_rep_slot(tiny_causal_lm):
    dec = SpliceDecoder(tiny_causal_lm)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    reps = torch.randn(2, 4, HIDDEN)
    rep_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    sb = dec.build_batch(prefix, reps, rep_mask, suffix)
    assert sb.slot_start.tolist() == [3, 2] and sb.slot_len.tolist() == [3, 2]
    assert torch.allclose(sb.inputs_embeds[0, 3:6], reps[0, :3])
    sbl = dec.build_batch(prefix, reps, rep_mask, suffix, pad_side="left")
    tmax = sbl.inputs_embeds.shape[1]
    assert sbl.slot_start.tolist() == [tmax - 10 + 3, tmax - 7 + 2]
    assert torch.allclose(
        sbl.inputs_embeds[1, sbl.slot_start[1] : sbl.slot_start[1] + 2], reps[1, :2]
    )


def test_attention_importance_from_answer_positions(small_ids):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    torch.manual_seed(0)
    cfg = tiny_config()
    cfg._attn_implementation = "sdpa"
    dec = SpliceDecoder(Lfm2ForCausalLM(cfg)).eval()
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    mask = torch.ones_like(small_ids)
    tb = dec.build_batch(prefix, dec.embed(small_ids), mask.bool(), suffix)
    imp = dec.attention_importance(tb)
    assert dec.base_lm.model.config._attn_implementation == "sdpa"  # restored after the pass
    dec.attention_importance_chunk = 1
    assert torch.allclose(dec.attention_importance(tb), imp, atol=1e-6)  # chunking is exact
    assert imp.shape == tb.labels.shape
    assert bool((imp >= 0).all()) and torch.isfinite(imp).all()
    # mass lands only on valid keys, and there is mass inside the context slot
    assert float(imp[tb.attention_mask == 0].abs().sum()) == 0.0
    st, ln = int(tb.slot_start[0]), int(tb.slot_len[0])
    assert float(imp[0, st : st + ln].sum()) > 0
    # each answer row's attention sums to 1 per layer, so per-row mass is <= 1
    assert float(imp.sum(dim=-1).max()) <= 1.0 + 1e-4


def test_utility_gradient_selection_trains_the_head_over_all_positions(tiny_lfm2_model, small_ids):
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model).train()
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.selection_weight = 1.0
    core.cfg2.kd.selection_signal = "utility_grad"
    core.encoder = enc
    mask = torch.ones_like(small_ids)
    cap: dict = {}
    enc.utility_capture = cap
    out = enc(small_ids, mask, ratio=0.5)
    assert enc.utility_capture is None  # consumed by the forward
    assert out.head_input is not None and out.eligible is not None and out.raw is not None
    cap["out"] = out
    core._utility_capture = cap
    (out.reps.float().pow(2).sum()).backward()  # stand-in for the decoder loss
    assert "grad" in cap and cap["grad"].shape == out.head_input.shape  # ALL positions
    enc.zero_grad()
    assert "handle" in cap
    val = core.utility_selection_backward()
    assert val is not None and val >= 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.head.parameters())
    assert 0 < core._utility_teacher_rate <= 1.0
    assert core._utility_capture is None
    assert cap == {}  # the cycle is broken: nothing retained after consumption


def test_head_scale_anchor_pulls_the_raw_std_toward_its_target(tiny_lfm2_model, small_ids):
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model).train()
    with torch.no_grad():
        enc.head.net[-1].weight.mul_(50.0)  # blow the raw scale up
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.head_scale_anchor_weight = 0.01
    core.cfg2.kd.head_scale_target = 1.0
    core.encoder = enc
    out = enc(small_ids, torch.ones_like(small_ids), ratio=0.5)
    loss, std = core.head_scale_terms(out)
    assert std is not None and loss is not None and float(loss) > 0
    loss.backward()
    g = enc.head.net[-1].weight.grad
    # gradient descent moves the scale toward the target from either side:
    # (g . w) > 0 means the step shrinks |w| (std above target), < 0 grows it
    import math

    assert g is not None
    assert math.copysign(1.0, float((g * enc.head.net[-1].weight).sum())) == math.copysign(
        1.0, math.log(std / 1.0)
    )
    rand = enc(small_ids, torch.ones_like(small_ids), ratio=0.5, selection="random")
    assert core.head_scale_terms(rand) == (None, None)


def test_selection_loss_trains_the_head_toward_teacher_attention(tiny_lfm2_model, small_ids):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model).train()
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.selection_weight = 1.0
    core.decoder = SpliceDecoder(Lfm2ForCausalLM(tiny_config())).eval()
    mask = torch.ones_like(small_ids)
    prefix = [torch.tensor([1, 5, 6]), torch.tensor([1, 5])]
    suffix = [torch.tensor([7, 8, 9, 2]), torch.tensor([7, 8, 2])]
    tb = core.decoder.build_batch(prefix, core.decoder.embed(small_ids), mask.bool(), suffix)

    class B:  # the two Batch fields selection_loss reads
        attention_mask = mask

    out = enc(small_ids, mask, ratio=0.5)
    loss = core.selection_loss(B(), out, tb)
    assert torch.isfinite(loss) and float(loss) >= 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.head.parameters())


def test_anchor_phase_out_timer_overrides_progress_decay():
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.anchor.weight, core.cfg2.anchor.floor = 0.05, 0.0
    core.cfg2.anchor.phase_out_steps = 1000
    core.cfg2.ratio.adaptive = True
    core.ratio_schedule = core.build_ratio_schedule(0)
    assert abs(core.anchor_weight(0) - 0.05) < 1e-9
    assert abs(core.anchor_weight(500) - 0.025) < 1e-9
    assert core.anchor_weight(1000) == 0.0 and core.anchor_weight(5000) == 0.0


def test_prompt_conditioning_starts_at_zero_and_routes_the_prompt(tiny_lfm2_model, small_ids):
    """The explicit prompt route: zero-initialised (identical to the plain
    encoder), in a param group of its own, and once the maps are non-zero a
    different prompt changes every document position's input."""
    torch.manual_seed(0)
    enc = _encoder(tiny_lfm2_model, prompt_separator=True, prompt_conditioning=True).eval()
    ref = _encoder(tiny_lfm2_model, prompt_separator=True).eval()
    ref.load_state_dict(enc.state_dict(), strict=False)
    mask = torch.ones_like(small_ids)
    p_ids = torch.tensor([[0, 0, 9, 10, 11], [0, 12, 13, 14, 15]])
    p_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    a = enc(small_ids, mask, ratio=0.5, prompt_ids=p_ids, prompt_mask=p_mask)
    b = ref(small_ids, mask, ratio=0.5, prompt_ids=p_ids, prompt_mask=p_mask)
    assert torch.allclose(a.reps, b.reps, atol=1e-6) and torch.equal(a.positions, b.positions)
    groups = enc.trainable_groups()
    assert len(groups["prompt_cond"]) == 2
    assert any(p is enc.prompt_separator for p in groups["survive_embedding"])
    assert not any(p is q for p in groups["prompt_cond"] for q in groups["backbone"])
    # no prompt: the route is inert (no pooled prompt to add)
    assert enc(small_ids, mask, ratio=0.5).reps.shape == a.reps.shape
    with torch.no_grad():
        enc.prompt_cond_in.weight.fill_(0.05)
        enc.prompt_cond_hook.weight.fill_(0.05)
    c = enc(small_ids, mask, ratio=0.5, prompt_ids=p_ids, prompt_mask=p_mask)
    p2 = torch.tensor([[0, 0, 20, 21, 22], [0, 23, 24, 25, 26]])
    d = enc(small_ids, mask, ratio=0.5, prompt_ids=p2, prompt_mask=p_mask)
    gap_in = (c.reps - d.reps).norm() / c.reps.norm()
    gap_ref = (
        a.reps - ref(small_ids, mask, ratio=0.5, prompt_ids=p2, prompt_mask=p_mask).reps
    ).norm() / a.reps.norm()
    assert gap_in > gap_ref  # the explicit route makes the reps more prompt-sensitive
    # the maps receive gradient
    c.reps.sum().backward()
    assert (
        enc.prompt_cond_in.weight.grad is not None and enc.prompt_cond_hook.weight.grad is not None
    )
