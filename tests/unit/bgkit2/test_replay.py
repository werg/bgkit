"""The replay stream (kd.replay_*): plain conversations with no context slot,
self-distilled to the adapter-disabled base. Rows of the replay family never
reach the encoder; the rep rows of the same microbatch keep their sibling
batch (paired prompt contrast) after the split."""

from __future__ import annotations

import torch
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

from bgkit2.data.autoencode import Batch, Collator, Sample
from bgkit2.models.decoder import LoraSpec, SpliceDecoder

from .conftest import tiny_config
from .test_data_and_training import _tiny_trainer


def _chat_sample(doc: int, task: str = "chat_replay") -> Sample:
    return Sample(
        ctx_ids=torch.zeros(0, dtype=torch.long),
        target_ids=torch.tensor([7, 8, 9, 10]),
        task=task,
        store=-1,
        doc=doc,
        prefix_ids=torch.tensor([3, 4, 5, 6, 11, 12]),
        instr_ids=torch.zeros(0, dtype=torch.long),
    )


def _rep_sample(doc: int, alt: bool = False) -> Sample:
    sib = None
    if alt:
        sib = Sample(
            ctx_ids=torch.tensor([13, 14, 15, 16, 17, 18, 19, 20]),
            target_ids=torch.tensor([9, 8]),
            task="qa",
            store=-1,
            doc=doc + 100,
            prompt_ids=torch.tensor([21, 22]),
            prefix_ids=torch.tensor([3, 4]),
            instr_ids=torch.tensor([5]),
        )
    return Sample(
        ctx_ids=torch.tensor([13, 14, 15, 16, 17, 18, 19, 20]),
        target_ids=torch.tensor([7, 8]),
        task="qa",
        store=-1,
        doc=doc,
        prompt_ids=torch.tensor([23, 24]),
        prefix_ids=torch.tensor([3, 4]),
        instr_ids=torch.tensor([5]),
        alt_prompt_ids=torch.tensor([21, 22]) if alt else None,
        sibling=sib,
    )


def test_subset_keeps_the_siblings_of_the_kept_rows(tmp_path):
    tr = _tiny_trainer(tmp_path)
    batch: Batch = tr.collate([_rep_sample(0, alt=True), _chat_sample(1), _rep_sample(2, alt=True)])
    assert batch.sibling is not None and len(batch.sibling.tasks) == 2
    sub = batch.subset([0, 2])
    assert sub.sibling is not None and len(sub.sibling.tasks) == 2
    assert sub.sibling.doc_keys == batch.sibling.doc_keys
    sub1 = batch.subset([2])
    assert sub1.sibling is not None and len(sub1.sibling.tasks) == 1
    assert sub1.sibling.doc_keys == [batch.sibling.doc_keys[1]]
    assert batch.subset([1]).sibling is None


def test_replay_rows_skip_the_encoder_and_rep_rows_still_train(tmp_path):
    tr = _tiny_trainer(tmp_path)
    tr.cfg2.kd.replay_kl_weight = 1.0
    tr.cfg2.kd.replay_ce_weight = 0.5
    calls = []
    orig = tr.encode

    def spy(batch, ratio, selection=None):
        calls.append(list(batch.tasks))
        return orig(batch, ratio, selection)

    tr.encode = spy
    batch = tr.collate([_chat_sample(0), _rep_sample(1), _chat_sample(2)])
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total)
    assert all("chat_replay" not in c for c in calls) and calls, calls
    assert stats["replay_rows"] == 2.0
    assert "loss/chat_replay" in stats and "loss/chat_replay_kl" in stats
    # No adapter on this decoder: the base IS the student, KL exactly 0.
    assert stats["loss/chat_replay_kl"] == 0.0
    assert "loss" in stats and "rep/shared_frac" in stats
    total.backward()
    # A replay-only microbatch is legal too.
    total2, stats2 = tr.training_loss(tr.collate([_chat_sample(3), _chat_sample(4)]))
    assert torch.isfinite(total2) and stats2["loss"] == stats2["loss/chat_replay"]
    assert "rep/shared_frac" not in stats2


def test_replay_kl_is_positive_for_an_ungated_adapter_with_a_delta(tmp_path):
    tr = _tiny_trainer(tmp_path)
    torch.manual_seed(0)
    dec = SpliceDecoder(
        Lfm2ForCausalLM(tiny_config()),
        lora=LoraSpec(rank=4, alpha=8, layers=[0, 1], rep_gated=False),
    )
    tr.decoder = dec
    batch = tr.collate([_chat_sample(0), _chat_sample(1)])
    _, st0 = tr.replay_loss(batch)
    assert st0["loss/chat_replay_kl"] == 0.0  # peft zero-inits lora_B
    for n, p in dec.lm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    loss, st1 = tr.replay_loss(batch)
    assert st1["loss/chat_replay_kl"] > 0.0
    loss.backward()
    grads = [p.grad for n, p in dec.lm.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    m = tr.replay_eval_metrics([batch])
    assert m["kl_to_base"] > 0.0 and abs(m["ce_delta"] - (m["ce"] - m["ce_base"])) < 1e-6


def test_collator_layout_for_chat_rows(tmp_path):
    tr = _tiny_trainer(tmp_path)
    b = tr.collate([_chat_sample(0)])
    assert b.input_ids.shape == (1, 0) and b.attention_mask.shape == (1, 0)
    assert b.suffix_label_start == [0]
    # suffix = target + end-of-turn, every token labelled
    assert b.suffix_ids[0].tolist() == [7, 8, 9, 10, tr.templates.eos_id]
    sb = tr.splice(b, None, None)
    assert int((sb.labels != -100).sum()) == 5 and int(sb.slot_len[0]) == 0


def test_prompt_contrast_is_skipped_while_the_encoder_is_frozen(tmp_path):
    """The contrast backward detaches the decoder and pushes the encoder; in
    the decoder-first phase the encoder is frozen too, and a loss with no
    grad_fn crashed P24 at step 0 (2026-09-13 21:28Z)."""
    tr = _tiny_trainer(tmp_path)
    tr.cfg2.kd.prompt_contrast_weight = 1.0
    tr.cfg2.kd.prompt_contrast_paired = True
    batch = tr.collate([_rep_sample(0, alt=True), _rep_sample(1, alt=True)])
    total, _ = tr.training_loss(batch)
    total.backward()
    assert tr._prompt_contrast_capture is not None
    for p in tr.encoder.parameters():
        p.requires_grad_(False)
    assert tr.prompt_contrast_backward() is None
    for p in tr.encoder.parameters():
        p.requires_grad_(True)
    total, _ = tr.training_loss(batch)
    total.backward()
    out = tr.prompt_contrast_backward()
    assert out is not None and "prompt_gap" in out


def test_truncated_target_gets_no_loss_on_the_end_of_turn_token(tmp_path):
    """2026-09-14 audit: targets cut at max_target_tokens ended with a
    loss-bearing <|im_end|> (9-13% of tool_slots targets)."""
    tr = _tiny_trainer(tmp_path)
    full = _chat_sample(0)
    cut = _chat_sample(1)
    cut.target_truncated = True
    b = tr.collate([full, cut])
    assert b.suffix_loss_mask is not None
    assert b.suffix_loss_mask[0] is None  # complete target: default (all loss)
    m = b.suffix_loss_mask[1]
    assert m is not None and bool(m[:-1].all()) and not bool(m[-1])  # eos masked out


def test_encode_target_reports_truncation(tmp_path):
    from bgkit2.data.tasks import ChatRenderer

    class Tok:
        bos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3] * len(text)}

        def convert_tokens_to_ids(self, t):
            return 5

    r = ChatRenderer(tok=Tok(), sentinel_id=5, eos_id=2, bos_id=1, pad_id=0)
    ids, tr = r.encode_target("abcdef", 4)
    assert ids.shape[0] == 4 and tr is True
    ids, tr = r.encode_target("abc", 4)
    assert ids.shape[0] == 3 and tr is False


def test_prompt_contrast_trains_the_space_adapter_when_the_encoder_is_frozen(tmp_path):
    """8B-A (frozen encoder, decoder-space adapter P8): the contrast's encoder-side
    pressure goes to P8, the one trainable map between the reps and the decoder; the
    decoder's own adapter still takes no contrast gradient."""
    from bgkit2.models.space_adapter import DecoderSpaceAdapter

    tr = _tiny_trainer(tmp_path)
    tr.cfg2.kd.prompt_contrast_weight = 1.0
    tr.cfg2.kd.prompt_contrast_paired = True
    d = tr.decoder.embed_tokens.weight.shape[1]
    tr.decoder.space_adapter = DecoderSpaceAdapter(d, d)
    for p in tr.encoder.parameters():
        p.requires_grad_(False)
    batch = tr.collate([_rep_sample(0, alt=True), _rep_sample(1, alt=True)])
    total, _ = tr.training_loss(batch)
    total.backward()
    tr.decoder.zero_grad(set_to_none=True)
    out = tr.prompt_contrast_backward()
    assert out is not None
    sa_grads = [p.grad for p in tr.decoder.space_adapter.parameters() if p.requires_grad]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in sa_grads)
    sa_ids = {id(p) for p in tr.decoder.space_adapter.parameters()}
    rest = [p for p in tr.decoder.parameters() if p.requires_grad and id(p) not in sa_ids]
    assert all(p.grad is None for p in rest)  # the rest of the decoder takes no contrast grad
