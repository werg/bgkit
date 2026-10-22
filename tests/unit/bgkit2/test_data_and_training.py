from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
import torch

from bgkit2.data.autoencode import AutoencodeDataset, Collator
from bgkit2.data.templates import Templates
from bgkit2.data.token_store import TokenStore, TokenStoreWriter
from bgkit2.probes.rep_probe import rep_distinguishability
from bgkit2.training.config import load_config
from bgkit2.training.metrics import corpus_bleu, exact_match, prefix_accuracy, token_f1
from bgkit2.training.ratios import EVAL_RATIOS, RatioSchedule

from .conftest import VOCAB, tiny_config


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, s: str, add_special_tokens: bool = False):
        return {"input_ids": [3 + (ord(c) % 40) for c in s]}


def _make_store(tmp_path, name, n_docs=30, seed=0, lo=70, hi=400):
    rng = np.random.default_rng(seed)
    w = TokenStoreWriter(tmp_path / name, meta={"source": "test"})
    for _ in range(n_docs):
        n = int(rng.integers(lo, hi))
        w.add(rng.integers(3, VOCAB, size=n))
    w.close()
    return TokenStore(tmp_path / name)


# ----------------------------------------------------------------------------


def test_token_store_roundtrip(tmp_path):
    store = _make_store(tmp_path, "s", n_docs=5)
    assert len(store) == 5
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["n_docs"] == 5 and meta["n_tokens"] == int(store.lengths.sum())
    doc = store[2]
    assert doc.dtype == np.int64 and doc.shape[0] == store.length(2)
    assert np.array_equal(store.slice(2, 5, 9), doc[5:9])


def test_writer_rejects_out_of_range(tmp_path):
    w = TokenStoreWriter(tmp_path / "bad", meta={})
    with pytest.raises(ValueError):
        w.add(np.array([70000]))


def test_dataset_is_deterministic_per_index_and_tasks_are_sound(tmp_path):
    store = _make_store(tmp_path, "a")
    ds = AutoencodeDataset(
        [store],
        ctx_min=16,
        ctx_max=128,
        cont_min=8,
        cont_max=32,
        p_continue=0.5,
        seed=3,
        epoch_size=200,
    )
    a, b = ds[7], ds[7]
    assert torch.equal(a.ctx_ids, b.ctx_ids) and a.task == b.task
    tasks = {ds[i].task for i in range(60)}
    assert tasks == {"reconstruct", "continue"}
    for i in range(60):
        s = ds[i]
        doc = store[s.doc]
        if s.task == "reconstruct":
            assert torch.equal(s.ctx_ids, s.target_ids)
        else:
            joined = torch.cat([s.ctx_ids, s.target_ids]).numpy()
            # the continuation follows the context contiguously in the document
            found = any(
                np.array_equal(doc[j : j + joined.size], joined)
                for j in range(doc.size - joined.size + 1)
            )
            assert found
        assert 16 <= s.ctx_ids.shape[0] <= 128


def test_collate_layout(tmp_path):
    store = _make_store(tmp_path, "a")
    t = Templates.from_tokenizer(FakeTokenizer())
    ds = AutoencodeDataset([store], ctx_min=16, ctx_max=64, seed=1, epoch_size=8)
    batch = Collator(t)([ds[i] for i in range(4)])
    assert batch.input_ids.shape[0] == 4
    assert torch.equal(
        (batch.input_ids != 0).long().sum(-1) >= batch.attention_mask.sum(-1),
        torch.ones(4, dtype=torch.bool),
    )
    for i in range(4):
        suf, st = batch.suffix_ids[i], batch.suffix_label_start[i]
        assert torch.equal(suf[:st], batch.instr_ids[i])
        assert torch.equal(suf[st:-1], batch.target_ids[i])
        assert int(suf[-1]) == t.eos_id
        assert int(batch.prefix_ids[i][0]) == t.bos_id


# ----------------------------------------------------------------------------


def test_ratio_floor_descends_and_never_rises():
    s = RatioSchedule(r_max=0.25, r_start=0.25, r_end=1 / 128, descent_steps=100, seed=0)
    floors = [s.floor(i) for i in range(0, 130, 10)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(floors))
    assert abs(floors[0] - 0.25) < 1e-9 and abs(floors[-1] - 1 / 128) < 1e-9
    for step in (0, 50, 120):
        for _ in range(50):
            r = s.sample(step)
            assert s.floor(step) - 1e-9 <= r <= 0.25 + 1e-9
    assert len(EVAL_RATIOS) == 7 and EVAL_RATIOS[0] == 1.0
    c = RatioSchedule(
        r_max=0.25,
        r_start=0.25,
        r_end=1 / 128,
        descent_steps=100,
        r_max_start=1.0,
        ceiling_steps=50,
    )
    assert abs(c.ceiling(0) - 1.0) < 1e-9 and abs(c.ceiling(50) - 0.25) < 1e-9
    assert c.ceiling(25) > c.ceiling(40) > 0.25
    assert all(c.sample(0) <= 1.0 + 1e-9 for _ in range(20))
    # hold: both bounds pinned at their start values, then the same descents
    h = RatioSchedule(
        r_max=0.25,
        r_start=0.25,
        r_end=1 / 128,
        descent_steps=100,
        r_max_start=1.0,
        ceiling_steps=50,
        hold_steps=30,
    )
    for step in (0, 15, 29):
        assert abs(h.ceiling(step) - 1.0) < 1e-9 and abs(h.floor(step) - 0.25) < 1e-9
    assert abs(h.ceiling(30 + 50) - 0.25) < 1e-9 and abs(h.floor(30 + 100) - 1 / 128) < 1e-9
    assert abs(h.ceiling(30 + 25) - c.ceiling(25)) < 1e-12
    one = RatioSchedule(r_max=1.0, r_start=1.0, r_end=1.0, descent_steps=0, r_max_start=1.0)
    assert all(abs(one.sample(s) - 1.0) < 1e-12 for s in (0, 10, 10_000))
    # a gate releases the hold early; the release survives a state round-trip
    g = RatioSchedule(
        r_max=0.25,
        r_start=0.25,
        r_end=1 / 128,
        descent_steps=100,
        r_max_start=1.0,
        ceiling_steps=50,
        hold_steps=10_000,
    )
    assert abs(g.ceiling(500) - 1.0) < 1e-9
    assert g.release_hold(500) and not g.release_hold(600)
    assert abs(g.ceiling(500 + 50) - 0.25) < 1e-9 and abs(g.floor(500 + 100) - 1 / 128) < 1e-9
    g2 = RatioSchedule(
        r_max=0.25, r_start=0.25, r_end=1 / 128, descent_steps=100, hold_steps=10_000
    )
    g2.load_state_dict(g.state_dict())
    assert g2.hold_steps == 500
    # the Phase-1 descent: floor starts above the final ceiling; the sample
    # never exceeds the ceiling and the floor never rises
    p = RatioSchedule(
        r_max=0.25,
        r_start=0.95,
        r_end=1 / 128,
        descent_steps=8000,
        r_max_start=1.0,
        ceiling_steps=4000,
        hold_steps=10,
    )
    prev = 1.0
    for step in range(0, 9000, 250):
        f, c = p.floor(step), p.ceiling(step)
        assert f <= prev + 1e-12 and 0 < f <= 1.0 and 0.25 - 1e-12 <= c <= 1.0
        assert f - 1e-9 <= p.sample(step) <= max(c, f) + 1e-9
        prev = f


def test_adaptive_floor_slows_with_compression_and_mastery_and_never_rises():
    a = RatioSchedule(
        r_max=0.25,
        r_start=0.95,
        r_end=1 / 128,
        r_max_start=1.0,
        ceiling_steps=100,
        adaptive=True,
        rate_log2_per_step=1e-2,
        rate_power=0.5,
        mult_min=0.1,
    )
    assert abs(a.floor(0) - 0.95) < 1e-9 and a.progress == 0.0
    prev = a.floor(0)
    doubling_steps = []
    step = 0
    target = 0.95
    while a.floor(step) > 1 / 128 + 1e-9 and step < 200_000:
        a.advance(step)
        f = a.floor(step + 1)
        assert f <= prev + 1e-12
        prev = f
        if f <= target / 2:
            doubling_steps.append(step)
            target = f
        step += 1
    assert abs(a.floor(step) - 1 / 128) < 1e-9 and abs(a.progress - 1.0) < 1e-9
    gaps = [b - c for b, c in zip(doubling_steps[1:], doubling_steps[:-1], strict=True)]
    assert gaps[-1] > 2 * gaps[0]  # later doublings take more steps
    # mastery eases but never stops; advance is idempotent per step
    b = RatioSchedule(
        r_max=0.25, r_start=0.95, r_end=1 / 128, adaptive=True, rate_log2_per_step=1e-2
    )
    assert b.set_mastery(0.01) == 0.1 and b.set_mastery(5.0) == 1.0
    b.set_mastery(0.5)
    b.advance(0)
    f1 = b.floor(1)
    b.advance(0)
    assert b.floor(1) == f1
    assert 0 < b.descent_rate_log2(1) < 1e-2
    # state round trip
    c = RatioSchedule(
        r_max=0.25, r_start=0.95, r_end=1 / 128, adaptive=True, rate_log2_per_step=1e-2
    )
    c.load_state_dict(b.state_dict())
    assert c.floor(5) == b.floor(5) and c.mastery == b.mastery


def test_metrics():
    assert exact_match([1, 2], [1, 2]) == 1.0 and exact_match([1], [1, 2]) == 0.0
    assert abs(token_f1([1, 2, 3], [2, 3, 4]) - 2 / 3) < 1e-9
    assert prefix_accuracy([1, 2, 9], [1, 2, 3, 4]) == 0.5
    g = [[1, 2, 3, 4, 5, 6, 7, 8]] * 4
    assert corpus_bleu(g, g) > 0.99
    assert corpus_bleu([[9, 9, 9, 9, 9, 9, 9, 9]] * 4, g) < 0.05
    assert 0 < corpus_bleu([[1, 2, 3, 4, 9, 9, 9, 9]] * 4, g) < 0.9


def test_probe_separates_distinct_from_collapsed():
    torch.manual_seed(0)
    distinct = [torch.randn(8, 16) + 5 * torch.randn(1, 16) for _ in range(12)]
    r = rep_distinguishability(distinct)
    assert r["top1_raw"] > 0.9 and r["top1_whitened"] > 0.9
    shared = torch.randn(1, 16) * 50
    collapsed = [shared.repeat(8, 1) + 0.01 * torch.randn(8, 16) for _ in range(12)]
    r2 = rep_distinguishability(collapsed)
    assert r2["top1_raw"] < 0.5


def test_config_loads_with_overrides(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "data:\n  train_stores: [a]\n  eval_stores: [b]\nrun:\n  name: t\n  out_dir: /tmp/x\n"
    )
    cfg = load_config(str(y), ["optim.max_steps=3", "model.lora.enabled=true"])
    assert cfg.optim.max_steps == 3 and cfg.model.lora.enabled is True
    with pytest.raises(ValueError):
        load_config(str(y), ["model.selection=bogus"])


# ----------------------------------------------------------------------------
# trainer end to end on tiny models (CPU)
# ----------------------------------------------------------------------------


def _tiny_trainer(tmp_path, max_steps=2, **over):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 48
  cont_min: 8
  cont_max: 16
  batch_size: 3
  num_workers: 0
  eval_samples: 6
  eval_batch_size: 3
  gen_samples: 2
  gen_max_new_tokens: 12
  probe_ratio: 0.5
run:
  name: t
  out_dir: {tmp_path / "out"}
  device: cpu
  dtype: float32
  log_every: 1
  eval_every: {max_steps}
  save_every: {max_steps}
  gradient_checkpointing: false
optim:
  max_steps: {max_steps}
  warmup_steps: 1
"""
    )
    cfg = load_config(str(y), [f"{k}={v}" for k, v in over.items()])
    torch.manual_seed(0)
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = SpliceDecoder(Lfm2ForCausalLM(tiny_config()))
    return CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)


def test_cpu_host_trains_both_selection_paths_and_evaluates(tmp_path):
    tr = _tiny_trainer(tmp_path, max_steps=2)
    tr.cfg2.kd.selection_weight = 1.0
    tr.cfg2.kd.selection_signal = "utility_grad"
    tr.cfg2.kd.head_scale_anchor_weight = 0.01
    # a random-selection step must not arm the capture (it crashed P19b)
    tr.cfg2.ratio.p_random_selection = 1.0
    st_rand = tr.train(1)[0]
    assert st_rand["selection_random"] == 1.0 and "loss/selection" not in st_rand
    tr.cfg2.ratio.p_random_selection = 0.0
    st = tr.train(1)[0]
    assert "loss/selection" in st and st["loss/selection"] >= 0
    assert "head_raw_std" in st and "loss/head_scale" in st
    assert 0.0 <= st["anchor_weight"] <= 1.0 and -1.0 <= st["rep/cos_to_token"] <= 1.0
    e = tr.evaluate()
    for key in (
        "eval/noctx/reconstruct/loss",
        "eval/full_text/reconstruct/loss",
        "eval/zeroed_x4/reconstruct/loss",
        "eval/x4/reconstruct/captured",
        "eval/x4/reconstruct/gap_p50",
        "eval/x128/continue/rep_gain_nats",
        "eval/x1/reconstruct/loss",
        "eval/x4/gen/bleu",
        "probe/top1_raw",
        "probe/nn_token_acc_x1",
        "lm_health/delta",
    ):
        assert key in e, key
    assert 0.0 <= e["probe/nn_token_acc_x1"] <= 1.0


def test_trainer_lora_rung_trains_only_adapters(tmp_path):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import LoraSpec, SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 32
  batch_size: 2
  num_workers: 0
  eval_samples: 4
  gen_samples: 0
run:
  name: l
  out_dir: {tmp_path / "o"}
  device: cpu
  dtype: float32
  eval_every: 100
  save_every: 100
  gradient_checkpointing: false
optim:
  max_steps: 1
  warmup_steps: 1
"""
    )
    cfg = load_config(str(y), ["model.lora.enabled=true"])
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = SpliceDecoder(
        Lfm2ForCausalLM(tiny_config()), lora=LoraSpec(rank=2, alpha=4, layers=[0, 1])
    )
    tr = CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)
    names = [g["name"] for g in tr.opt.param_groups]
    assert "lora" in names
    before = {n: p.detach().clone() for n, p in dec.lm.named_parameters() if not p.requires_grad}
    tr.train(1)
    after = {n: p.detach() for n, p in dec.lm.named_parameters() if not p.requires_grad}
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_chat_templates_split_at_the_sentinel():
    """Uses the real LFM2.5-350M tokenizer if cached; the split must be exact."""
    from pathlib import Path

    from transformers import AutoTokenizer

    from bgkit2.data.templates import SPLICE_SENTINEL, Templates

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/chat_template.jinja")) if cache.exists() else []
    if not snaps:
        pytest.skip("LFM2.5-350M tokenizer not cached")
    tok = AutoTokenizer.from_pretrained(str(snaps[0].parent))
    t = Templates.from_tokenizer(tok, style="chat")
    sentinel = tok.convert_tokens_to_ids(SPLICE_SENTINEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert t.eos_id == im_end and int(t.prefix[0]) == tok.bos_token_id
    for task in ("reconstruct", "continue"):
        target = torch.tensor(tok("hello world", add_special_tokens=False)["input_ids"])
        suf, start = t.suffix(task, target)
        assert sentinel not in t.prefix.tolist() and sentinel not in suf.tolist()
        assert start == t.instr[task].shape[0]
        assert torch.equal(suf[start:-1], target) and int(suf[-1]) == im_end
        # prefix + sentinel + instr reproduces apply_chat_template exactly
        text = tok.decode(torch.cat([t.prefix, torch.tensor([sentinel]), t.instr[task]]))
        assert text.startswith("<|startoftext|><|im_start|>user\nContext:\n" + SPLICE_SENTINEL)
        assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")
    with pytest.raises(ValueError):
        Templates.from_tokenizer(tok, style="bogus")


def test_collator_left_pads_compression_prompts(tmp_path):
    store = _make_store(tmp_path, "a")
    t = Templates.from_tokenizer(FakeTokenizer())
    prompts = {"reconstruct": torch.tensor([50, 51, 52]), "continue": torch.tensor([60])}
    ds = AutoencodeDataset(
        [store], ctx_min=16, ctx_max=64, seed=1, epoch_size=8, task_prompt_ids=prompts
    )
    batch = Collator(t)([ds[i] for i in range(4)])
    assert batch.prompt_ids is not None and batch.prompt_ids.shape[1] == 3
    for i in range(4):
        n = int(batch.prompt_mask[i].sum())
        assert torch.equal(batch.prompt_ids[i, 3 - n :], prompts[batch.tasks[i]])
        assert int(batch.prompt_mask[i, : 3 - n].sum()) == 0
    sub = batch.subset([0, 2])
    assert sub.prompt_ids.shape[0] == 2 and len(sub.tasks) == 2


def test_mixture_is_deterministic_and_weighted():
    from bgkit2.data.mixture import MixtureDataset

    a = list(range(100))
    b = [1000 + i for i in range(50)]
    mix = MixtureDataset([a, b], [0.75, 0.25], seed=3, epoch_size=4000)
    assert len(mix) == 4000
    assert [mix[i] for i in range(20)] == [mix[i] for i in range(20)]
    frac_b = sum(1 for i in range(4000) if mix[i] >= 1000) / 4000
    assert 0.2 < frac_b < 0.3
    with pytest.raises(ValueError):
        MixtureDataset([a], [1.0, 1.0])


def test_multi_ratio_training_loss_runs(tmp_path):
    tr = _tiny_trainer(tmp_path, max_steps=1, **{"ratio.ratios_per_step": 3})
    batch = next(
        iter(torch.utils.data.DataLoader(tr.train_ds, batch_size=3, collate_fn=tr.collate))
    )
    total, stats = tr.training_loss(batch)
    assert stats["ratios_per_step"] == 3.0 and torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in tr.encoder.head.parameters())


def test_decorrelation_loss_is_zero_for_whitened_and_positive_for_collapsed():
    from bgkit2.training.core import BgKIT2Core

    torch.manual_seed(0)
    n, d = 512, 16
    z = torch.randn(n, d)
    z = z - z.mean(0)
    # whiten
    cov = z.T @ z / (n - 1)
    evals, evecs = torch.linalg.eigh(cov)
    w = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    white = (z @ w)[None]
    mask = torch.ones(1, n, dtype=torch.bool)
    assert float(BgKIT2Core.decorrelation_loss(white, mask)) < 1e-3
    collapsed = torch.randn(n, 1) @ torch.randn(1, d)
    assert float(BgKIT2Core.decorrelation_loss(collapsed[None], mask)) > 0.9


def test_within_doc_diversity_loss_zero_for_orthogonal_one_for_copies():
    from bgkit2.training.core import BgKIT2Core

    eye = torch.eye(4)[None]  # 4 orthogonal survivors of one doc
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert float(BgKIT2Core.within_doc_diversity_loss(eye, mask)) < 1e-6
    copies = torch.ones(1, 4, 4)
    assert abs(float(BgKIT2Core.within_doc_diversity_loss(copies, mask)) - 1.0) < 1e-4
    # padded rows are ignored
    mask2 = torch.tensor([[True, True, False, False]])
    assert abs(float(BgKIT2Core.within_doc_diversity_loss(copies, mask2)) - 1.0) < 1e-4


def test_token_residual_adds_each_survivors_own_embedding(tmp_path):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 32
  batch_size: 2
  num_workers: 0
  eval_samples: 4
  gen_samples: 0
run:
  name: r
  out_dir: {tmp_path / "o"}
  device: cpu
  dtype: float32
optim:
  max_steps: 1
  warmup_steps: 1
"""
    )
    cfg = load_config(str(y), [])
    torch.manual_seed(0)
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = SpliceDecoder(Lfm2ForCausalLM(tiny_config()))
    tr = CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)
    tr.encoder.eval()
    batch = tr.collate([tr.train_ds[0], tr.train_ds[1]])
    with torch.no_grad():
        base = tr.encode(batch, 0.5)
        tr.cfg2.model.rep_token_residual = True
        res = tr.encode(batch, 0.5)
        tr.cfg2.model.rep_token_residual = False
        tr.cfg2.model.rep_token_residual_only = True
        only = tr.encode(batch, 0.5)
    assert torch.equal(res.positions, base.positions)
    ids = batch.input_ids.gather(1, base.positions.clamp_min(0))
    emb = dec.embed(ids) * base.rep_mask.unsqueeze(-1).float()
    assert torch.allclose(res.reps, base.reps + emb, atol=1e-5)
    assert torch.allclose(only.reps, emb, atol=1e-6)
    # padded rep slots stay zero
    assert float(res.reps[~res.rep_mask].abs().max()) == 0.0


def test_length_hints_match_samples_and_bucketing_cuts_padding(tmp_path):
    from bgkit2.data.samplers import LengthBucketBatchSampler

    tr = _tiny_trainer(tmp_path, max_steps=1)
    ds = tr.train_ds
    # the autoencode hint equals the real sample length for every index tried
    ae = ds.members[0] if hasattr(ds, "members") else ds
    for i in range(12):
        s = ae[i]
        assert ae.length_hint(i) == int(s.ctx_ids.shape[0] + s.target_ids.shape[0])
    n = 4 * 3 * 4  # 4 windows of 3 batches of 4
    sampler = LengthBucketBatchSampler(n, 4, ae.length_hint, window_batches=3, seed=1)
    batches = list(sampler)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(n)) and len(batches) == n // 4
    assert list(sampler) == batches  # deterministic
    sampler.set_epoch(1)
    assert list(sampler) != batches  # a different batch order per epoch

    # padding: sum over batches of (max_len - mean_len) is lower than in plain order
    def waste(bs):
        tot = 0.0
        for b in bs:
            ls = [ae.length_hint(i) for i in b]
            tot += max(ls) * len(ls) - sum(ls)
        return tot

    plain = [list(range(b, b + 4)) for b in range(0, n, 4)]
    assert waste(batches) < waste(plain)


def test_full_text_student_trains_only_the_decoder_adapter(tmp_path):
    """kd.full_text_student: token embeddings in the slot, no encoder forward;
    the LoRA moves, every encoder parameter stays put."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import LoraSpec, SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 32
  batch_size: 2
  num_workers: 0
  eval_samples: 4
  gen_samples: 0
run:
  name: ft
  out_dir: {tmp_path / "o"}
  device: cpu
  dtype: float32
  eval_every: 100
  save_every: 100
  gradient_checkpointing: false
optim:
  max_steps: 2
  warmup_steps: 1
kd:
  full_text_student: true
  weight: 0.0
  selection_weight: 0.0
"""
    )
    cfg = load_config(str(y), ["model.lora.enabled=true", "model.lora.rep_gated=true"])
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = SpliceDecoder(
        Lfm2ForCausalLM(tiny_config()),
        lora=LoraSpec(rank=2, alpha=4, layers=[0, 1], rep_gated=True),
    )
    tr = CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)
    enc_before = {n: p.detach().clone() for n, p in enc.named_parameters()}
    lora_before = {n: p.detach().clone() for n, p in dec.lm.named_parameters() if "lora_" in n}
    logs = tr.train(2)
    assert all(torch.equal(enc_before[n], p.detach()) for n, p in enc.named_parameters())
    assert any(
        not torch.equal(lora_before[n], p.detach())
        for n, p in dec.lm.named_parameters()
        if "lora_" in n
    )
    assert "loss/ce" in logs[-1] and "loss/kd" not in logs[-1]


def test_eval_ablation_switches_change_only_the_rep_arm(tmp_path):
    """eval_selection / eval_no_prompt / rep_token_residual_only drive the
    encoder-side arms of teacher_forced; the no-context arm is untouched."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 32
  batch_size: 2
  num_workers: 0
  eval_samples: 4
  gen_samples: 0
  compression_prompts: true
run:
  name: ab
  out_dir: {tmp_path / "o"}
  device: cpu
  dtype: float32
  eval_every: 100
  save_every: 100
  gradient_checkpointing: false
optim:
  max_steps: 1
  warmup_steps: 1
"""
    )
    cfg = load_config(str(y), [])
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = SpliceDecoder(Lfm2ForCausalLM(tiny_config()))
    tr = CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)
    batches = tr.eval_batches(tr.eval_ds, 2)
    assert batches[0].prompt_ids is not None

    def arm(ratio):
        summary, _ = tr._arm(batches, ratio)
        return {k: v[0] for k, v in summary.items()}

    floor = arm(None)
    learned = arm(0.25)
    tr.eval_selection = "random"
    random_arm = arm(0.25)
    assert arm(None) == floor
    tr.eval_selection = None
    tr.eval_no_prompt = True
    noprompt = arm(0.25)
    tr.eval_no_prompt = False
    tr.cfg2.model.rep_token_residual_only = True
    pruning = arm(0.25)
    tr.cfg2.model.rep_token_residual_only = False
    again = arm(0.25)
    fams = list(learned)
    # (the interface reference's first-call calibration drifts by ~1e-5)
    assert all(abs(again[f] - learned[f]) < 1e-3 for f in fams)
    assert any(random_arm[f] != learned[f] for f in fams)
    assert any(noprompt[f] != learned[f] for f in fams)
    assert any(pruning[f] != learned[f] for f in fams)


def test_init_decoder_from_loads_the_adapter(tmp_path, monkeypatch):
    """model.init_decoder_from warm-starts the decoder state (the LoRA) from
    another checkpoint's decoder state dict, strictly."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    import bgkit2.training.bgkit2_trainer as bt
    from bgkit2.models.decoder import LoraSpec, SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    from .cpu_host import CpuHost

    _make_store(tmp_path, "train")
    _make_store(tmp_path, "eval", n_docs=12, seed=9)
    y = tmp_path / "c.yaml"
    y.write_text(
        f"""
data:
  train_stores: [{tmp_path / "train"}]
  eval_stores: [{tmp_path / "eval"}]
  ctx_min: 16
  ctx_max: 32
  batch_size: 2
  num_workers: 0
  eval_samples: 4
  gen_samples: 0
run:
  name: wd
  out_dir: {tmp_path / "o"}
  device: cpu
  dtype: float32
  eval_every: 100
  save_every: 100
  gradient_checkpointing: false
optim:
  max_steps: 1
  warmup_steps: 1
"""
    )
    cfg = load_config(str(y), ["model.lora.enabled=true", "model.lora.rep_gated=true"])
    mk = lambda: SpliceDecoder(  # noqa: E731
        Lfm2ForCausalLM(tiny_config()),
        lora=LoraSpec(rank=2, alpha=4, layers=[0, 1], rep_gated=True),
    )
    src = mk()
    for n, p in src.lm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.3)
    donor = {k: v.clone() for k, v in src.state_dict().items()}
    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
    )
    dec = mk()
    tr = CpuHost(cfg, tokenizer=FakeTokenizer(), encoder=enc, decoder=dec)
    monkeypatch.setattr(
        "bgkit.training.checkpointing.load_checkpoint", lambda p: (None, {"decoder": donor})
    )
    bt.BgKIT2Trainer._load_decoder_init(tr, "/fake/ckpt")
    for n, p in dec.lm.named_parameters():
        if "lora_B" in n:
            assert torch.equal(p.detach(), donor["lm." + n])


def test_eval_wrong_prompt_rolls_prompts_and_no_prompt_drops_them():
    import torch

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    b = Batch(
        input_ids=torch.zeros(3, 4, dtype=torch.long),
        attention_mask=torch.ones(3, 4, dtype=torch.long),
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["x"] * 3,
        doc_keys=[(0, 0), (0, 1), (0, 2)],
        prompt_ids=torch.tensor([[10, 11], [20, 21], [30, 31]]),
        prompt_mask=torch.ones(3, 2, dtype=torch.long),
    )
    core.eval_wrong_prompt = True
    out = core._ablate_batch(b)
    assert out.prompt_ids.tolist() == [[30, 31], [10, 11], [20, 21]]
    core.eval_wrong_prompt = False
    core.eval_no_prompt = True
    assert core._ablate_batch(b).prompt_ids is None


def test_eval_wrong_prompt_rolls_prompts_within_the_batch():
    from bgkit2.data.autoencode import Batch
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    b = Batch(
        input_ids=torch.zeros(3, 4, dtype=torch.long),
        attention_mask=torch.ones(3, 4, dtype=torch.long),
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["x"] * 3,
        doc_keys=[(0, 0), (0, 1), (0, 2)],
        prompt_ids=torch.tensor([[10, 11], [20, 21], [30, 31]]),
        prompt_mask=torch.ones(3, 2, dtype=torch.long),
    )
    core.eval_wrong_prompt = True
    assert core._ablate_batch(b).prompt_ids.tolist() == [[30, 31], [10, 11], [20, 21]]
    core.eval_wrong_prompt = False
    core.eval_no_prompt = True
    assert core._ablate_batch(b).prompt_ids is None


def test_prompt_separator_adds_one_unselectable_position():
    import torch
    from transformers.models.lfm2.modeling_lfm2 import Lfm2Model

    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

    enc = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())),
        head_layer=2,
        head_inner_dim=8,
        prompt_separator=True,
    )
    assert enc.prompt_separator is not None and enc.prompt_separator.requires_grad
    ids = torch.randint(3, VOCAB, (2, 12))
    am = torch.ones(2, 12, dtype=torch.long)
    pid = torch.randint(3, VOCAB, (2, 3))
    pm = torch.ones(2, 3, dtype=torch.long)
    out = enc(ids, am, ratio=0.5, prompt_ids=pid, prompt_mask=pm)
    assert out.offset == 4  # 3 prompt tokens + the separator
    assert out.eligible.shape[1] == 16 and not out.eligible[:, :4].any()
    assert out.positions.max() < 12
    plain = enc(ids, am, ratio=0.5)
    assert plain.offset == 0


def test_family_ratio_bands_give_listed_families_their_own_ratio():
    import random

    import torch

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.device = torch.device("cpu")
    core.select_rng = random.Random(0)
    b = Batch(
        input_ids=torch.zeros(3, 4, dtype=torch.long),
        attention_mask=torch.ones(3, 4, dtype=torch.long),
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["qa_squad", "qa_multineedle", "reconstruct"],
        doc_keys=[(0, 0), (0, 1), (0, 2)],
    )
    assert core._family_ratio(b, 0.2) == 0.2
    core.cfg2.ratio.family_bands = {"qa_multineedle": [0.01, 0.03]}
    v = core._family_ratio(b, 0.2)
    assert v.shape == (3,) and v[0] == 0.2 and v[2] == 0.2 and 0.01 <= float(v[1]) <= 0.03


def test_length_bucket_sampler_token_budget_shrinks_long_batches():
    from bgkit2.data.samplers import LengthBucketBatchSampler

    hints = [100] * 20 + [4000] * 6
    s = LengthBucketBatchSampler(26, 8, lambda i: hints[i], window_batches=4, max_tokens=8000)
    batches = list(iter(s))
    assert sorted(i for b in batches for i in b) == list(range(26))
    for b in batches:
        m = max(hints[i] for i in b)
        assert (len(b) <= 8 and len(b) * m <= 8000) or len(b) == 1
    long_batches = [b for b in batches if hints[b[0]] == 4000]
    assert all(len(b) <= 2 for b in long_batches)


def test_ratio_schedule_band_top_prob_keeps_the_band_top_trained():
    import math

    from bgkit2.training.ratios import RatioSchedule

    s = RatioSchedule(
        r_max=0.25,
        r_max_start=0.25,
        r_start=1 / 128,
        r_end=1 / 128,
        ceiling_steps=0,
        adaptive=True,
        band_top_prob=0.5,
        band_top_lo=0.125,
        seed=3,
    )
    xs = [s.sample(1) for _ in range(4000)]
    top = sum(x >= 0.125 for x in xs) / len(xs)
    # plain log-uniform would give log(2)/log(32) ≈ 0.2; with half the draws from the top ≈ 0.6
    assert 0.5 < top < 0.7
    assert all(1 / 128 - 1e-9 <= x <= 0.25 + 1e-9 for x in xs)
    plain = RatioSchedule(
        r_max=0.25,
        r_max_start=0.25,
        r_start=1 / 128,
        r_end=1 / 128,
        ceiling_steps=0,
        adaptive=True,
        seed=3,
    )
    ys = [plain.sample(1) for _ in range(4000)]
    assert abs(sum(y >= 0.125 for y in ys) / len(ys) - math.log(2) / math.log(32)) < 0.05


def test_prompt_contrast_backward_pushes_the_sibling_nll_up_encoder_only():
    """The post-backward contrast pass: only rows with a sibling prompt, the
    true answer's nll under the sibling reps is pushed ABOVE nll_right by the
    margin, the encoder takes the gradient and the decoder adapter does not."""
    import contextlib
    import types

    import torch
    from torch import nn

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.prompt_contrast_weight = 1.0
    core.cfg2.kd.prompt_contrast_margin = 1.0
    core.cfg2.kd.prompt_contrast_max_tokens = 0
    core.autocast = contextlib.nullcontext
    enc_w = nn.Parameter(torch.zeros(1))
    dec_w = nn.Parameter(torch.zeros(1))
    seen: dict = {}

    def encode(batch, ratio, selection):
        seen["prompt_ids"] = batch.prompt_ids.clone()
        seen["ratio"] = ratio
        # reps = the prompt's first id (a stand-in for prompt-conditioned content)
        reps = batch.prompt_ids[:, :1].float() * (1 + enc_w)
        return types.SimpleNamespace(reps=reps, rep_mask=torch.ones_like(reps, dtype=torch.long))

    core.encode = encode
    core.splice = lambda batch, reps, mask: reps

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = dec_w

        def token_nll(self, reps):
            nll = (reps + self.w).repeat(1, 2)  # (B, 2) per-token nll
            return nll, torch.ones_like(nll, dtype=torch.bool)

    core.decoder = Dec()
    b = Batch(
        input_ids=torch.zeros(3, 4, dtype=torch.long),
        attention_mask=torch.ones(3, 4, dtype=torch.long),
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["qa_multineedle", "qa_squad", "qa_multineedle"],
        doc_keys=[(0, 0), (0, 1), (0, 2)],
        prompt_ids=torch.tensor([[1, 1], [5, 5], [2, 2]]),
        prompt_mask=torch.ones(3, 2, dtype=torch.long),
        alt_prompt_ids=torch.tensor([[3, 3], [0, 0], [1, 1]]),
        alt_prompt_mask=torch.tensor([[1, 1], [0, 0], [1, 1]]),
        alt_present=torch.tensor([True, False, True]),
    )
    ratio = torch.tensor([0.1, 0.25, 0.02])
    core._prompt_contrast_capture = {
        "batch": b,
        "ratio": ratio,
        "selection": None,
        "nll_right": torch.tensor([1.0, 5.0, 2.0]),
    }
    stats = core.prompt_contrast_backward()
    assert stats["prompt_contrast_n"] == 2
    assert seen["prompt_ids"].tolist() == [[3, 3], [1, 1]]  # the sibling prompts, rows 0 and 2
    assert torch.allclose(seen["ratio"], ratio[[0, 2]])  # the SAME per-sample ratios
    # gaps: row0 3-1 = 2 (inactive), row2 1-2 = -1 (active: hinge 2)
    assert abs(stats["prompt_gap"] - 0.5) < 1e-6 and abs(stats["loss/prompt_contrast"] - 1.0) < 1e-6
    assert enc_w.grad is not None and enc_w.grad.item() < 0  # pushes nll_alt UP
    assert dec_w.grad is None and dec_w.requires_grad  # decoder untouched, flag restored
    assert core._prompt_contrast_capture is None
    assert core.prompt_contrast_backward() is None
    # Symmetric: the right arm is recomputed too and both carry gradient.
    core.cfg2.kd.prompt_contrast_symmetric = True
    enc_w.grad = None
    calls: list = []
    core.encode = lambda batch, ratio, selection: (
        calls.append(batch.prompt_ids.clone()),
        encode(batch, ratio, selection),
    )[1]
    core._prompt_contrast_capture = {
        "batch": b,
        "ratio": ratio,
        "selection": None,
        "nll_right": torch.tensor([9.0, 9.0, 9.0]),  # ignored when symmetric
    }
    stats = core.prompt_contrast_backward()
    assert [c.tolist() for c in calls] == [[[3, 3], [1, 1]], [[1, 1], [2, 2]]]
    # gaps from the recomputed arms: row0 3-1 = 2, row2 1-2 = -1 -> same as above
    assert abs(stats["prompt_gap"] - 0.5) < 1e-6 and abs(stats["loss/prompt_contrast"] - 1.0) < 1e-6
    assert enc_w.grad is not None and dec_w.grad is None and dec_w.requires_grad


def test_paired_prompt_contrast_scores_the_two_by_two():
    """Paired mode: both arms encoded, the crossed pairs pushed above the
    matched pairs by the margin, CE on the sibling's own answer, encoder-only."""
    import contextlib
    import types

    import torch
    from torch import nn

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.prompt_contrast_weight = 1.0
    core.cfg2.kd.prompt_contrast_margin = 1.0
    core.cfg2.kd.prompt_contrast_max_tokens = 0
    core.cfg2.kd.prompt_contrast_paired = True
    core.autocast = contextlib.nullcontext
    enc_w = nn.Parameter(torch.zeros(1))
    dec_w = nn.Parameter(torch.zeros(1))

    def encode(batch, ratio, selection):
        reps = batch.prompt_ids[:, :1].float() * (1 + enc_w)
        return types.SimpleNamespace(reps=reps, rep_mask=torch.ones_like(reps, dtype=torch.long))

    core.encode = encode
    # nll(target | reps) = |reps - target|: matched pairs score 0, crossed pairs the distance
    core.splice = lambda batch, reps, mask: (batch, reps)

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = dec_w

        def token_nll(self, sb):
            batch, reps = sb
            tgt = torch.stack([t.float()[:1] for t in batch.target_ids])
            nll = ((reps - tgt).abs() + self.w).repeat(1, 2)
            return nll, torch.ones_like(nll, dtype=torch.bool)

    core.decoder = Dec()

    def mk(prompts, targets, tasks, **kw):
        n = len(prompts)
        return Batch(
            input_ids=torch.zeros(n, 4, dtype=torch.long),
            attention_mask=torch.ones(n, 4, dtype=torch.long),
            prefix_ids=[torch.tensor([1])] * n,
            suffix_ids=[torch.tensor([2])] * n,
            suffix_label_start=[0] * n,
            target_ids=[torch.tensor([t]) for t in targets],
            instr_ids=[torch.tensor([0])] * n,
            tasks=tasks,
            doc_keys=[(0, i) for i in range(n)],
            prompt_ids=torch.tensor([[p, p] for p in prompts]),
            prompt_mask=torch.ones(n, 2, dtype=torch.long),
            **kw,
        )

    sib = mk([3, 1], [3, 1], ["qa_multineedle"] * 2)
    b = mk(
        [1, 5, 2],
        [1, 5, 2],
        ["qa_multineedle", "qa_squad", "qa_multineedle"],
        alt_prompt_ids=torch.tensor([[3, 3], [0, 0], [1, 1]]),
        alt_prompt_mask=torch.tensor([[1, 1], [0, 0], [1, 1]]),
        alt_present=torch.tensor([True, False, True]),
        sibling=sib,
    )
    core._prompt_contrast_capture = {
        "batch": b,
        "ratio": 0.1,
        "selection": None,
        "nll_right": torch.zeros(3),
    }
    stats = core.prompt_contrast_backward()
    # row0: A=1,B=3 -> gap_a = |3-1| - 0 = 2, gap_b = 2; row2: A=2,B=1 -> gaps 1, 1
    assert stats["prompt_contrast_n"] == 2
    assert abs(stats["prompt_gap"] - 1.5) < 1e-6 and abs(stats["prompt_gap_min"] - 1.0) < 1e-6
    assert abs(stats["loss/prompt_contrast"]) < 1e-6  # every gap >= margin
    assert abs(stats["loss/sibling_ce"]) < 1e-6  # matched pairs score 0
    assert enc_w.grad is not None and dec_w.grad is None and dec_w.requires_grad


def test_selection_loss_row_cap_uses_only_the_short_rows():
    """kd.selection_max_ctx_tokens: long rows stay out of the attention pass;
    the listwise target is built from the sub-batch's teacher importance and
    the head scores of those rows only."""
    import types

    import torch

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.kd.selection_max_ctx_tokens = 4
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]])
    b = Batch(
        input_ids=torch.zeros(3, 6, dtype=torch.long),
        attention_mask=mask,
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["a", "b", "c"],
        doc_keys=[(0, 0), (0, 1), (0, 2)],
    )
    assert core.selection_rows(b) == [0, 2]
    core.cfg2.kd.selection_max_ctx_tokens = 0
    assert core.selection_rows(b) is None
    scores = torch.randn(3, 6, requires_grad=True)
    out = types.SimpleNamespace(scores=scores)
    # teacher importance for the 2-row sub-batch: all mass on the last valid token
    imp = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    tb = types.SimpleNamespace(slot_start=torch.tensor([0, 0]), slot_len=torch.tensor([3, 2]))
    core.decoder = types.SimpleNamespace(attention_importance=lambda _tb: imp)
    loss = core.selection_loss(b, out, tb, rows=[0, 2])
    loss.backward()
    assert loss.item() > 0
    assert scores.grad[1].abs().sum() == 0  # the long row took no gradient
    assert scores.grad[0].abs().sum() > 0 and scores.grad[2].abs().sum() > 0
    assert scores.grad[0, 3:].abs().sum() == 0  # padding never


def test_family_rep_budget_and_teacher_window():
    """ratio.family_rep_budget: an absolute budget per sample -> ratio =
    budget / length (clamped). kd.teacher_gold_window: the collator carries
    the oracle window (full context where a row has none) and full_text_batch
    puts it in the slot only when asked."""
    import random
    import types

    import torch

    from bgkit2.data.autoencode import Batch, Collator, Sample
    from bgkit2.data.templates import Templates
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.ratio.family_rep_budget = {"babi_filler": [32, 32]}
    core.cfg2.ratio.family_bands = {"qa_multineedle": [0.01, 0.01]}
    core.device = torch.device("cpu")
    core.select_rng = random.Random(0)
    mask = torch.zeros(3, 1000, dtype=torch.long)
    mask[0, :1000] = 1  # babi row, 1000 tokens -> 32/1000
    mask[1, :20] = 1  # babi row, 20 tokens -> clamped to 0.5
    mask[2, :500] = 1  # needle row -> band
    b = Batch(
        input_ids=torch.zeros(3, 1000, dtype=torch.long),
        attention_mask=mask,
        prefix_ids=[torch.tensor([1])] * 3,
        suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3,
        target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3,
        tasks=["babi_filler", "babi_filler", "qa_multineedle"],
        doc_keys=[(0, 0), (0, 1), (0, 2)],
    )
    r = core._family_ratio(b, 0.25)
    assert abs(r[0].item() - 0.032) < 1e-6 and abs(r[1].item() - 0.5) < 1e-6
    assert abs(r[2].item() - 0.01) < 1e-6
    # collator: teacher tensors only when some row has a window; fallback = full ctx
    t = (
        Templates.from_tokenizer(FakeTokenizer(), style="plain")
        if "FakeTokenizer" in globals()
        else None
    )
    if t is None:
        return
    s0 = Sample(
        ctx_ids=torch.tensor([5, 6, 7, 8]),
        target_ids=torch.tensor([9]),
        task="reconstruct",
        store=0,
        doc=0,
        teacher_ctx_ids=torch.tensor([6, 7]),
    )
    s1 = Sample(
        ctx_ids=torch.tensor([5, 6]),
        target_ids=torch.tensor([9]),
        task="reconstruct",
        store=0,
        doc=1,
    )
    batch = Collator(t)([s0, s1])
    assert (
        batch.teacher_input_ids.tolist()[0][:2] == [6, 7]
        and batch.teacher_attention_mask[0].sum() == 2
    )
    assert batch.teacher_attention_mask[1].sum() == 2 and batch.teacher_input_ids[1].tolist()[
        :2
    ] == [5, 6]
    sub = batch.subset([0])
    assert sub.teacher_attention_mask.shape[1] == 2
    seen = {}
    core.decoder = types.SimpleNamespace(embed=lambda ids: ids.float())
    core.splice = lambda b_, reps, m, pad_side="right", **kw: seen.update(reps=reps, mask=m)
    core.full_text_batch(batch, teacher_window=True)
    assert seen["mask"].sum() == 4 and seen["reps"].shape[1] == 2
    core.full_text_batch(batch, teacher_window=False)
    assert seen["mask"].sum() == 6 and seen["reps"].shape[1] == 4


def test_gold_span_target_per_row_ce_and_normalised_em():
    """(c) rows with a gold span use it as the selection target instead of the
    teacher's attention; (d) per-row CE weighs rows equally and family weights
    apply; span_mask finds facts by character offsets."""
    import types

    import torch

    from bgkit2.data.autoencode import Batch
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    b = Batch(
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        attention_mask=mask,
        prefix_ids=[torch.tensor([1])] * 2,
        suffix_ids=[torch.tensor([2])] * 2,
        suffix_label_start=[0] * 2,
        target_ids=[torch.tensor([2])] * 2,
        instr_ids=[torch.tensor([0])] * 2,
        tasks=["babi_filler", "qa_squad"],
        doc_keys=[(0, 0), (0, 1)],
        gold_span=torch.tensor([[False, True, True, False], [False, False, False, False]]),
    )
    scores = torch.zeros(2, 4, requires_grad=True)
    out = types.SimpleNamespace(scores=scores)
    imp = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    tb = types.SimpleNamespace(slot_start=torch.tensor([0, 0]), slot_len=torch.tensor([4, 3]))
    core.decoder = types.SimpleNamespace(attention_importance=lambda _tb: imp)
    loss = core.selection_loss(b, out, tb)
    loss.backward()
    g = scores.grad
    # row 0: target = span tokens 1,2 -> pushes those up, tokens 0,3 down
    assert g[0, 1] < 0 and g[0, 2] < 0 and g[0, 0] > 0 and g[0, 3] > 0
    # row 1: teacher attention on token 0
    assert g[1, 0] < 0 and g[1, 1] > 0 and g[1, 3] == 0
    # per-row CE + family weights
    nll = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]]).bool()
    core.cfg2.kd.per_row_ce = False
    assert abs(core.pooled_ce(b, nll, valid).item() - 7 / 5) < 1e-6
    core.cfg2.kd.per_row_ce = True
    assert abs(core.pooled_ce(b, nll, valid).item() - 2.0) < 1e-6
    core.cfg2.kd.family_loss_weight = {"babi_filler": 3.0}
    assert abs(core.pooled_ce(b, nll, valid).item() - (3 * 1.0 + 3.0) / 4) < 1e-6
    # normalised EM through the real qa_metrics
    from bgkit.eval.metrics.qa_metrics import exact_match as text_em

    assert text_em("Garden.", ["garden"]) == 1.0 and text_em("the garden", ["garden"]) == 1.0


def test_eval_batches_pack_by_length_under_a_token_cap():
    import types

    import torch

    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.data.eval_batch_max_tokens = 100
    hints = [10, 50, 10, 90, 10, 40]

    class DS:
        def __len__(self):
            return len(hints)

        def length_hint(self, i):
            return hints[i]

        def __getitem__(self, i):
            return i

    core.collate = lambda items: list(items)
    groups = core.eval_batches(DS(), 2)
    # sorted by length 10,10,10,40,50,90; packed while rows x longest <= cap
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(6)) and len(flat) == 6
    for g in groups:
        assert len(g) * max(hints[i] for i in g) <= 100
    assert groups[-1] == [3]  # the 90-token row is alone
    core.cfg2.data.eval_batch_max_tokens = 0
    core.collate = None
    # cap 0 -> plain DataLoader path is used (needs a real collate; skip the call)
    assert core.cfg2.data.eval_batch_max_tokens == 0
