"""BgKIT2Trainer on BaseTrainer, exercised with tiny models and a minimal Hydra cfg."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

from bgkit2.data.token_store import TokenStoreWriter
from bgkit2.models.decoder import SpliceDecoder
from bgkit2.models.encoder import BgKIT2Encoder
from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone
from bgkit2.training.bgkit2_trainer import BgKIT2Trainer

from .conftest import VOCAB, tiny_config
from .test_data_and_training import FakeTokenizer


def _store(tmp_path, name, n=24, seed=0):
    rng = np.random.default_rng(seed)
    w = TokenStoreWriter(tmp_path / name, meta={})
    for _ in range(n):
        w.add(rng.integers(3, VOCAB, size=int(rng.integers(70, 200))))
    w.close()
    return str(tmp_path / name)


class _TinyTrainer(BgKIT2Trainer):
    def build_encoder(self):
        torch.manual_seed(0)
        return BgKIT2Encoder(
            Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())), head_layer=2, head_inner_dim=8
        )

    def build_decoder(self):
        torch.manual_seed(1)
        lm = Lfm2ForCausalLM(tiny_config())
        lm.config.use_cache = False
        return SpliceDecoder(lm)


@pytest.fixture
def hydra_cfg(tmp_path):
    train = _store(tmp_path, "train")
    ev = _store(tmp_path, "eval", n=12, seed=9)
    return OmegaConf.create(
        {
            "seed": 3,
            "run_name": "tiny",
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "resume_checkpoint": "none",
            "wandb": {"enabled": False},
            "compute": {"memory": {"log_every_steps": 0}},
            "training": {
                "phase": "bgkit2",
                "lr": 1e-3,
                "max_steps": 2,
                "warmup_steps": 1,
                "eval_every": 2,
                "save_every": 2,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "optimizer": "adamw",
                "gradient_checkpointing": False,
                "bgkit2": {
                    "data": {
                        "train_stores": [train],
                        "eval_stores": [ev],
                        "ctx_min": 16,
                        "ctx_max": 48,
                        "cont_min": 8,
                        "cont_max": 16,
                        "batch_size": 3,
                        "num_workers": 0,
                        "eval_samples": 6,
                        "eval_batch_size": 3,
                        "gen_samples": 2,
                        "gen_max_new_tokens": 12,
                        "probe_ratio": 0.5,
                    },
                    "ratio": {"ceiling_steps": 10},
                    "run": {"dtype": "float32"},
                },
            },
        }
    )


def test_bgkit2_trainer_setup_step_eval_checkpoint(hydra_cfg, monkeypatch):
    monkeypatch.setattr(
        "bgkit2.training.bgkit2_trainer.load_tokenizer", lambda *_a, **_k: FakeTokenizer()
    )
    tr = _TinyTrainer(hydra_cfg)
    tr.setup()
    assert tr.SPLICES_REPS and tr.current_step == 0
    assert [g["name"] for g in tr.optimizer.param_groups] == ["head", "projection"]
    batch = next(iter(tr.train_dataloader))
    tr._accum_steps = 1
    stats = tr._forward_backward(batch)
    assert "loss" in stats and "loss/anchor" in stats and "rep/norm_ratio" in stats
    assert any(p.grad is not None for p in tr.trainable_parameters())
    m = tr.evaluate()
    assert "eval/x4/reconstruct/rep_gain_nats" in m and "probe/top1_raw" in m
    assert "lm_health/delta" in m
    norm = tr._normalize_eval_metrics(m)
    assert all(k.startswith("eval/") for k in norm)
    assert set(tr.checkpoint_models()) == {"encoder", "decoder"}
    st = tr._build_training_state(None, 0, None)
    assert "bgkit2_ratio_schedule" in st and "bgkit2_select_rng" in st
    tr._restore_training_state(st)


def test_training_state_round_trips_through_json(hydra_cfg, monkeypatch):
    import json
    import random

    monkeypatch.setattr(
        "bgkit2.training.bgkit2_trainer.load_tokenizer", lambda *_a, **_k: FakeTokenizer()
    )
    tr = _TinyTrainer(hydra_cfg)
    tr.setup()
    for _ in range(5):
        tr.select_rng.random()
        tr.ratio_schedule.sample(3)
    st = json.loads(json.dumps(tr._build_training_state(None, 0, None)))
    expected = [tr.select_rng.random() for _ in range(3)]
    expected_r = [tr.ratio_schedule.sample(4) for _ in range(3)]
    tr.select_rng = random.Random(0)
    tr._restore_training_state(st)
    assert [tr.select_rng.random() for _ in range(3)] == expected
    assert [tr.ratio_schedule.sample(4) for _ in range(3)] == expected_r


def test_encoder_warm_start_loads_strictly(hydra_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "bgkit2.training.bgkit2_trainer.load_tokenizer", lambda *_a, **_k: FakeTokenizer()
    )
    src = _TinyTrainer(hydra_cfg)
    src.setup()
    state = {k: v.clone() for k, v in src.encoder.state_dict().items()}
    with torch.no_grad():
        src.encoder.survive_embedding.add_(1.0)
    marked = {k: v.clone() for k, v in src.encoder.state_dict().items()}
    monkeypatch.setattr(
        "bgkit.training.checkpointing.load_checkpoint", lambda p: (None, {"encoder": marked})
    )
    cfg = hydra_cfg.copy()
    cfg.training.bgkit2.model = {"init_encoder_from": str(tmp_path)}
    tr = _TinyTrainer(cfg)
    tr.setup()
    assert torch.allclose(tr.encoder.survive_embedding, marked["survive_embedding"])
    assert not torch.allclose(tr.encoder.survive_embedding, state["survive_embedding"])
    assert tr._startup_sources["encoder"] == str(tmp_path)


def test_live_config_handlers_change_the_curriculum_and_losses(hydra_cfg, monkeypatch):
    tr = _TinyTrainer(hydra_cfg)
    tr.setup()
    tr.apply_live_config(
        {
            "anchor_weight": 0.7,
            "kd_weight": 2.0,
            "selection_weight": 0.5,
            "selection_every": 3,
            "p_random_selection": 0.25,
            "ratio_rate_log2_per_step": 0.004,
            "ratio_mult_min": 0.2,
            "mastery_target_gap": 0.1,
            "ratio_ceiling_steps": 123,
            "ratio_rate_power": 9.0,  # out of range: ignored
            "kd_weight_typo": 1.0,  # unknown: ignored
        }
    )
    assert tr.cfg2.anchor.weight == 0.7 and tr.cfg2.kd.weight == 2.0
    assert tr.cfg2.kd.selection_weight == 0.5 and tr.cfg2.kd.selection_every == 3
    assert tr.cfg2.ratio.p_random_selection == 0.25 and tr.cfg2.ratio.mastery_target_gap == 0.1
    assert tr.ratio_schedule.rate_log2_per_step == 0.004 and tr.ratio_schedule.mult_min == 0.2
    assert tr.ratio_schedule.ceiling_steps == 123 and tr.ratio_schedule.rate_power == 0.5


def test_encoder_freeze_phase_holds_the_encoder_then_releases_it(hydra_cfg, monkeypatch):
    """optim.encoder_freeze_steps (P24's decoder-first phase): before the
    boundary every trainable encoder param has requires_grad False, at the
    boundary they are restored — exactly once each way."""
    monkeypatch.setattr(
        "bgkit2.training.bgkit2_trainer.load_tokenizer", lambda *_a, **_k: FakeTokenizer()
    )
    hydra_cfg.training.bgkit2.optim = {"encoder_freeze_steps": 2}
    tr = _TinyTrainer(hydra_cfg)
    tr.setup()
    enc_trainable = [p for p in tr.encoder.parameters() if p.requires_grad]
    assert enc_trainable
    tr.global_step = 0
    assert tr._apply_encoder_freeze() is True
    assert all(not p.requires_grad for p in enc_trainable)
    tr.global_step = 1
    assert tr._apply_encoder_freeze() is True
    assert all(not p.requires_grad for p in enc_trainable)
    tr.global_step = 2
    assert tr._apply_encoder_freeze() is False
    assert all(p.requires_grad for p in enc_trainable)
    assert len(tr._encoder_trainable) == len(enc_trainable)


# ----------------------------------------------------------------------------
# Multi-level: Stage C tree training and the grouped Stage B path
# ----------------------------------------------------------------------------


class _SmallIds:
    """Tokenizer stand-in that keeps ids inside the tiny vocab (257). The real
    LFM2.5 tokenizers emit ids in the tens of thousands, which the tiny models
    turn into a CUDA gather assert."""

    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": [3 + (ord(c) % 40) for c in str(text)][:64]}

    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()


def _patch_tokenizers(monkeypatch):
    monkeypatch.setattr(
        "bgkit2.training.bgkit2_trainer.load_tokenizer", lambda *_a, **_k: FakeTokenizer()
    )
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _SmallIds.from_pretrained)


class _TreeTrainer(_TinyTrainer):
    def build_encoder(self):
        torch.manual_seed(0)
        return BgKIT2Encoder(
            Lfm2BidirectionalBackbone(Lfm2Model(tiny_config())),
            head_layer=2,
            head_inner_dim=8,
            repro_head=True,
        )

    def build_decoder(self, attn_implementation: str = "sdpa", lora_cfg=None):
        """Honour cfg.model.lora so Stage D's 'only the adapter trains' can be
        tested; _TinyTrainer ignores it."""
        from bgkit2.models.decoder import LoraSpec

        torch.manual_seed(1)
        lm = Lfm2ForCausalLM(tiny_config())
        lm.config.use_cache = False
        lc = lora_cfg if lora_cfg is not None else self.cfg2.model.lora
        lora = LoraSpec(rank=int(lc.rank), alpha=int(lc.alpha)) if lc.enabled else None
        return SpliceDecoder(lm, lora=lora)


def _tree_store(tmp_path, n=6):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = torch.Generator().manual_seed(4)
    rows = []
    for i in range(n):
        paths = ["a.py", "pkg/b.py", "pkg/c.py", "pkg/sub/d.py"]
        rows.append(
            {
                "repo": f"r{i}",
                "question": "Which file defines the class Foo?",
                "answer": "pkg/sub/d.py",
                "n_files": 4,
                "paths": paths,
                "file_ids": [torch.randint(3, 50, (20,), generator=rng).tolist() for _ in paths],
                "split": "eval" if i == n - 1 else "train",
            }
        )
    p = tmp_path / "trees.parquet"
    pq.write_table(pa.table({k: pa.array([r[k] for r in rows]) for k in rows[0]}), p)
    return str(p)


def test_repo_tree_dataset_and_collate(tmp_path):
    from bgkit2.data.repo_tree import RepoTreeDataset, collate_trees

    ds = RepoTreeDataset(_tree_store(tmp_path), "train", max_files=3, file_tokens=8,
                         shuffle=False)
    assert len(ds) == 5
    item = ds[0]
    # max_files below the repo's size keeps the ANSWER file in the tree (it
    # replaces the last kept file); the rest is the store's order.
    assert item.paths[:2] == ["a.py", "pkg/b.py"] and item.answer in item.paths
    assert len(item.paths) == 3 and item.file_ids[0].shape[0] == 8
    batch = collate_trees([ds[0], ds[1]])
    assert len(batch) == 2 and batch[1].answer == "pkg/sub/d.py"


def test_stage_c_tree_training_and_eval(hydra_cfg, monkeypatch, tmp_path):
    """The decoder reads the ROOT of a recursive tree encoding and is scored on
    the answer path; eval reports the zeroed-root floor beside it."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.repo_tree_store = _tree_store(tmp_path)
    b.data.repo_tree_max_files = 4
    b.data.repo_tree_file_tokens = 16
    b.data.batch_size = 2
    b.model = {"repro_head": True, "lora": {"enabled": False}}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 3}
    tr = _TreeTrainer(cfg)
    tr.setup()
    from bgkit2.data.repo_tree import RepoTreeDataset

    assert isinstance(tr.train_ds, RepoTreeDataset)
    batch = next(iter(tr.train_dataloader))
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and "loss/tree_nav" in stats and stats["tree/files"] == 4.0
    total.backward()  # gradients reach the encoder side, not the frozen decoder
    assert any(p.grad is not None for p in tr.encoder.repro.parameters())
    assert all(p.grad is None for p in tr.decoder.parameters() if p.requires_grad)
    m = tr.evaluate()
    assert "eval/tree/nav_gain_nats" in m and "eval/tree/nav_ce_zeroed" in m
    assert "eval/tree/nav_ce_noslot" in m and "eval/tree/nav_gain_vs_noslot" in m  # honest floor


def test_stage_b_grouped_path_b_matches_teacher_survivor_sets(hydra_cfg, monkeypatch):
    """Grouped Path B: the teacher encodes the flat concatenation of a group,
    the student encodes it as a tree, and the forced survivor sets agree."""
    import copy

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.batch_size = 4
    b.model = {"repro_head": True}
    b.multilevel = {
        "enabled": False,  # avoid the checkpoint load; the teacher is set by hand below
        "teacher_ratio": 0.5,
        "group_size": 2,
        "path_a_prob": 0.0,
        "frac_extras_start": 0.5,
        "frac_extras_end": 0.5,
    }
    tr = _TreeTrainer(cfg)
    tr.setup()
    tr.encoder_teacher = copy.deepcopy(tr.encoder).eval()
    for p in tr.encoder_teacher.parameters():
        p.requires_grad_(False)
    batch = next(iter(tr.train_dataloader)).to(tr.device)
    out = tr.multilevel_distill(batch, step=0)
    assert out is not None
    loss, st = out
    assert torch.isfinite(loss) and st["path"] == 1.0 and st["groups"] >= 1
    assert 0.0 <= st["cos"] <= 1.0


def test_tree_bridge_option_selects_identity_or_repro(hydra_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.repo_tree_store = _tree_store(tmp_path)
    b.data.repo_tree_file_tokens = 16
    b.data.batch_size = 1
    b.model = {"repro_head": True, "lora": {"enabled": False}}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 2, "bridge": "identity"}
    tr = _TreeTrainer(cfg)
    tr.setup()
    x = torch.randn(2, 3, 32)
    assert torch.equal(tr.tree_bridge()(x), x)          # identity passes through
    tr.cfg2.multilevel.bridge = "repro"
    assert tr.tree_bridge() is None                    # None = encoder.reproduce
    tr.cfg2.multilevel.bridge = "bogus"
    with pytest.raises(ValueError, match="bridge"):
        tr.tree_bridge()


def test_freeze_encoder_leaves_only_the_decoder_adapter_in_the_optimizer(hydra_cfg, monkeypatch):
    """Stage D's memory constraint: the encoder tree is out of the optimizer."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.model = {"repro_head": True, "lora": {"enabled": True, "rank": 4, "alpha": 8}}
    b.optim = {"freeze_encoder": True}
    tr = _TreeTrainer(cfg)
    tr.setup()
    names = [g["name"] for g in tr.optimizer.param_groups]
    assert names == ["lora"]
    assert all(not p.requires_grad for p in tr.encoder.parameters())


def test_stage_d_interleaves_tree_steps(hydra_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.repo_tree_store = _tree_store(tmp_path)
    b.data.repo_tree_file_tokens = 16
    b.data.repo_tree_every = 2
    b.data.batch_size = 2
    b.model = {"repro_head": True, "lora": {"enabled": True, "rank": 4, "alpha": 8}}
    b.optim = {"freeze_encoder": True}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 2}
    tr = _TreeTrainer(cfg)
    tr.setup()
    assert tr.tree_dataloader is not None and tr.collate is not tr.tree_dataloader.collate_fn
    corpus_batch = next(iter(tr.train_dataloader))
    tr.global_step = 1
    s1 = tr._forward_backward(corpus_batch)           # odd step: corpus
    assert "loss/tree_nav" not in s1
    tr.global_step = 2
    s2 = tr._forward_backward(corpus_batch)           # even step: swapped for trees
    assert "loss/tree_nav" in s2
    # Stage D eval reports BOTH sides: the tree gain AND the task/corpus eval
    # the mix must preserve (P28's step-6000 eval had only the tree; the
    # pruning metric keys on the task side). eval/loss is the task eval's.
    m = tr.evaluate()
    assert "eval/tree/nav_gain_nats" in m
    assert "eval/loss" not in m or m["eval/loss"] != m["eval/tree/nav_ce"]
    assert any(k.startswith("eval/") and "/tree/" not in k for k in m)


def test_repo_tree_dataset_shuffles_so_the_answer_is_not_always_first(tmp_path):
    """The builder puts the evidence file first; unshuffled, 'first file's
    header' answers the question with no navigation (M3 +9.3 nats at step 250)."""
    from bgkit2.data.repo_tree import RepoTreeDataset

    ds = RepoTreeDataset(_tree_store(tmp_path), "train", max_files=4, file_tokens=8)
    firsts = {ds[0].paths[0] for _ in range(12)}
    assert len(firsts) > 1                      # train: varies across draws
    ev = RepoTreeDataset(_tree_store(tmp_path), "eval", max_files=4, file_tokens=8)
    assert ev[0].paths == ev[0].paths           # eval: fixed per index
    plain = RepoTreeDataset(_tree_store(tmp_path), "train", max_files=4, file_tokens=8,
                            shuffle=False)
    assert plain[0].paths[0] == "a.py"          # unshuffled keeps builder order


def test_repo_tree_dataset_v4_joins_the_per_repo_file_table(tmp_path):
    """v4 whole-repo stores keep the files ONCE per repo in <path>.repos.parquet;
    question rows reference the repo. The answer file survives max_files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bgkit2.data.repo_tree import RepoTreeDataset

    p = str(tmp_path / "trees.parquet")
    pq.write_table(pa.table({
        "repo": ["r1", "r1", "r2"], "question": ["q1", "q2", "q3"],
        "answer": ["pkg/z.py", "a.py", "m.py"], "n_files": [3, 3, 2],
        "split": ["train", "eval", "train"],
    }), p)
    pq.write_table(pa.table({
        "repo": ["r1", "r2"],
        "paths": [["a.py", "pkg/b.py", "pkg/z.py"], ["m.py", "n.py"]],
        "file_ids": pa.array([[[5, 6, 7], [8, 9], [10, 11, 12, 13]], [[1, 2], [3]]],
                             type=pa.list_(pa.list_(pa.int32()))),
    }), p + ".repos.parquet")
    ds = RepoTreeDataset(p, "train", max_files=2, file_tokens=3, shuffle=False)
    assert len(ds) == 2 and set(ds.repos) == {"r1", "r2"}
    it = ds[0]
    assert it.repo == "r1" and len(it.paths) == 2 and "pkg/z.py" in it.paths  # answer kept
    assert all(t.shape[0] <= 3 for t in it.file_ids)
    assert ds.length_hints() == [3 + 2, 2 + 1]  # real token counts, capped per file
    assert ds.length_hint(1) == 3                # callable, like TaskDataset's
    ev = RepoTreeDataset(p, "eval", max_files=10, file_tokens=8)
    assert ev[0].answer == "a.py" and len(ev[0].paths) == 3
    # a PARTIAL node cache: keep only the repos it holds (rows AND walk steps)
    w = RepoTreeDataset(p, "train", max_files=10, file_tokens=8, walk_depth=3, load_files=False)
    n_all = len(w)
    assert w.restrict_to({"r2"}) < n_all and {w.rows[ri]["repo"] for ri, _n, _t in w.steps} == {"r2"}
    assert w[0].repo == "r2" and w.length_hint(0) > 0


def test_stage_d_reads_root_reps_from_the_cache_without_the_encoder(hydra_cfg, monkeypatch, tmp_path):
    """Frozen encoder: root reps pre-computed per repo (bgkit2_cache_repo_trees.py)
    feed the tree batches and the tree eval; the encoder is never called."""
    from safetensors.torch import save_file

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    store = _tree_store(tmp_path)
    b.data.repo_tree_store = store
    b.data.repo_tree_file_tokens = 16
    b.data.repo_tree_every = 2
    b.data.batch_size = 2
    b.model = {"repro_head": True, "lora": {"enabled": True, "rank": 4, "alpha": 8}}
    b.optim = {"freeze_encoder": True}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 2}
    tr = _TreeTrainer(cfg)
    tr.setup()
    d = tr.decoder.embed_tokens.weight.shape[1]
    repos = {r["repo"] for r in tr.tree_eval_ds.rows} | {r["repo"] for r in tr.tree_dataloader.dataset.rows}
    blob = {}
    for name in repos:
        blob[f"{name}/reps"] = torch.randn(3, d, dtype=torch.bfloat16)
        blob[f"{name}/mask"] = torch.ones(3, dtype=torch.bool)
    cache = str(tmp_path / "roots.safetensors")
    save_file(blob, cache)
    tr.load_tree_root_cache(cache)

    def boom(*a, **k):
        raise AssertionError("encoder must not run when the root cache is loaded")

    tr.encoder.forward = boom
    trees = next(iter(tr.tree_dataloader))
    total, stats = tr.tree_training_loss(trees)
    assert torch.isfinite(total) and stats["tree/root_reps"] == 3.0
    m = tr.evaluate_tree(max_items=2)
    assert "eval/tree/nav_gain_nats" in m


def test_path_walk_training_and_eval(hydra_cfg, monkeypatch, tmp_path):
    """PATH WALK: one item per step; the decoder reads the NODE named by the
    step and is scored on the next component; eval reports step accuracy by
    depth (and full greedy walks when the tokenizer can decode)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.repo_tree_store = _tree_store(tmp_path)
    b.data.repo_tree_max_files = 8
    b.data.repo_tree_file_tokens = 16
    b.data.batch_size = 2
    b.model = {"repro_head": True, "lora": {"enabled": False}}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 3, "walk": True,
                    "bridge": "identity"}
    tr = _TreeTrainer(cfg)
    tr.setup()
    ds = tr.train_ds
    assert ds.walk_depth == 3 and len(ds) == len(ds.steps) > len(ds.rows)
    nodes = {n for _ri, n, _t in ds.steps}
    assert "" in nodes and any(n for n in nodes)          # root steps and deeper ones
    item = next(ds[i] for i in range(len(ds)) if ds[i].node)
    q_text, a_text = tr.tree_prompt(item)
    assert f"Current directory: {item.node}" in q_text and a_text.endswith("\n")
    total, stats = tr.training_loss([ds[0], item])
    assert torch.isfinite(total) and "loss/tree_nav" in stats
    total.backward()
    m = tr.evaluate()
    assert "eval/tree/walk_step_exact" in m and "eval/tree/walk_step_exact_depth0" in m
    assert "eval/tree/gen_path_exact" not in m


def test_encoder_adapter_run_with_a_tree_cache_and_live_small_trees(
    hydra_cfg, monkeypatch, tmp_path
):
    """optim.encoder_mode adapter on the production trainer: the cache loads
    (the base encoder stays frozen), the optimizer holds the encoder adapter
    and the decoder adapter but no base encoder tensor, small trees encode live
    through the adapter with a gradient, and a FROZEN run's encoder state
    resumes into it through the adapter key mapping."""
    from safetensors.torch import save_file

    monkeypatch.setenv("WANDB_MODE", "disabled")
    _patch_tokenizers(monkeypatch)
    cfg = hydra_cfg
    b = cfg.training.bgkit2
    b.data.repo_tree_store = _tree_store(tmp_path)
    b.data.repo_tree_file_tokens = 16
    b.data.repo_tree_every = 2
    b.data.batch_size = 2
    b.model = {"repro_head": True, "lora": {"enabled": True, "rank": 4, "alpha": 8},
               "encoder_adapter": {"rank": 4, "alpha": 8}}
    b.optim = {"encoder_mode": "adapter", "encoder_grad_token_budget": 64}
    b.multilevel = {"ratio_leaf": 0.5, "ratio_node": 0.5, "max_depth": 2, "bridge": "identity",
                    "live_tree_max_files": 4, "grad_levels": 1}
    d = 32
    blob = {f"r{i}/reps": torch.randn(3, d) for i in range(6)}
    blob.update({f"r{i}/mask": torch.ones(3, dtype=torch.bool) for i in range(6)})
    cache = str(tmp_path / "roots.safetensors")
    save_file(blob, cache)
    b.data.repo_tree_root_cache = cache
    frozen_state = {k: v.clone() for k, v in _TreeTrainer(cfg).build_encoder().state_dict().items()}
    tr = _TreeTrainer(cfg)
    tr.setup()
    names = {g["name"] for g in tr.optimizer.param_groups}
    assert {"encoder_adapter", "lora"} <= names and "backbone" not in names and "head" not in names
    in_opt = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
    for n, p in tr.encoder.named_parameters():
        if id(p) in in_opt:
            assert n.startswith("slot_parts.") or ".slots." in n, n
    trees = next(iter(tr.tree_dataloader))
    assert all(len(t.file_ids) == len(t.paths) for t in trees)  # live trees need the files
    tr.encoder.train()
    total, _stats = tr.tree_training_loss(trees)
    total.backward()
    assert any(p.grad is not None for p in tr.encoder.slot_adapter_parameters())
    # resume of a frozen-run encoder state: base weights land under peft's
    # base_layer names, the adapter copies start from them
    migrated = tr._migrate_state_dict("encoder", frozen_state)
    missing, unexpected = tr.encoder.load_state_dict(migrated, strict=False)
    assert unexpected == [] and all(".slots." in k for k in missing)
