"""Smoke tests for the KB-scale trainer's offline/online building blocks.

These tests exercise the browse-tree/trajectory/L0-cache/tool-template stack
without spinning up a real encoder or decoder. The full stage-A live-L0
smoke test is gated behind GPU and lives in ``tests/integration``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.bgkit_tool_template import (
    TrajectoryTurn,
    trajectory_from_json,
    trajectory_to_json,
)
from bgkit.data.browse_tree import BrowseTree
from bgkit.data.datasets.phase2_kb_dataset import KBTrajectoryDataset
from bgkit.data.l0_cache import L0Cache, L0CacheWriter, update_dataset_index
from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig
from bgkit.data.teacher_trajectories import TrajectoryConfig, build_trajectory


def _build_kb_parquet(tmp_path: Path) -> tuple[Path, BrowseTree]:
    builder = BrowseTreeBuilder(
        TaggingConfig(dataset="toy", leaf_cap=10, fanout_cap=10),
    )
    for top in ["Physics", "Biology"]:
        for sub in [f"{top}_sub{j}" for j in range(3)]:
            for k in range(3):
                builder.add_article(f"{sub}_a{k}", [top, sub])
    tree = builder.build()

    cfg = TrajectoryConfig(exploration_fraction=0.0)
    rows = []
    for i, gold in enumerate(["Physics_sub1_a2", "Biology_sub0_a1"]):
        trajectory = build_trajectory(
            tree, f"question_{i}", gold, f"answer_{i}", cfg, sample_idx=i,
        )
        rows.append({
            "dataset_name": "toy",
            "scope_template": "topic_list",
            "scope_description": "",
            "topic_list_json": json.dumps(tree.top_level_topic_list()),
            "question": f"question_{i}",
            "gold_answer": "answer",
            "trajectory_json": trajectory_to_json(trajectory),
        })
    path = tmp_path / "toy.parquet"
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("dataset_name", pa.string()),
        ("scope_template", pa.string()),
        ("scope_description", pa.string()),
        ("topic_list_json", pa.string()),
        ("question", pa.string()),
        ("gold_answer", pa.string()),
        ("trajectory_json", pa.string()),
    ]))
    pq.write_table(table, path)
    return path, tree


def test_kb_trajectory_dataset_roundtrip(tmp_path):
    path, tree = _build_kb_parquet(tmp_path)
    ds = KBTrajectoryDataset(path)
    assert len(ds) == 2
    sample = ds[0]
    assert sample.dataset_name == "toy"
    assert sample.scope_template == "topic_list"
    assert sample.topic_list == tree.top_level_topic_list()
    # Trajectory must have at least one browse and one bgkit turn
    kinds = [t.kind for t in sample.trajectory]
    assert "browse" in kinds and "bgkit" in kinds


def test_l0_cache_extend_only(tmp_path):
    # Simulate the Stage A → B pre-compute flow: add a shard, read it, then
    # add another shard and re-read to verify both are visible.
    writer1 = L0CacheWriter(tmp_path, "toy", "shard_0000")
    writer1.add("art_a", np.ones((2, 8), dtype=np.float16))
    _, rows1 = writer1.finalize()
    update_dataset_index(tmp_path, "toy", "shard_0000", rows1)

    cache = L0Cache(tmp_path)
    assert cache.has("toy", "art_a")

    writer2 = L0CacheWriter(tmp_path, "toy", "shard_0001")
    writer2.add("art_b", np.full((3, 8), 5, dtype=np.float16))
    _, rows2 = writer2.finalize()
    update_dataset_index(tmp_path, "toy", "shard_0001", rows2)

    # A fresh L0Cache instance sees both shards.
    cache2 = L0Cache(tmp_path)
    assert cache2.has("toy", "art_a")
    assert cache2.has("toy", "art_b")
    b = cache2.get("toy", "art_b")
    assert b.shape == (3, 8)
    assert torch.allclose(b.float(), torch.full((3, 8), 5.0))


def test_build_decoder_segments_interleaves_tokens_and_survivors():
    """_build_decoder_segments walks a rendered trajectory and emits a
    TokenSegment/EmbeddingSegment interleaving aligned with the bgkit
    sentinel positions. We stub out the L1 prepare/batch methods and the
    chat template to keep the test pure-CPU and tokenizer-free."""
    import types

    from bgkit.data.bgkit_tool_template import (
        BGKIT_SENTINEL,
        TrajectoryTurn,
    )
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.models.decoder import EmbeddingSegment, TokenSegment
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer.topic_embeddings = None
    trainer._ablation_mode = None

    class _FakeDecoder:
        hidden_dim = 4

    trainer.decoder = _FakeDecoder()

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == BGKIT_SENTINEL:
                return [200, 201]
            return [ord(c) % 50 + 1 for c in text]

        def apply_chat_template(
            self, messages, tokenize=False,
            add_generation_prompt=False, tools=None,
        ):
            parts = []
            for m in messages:
                role = m.get("role", "user")
                if role == "assistant" and m.get("tool_calls"):
                    call = m["tool_calls"][0]["function"]
                    parts.append(f"<A:call {call['name']}>")
                elif role == "tool":
                    parts.append(f"<T>{m.get('content', '')}</T>")
                else:
                    first = role[0].upper()
                    parts.append(f"<{first}>{m.get('content', '')}</{first}>")
            return "".join(parts)

    trainer.tokenizer = _FakeTokenizer()

    prepare_log = []
    batch_log = []

    def fake_prepare(self, dataset, ids, query):
        prepare_log.append((dataset, tuple(ids), query))
        # Return a sentinel dict; real contents are ignored by fake_batch.
        return {"dataset": dataset, "ids": tuple(ids), "query": query}

    def fake_batch(self, prepared):
        batch_log.append(len(prepared))
        results = []
        for entry in prepared:
            if entry is None:
                results.append(torch.zeros(1, 4))
            else:
                # Deterministic survivors: K = len(ids) + 2, D = 4.
                k = len(entry["ids"]) + 2
                results.append(
                    torch.arange(k * 4, dtype=torch.float32).reshape(k, 4)
                )
        return results

    trainer._prepare_l1_turn = types.MethodType(fake_prepare, trainer)
    trainer._run_l1_batch = types.MethodType(fake_batch, trainer)
    trainer._system_prompt_for = types.MethodType(
        lambda self, sample: "SYSTEM",
        trainer,
    )

    sample = KBSample(
        dataset_name="toy",
        scope_template="topic_list",
        scope_description="",
        topic_list=["Physics"],
        question="q?",
        gold_answer="a",
        trajectory=[
            TrajectoryTurn(kind="browse", args={"id": "Physics"}, response="kids", loss=True),
            TrajectoryTurn(
                kind="bgkit",
                args={"ids": ["Physics/sub1"], "query": "q?"},
                loss=True,
            ),
            TrajectoryTurn(kind="answer", response="42", loss=True),
        ],
    )

    segments, answer_span = trainer._build_decoder_segments(sample)

    emb_segs = [s for s in segments if isinstance(s, EmbeddingSegment)]
    tok_segs = [s for s in segments if isinstance(s, TokenSegment)]
    assert len(emb_segs) == 1, f"expected 1 EmbeddingSegment, got {len(emb_segs)}"
    assert len(tok_segs) >= 2, "should have tokens before and after the bgkit"

    # Faked L1 output shape: K = len(ids)+2 = 3, D = 4 → (1, 3, 4) after unsqueeze.
    assert emb_segs[0].embeddings.shape == (1, 3, 4)
    assert prepare_log == [("toy", ("Physics/sub1",), "q?")]
    # Single batch call covers all turns in the sample.
    assert batch_log == [1]

    for seg in tok_segs:
        assert seg.loss_mask is not None
        assert seg.loss_mask.shape == seg.token_ids.shape

    # The sample has an answer turn ("42"), so answer_span must be set
    # and point to a range that's non-empty and within the concatenated
    # sequence length.
    assert answer_span is not None
    a_start, a_end = answer_span
    assert 0 <= a_start < a_end
    total_concat_len = 0
    for seg in segments:
        if isinstance(seg, TokenSegment):
            total_concat_len += seg.token_ids.size(1)
        else:
            total_concat_len += seg.embeddings.size(1)
    assert a_end <= total_concat_len


def test_kb_model_lora_state_dict_roundtrip():
    """Build a _KBModel with installed LoRA, save its state dict, load
    into a fresh _KBModel, verify every adapter and base weight matches."""
    from bgkit.models.lora_encoder import (
        DEFAULT_LORA_TARGETS,
        LoRARouter,
    )
    from bgkit.training.phase2.kr_kb_trainer import _KBModel

    torch.manual_seed(101)

    class _MiniEncoder(torch.nn.Module):
        """Stand-in for BgKITEncoder with enough structure that
        LoRARouter.install finds targets."""

        def __init__(self, hidden: int = 8):
            super().__init__()

            class _Attn(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
                    self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
                    self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
                    self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)

            self.attn = _Attn()

    class _MiniDecoder(torch.nn.Module):
        def __init__(self, hidden: int = 8):
            super().__init__()
            self.hidden_dim = hidden
            self.linear = torch.nn.Linear(hidden, hidden)

    class _MiniIce(torch.nn.Module):
        def __init__(self, hidden: int = 8):
            super().__init__()
            self.proj = torch.nn.Linear(hidden, 1)

    # Model 1: fresh build, install LoRA, randomize adapter weights.
    enc1 = _MiniEncoder()
    dec1 = _MiniDecoder()
    ice1 = _MiniIce()
    router1 = LoRARouter.install(
        enc1,
        target_names=DEFAULT_LORA_TARGETS,
        levels={"l0": 4, "l1": 4},
    )
    LoRARouter.bind(router1)
    with torch.no_grad():
        for w in router1._wrappers:
            for level in ("l0", "l1"):
                w.adapters[level].lora_B.copy_(
                    torch.randn_like(w.adapters[level].lora_B)
                )
    model1 = _KBModel(
        encoder=enc1, decoder=dec1, ice=ice1, lora_router=router1,
    )
    saved_state = model1.state_dict()

    # Model 2: fresh rebuild, install LoRA same way, load the state dict.
    enc2 = _MiniEncoder()
    dec2 = _MiniDecoder()
    ice2 = _MiniIce()
    router2 = LoRARouter.install(
        enc2,
        target_names=DEFAULT_LORA_TARGETS,
        levels={"l0": 4, "l1": 4},
    )
    model2 = _KBModel(
        encoder=enc2, decoder=dec2, ice=ice2, lora_router=router2,
    )
    missing, unexpected = model2.load_state_dict(saved_state, strict=True)
    assert missing == []
    assert unexpected == []

    # Every adapter tensor matches across the two models.
    for w1, w2 in zip(router1._wrappers, router2._wrappers, strict=True):
        assert torch.allclose(w1.base_layer.weight, w2.base_layer.weight)
        for level in ("l0", "l1"):
            assert torch.allclose(
                w1.adapters[level].lora_A, w2.adapters[level].lora_A,
            )
            assert torch.allclose(
                w1.adapters[level].lora_B, w2.adapters[level].lora_B,
            )

    LoRARouter.bind(None)


def test_load_checkpoint_remap_pre_lora_keys_defensive():
    """When a pre-LoRA state dict is fed into a LoRA-wrapped _KBModel,
    KRKBTrainer._remap_pre_lora_state_dict rewrites keys so the load
    succeeds via strict=False (adapter params stay at their init)."""
    from bgkit.models.lora_encoder import (
        DEFAULT_LORA_TARGETS,
        LoRARouter,
    )
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer, _KBModel

    class _MiniEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            class _Attn(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = torch.nn.Linear(8, 8, bias=False)
                    self.k_proj = torch.nn.Linear(8, 8, bias=False)

            self.attn = _Attn()

    # Pre-LoRA encoder: capture its state dict BEFORE installing LoRA.
    pre_encoder = _MiniEncoder()
    pre_encoder_state = {
        f"encoder.{k}": v for k, v in pre_encoder.state_dict().items()
    }

    # Build a "post-LoRA" encoder + _KBModel.
    post_encoder = _MiniEncoder()
    router = LoRARouter.install(
        post_encoder,
        target_names=DEFAULT_LORA_TARGETS,
        levels={"l0": 4, "l1": 4},
    )
    model = _KBModel(
        encoder=post_encoder,
        decoder=torch.nn.Linear(8, 8),
        ice=torch.nn.Linear(8, 1),
        lora_router=router,
    )

    # Strict load of pre-LoRA keys should FAIL.
    with pytest.raises((RuntimeError, KeyError)):
        model.load_state_dict(pre_encoder_state, strict=True)

    # Now the defensive remap should succeed.
    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.encoder = post_encoder
    remapped = trainer._remap_pre_lora_state_dict(pre_encoder_state)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert unexpected == []
    # Every encoder-side missing key is a LoRA adapter param (not present
    # in a pre-LoRA checkpoint). Decoder/ice keys are missing because the
    # test input only supplied encoder keys — that's expected.
    encoder_missing = [k for k in missing if k.startswith("encoder.")]
    for k in encoder_missing:
        assert ".adapters." in k and ("lora_A" in k or "lora_B" in k)
    # Verify the remap actually produced base_layer keys in the remapped dict
    assert any("base_layer.weight" in k for k in remapped)


def test_topic_embeddings_from_browse_tree_integration():
    """Build a TagTaxonomy from a real BrowseTree, wire it through the
    KB trainer, verify _build_decoder_segments prepends a topic
    EmbeddingSegment populated from the sample's bgkit turn IDs."""
    import types

    from bgkit.data.bgkit_tool_template import BGKIT_SENTINEL, TrajectoryTurn
    from bgkit.data.browse_tree import BrowseNode, BrowseTree
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.data.taxonomy import TagTaxonomy
    from bgkit.models.decoder import EmbeddingSegment, TokenSegment
    from bgkit.models.topic_embeddings import TopicEmbeddingModule
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    # Minimal browse tree: root → Physics → Physics/sub1 (leaf, 2 articles)
    nodes = {
        "root": BrowseNode(
            id="root", parent=None, kind="sub-tag", size=2,
            children=("Physics",), articles=(),
        ),
        "Physics": BrowseNode(
            id="Physics", parent="root", kind="sub-tag", size=2,
            children=("Physics/sub1",), articles=(),
        ),
        "Physics/sub1": BrowseNode(
            id="Physics/sub1", parent="Physics", kind="sub-tag", size=2,
            children=(), articles=("art_a", "art_b"),
        ),
    }
    tree = BrowseTree(dataset="toy", nodes=nodes)

    # Build taxonomy from the tree — non-article nodes become tags.
    taxonomy = TagTaxonomy.from_browse_tree(tree)
    assert "Physics" in taxonomy
    assert "Physics/sub1" in taxonomy
    # Size-derived frequencies
    assert taxonomy.frequency("Physics") == 2

    # Topic embedding module (small hidden dim for test speed).
    topic_emb = TopicEmbeddingModule(
        taxonomy, positions_per_tag=3, hidden_dim=4,
    )

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer.topic_embeddings = topic_emb
    trainer._ablation_mode = None
    trainer._trees = {"toy": tree}

    class _FakeDecoder:
        hidden_dim = 4

    trainer.decoder = _FakeDecoder()

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == BGKIT_SENTINEL:
                return [200, 201]
            return [ord(c) % 50 + 1 for c in text]

        def apply_chat_template(self, messages, **kwargs):
            parts = []
            for m in messages:
                role = m.get("role", "user")
                if role == "assistant" and m.get("tool_calls"):
                    parts.append(f"<A:{m['tool_calls'][0]['function']['name']}>")
                elif role == "tool":
                    parts.append(f"<T>{m.get('content', '')}</T>")
                else:
                    first = role[0].upper()
                    parts.append(f"<{first}>{m.get('content', '')}</{first}>")
            return "".join(parts)

    trainer.tokenizer = _FakeTokenizer()

    def fake_prepare(self, dataset, ids, query):
        return {"ids": tuple(ids)}

    def fake_batch(self, prepared):
        return [
            torch.zeros(2, 4) if p is not None else torch.zeros(1, 4)
            for p in prepared
        ]

    trainer._prepare_l1_turn = types.MethodType(fake_prepare, trainer)
    trainer._run_l1_batch = types.MethodType(fake_batch, trainer)
    trainer._system_prompt_for = types.MethodType(
        lambda self, sample: "SYSTEM", trainer,
    )

    sample = KBSample(
        dataset_name="toy",
        scope_template="topic_list",
        scope_description="",
        topic_list=["Physics"],
        question="q?",
        gold_answer="a",
        trajectory=[
            TrajectoryTurn(
                kind="bgkit",
                args={"ids": ["Physics/sub1"], "query": "q?"},
                loss=True,
            ),
            TrajectoryTurn(kind="answer", response="42", loss=True),
        ],
    )

    segments, answer_span = trainer._build_decoder_segments(sample)

    # Topic-knowledge now arrives as an in-stream EmbeddingSegment spliced
    # at the BGKIT_TOPIC_SENTINEL position (which lives inside the tool
    # response right after the user question). Find the topic embedding
    # segment — it must be the FIRST EmbeddingSegment in the list, since
    # the topic tool-call pair is injected before any bgkit turns.
    emb_segs = [
        (i, s) for i, s in enumerate(segments) if isinstance(s, EmbeddingSegment)
    ]
    assert len(emb_segs) >= 2, "expected topic EmbeddingSegment + ≥1 bgkit EmbeddingSegment"
    topic_idx, topic_seg = emb_segs[0]
    assert topic_seg.embeddings.size(-1) == 4
    assert topic_seg.embeddings.size(1) >= 3  # ≥1 tag * 3 positions_per_tag

    # Topic segment is preceded by some token segment (the chat template
    # opener + assistant tool-call tokens) — it's not at index 0 anymore.
    assert topic_idx >= 1
    assert isinstance(segments[topic_idx - 1], TokenSegment)

    # Answer span still valid — remap accounts for both the topic sentinel
    # and the bgkit sentinel splices.
    assert answer_span is not None
    a_start, a_end = answer_span
    assert a_end > a_start


def test_top_tag_whitelist_filters_samples_by_top_level_topic():
    """_apply_top_tag_whitelist should keep only samples whose bgkit
    turns' referenced tags descend from a whitelisted top-level topic."""
    import types

    from bgkit.data.bgkit_tool_template import TrajectoryTurn
    from bgkit.data.browse_tree import BrowseNode, BrowseTree
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    # Browse tree: root → {Physics → Physics/sub, Biology → Biology/sub}
    nodes = {
        "root": BrowseNode(
            id="root", parent=None, kind="sub-tag", size=2,
            children=("Physics", "Biology"), articles=(),
        ),
        "Physics": BrowseNode(
            id="Physics", parent="root", kind="sub-tag", size=1,
            children=("Physics/sub",), articles=(),
        ),
        "Physics/sub": BrowseNode(
            id="Physics/sub", parent="Physics", kind="sub-tag", size=1,
            children=(), articles=("p_a",),
        ),
        "Biology": BrowseNode(
            id="Biology", parent="root", kind="sub-tag", size=1,
            children=("Biology/sub",), articles=(),
        ),
        "Biology/sub": BrowseNode(
            id="Biology/sub", parent="Biology", kind="sub-tag", size=1,
            children=(), articles=("b_a",),
        ),
    }
    tree = BrowseTree(dataset="toy", nodes=nodes)

    # Fake dataset: list of 3 KBSample objects, 2 Physics + 1 Biology.
    samples = [
        KBSample(
            dataset_name="toy",
            scope_template="topic_list",
            scope_description="",
            topic_list=[],
            question="q1",
            gold_answer="a1",
            trajectory=[
                TrajectoryTurn(
                    kind="bgkit", args={"ids": ["Physics/sub"], "query": "q1"}, loss=True,
                ),
                TrajectoryTurn(kind="answer", response="a1", loss=True),
            ],
        ),
        KBSample(
            dataset_name="toy",
            scope_template="topic_list",
            scope_description="",
            topic_list=[],
            question="q2",
            gold_answer="a2",
            trajectory=[
                TrajectoryTurn(
                    kind="bgkit", args={"ids": ["Biology/sub"], "query": "q2"}, loss=True,
                ),
                TrajectoryTurn(kind="answer", response="a2", loss=True),
            ],
        ),
        KBSample(
            dataset_name="toy",
            scope_template="topic_list",
            scope_description="",
            topic_list=[],
            question="q3",
            gold_answer="a3",
            trajectory=[
                TrajectoryTurn(
                    kind="bgkit", args={"ids": ["Physics/sub"], "query": "q3"}, loss=True,
                ),
                TrajectoryTurn(kind="answer", response="a3", loss=True),
            ],
        ),
    ]

    class _ListDataset:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            return self._items[idx]

    ds = _ListDataset(samples)

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._trees = {"toy": tree}
    trainer._sample_tags_for = types.MethodType(
        lambda self, sample: [
            tid for turn in sample.trajectory if turn.kind == "bgkit"
            for tid in turn.args.get("ids", [])
        ],
        trainer,
    )

    # Whitelist only Physics → only 2 samples should survive.
    subset = trainer._apply_top_tag_whitelist(ds, {"Physics"})
    assert len(subset) == 2
    kept_indices = subset.indices
    assert 0 in kept_indices and 2 in kept_indices  # Physics samples
    assert 1 not in kept_indices  # Biology sample dropped

    # Whitelist both → all 3 samples pass
    subset_both = trainer._apply_top_tag_whitelist(ds, {"Physics", "Biology"})
    assert len(subset_both) == 3


def test_training_time_ablation_rolls_when_training():
    """_training_ablation_override rolls random ablations during
    training and leaves self._ablation_mode untouched during eval."""
    import random

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._ablation_mode = None
    trainer._p_skip_bgkit_training = 1.0  # always roll skip_bgkit
    trainer._p_skip_topic_training = 1.0  # always roll skip_topic
    trainer._p_noise_bgkit_training = 0.0
    trainer._ablation_rng = random.Random(0)

    class _FakeModel:
        def __init__(self, training):
            self.training = training

    # Training mode — both rolls fire → ABLATION_NEITHER
    trainer.model = _FakeModel(training=True)
    with trainer._training_ablation_override():
        assert trainer._ablation_mode == trainer.ABLATION_NEITHER
    # Restored after context
    assert trainer._ablation_mode is None

    # Eval mode — no roll, mode stays None
    trainer.model = _FakeModel(training=False)
    with trainer._training_ablation_override():
        assert trainer._ablation_mode is None

    # Explicit eval ablation mode is preserved (takes precedence over rolls)
    trainer._ablation_mode = trainer.ABLATION_ZEROED
    trainer.model = _FakeModel(training=True)
    with trainer._training_ablation_override():
        assert trainer._ablation_mode == trainer.ABLATION_ZEROED
    assert trainer._ablation_mode == trainer.ABLATION_ZEROED


def test_query_conditioning_produces_different_survivors_for_different_queries():
    """Verifies the query-conditioning plumbing: ``_prepare_l1_turn`` +
    ``_run_l1_batch`` must pass the encoder a different prompt embedding
    when the query string changes, and the resulting survivor tensors
    must differ.

    Uses a stub encoder whose output adds the prompt's mean vector to
    each position, so different queries produce observably different
    survivors without needing a real transformer.
    """
    import types

    from bgkit.data.bgkit_tool_template import TrajectoryTurn
    from bgkit.data.browse_tree import BrowseNode, BrowseTree
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    # Mini browse tree with one leaf containing one article.
    nodes = {
        "root": BrowseNode(
            id="root", parent=None, kind="sub-tag", size=1,
            children=("Topic",), articles=(),
        ),
        "Topic": BrowseNode(
            id="Topic", parent="root", kind="sub-tag", size=1,
            children=(), articles=("art",),
        ),
    }
    tree = BrowseTree(dataset="toy", nodes=nodes)

    hidden_dim = 4

    class _StubBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._embed = torch.nn.Embedding(50, hidden_dim)

        def get_input_embeddings(self):
            return self._embed

    class _StubCompressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _StubBackbone()
            self.hidden_dim = hidden_dim

    class _StubEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compressor = _StubCompressor()
            self.calls: list = []

        def __call__(self, **kwargs):
            # Pass prompt mean through to output so the stub's result
            # depends on the query.
            self.calls.append(kwargs)
            content = kwargs["input_embeddings"]
            prompt = kwargs["prompt_embeddings"]
            prompt_mask = kwargs["prompt_attention_mask"]
            prompt_mean = (
                prompt * prompt_mask.unsqueeze(-1)
            ).sum(dim=1) / prompt_mask.sum(dim=1, keepdim=True).clamp(min=1)
            # Survivors = content + prompt_mean broadcast.
            out_content = content + prompt_mean.unsqueeze(1)
            mask = kwargs["survivor_mask"]
            return types.SimpleNamespace(
                survivor_embeddings=out_content,
                survivor_attention_mask=mask,
            )

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer.encoder = _StubEncoder()
    trainer.encoder_tokenizer = type(
        "T", (), {
            "encode": lambda self, text, add_special_tokens=False: [
                (ord(c) % 40) + 1 for c in text
            ],
        },
    )()
    trainer._trees = {"toy": tree}
    trainer._title_to_doc_id = {}
    trainer._doc_id_to_title = {}
    trainer._l1_calibrator = None
    trainer._l1_retention = 1.0
    trainer._live_l0 = True

    # Fake L0 for the single article — a deterministic (1, K, D) tensor
    class _TokenStore:
        def has(self, dataset, aid):
            return True

        def get_batch(self, dataset, ids):
            n = len(ids)
            tokens = torch.arange(n * 3).reshape(n, 3)
            mask = torch.ones(n, 3, dtype=torch.bool)
            return tokens, mask

    trainer._token_store = _TokenStore()
    trainer._missing_article_counts = {}
    trainer._checkpoint_encoder = False

    # Provide a tiny ICE stub
    class _IceStub(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=-1)

    trainer.ice = _IceStub()

    # Mock _l0_for_articles directly to skip the encoder-based L0 path.
    def fake_l0(self, dataset, article_ids):
        n = len(article_ids)
        batch = torch.arange(n * 2 * hidden_dim, dtype=torch.float32).reshape(
            n, 2, hidden_dim,
        )
        mask = torch.ones(n, 2, dtype=torch.bool)
        return batch, mask

    trainer._l0_for_articles = types.MethodType(fake_l0, trainer)

    # Two different queries of EQUAL length so the prompt embeddings
    # are directly comparable (and the difference isn't just padding).
    turn_a = trainer._prepare_l1_turn("toy", ["Topic"], "hello")
    turn_b = trainer._prepare_l1_turn("toy", ["Topic"], "world")

    # Same content (same articles), different query embeddings.
    assert torch.allclose(turn_a["content"], turn_b["content"])
    assert not torch.allclose(turn_a["query_emb"], turn_b["query_emb"]), (
        "different queries should produce different prompt embeddings"
    )

    # Run the batched encoder forward on both prepared turns and verify
    # the survivor tensors are observably different (the stub adds prompt
    # mean to content, so different prompts → different survivors).
    out_a = trainer._run_l1_batch([turn_a])
    out_b = trainer._run_l1_batch([turn_b])
    assert not torch.allclose(out_a[0], out_b[0]), (
        "stub encoder should propagate query conditioning into survivors"
    )


def test_training_time_ablation_partial_rolls():
    """With only p_skip_bgkit>0, the rolled mode is TOPICS_ONLY."""
    import random

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._ablation_mode = None
    trainer._p_skip_bgkit_training = 1.0
    trainer._p_skip_topic_training = 0.0
    trainer._p_noise_bgkit_training = 0.0
    trainer._ablation_rng = random.Random(1)

    class _FakeModel:
        training = True

    trainer.model = _FakeModel()
    with trainer._training_ablation_override():
        assert trainer._ablation_mode == trainer.ABLATION_TOPICS_ONLY

    # Only p_skip_topic → NO_TOPICS
    trainer._p_skip_bgkit_training = 0.0
    trainer._p_skip_topic_training = 1.0
    with trainer._training_ablation_override():
        assert trainer._ablation_mode == trainer.ABLATION_NO_TOPICS


def test_checkpointed_encoder_matches_plain_forward():
    """_checkpointed_encoder should return the same output as a direct
    encoder call when activation checkpointing is enabled, and fall back
    cleanly when disabled or when no tensor requires grad."""
    import types

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer._checkpoint_encoder = True

    class _FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, **kwargs):
            x = kwargs["input_embeddings"]
            return types.SimpleNamespace(
                survivor_embeddings=self.linear(x),
                survivor_attention_mask=torch.ones(
                    x.size(0), x.size(1), dtype=torch.bool,
                ),
            )

    trainer.encoder = _FakeEncoder()
    trainer.encoder.train()

    x = torch.randn(1, 3, 4, requires_grad=True)
    # Plain forward reference
    plain = trainer.encoder(input_embeddings=x).survivor_embeddings

    # Checkpointed forward
    ckpt = trainer._checkpointed_encoder(input_embeddings=x).survivor_embeddings
    torch.testing.assert_close(ckpt, plain)

    # Gradient still flows through the checkpointed path
    loss = ckpt.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # With no grad-tracking input, checkpointed path falls back to plain.
    x_nograd = torch.randn(1, 3, 4, requires_grad=False)
    out_nograd = trainer._checkpointed_encoder(input_embeddings=x_nograd)
    assert out_nograd.survivor_embeddings.shape == (1, 3, 4)


def test_trajectory_json_roundtrip():
    turns = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="x", loss=True),
        TrajectoryTurn(kind="bgkit", args={"ids": ["x"], "query": "q"}, loss=False),
        TrajectoryTurn(kind="answer", response="answer", loss=True),
    ]
    blob = trajectory_to_json(turns)
    restored = trajectory_from_json(blob)
    assert len(restored) == len(turns)
    assert restored[1].args["ids"] == ["x"]
    assert restored[1].loss is False
    assert restored[2].response == "answer"


# ---------------------------------------------------------------------------
# L0 retention curriculum
# ---------------------------------------------------------------------------


def test_l0_retention_static():
    """Static per-dataset retention returns the configured value."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._l0_retention = {"pubmedqa": 0.20, "kilt_wikipedia": 0.05}
    trainer.global_step = 999
    trainer.step_cfg = {}
    assert trainer._l0_retention_for("pubmedqa") == pytest.approx(0.20)
    assert trainer._l0_retention_for("kilt_wikipedia") == pytest.approx(0.05)
    # Missing dataset uses default
    assert trainer._l0_retention_for("unknown") == pytest.approx(0.10)


def test_l0_retention_curriculum_ramp():
    """Curriculum config ramps from start → end over ramp_steps."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._l0_retention = {
        "pubmedqa": {"start": 0.20, "end": 0.01, "ramp_steps": 1000},
    }
    trainer.step_cfg = {}

    trainer.global_step = 0
    assert trainer._l0_retention_for("pubmedqa") == pytest.approx(0.20)

    trainer.global_step = 500
    assert trainer._l0_retention_for("pubmedqa") == pytest.approx(0.105)

    trainer.global_step = 1000
    assert trainer._l0_retention_for("pubmedqa") == pytest.approx(0.01)

    # Past ramp_steps: stays at end
    trainer.global_step = 5000
    assert trainer._l0_retention_for("pubmedqa") == pytest.approx(0.01)
