"""End-to-end integration test for the Phase 2 KB-scale data prep pipeline.

Exercises the full chain on a tiny synthetic corpus without any GPU, HF model
download, or real Phase 1 checkpoint:

1. Build a toy corpus of 5 topics x 10 leaves x 5 articles (250 articles).
2. Build the browse tree via :class:`bgkit.data.tagging.BrowseTreeBuilder`.
3. Emit a synthetic ``metadata.parquet`` with ``document_id`` +
   ``provenance_json`` columns (the schema downstream provenance scripts
   read).
4. Generate teacher trajectories via
   :func:`bgkit.data.teacher_trajectories.build_trajectory` and serialize
   them in the schema :class:`bgkit.data.datasets.phase2_kb_dataset.KBTrajectoryDataset`
   consumes.
5. Round-trip via :func:`bgkit.data.bgkit_tool_template.articles_referenced_by_trajectory`
   to get the trajectory's article set, and populate an
   :class:`bgkit.data.l0_cache.L0Cache` with stubbed L0 survivor rows via
   :class:`bgkit.data.l0_cache.L0CacheWriter`.
6. Build a tiny :class:`bgkit.training.phase2.kr_kb_trainer.KRKBTrainer`
   with a CPU-compatible decoder backbone and stubbed encoder/tokenizer,
   and run 3 training steps over a small batch. Asserts loss is finite
   throughout and decreases from step 0 to step 2.

Marked :py:mod:`integration` so the default unit sweep skips it; run via
``make test-integration`` or ``pytest tests/integration -v``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Toy corpus construction
# ---------------------------------------------------------------------------


TOPICS = [f"T{i}" for i in range(5)]
LEAVES_PER_TOPIC = 10
ARTICLES_PER_LEAF = 5
TOTAL_ARTICLES = len(TOPICS) * LEAVES_PER_TOPIC * ARTICLES_PER_LEAF  # 250


def _synthetic_corpus() -> list[tuple[str, list[str]]]:
    """Return a list of ``(article_id, tag_path)`` pairs for the toy corpus.

    Each article lives under a two-level hierarchy: topic -> leaf.
    Article IDs encode their path deterministically, e.g. ``T2_L7_a4``.
    """
    rows: list[tuple[str, list[str]]] = []
    for topic in TOPICS:
        for j in range(LEAVES_PER_TOPIC):
            leaf = f"{topic}_L{j}"
            for k in range(ARTICLES_PER_LEAF):
                aid = f"{topic}_L{j}_a{k}"
                rows.append((aid, [topic, leaf]))
    return rows


def _article_token_ids(article_id: str, vocab_size: int) -> list[int]:
    """Deterministic 20-50 fake token IDs for an article."""
    rng = np.random.default_rng(abs(hash(article_id)) % (2**32))
    length = int(rng.integers(20, 51))
    return [int(x) for x in rng.integers(1, vocab_size, size=length)]


# ---------------------------------------------------------------------------
# Stubbed decoder / tokenizer / encoder
# ---------------------------------------------------------------------------


class _TinyInnerModel(torch.nn.Module):
    """Stand-in for the Qwen3.5 inner model: a single Linear over inputs_embeds.

    Exposes the surface :meth:`ReconstructionDecoder._inner_forward` calls
    into: ``forward(inputs_embeds, attention_mask) -> SimpleNamespace``
    carrying ``last_hidden_state``. Also satisfies
    ``get_input_embeddings()`` for any call path that unwraps through it.
    """

    def __init__(self, embed: torch.nn.Embedding):
        super().__init__()
        self.embed_tokens = embed
        self.layer = torch.nn.Linear(embed.embedding_dim, embed.embedding_dim)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,  # absorb FA4 packed args (position_ids / cu_seqlens / max_seqlen)
    ) -> SimpleNamespace:
        del attention_mask, kwargs  # unused in toy model
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))


class _TinyBackbone(torch.nn.Module):
    """Minimal HF-AutoModelForCausalLM stand-in matching the surface the
    real :class:`bgkit.models.decoder.ReconstructionDecoder` expects.

    Provides ``.model`` (inner transformer), ``.lm_head`` (vocab projection),
    and ``.get_input_embeddings()``. The decoder's
    :meth:`_get_inner_model_and_head` reads ``backbone.model`` and
    ``backbone.lm_head`` directly when there is no ``peft`` wrapper.
    """

    def __init__(self, vocab_size: int = 512, hidden_dim: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = _TinyInnerModel(self.embed)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed


class _FakeTokenizer:
    """Append-idempotent fake chat template + small-vocab encoder.

    :func:`bgkit.data.bgkit_tool_template.tokenize_trajectory` calls
    ``apply_chat_template`` with progressively-longer message lists and
    relies on the rendered strings forming a prefix chain. Our concatenation
    strategy satisfies that contract because each message maps to a
    deterministic role-tagged chunk of text that depends only on its own
    contents.
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def _format_message(self, m: dict) -> str:
        role = m.get("role", "user")
        if role == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]["function"]
            args_str = json.dumps(call.get("arguments", {}), sort_keys=True)
            return f"<assistant_call name={call['name']} args={args_str}>"
        if role == "tool":
            return f"<tool name={m.get('name', '')}>{m.get('content', '')}</tool>"
        content = m.get("content", "")
        return f"<{role}>{content}</{role}>"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list | None = None,
    ) -> str:
        del tokenize, add_generation_prompt, tools
        return "".join(self._format_message(m) for m in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        # Deterministic small-vocab encoding — one ID per character, mapped
        # into [1, vocab_size) so we never collide with pad/EOS sentinels.
        return [((ord(c) * 131 + 17) % (self.vocab_size - 1)) + 1 for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)


# ---------------------------------------------------------------------------
# Trainer stubbing
# ---------------------------------------------------------------------------


def _new_trainer(
    *,
    decoder,
    tokenizer: _FakeTokenizer,
    tree,
    l0_cache,
    device: torch.device,
    hidden_dim: int,
):
    """Build a KRKBTrainer with just the attributes the forward path needs.

    Follows the same construction pattern as
    ``tests/unit/training/test_kr_kb_trainer_pieces.py`` — skip ``__init__``
    via ``__new__`` and fill in the attributes read by
    :meth:`_build_decoder_segments_core` and :meth:`_forward_backward`.
    """
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = device
    trainer.decoder = decoder
    trainer.tokenizer = tokenizer
    trainer.encoder_tokenizer = tokenizer
    trainer._trees = {"toy": tree}
    trainer._l0_cache = l0_cache
    trainer._live_l0 = False
    trainer._l0_retention = {}
    trainer._l1_retention = 0.5
    trainer._missing_article_counts = {}
    trainer._ablation_mode = None
    trainer.topic_embeddings = None
    trainer.taxonomy = None
    trainer._accum_steps = 1
    trainer.global_step = 0
    trainer._hidden_dim = hidden_dim

    # Stub _prepare_l1_turn: resolve bgkit IDs to articles, pull stub
    # survivors from the L0 cache. Matches the interface the real
    # :meth:`KRKBTrainer._prepare_l1_turn` presents to
    # :meth:`_run_l1_batch`.
    def fake_prepare(self, dataset: str, tag_or_article_ids: list[str], query: str):
        resolved: list[str] = []
        t = self._trees[dataset]
        for raw in tag_or_article_ids:
            if raw not in t:
                continue
            node = t.get(raw)
            if node.is_article:
                resolved.append(raw)
            elif node.is_leaf_tag:
                resolved.extend(node.articles)
            else:
                resolved.extend(t.articles(raw))
        # Drop anything the cache doesn't have (so we never crash on lookup).
        resolved = [a for a in resolved if self._l0_cache.has(dataset, a)]
        if not resolved:
            return None
        return {
            "dataset": dataset,
            "article_ids": resolved,
            "query": query,
        }

    # Stub _run_l1_batch: concatenate cached L0 rows per turn. We don't run
    # any encoder forward, so the survivor tensors are leaves w.r.t. the
    # autograd graph — the decoder's embedding table and head still carry
    # the trainable state and backprop correctly.
    def fake_run_l1_batch(self, prepared):
        results: list[torch.Tensor] = []
        zero_fallback = torch.zeros(1, self._hidden_dim, device=self.device)
        for entry in prepared:
            if entry is None:
                results.append(zero_fallback)
                continue
            ds = entry.get("dataset", "toy")
            if "mode" in entry:
                # Drill-down navigation turn (head/node). The real trainer
                # resolves these from the per-repo shared tree; the QA cached-tree
                # runtime is deferred, so here we stub-resolve the node to its
                # articles (same L0-cache path as a leaf drill) so the pipeline
                # still exercises the multi-turn splice.
                nid = entry.get("node_id", "")
                t = self._trees[ds]
                if nid in t:
                    node = t.get(nid)
                    aids = list(node.articles) if node.is_leaf_tag else list(t.articles(nid))
                else:
                    aids = []
                aids = [a for a in aids if self._l0_cache.has(ds, a)]
            else:
                aids = list(entry["article_ids"])
            rows_list = [
                self._l0_cache.get(ds, aid).float().to(self.device)
                for aid in aids
            ]
            if not rows_list:
                results.append(zero_fallback)
                continue
            cat = torch.cat(rows_list, dim=0)
            if cat.size(-1) != self._hidden_dim:
                raise AssertionError(
                    f"L0 cache hidden dim {cat.size(-1)} != decoder {self._hidden_dim}"
                )
            results.append(cat)
        return results

    import types

    trainer._prepare_l1_turn = types.MethodType(fake_prepare, trainer)
    trainer._run_l1_batch = types.MethodType(fake_run_l1_batch, trainer)
    trainer._system_prompt_for = types.MethodType(
        lambda self, sample: (
            "SYSTEM: topics=" + ",".join(sample.topic_list[:3])
        ),
        trainer,
    )
    return trainer


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_phase2_kb_e2e_pipeline(tmp_path: Path):
    """Run the full Phase 2 KB data-prep pipeline on a toy corpus and train 3 steps."""
    from bgkit.data.bgkit_tool_template import (
        articles_referenced_by_trajectory,
        trajectory_from_json,
        trajectory_to_json,
    )
    from bgkit.data.datasets.phase2_kb_dataset import KBTrajectoryDataset
    from bgkit.data.l0_cache import L0Cache, L0CacheWriter, update_dataset_index
    from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig
    from bgkit.data.teacher_trajectories import TrajectoryConfig, build_trajectory
    from bgkit.models.decoder import ReconstructionDecoder

    torch.manual_seed(0)
    np.random.seed(0)

    dataset_name = "toy"
    hidden_dim = 16
    vocab_size = 512

    # ------------------------------------------------------------------
    # Stage 1: Browse tree via BrowseTreeBuilder (real API call).
    # ------------------------------------------------------------------
    builder = BrowseTreeBuilder(TaggingConfig(
        dataset=dataset_name, leaf_cap=10, fanout_cap=20,
    ))
    corpus = _synthetic_corpus()
    for aid, tag_path in corpus:
        builder.add_article(aid, tag_path)
    tree = builder.build()

    # Persist the tree and verify round-trip load.
    browse_tree_dir = tmp_path / "browse_trees"
    browse_tree_dir.mkdir()
    browse_tree_path = browse_tree_dir / f"{dataset_name}.parquet"
    tree.save(browse_tree_path)
    assert browse_tree_path.exists()

    from bgkit.data.browse_tree import BrowseTree

    loaded_tree = BrowseTree.load(browse_tree_path, dataset=dataset_name)
    assert len(loaded_tree) >= TOTAL_ARTICLES  # tree also contains tag nodes
    for aid, _ in corpus:
        assert aid in loaded_tree, f"article {aid!r} missing from loaded tree"

    # ------------------------------------------------------------------
    # Stage 2: Synthetic metadata.parquet (mmap-layout input format for
    # per-dataset provenance builders like scripts/build_provenance_kilt.py).
    # ------------------------------------------------------------------
    mmap_dir = tmp_path / "mmap_phase2" / dataset_name
    mmap_dir.mkdir(parents=True)
    metadata_rows = [
        {
            "document_id": aid,
            # provenance_json is a JSON list of article IDs (KILT style);
            # for our toy we just point each document at itself.
            "provenance_json": json.dumps([aid]),
        }
        for aid, _ in corpus
    ]
    metadata_schema = pa.schema([
        ("document_id", pa.string()),
        ("provenance_json", pa.string()),
    ])
    pq.write_table(
        pa.Table.from_pylist(metadata_rows, schema=metadata_schema),
        mmap_dir / "metadata.parquet",
    )

    # Sanity-check: the metadata parquet round-trips and matches the toy corpus.
    metadata_back = pq.read_table(mmap_dir / "metadata.parquet").to_pylist()
    assert len(metadata_back) == TOTAL_ARTICLES
    assert {row["document_id"] for row in metadata_back} == {aid for aid, _ in corpus}

    # ------------------------------------------------------------------
    # Stage 3: Teacher trajectories via build_trajectory (real API call).
    # ------------------------------------------------------------------
    # Pick 10 gold articles spread across topics and leaves.
    gold_samples = [
        ("What is T0 L0 a1?", "answer_T0_L0_a1", "T0_L0_a1"),
        ("What is T1 L2 a3?", "answer_T1_L2_a3", "T1_L2_a3"),
        ("Where is T2 L5 a0?", "answer_T2_L5_a0", "T2_L5_a0"),
        ("Who owns T3 L7 a4?", "answer_T3_L7_a4", "T3_L7_a4"),
        ("What is T4 L1 a2?", "answer_T4_L1_a2", "T4_L1_a2"),
        ("Describe T0 L9 a3", "answer_T0_L9_a3", "T0_L9_a3"),
        ("Describe T1 L6 a1", "answer_T1_L6_a1", "T1_L6_a1"),
        ("Describe T2 L3 a4", "answer_T2_L3_a4", "T2_L3_a4"),
        ("Describe T3 L0 a0", "answer_T3_L0_a0", "T3_L0_a0"),
        ("Describe T4 L8 a2", "answer_T4_L8_a2", "T4_L8_a2"),
    ]
    traj_cfg = TrajectoryConfig(exploration_fraction=0.3, seed=42)
    rows = []
    for i, (question, answer, gold_article) in enumerate(gold_samples):
        trajectory = build_trajectory(
            loaded_tree, question, gold_article, answer,
            traj_cfg, sample_idx=i,
        )
        # Assertion: trajectory must contain at least one bgkit turn and an
        # answer turn — the trainer relies on both to compute loss.
        kinds = [t.kind for t in trajectory]
        assert "bgkit" in kinds, f"trajectory missing bgkit turn: {kinds}"
        rows.append({
            "dataset_name": dataset_name,
            "scope_template": "topic_list",
            "scope_description": "",
            "topic_list_json": json.dumps(loaded_tree.top_level_topic_list()),
            "question": question,
            "gold_answer": answer,
            "trajectory_json": trajectory_to_json(trajectory),
        })

    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    trajectory_path = trajectory_dir / f"{dataset_name}.parquet"
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("dataset_name", pa.string()),
        ("scope_template", pa.string()),
        ("scope_description", pa.string()),
        ("topic_list_json", pa.string()),
        ("question", pa.string()),
        ("gold_answer", pa.string()),
        ("trajectory_json", pa.string()),
    ]))
    pq.write_table(table, trajectory_path)
    assert trajectory_path.exists()

    # Stage 4: KBTrajectoryDataset round-trip.
    kb_dataset = KBTrajectoryDataset(trajectory_path)
    assert len(kb_dataset) == len(gold_samples)
    sample0 = kb_dataset[0]
    assert sample0.dataset_name == dataset_name
    assert sample0.scope_template == "topic_list"
    assert sample0.topic_list == loaded_tree.top_level_topic_list()
    assert any(t.kind == "bgkit" for t in sample0.trajectory)

    # ------------------------------------------------------------------
    # Stage 5: Trajectory set + L0 cache (stubbed survivors).
    # ------------------------------------------------------------------
    referenced_article_ids: set[str] = set()
    for row in rows:
        traj = trajectory_from_json(row["trajectory_json"])
        for ref_id in articles_referenced_by_trajectory(traj):
            if ref_id not in loaded_tree:
                continue
            node = loaded_tree.get(ref_id)
            if node.is_article:
                referenced_article_ids.add(ref_id)
            else:
                referenced_article_ids.update(loaded_tree.articles(ref_id))
    # The trajectory set must be non-empty — otherwise the trainer would
    # have nothing to splice survivors for.
    assert len(referenced_article_ids) > 0, "trajectory set is empty"

    l0_cache_dir = tmp_path / "l0_cache"
    # Single shard is enough for a 250-article toy.
    shard_id = "shard_0000"
    writer = L0CacheWriter(l0_cache_dir, dataset_name, shard_id)
    rng = np.random.default_rng(1)
    for aid in sorted(referenced_article_ids):
        # Each article gets K in [2, 6] stub survivor rows.
        k = int(rng.integers(2, 7))
        survivors = rng.standard_normal((k, hidden_dim)).astype(np.float16) * 0.1
        writer.add(aid, survivors)
    _, index_rows = writer.finalize()
    update_dataset_index(l0_cache_dir, dataset_name, shard_id, index_rows)

    # Stage 6: L0Cache read-back — every referenced article must be present.
    l0_cache = L0Cache(l0_cache_dir)
    for aid in referenced_article_ids:
        assert l0_cache.has(dataset_name, aid), f"missing {aid} in L0 cache"
        rows_t = l0_cache.get(dataset_name, aid)
        assert rows_t.ndim == 2
        assert rows_t.size(-1) == hidden_dim

    # ------------------------------------------------------------------
    # Stage 7: Tiny KRKBTrainer + 3 training steps.
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    backbone = _TinyBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim)
    decoder = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)
    decoder.train()
    tokenizer = _FakeTokenizer(vocab_size=vocab_size)

    trainer = _new_trainer(
        decoder=decoder,
        tokenizer=tokenizer,
        tree=loaded_tree,
        l0_cache=l0_cache,
        device=device,
        hidden_dim=hidden_dim,
    )

    # Sanity: a single forward pass on one sample produces finite, positive loss.
    with torch.no_grad():
        first_segments, _ = trainer._build_decoder_segments(sample0)
        sanity_loss = decoder.forward_interleaved_with_loss(first_segments)
    assert torch.isfinite(sanity_loss)
    assert float(sanity_loss) > 0.0

    # Fixed small batch replay over 3 optimizer steps. Using SGD with a
    # relatively high lr is robust for a toy loss surface — Adam's momentum
    # can overshoot on single-batch replays.
    #
    # Note: we drive training via the trainer's :meth:`_compute_sample_loss`
    # (single-sample path) rather than :meth:`_forward_backward`. The
    # real ``_forward_backward`` reads ``self.encoder.l1.hidden_dim``
    # and the ``content`` tensor produced by the real ``_prepare_l1_turn``
    # — both require a real encoder. The single-sample path flows through
    # ``_build_decoder_segments`` → ``_prepare_sample_for_decode`` →
    # ``_run_l1_batch``, all of which our stubs satisfy.
    batch = [kb_dataset[0], kb_dataset[1]]
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_tokens = 0
        n_samples = 0
        for s in batch:
            sample_loss, sample_tokens = trainer._compute_sample_loss(s)
            if sample_tokens == 0:
                continue
            total_loss = total_loss + sample_loss
            total_tokens += sample_tokens
            n_samples += 1
        assert n_samples > 0, "every sample had zero loss tokens"
        mean_loss = total_loss / n_samples
        mean_loss.backward()
        loss_val = float(mean_loss.detach())
        assert np.isfinite(loss_val), f"non-finite loss: {loss_val}"
        losses.append(loss_val)
        optimizer.step()

    # Loss must decrease over the 3 steps on this fixed batch.
    assert losses[-1] < losses[0], (
        f"loss did not decrease: {losses}"
    )
    # Every intermediate loss must still be finite.
    for i, lv in enumerate(losses):
        assert np.isfinite(lv), f"loss[{i}] not finite: {lv}"

    # ------------------------------------------------------------------
    # Stage 8: Artifact summary — each pipeline stage produced a valid file.
    # ------------------------------------------------------------------
    assert browse_tree_path.is_file()
    assert (mmap_dir / "metadata.parquet").is_file()
    assert trajectory_path.is_file()
    index_parquet = l0_cache_dir / dataset_name / "index.parquet"
    assert index_parquet.is_file()
    shard_survivors = l0_cache_dir / dataset_name / shard_id / "survivors.npy"
    shard_offsets = l0_cache_dir / dataset_name / shard_id / "offsets.npy"
    assert shard_survivors.is_file()
    assert shard_offsets.is_file()
