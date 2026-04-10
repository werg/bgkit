"""Tests for Phase 2 enriched dataset wrappers.

Each wrapper extends Phase2QADataset with dataset-specific metadata parsing,
query grouping, and episode/chain access patterns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Shared helpers for creating tiny mmap Phase 2 artifacts
# ---------------------------------------------------------------------------


def _write_base_artifacts(base: Path, n_rows: int, metadata_table: pa.Table) -> Path:
    """Write the standard Phase 2 mmap files with *n_rows* dummy samples."""
    base.mkdir(parents=True, exist_ok=True)

    # Tokens: each sample gets 3 content tokens, 2 question tokens, 2 answer tokens
    content_toks = list(range(10, 10 + n_rows * 3))
    question_toks = list(range(100, 100 + n_rows * 2))
    answer_toks = list(range(200, 200 + n_rows * 2))

    np.save(base / "tokens.npy", np.array(content_toks, dtype=np.int32))
    np.save(
        base / "offsets.npy",
        np.array([i * 3 for i in range(n_rows + 1)], dtype=np.int64),
    )
    np.save(base / "question_tokens.npy", np.array(question_toks, dtype=np.int32))
    np.save(
        base / "question_offsets.npy",
        np.array([i * 2 for i in range(n_rows + 1)], dtype=np.int64),
    )
    np.save(base / "answer_tokens.npy", np.array(answer_toks, dtype=np.int32))
    np.save(
        base / "answer_offsets.npy",
        np.array([i * 2 for i in range(n_rows + 1)], dtype=np.int64),
    )
    (base / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "row_count": n_rows,
            "total_tokens": n_rows * 3,
            "dataset_name": "test",
        })
    )
    pq.write_table(metadata_table, base / "metadata.parquet")
    return base


# ===================================================================
# MSMARCODataset
# ===================================================================


class TestMSMARCODataset:
    @pytest.fixture()
    def msmarco_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["s0", "s1", "s2"]),
            "document_id": pa.array(["d0", "d1", "d2"]),
            "language": pa.array(["text", "text", "text"]),
            "tag_list_json": pa.array(['[]', '[]', '[]']),
            "query_id": pa.array(["q1", "q1", "q2"]),
            "is_selected_json": pa.array([
                json.dumps([1, 0]),
                json.dumps([0, 0]),
                json.dumps([1, 1]),
            ]),
            "query_type": pa.array(["numeric", "entity", "numeric"]),
        })
        return _write_base_artifacts(tmp_path / "msmarco", 3, meta)

    def test_getitem_injects_query_id(self, msmarco_dir):
        from bgkit.data.datasets.msmarco_dataset import MSMARCODataset

        ds = MSMARCODataset(str(msmarco_dir))
        sample = ds[0]
        assert sample.metadata["query_id"] == "q1"

    def test_get_relevant_doc_ids_returns_correct_indices(self, msmarco_dir):
        from bgkit.data.datasets.msmarco_dataset import MSMARCODataset

        ds = MSMARCODataset(str(msmarco_dir))
        # Index 0 has query_id=q1 with is_selected=[1,0] -> relevant
        # Index 1 has query_id=q1 with is_selected=[0,0] -> no positive, but included by default
        relevant = ds.get_relevant_doc_ids(0)
        # Should return index 1 (same query q1, excludes self=0)
        assert 1 in relevant
        assert 0 not in relevant

    def test_relevance_judgments_parses_json(self, msmarco_dir):
        from bgkit.data.datasets.msmarco_dataset import MSMARCODataset

        ds = MSMARCODataset(str(msmarco_dir))
        judgments = ds.relevance_judgments
        assert judgments[0] == [1, 0]
        assert judgments[2] == [1, 1]

    def test_query_type_tag_added(self, msmarco_dir):
        from bgkit.data.datasets.msmarco_dataset import MSMARCODataset

        ds = MSMARCODataset(str(msmarco_dir))
        sample = ds[0]
        assert "numeric" in sample.tags


# ===================================================================
# KILTDataset
# ===================================================================


class TestKILTDataset:
    @pytest.fixture()
    def kilt_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["k0", "k1", "k2", "k3"]),
            "document_id": pa.array(["d0", "d1", "d2", "d3"]),
            "language": pa.array(["text"] * 4),
            "tag_list_json": pa.array(['[]'] * 4),
            "task_name": pa.array(["nq", "nq", "hotpotqa", "hotpotqa"]),
            "provenance_json": pa.array([
                json.dumps(["wiki_1", "wiki_2"]),
                json.dumps(["wiki_3"]),
                json.dumps([]),
                "invalid-json",
            ]),
        })
        return _write_base_artifacts(tmp_path / "kilt", 4, meta)

    def test_task_name_returns_expected(self, kilt_dir):
        from bgkit.data.datasets.kilt_dataset import KILTDataset

        ds = KILTDataset(str(kilt_dir))
        assert ds.task_name(0) == "nq"
        assert ds.task_name(2) == "hotpotqa"

    def test_provenance_parses_json(self, kilt_dir):
        from bgkit.data.datasets.kilt_dataset import KILTDataset

        ds = KILTDataset(str(kilt_dir))
        assert ds.provenance(0) == ["wiki_1", "wiki_2"]
        assert ds.provenance(2) == []
        # Invalid JSON returns empty list
        assert ds.provenance(3) == []

    def test_sample_task_balanced_respects_weights(self, kilt_dir):
        from bgkit.data.datasets.kilt_dataset import KILTDataset

        ds = KILTDataset(str(kilt_dir), task_weights={"nq": 10.0, "hotpotqa": 0.01})
        sampled = ds.sample_task_balanced(100, seed=42)
        # With huge nq weight, most should be nq indices (0 or 1)
        nq_count = sum(1 for idx in sampled if ds.task_name(idx) == "nq")
        assert nq_count > 80  # nq should dominate

    def test_task_indices_mapping(self, kilt_dir):
        from bgkit.data.datasets.kilt_dataset import KILTDataset

        ds = KILTDataset(str(kilt_dir))
        ti = ds.task_indices
        assert sorted(ti.keys()) == ["hotpotqa", "nq"]
        assert len(ti["nq"]) == 2
        assert len(ti["hotpotqa"]) == 2

    def test_getitem_injects_task_and_provenance(self, kilt_dir):
        from bgkit.data.datasets.kilt_dataset import KILTDataset

        ds = KILTDataset(str(kilt_dir))
        sample = ds[0]
        assert sample.metadata["task_name"] == "nq"
        assert sample.metadata["provenance"] == ["wiki_1", "wiki_2"]
        assert "nq" in sample.tags


# ===================================================================
# GitHistoryDataset
# ===================================================================


class TestGitHistoryDataset:
    @pytest.fixture()
    def git_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["g0", "g1", "g2", "g3"]),
            "document_id": pa.array(["d0", "d1", "d2", "d3"]),
            "language": pa.array(["python"] * 4),
            "tag_list_json": pa.array(['[]'] * 4),
            "question_type": pa.array([
                "factual_recall", "rationale", "factual_recall", "diff_grounded",
            ]),
            "repo_path": pa.array(["owner/repo_a", "owner/repo_a", "owner/repo_b", "owner/repo_b"]),
            "commit_sha": pa.array(["aaa", "bbb", "ccc", "ddd"]),
        })
        return _write_base_artifacts(tmp_path / "git", 4, meta)

    def test_question_type_returns_metadata_value(self, git_dir):
        from bgkit.data.datasets.git_history_dataset import GitHistoryDataset

        ds = GitHistoryDataset(str(git_dir))
        assert ds.question_type(0) == "factual_recall"
        assert ds.question_type(1) == "rationale"
        assert ds.question_type(3) == "diff_grounded"

    def test_get_chain_samples_returns_same_repo(self, git_dir):
        from bgkit.data.datasets.git_history_dataset import GitHistoryDataset

        ds = GitHistoryDataset(str(git_dir))
        chain = ds.get_chain_samples("owner/repo_a")
        assert len(chain) == 2
        for s in chain:
            assert s.metadata["repo_path"] == "owner/repo_a"

    def test_get_chain_samples_respects_max_commits(self, git_dir):
        from bgkit.data.datasets.git_history_dataset import GitHistoryDataset

        ds = GitHistoryDataset(str(git_dir))
        chain = ds.get_chain_samples("owner/repo_b", max_commits=1)
        assert len(chain) == 1

    def test_get_chain_samples_unknown_repo_returns_empty(self, git_dir):
        from bgkit.data.datasets.git_history_dataset import GitHistoryDataset

        ds = GitHistoryDataset(str(git_dir))
        assert ds.get_chain_samples("nonexistent/repo") == []

    def test_repo_indices_and_ids(self, git_dir):
        from bgkit.data.datasets.git_history_dataset import GitHistoryDataset

        ds = GitHistoryDataset(str(git_dir))
        assert sorted(ds.repo_ids) == ["owner/repo_a", "owner/repo_b"]
        ri = ds.repo_indices
        assert len(ri["owner/repo_a"]) == 2


# ===================================================================
# MemoryDataset
# ===================================================================


class TestMemoryDataset:
    @pytest.fixture()
    def mem_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["m0", "m1", "m2", "m3", "m4"]),
            "document_id": pa.array(["ep1", "ep1", "ep1", "ep2", "ep2"]),
            "language": pa.array(["text"] * 5),
            "tag_list_json": pa.array(['[]'] * 5),
            "memory_type": pa.array([
                "persona", "knowledge_update", "mutual_events", "profile", "temporal",
            ]),
        })
        return _write_base_artifacts(tmp_path / "memory", 5, meta)

    def test_get_episode_sessions_groups_by_episode_id(self, mem_dir):
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        ds = MemoryDataset(str(mem_dir))
        ep1 = ds.get_episode_sessions("ep1")
        assert len(ep1) == 3
        ep2 = ds.get_episode_sessions("ep2")
        assert len(ep2) == 2
        assert ds.get_episode_sessions("nonexistent") == []

    def test_get_distractor_sessions_excludes_specified(self, mem_dir):
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        ds = MemoryDataset(str(mem_dir))
        distractors = ds.get_distractor_sessions("ep1", exclude={0}, n=10, seed=42)
        assert 0 not in distractors
        assert len(distractors) <= 2  # ep1 has 3 samples, minus 1 excluded

    def test_get_answer_sessions_filters_by_memory_type(self, mem_dir):
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        ds = MemoryDataset(str(mem_dir))
        # ep1: indices 0 (persona), 1 (knowledge_update), 2 (mutual_events)
        # answer types include persona and knowledge_update but not mutual_events
        answer_sessions = ds.get_answer_sessions("ep1")
        assert 0 in answer_sessions  # persona
        assert 1 in answer_sessions  # knowledge_update
        assert 2 not in answer_sessions  # mutual_events is not in answer_types

    def test_get_answer_sessions_fallback_when_no_match(self, tmp_path):
        """When no sessions match answer types, all are returned."""
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        meta = pa.table({
            "id": pa.array(["x0", "x1"]),
            "document_id": pa.array(["ep_fallback", "ep_fallback"]),
            "language": pa.array(["text", "text"]),
            "tag_list_json": pa.array(['[]', '[]']),
            "memory_type": pa.array(["mutual_events", "shared_memories"]),
        })
        d = _write_base_artifacts(tmp_path / "mem_fb", 2, meta)
        ds = MemoryDataset(str(d))
        sessions = ds.get_answer_sessions("ep_fallback")
        assert len(sessions) == 2  # fallback returns all

    def test_episode_ids_sorted(self, mem_dir):
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        ds = MemoryDataset(str(mem_dir))
        assert ds.episode_ids == ["ep1", "ep2"]

    def test_getitem_injects_memory_type(self, mem_dir):
        from bgkit.data.datasets.memory_dataset import MemoryDataset

        ds = MemoryDataset(str(mem_dir))
        sample = ds[0]
        assert sample.metadata["memory_type"] == "persona"
        assert "persona" in sample.tags


# ===================================================================
# SearchQADataset
# ===================================================================


class TestSearchQADataset:
    @pytest.fixture()
    def searchqa_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["s0", "s1", "s2"]),
            "document_id": pa.array(["d0", "d1", "d2"]),
            "language": pa.array(["text"] * 3),
            "tag_list_json": pa.array(['[]'] * 3),
            "query_id": pa.array(["qa1", "qa1", "qa2"]),
            "snippet_count": pa.array([5, 3, 7]),
        })
        return _write_base_artifacts(tmp_path / "searchqa", 3, meta)

    def test_query_indices_groups_correctly(self, searchqa_dir):
        from bgkit.data.datasets.searchqa_dataset import SearchQADataset

        ds = SearchQADataset(str(searchqa_dir))
        qi = ds.query_indices
        assert len(qi["qa1"]) == 2
        assert len(qi["qa2"]) == 1

    def test_getitem_has_query_id_in_metadata(self, searchqa_dir):
        from bgkit.data.datasets.searchqa_dataset import SearchQADataset

        ds = SearchQADataset(str(searchqa_dir))
        sample = ds[0]
        assert sample.metadata["query_id"] == "qa1"

    def test_snippet_count_accessor(self, searchqa_dir):
        from bgkit.data.datasets.searchqa_dataset import SearchQADataset

        ds = SearchQADataset(str(searchqa_dir))
        assert ds.snippet_count(0) == 5
        assert ds.snippet_count(2) == 7


# ===================================================================
# NarrativeQADataset
# ===================================================================


class TestNarrativeQADataset:
    @pytest.fixture()
    def nqa_dir(self, tmp_path):
        meta = pa.table({
            "id": pa.array(["n0", "n1"]),
            "document_id": pa.array(["d0", "d1"]),
            "language": pa.array(["text", "text"]),
            "tag_list_json": pa.array(['[]', '[]']),
            "document_type": pa.array(["book", "movie_script"]),
        })
        return _write_base_artifacts(tmp_path / "nqa", 2, meta)

    def test_default_max_document_len(self, nqa_dir):
        from bgkit.data.datasets.narrativeqa_dataset import NarrativeQADataset

        ds = NarrativeQADataset(str(nqa_dir))
        assert ds._max_document_len == 65536

    def test_custom_max_document_len(self, nqa_dir):
        from bgkit.data.datasets.narrativeqa_dataset import NarrativeQADataset

        ds = NarrativeQADataset(str(nqa_dir), max_document_len=1024)
        assert ds._max_document_len == 1024

    def test_document_type_returns_expected(self, nqa_dir):
        from bgkit.data.datasets.narrativeqa_dataset import NarrativeQADataset

        ds = NarrativeQADataset(str(nqa_dir))
        assert ds.document_type(0) == "book"
        assert ds.document_type(1) == "movie_script"

    def test_getitem_injects_document_type(self, nqa_dir):
        from bgkit.data.datasets.narrativeqa_dataset import NarrativeQADataset

        ds = NarrativeQADataset(str(nqa_dir))
        sample = ds[0]
        assert sample.metadata["document_type"] == "book"
        assert "book" in sample.tags
