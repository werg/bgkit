"""Tests for the packed collators in bgkit.data.collators.

All collators produce flat (N,) tensors + cu_seqlens / position_ids with
per-sample position restart.  No attention_mask is emitted.

Synthetic dataset items are used throughout -- no tokenizer is loaded.
"""

from __future__ import annotations

import pytest
import torch

from bgkit.data.collators import (
    _collate_file_samples,
    _collate_repo_samples,
    collate_chat_repro,
    collate_compression,
    collate_qa,
    collate_token_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tok(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.long)


def _make_file_sample(**kwargs):
    """Build a minimal FileCompressionSample with overridable fields."""
    from bgkit.data.datasets.compression_dataset import FileCompressionSample

    defaults = dict(
        objective="file_read_repro",
        content_token_ids=_tok([1, 2, 3]),
        content_attention_mask=torch.ones(3, dtype=torch.bool),
        compression_ratio=0.5,
        compression_level=0,
        target_token_ids=_tok([10, 11]),
        target_attention_mask=torch.ones(2, dtype=torch.bool),
        target_loss_mask=_tok([0, 1]),
        prefix_ids=_tok([5]),
        compression_prompt_ids=_tok([7, 8]),
        bgkit_splice_start=0,
        bgkit_splice_len=2,
    )
    defaults.update(kwargs)
    return FileCompressionSample(**defaults)


def _make_repo_sample(file_token_ids_list: list[list[int]], **kwargs):
    """Build a minimal RepoCompressionSample."""
    from bgkit.data.datasets.compression_dataset import RepoCompressionSample

    file_toks = [_tok(ids) for ids in file_token_ids_list]
    file_masks = [torch.ones(len(ids), dtype=torch.bool) for ids in file_token_ids_list]
    defaults = dict(
        objective="file_read_repro",
        file_token_ids=file_toks,
        file_attention_masks=file_masks,
        compression_ratio=0.3,
        compression_level=1,
        target_token_ids=_tok([20, 21, 22]),
        target_attention_mask=torch.ones(3, dtype=torch.bool),
        target_loss_mask=_tok([0, 1, 1]),
        prefix_ids=_tok([6]),
        compression_prompt_ids=_tok([30, 31, 32]),
        bgkit_splice_start=1,
        bgkit_splice_len=3,
    )
    defaults.update(kwargs)
    return RepoCompressionSample(**defaults)


def _make_qa_sample(**kwargs):
    """Build a minimal QASample."""
    from bgkit.data.datasets.qa_sample import QASample

    defaults = dict(
        objective="file_read_repro",
        content_token_ids=_tok([1, 2, 3, 4]),
        content_attention_mask=torch.ones(4, dtype=torch.bool),
        compression_ratio=0.5,
        compression_level=0,
        target_token_ids=_tok([10, 11, 12]),
        target_attention_mask=torch.ones(3, dtype=torch.bool),
        target_loss_mask=_tok([0, 1, 1]),
        prefix_ids=_tok([5, 6]),
        compression_prompt_ids=_tok([7, 8]),
        bgkit_splice_start=0,
        bgkit_splice_len=3,
        question_token_ids=_tok([100, 101]),
        answer_token_ids=_tok([200]),
        sample_id="s0",
        dataset_name="test_ds",
        document_id="doc0",
        tags=["tag1"],
        metadata={"key": "val"},
    )
    defaults.update(kwargs)
    return QASample(**defaults)


# ---------------------------------------------------------------------------
# collate_token_ids
# ---------------------------------------------------------------------------


class TestCollateTokenIds:
    def test_basic_invariants(self):
        """cu_seqlens boundaries, flat length, max_seqlen."""
        items = [
            {"token_ids": _tok([1, 2, 3])},
            {"token_ids": _tok([4, 5])},
            {"token_ids": _tok([6, 7, 8, 9])},
        ]
        out = collate_token_ids(items)
        num_samples = 3
        lengths = [3, 2, 4]
        total = sum(lengths)

        cu = out["cu_seqlens"]
        assert cu[0].item() == 0
        assert cu[-1].item() == total
        assert cu.shape[0] == num_samples + 1

        assert out["input_ids"].shape == (total,)
        assert out["position_ids"].shape == (total,)
        assert out["max_seqlen"] == max(lengths)

    def test_position_ids_restart(self):
        items = [
            {"token_ids": _tok([1, 2, 3])},
            {"token_ids": _tok([4, 5])},
        ]
        out = collate_token_ids(items)
        pos = out["position_ids"]
        # Sample 0: 0,1,2  -- Sample 1: 0,1
        expected = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        assert torch.equal(pos, expected)

    def test_no_attention_mask(self):
        items = [{"token_ids": _tok([1, 2])}]
        out = collate_token_ids(items)
        assert "attention_mask" not in out

    def test_input_ids_values(self):
        """Flat content matches concatenation of inputs."""
        items = [{"token_ids": _tok([1, 2])}, {"token_ids": _tok([3, 4, 5])}]
        out = collate_token_ids(items)
        assert torch.equal(out["input_ids"], _tok([1, 2, 3, 4, 5]))

    def test_single_sample(self):
        items = [{"token_ids": _tok([7, 8, 9])}]
        out = collate_token_ids(items)
        assert out["cu_seqlens"].tolist() == [0, 3]
        assert out["max_seqlen"] == 3
        assert torch.equal(out["position_ids"], _tok([0, 1, 2]))

    def test_max_seqlen_matches_lengths_max(self):
        items = [{"token_ids": _tok([1] * k)} for k in [2, 7, 4]]
        out = collate_token_ids(items)
        cu = out["cu_seqlens"]
        lengths = (cu[1:] - cu[:-1]).tolist()
        assert out["max_seqlen"] == max(lengths)


# ---------------------------------------------------------------------------
# collate_chat_repro
# ---------------------------------------------------------------------------


class TestCollateChatRepro:
    def _make_item(
        self, tok_len=5, content_len=3, prompt_len=2, prefix_len=1, lang="python",
    ):
        return {
            "token_ids": _tok(list(range(tok_len))),
            "loss_mask": _tok([0] * (tok_len - content_len) + [1] * content_len),
            "content_token_ids": _tok(list(range(100, 100 + content_len))),
            "compression_prompt_ids": _tok(list(range(200, 200 + prompt_len))),
            "prefix_ids": _tok(list(range(300, 300 + prefix_len))),
            "language": lang,
            "bgkit_splice_start": 2,
            "bgkit_splice_len": content_len,
        }

    def test_basic_invariants(self):
        batch = [self._make_item(5, 3, 2, 1), self._make_item(7, 4, 3, 2)]
        out = collate_chat_repro(batch)

        # Main sequence
        num_samples = 2
        tok_lens = [5, 7]
        total = sum(tok_lens)
        cu = out["cu_seqlens"]
        assert cu[0].item() == 0
        assert cu[-1].item() == total
        assert cu.shape[0] == num_samples + 1
        assert out["token_ids"].shape == (total,)
        assert out["position_ids"].shape == (total,)

    def test_flat_length_content(self):
        batch = [self._make_item(5, 3, 2, 1), self._make_item(7, 4, 3, 2)]
        out = collate_chat_repro(batch)
        content_total = out["content_cu_seqlens"][-1].item()
        assert out["content_token_ids"].shape == (content_total,)
        assert out["content_position_ids"].shape == (content_total,)
        assert out["loss_mask"].shape == out["token_ids"].shape

    def test_prompt_flat_length(self):
        batch = [self._make_item(5, 3, 2, 1), self._make_item(7, 4, 3, 2)]
        out = collate_chat_repro(batch)
        prompt_total = out["compression_prompt_cu_seqlens"][-1].item()
        assert out["compression_prompt_ids"].shape == (prompt_total,)

    def test_prefix_flat_length(self):
        batch = [self._make_item(5, 3, 2, 1), self._make_item(7, 4, 3, 2)]
        out = collate_chat_repro(batch)
        prefix_total = out["prefix_cu_seqlens"][-1].item()
        assert out["prefix_ids"].shape == (prefix_total,)

    def test_per_sample_scalars(self):
        item1 = self._make_item(5, 3)
        item1["bgkit_splice_start"] = 7
        item1["bgkit_splice_len"] = 3
        item2 = self._make_item(7, 4)
        item2["bgkit_splice_start"] = 0
        item2["bgkit_splice_len"] = 4
        out = collate_chat_repro([item1, item2])
        assert out["bgkit_splice_start"].tolist() == [7, 0]
        assert out["bgkit_splice_len"].tolist() == [3, 4]
        assert out["languages"] == ["python", "python"]

    def test_no_attention_mask(self):
        batch = [self._make_item()]
        out = collate_chat_repro(batch)
        assert "attention_mask" not in out
        assert "content_attention_mask" not in out

    def test_content_position_ids_restart(self):
        item1 = self._make_item(content_len=3)
        item2 = self._make_item(content_len=2)
        out = collate_chat_repro([item1, item2])
        pos = out["content_position_ids"]
        # Sample 0: 0,1,2  Sample 1: 0,1
        expected = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        assert torch.equal(pos, expected)

    def test_max_seqlen_matches_lengths_max(self):
        batch = [self._make_item(tok_len=k, content_len=1) for k in [3, 6, 4]]
        out = collate_chat_repro(batch)
        cu = out["cu_seqlens"]
        lengths = (cu[1:] - cu[:-1]).tolist()
        assert out["max_seqlen"] == max(lengths)

    def test_answer_position_mask_flat(self):
        """answer_position_mask flows through as flat (N_content,) tensor."""
        item1 = self._make_item(content_len=3)
        item1["answer_position_mask"] = torch.tensor([0, 1, 0], dtype=torch.bool)
        item2 = self._make_item(content_len=2)
        item2["answer_position_mask"] = torch.tensor([1, 0], dtype=torch.bool)
        out = collate_chat_repro([item1, item2])
        expected = torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool)
        assert torch.equal(out["answer_position_mask"], expected)

    def test_mixed_answer_position_mask_raises(self):
        item1 = self._make_item()
        item2 = self._make_item()
        item1["answer_position_mask"] = torch.ones(3, dtype=torch.bool)
        with pytest.raises(ValueError, match="answer_position_mask"):
            collate_chat_repro([item1, item2])


# ---------------------------------------------------------------------------
# collate_compression -- file variant
# ---------------------------------------------------------------------------


class TestCollateCompressionFile:
    def test_basic_invariants(self):
        s1 = _make_file_sample(content_token_ids=_tok([1, 2, 3]))
        s2 = _make_file_sample(content_token_ids=_tok([4, 5]))
        out = collate_compression([s1, s2])

        assert out["sample_type"] == "file"
        num_samples = 2
        cu = out["content_cu_seqlens"]
        assert cu[0].item() == 0
        assert cu[-1].item() == 5
        assert cu.shape[0] == num_samples + 1
        assert out["content_token_ids"].shape == (5,)
        assert out["content_position_ids"].shape == (5,)

    def test_target_flat_length(self):
        s1 = _make_file_sample(
            target_token_ids=_tok([10, 11, 12]),
            target_loss_mask=_tok([0, 1, 1]),
            target_attention_mask=torch.ones(3, dtype=torch.bool),
        )
        s2 = _make_file_sample(
            target_token_ids=_tok([13, 14]),
            target_loss_mask=_tok([1, 1]),
            target_attention_mask=torch.ones(2, dtype=torch.bool),
        )
        out = _collate_file_samples([s1, s2])
        tgt_total = out["target_cu_seqlens"][-1].item()
        assert out["target_token_ids"].shape == (tgt_total,)
        assert out["target_loss_mask"].shape == (tgt_total,)

    def test_prompt_flat_length(self):
        s1 = _make_file_sample(compression_prompt_ids=_tok([7, 8, 9]))
        s2 = _make_file_sample(compression_prompt_ids=_tok([10]))
        out = _collate_file_samples([s1, s2])
        prompt_total = out["prompt_cu_seqlens"][-1].item()
        assert out["compression_prompt_ids"].shape == (prompt_total,)

    def test_prefix_flat_length(self):
        s1 = _make_file_sample(prefix_ids=_tok([1, 2]))
        s2 = _make_file_sample(prefix_ids=_tok([3]))
        out = _collate_file_samples([s1, s2])
        prefix_total = out["prefix_cu_seqlens"][-1].item()
        assert out["prefix_ids"].shape == (prefix_total,)

    def test_no_attention_mask(self):
        s = _make_file_sample()
        out = _collate_file_samples([s])
        assert "content_attention_mask" not in out
        assert "target_attention_mask" not in out

    def test_content_position_ids_restart(self):
        s1 = _make_file_sample(content_token_ids=_tok([1, 2, 3]))
        s2 = _make_file_sample(content_token_ids=_tok([4, 5]))
        out = _collate_file_samples([s1, s2])
        pos = out["content_position_ids"]
        expected = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        assert torch.equal(pos, expected)

    def test_max_seqlen(self):
        s1 = _make_file_sample(content_token_ids=_tok([1, 2, 3]))
        s2 = _make_file_sample(content_token_ids=_tok([4, 5]))
        out = _collate_file_samples([s1, s2])
        cu = out["content_cu_seqlens"]
        lengths = (cu[1:] - cu[:-1]).tolist()
        assert out["content_max_seqlen"] == max(lengths)

    def test_per_sample_scalars(self):
        s1 = _make_file_sample(bgkit_splice_start=3, bgkit_splice_len=5)
        s2 = _make_file_sample(bgkit_splice_start=0, bgkit_splice_len=2)
        out = _collate_file_samples([s1, s2])
        assert out["bgkit_splice_start"].tolist() == [3, 0]
        assert out["bgkit_splice_len"].tolist() == [5, 2]


# ---------------------------------------------------------------------------
# collate_compression -- repo variant
# ---------------------------------------------------------------------------


class TestCollateCompressionRepo:
    def test_basic_invariants(self):
        """cu_file_seqlens covers all files; cu_repo_seqlens covers all repos."""
        # repo 0: 3 files, repo 1: 2 files, repo 2: 4 files
        r0 = _make_repo_sample([[1, 2], [3, 4, 5], [6]])           # 3 files
        r1 = _make_repo_sample([[7, 8, 9, 10], [11, 12]])           # 2 files
        r2 = _make_repo_sample([[13], [14, 15], [16, 17], [18, 19, 20]])  # 4 files

        out = _collate_repo_samples([r0, r1, r2])

        total_files = 3 + 2 + 4  # 9
        assert out["cu_file_seqlens"].shape[0] == total_files + 1
        assert out["cu_repo_seqlens"].tolist() == [0, 3, 5, 9]

        content_total = sum([2, 3, 1, 4, 2, 1, 2, 2, 3])  # all file lens
        assert out["content_token_ids"].shape == (content_total,)
        assert out["content_position_ids"].shape == (content_total,)

    def test_cu_file_seqlens_cumulative(self):
        """cu_file_seqlens[0]==0, cu_file_seqlens[-1]==N_content."""
        r0 = _make_repo_sample([[1, 2, 3], [4, 5]])    # files: len 3, 2
        r1 = _make_repo_sample([[6, 7, 8, 9], [10]])   # files: len 4, 1
        out = _collate_repo_samples([r0, r1])

        cu = out["cu_file_seqlens"]
        assert cu[0].item() == 0
        assert cu[-1].item() == out["content_token_ids"].shape[0]
        # Expected: [0, 3, 5, 9, 10]
        assert cu.tolist() == [0, 3, 5, 9, 10]

    def test_cu_repo_seqlens_indices_into_cu_file(self):
        """cu_repo_seqlens are *indices* into cu_file_seqlens, not token counts."""
        r0 = _make_repo_sample([[1, 2], [3, 4, 5]])  # 2 files
        r1 = _make_repo_sample([[6, 7, 8]])            # 1 file
        out = _collate_repo_samples([r0, r1])

        cu_repo = out["cu_repo_seqlens"]
        # Repo 0 has 2 files -> boundary at file-index 2
        # Repo 1 has 1 file  -> boundary at file-index 3
        assert cu_repo.tolist() == [0, 2, 3]

    def test_roundtrip_repo_file_indexing(self):
        """Round-trip: use cu_repo_seqlens to recover each repo's file slices."""
        # repo 0: 3 files with known lengths [2, 3, 1]
        # repo 1: 2 files with known lengths [4, 2]
        r0 = _make_repo_sample([[10, 20], [30, 40, 50], [60]])
        r1 = _make_repo_sample([[1, 2, 3, 4], [5, 6]])
        out = _collate_repo_samples([r0, r1])

        cu_file = out["cu_file_seqlens"]
        cu_repo = out["cu_repo_seqlens"]
        content = out["content_token_ids"]

        # Recover repo 0 files (file indices 0..2 in cu_file)
        repo0_file_start = cu_repo[0].item()   # 0
        repo0_file_end = cu_repo[1].item()     # 3
        repo0_token_start = cu_file[repo0_file_start].item()  # 0
        repo0_token_end = cu_file[repo0_file_end].item()      # 2+3+1=6

        repo0_content = content[repo0_token_start:repo0_token_end]
        expected_r0 = torch.cat([_tok([10, 20]), _tok([30, 40, 50]), _tok([60])])
        assert torch.equal(repo0_content, expected_r0)

        # Recover repo 1 files (file indices 3..4 in cu_file)
        repo1_file_start = cu_repo[1].item()   # 3
        repo1_file_end = cu_repo[2].item()     # 5
        repo1_token_start = cu_file[repo1_file_start].item()  # 6
        repo1_token_end = cu_file[repo1_file_end].item()      # 6+4+2=12

        repo1_content = content[repo1_token_start:repo1_token_end]
        expected_r1 = torch.cat([_tok([1, 2, 3, 4]), _tok([5, 6])])
        assert torch.equal(repo1_content, expected_r1)

    def test_prompt_tiling(self):
        """Each repo's prompt is tiled n_files times; prompt_cu_seqlens 1:1 with files."""
        # repo 0: 2 files, prompt=[7, 8, 9] (len 3)
        # repo 1: 3 files, prompt=[10]       (len 1)
        r0 = _make_repo_sample([[1, 2], [3, 4, 5]], compression_prompt_ids=_tok([7, 8, 9]))
        r1 = _make_repo_sample([[6], [7, 8], [9, 10, 11]], compression_prompt_ids=_tok([10]))

        out = _collate_repo_samples([r0, r1])
        total_files = 2 + 3  # 5
        assert out["prompt_cu_seqlens"].shape[0] == total_files + 1
        # Prompt tokens: r0: 3+3=6, r1: 1+1+1=3  -> total 9
        expected_prompt_total = 2 * 3 + 3 * 1
        assert out["prompt_token_ids"].shape == (expected_prompt_total,)

        # Verify the prompt content matches the tiling pattern
        prompt_cu = out["prompt_cu_seqlens"]
        prompt_toks = out["prompt_token_ids"]

        # Segment 0: r0 file 0 -> [7, 8, 9]
        s0 = prompt_toks[prompt_cu[0]:prompt_cu[1]]
        assert torch.equal(s0, _tok([7, 8, 9]))
        # Segment 1: r0 file 1 -> [7, 8, 9]
        s1 = prompt_toks[prompt_cu[1]:prompt_cu[2]]
        assert torch.equal(s1, _tok([7, 8, 9]))
        # Segment 2: r1 file 0 -> [10]
        s2 = prompt_toks[prompt_cu[2]:prompt_cu[3]]
        assert torch.equal(s2, _tok([10]))

    def test_prompt_cu_seqlens_length(self):
        """prompt_cu_seqlens has length total_files + 1."""
        r0 = _make_repo_sample([[1], [2, 3]])    # 2 files
        r1 = _make_repo_sample([[4, 5, 6]])      # 1 file
        out = _collate_repo_samples([r0, r1])
        assert out["prompt_cu_seqlens"].shape[0] == 3 + 1

    def test_content_position_ids_per_file_restart(self):
        """position_ids restart to 0 at each FILE boundary (not repo boundary)."""
        r0 = _make_repo_sample([[10, 20], [30, 40, 50]])  # files: len 2, 3
        out = _collate_repo_samples([r0])
        pos = out["content_position_ids"]
        expected = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)
        assert torch.equal(pos, expected)

    def test_target_per_repo(self):
        """target_token_ids is per-repo, not per-file."""
        r0 = _make_repo_sample([[1, 2], [3]], target_token_ids=_tok([20, 21]))
        r1 = _make_repo_sample([[4, 5, 6]], target_token_ids=_tok([30]))
        out = _collate_repo_samples([r0, r1])
        # 2 repos -> target_cu_seqlens has B+1=3 entries
        assert out["target_cu_seqlens"].shape[0] == 3
        assert out["target_token_ids"].shape == (3,)  # 2+1

    def test_dispatch_from_collate_compression(self):
        """collate_compression dispatches correctly to repo path."""
        r = _make_repo_sample([[1, 2], [3]])
        out = collate_compression([r])
        assert out["sample_type"] == "repo"
        assert "cu_file_seqlens" in out
        assert "cu_repo_seqlens" in out

    def test_mixed_batch(self):
        """Mixed file+repo batch produces 'mixed' dict."""
        from bgkit.data.datasets.compression_dataset import (
            FileCompressionSample,
            RepoCompressionSample,
        )

        f = _make_file_sample()
        r = _make_repo_sample([[1, 2]])
        assert isinstance(f, FileCompressionSample)
        assert isinstance(r, RepoCompressionSample)
        out = collate_compression([f, r])
        assert out["mixed"] is True
        assert "file_batch" in out
        assert "repo_batch" in out

    def test_no_attention_mask(self):
        r = _make_repo_sample([[1, 2], [3, 4, 5]])
        out = _collate_repo_samples([r])
        assert "file_attention_masks" not in out
        assert "content_attention_mask" not in out


# ---------------------------------------------------------------------------
# collate_qa
# ---------------------------------------------------------------------------


class TestCollateQA:
    def test_basic_invariants(self):
        s1 = _make_qa_sample(content_token_ids=_tok([1, 2, 3]))
        s2 = _make_qa_sample(content_token_ids=_tok([4, 5]))
        out = collate_qa([s1, s2])

        assert out["sample_type"] == "qa"
        num_samples = 2
        cu = out["content_cu_seqlens"]
        assert cu[0].item() == 0
        assert cu[-1].item() == 5
        assert cu.shape[0] == num_samples + 1

    def test_question_flat_length(self):
        s1 = _make_qa_sample(question_token_ids=_tok([100, 101, 102]))
        s2 = _make_qa_sample(question_token_ids=_tok([103]))
        out = collate_qa([s1, s2])
        q_total = out["question_cu_seqlens"][-1].item()
        assert out["question_token_ids"].shape == (q_total,)
        assert q_total == 4

    def test_answer_flat_length(self):
        s1 = _make_qa_sample(answer_token_ids=_tok([200, 201]))
        s2 = _make_qa_sample(answer_token_ids=_tok([202, 203, 204]))
        out = collate_qa([s1, s2])
        a_total = out["answer_cu_seqlens"][-1].item()
        assert out["answer_token_ids"].shape == (a_total,)
        assert a_total == 5

    def test_metadata_fields(self):
        s1 = _make_qa_sample(dataset_name="ds1", sample_id="a", document_id="d1", tags=["t1"])
        s2 = _make_qa_sample(
            dataset_name="ds2", sample_id="b", document_id="d2", tags=["t2", "t3"],
        )
        out = collate_qa([s1, s2])
        assert out["dataset_names"] == ["ds1", "ds2"]
        assert out["sample_ids"] == ["a", "b"]
        assert out["document_ids"] == ["d1", "d2"]
        assert out["tags"] == [["t1"], ["t2", "t3"]]

    def test_no_attention_mask(self):
        s = _make_qa_sample()
        out = collate_qa([s])
        assert "content_attention_mask" not in out
        assert "question_attention_mask" not in out
        assert "answer_attention_mask" not in out

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="empty"):
            collate_qa([])

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            collate_qa([{"token_ids": _tok([1, 2])}])

    def test_question_cu_seqlens_present(self):
        """collate_qa emits cu_seqlens for question/answer."""
        s = _make_qa_sample()
        out = collate_qa([s])
        assert "question_cu_seqlens" in out
        assert "answer_cu_seqlens" in out

    def test_max_seqlen_fields(self):
        s1 = _make_qa_sample(
            content_token_ids=_tok([1, 2, 3]),
            question_token_ids=_tok([10, 11]),
            answer_token_ids=_tok([20]),
        )
        s2 = _make_qa_sample(
            content_token_ids=_tok([4, 5]),
            question_token_ids=_tok([12, 13, 14]),
            answer_token_ids=_tok([21, 22]),
        )
        out = collate_qa([s1, s2])
        assert out["content_max_seqlen"] == 3
        assert out["question_max_seqlen"] == 3
        assert out["answer_max_seqlen"] == 2
