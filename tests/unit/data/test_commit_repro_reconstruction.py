from __future__ import annotations

import subprocess

import pytest

from bgkit.data.commit_repro import (
    FileChange,
    ReproCommit,
    apply_structured_patch,
    prepare_reconstruction_chains,
    reconstruction_chain,
    render_file_change_evidence,
    require_record_schema,
    walk_repo_commits_oldest_first,
)


def _hunk(old: str, new: str, *, old_start: int = 1) -> list[dict]:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    return [{
        "old_start": old_start if old_lines else 0,
        "old_lines": len(old_lines),
        "new_start": old_start if old_lines else 1,
        "new_lines": len(new_lines),
        "lines": (
            [{"origin": "-", "content": line} for line in old_lines]
            + [{"origin": "+", "content": line} for line in new_lines]
        ),
    }]


def _commit(ordinal: int, base: str, target: str) -> ReproCommit:
    return ReproCommit(
        repo="owner/repo",
        sha=f"sha-{ordinal}",
        parent_sha=f"sha-{ordinal - 1}" if ordinal else "",
        ordinal=ordinal,
        message=f"message {ordinal}",
        timestamp=ordinal,
        file_changes=[FileChange(
            file_idx=0,
            path="f.py",
            old_path="f.py",
            change_type="added" if ordinal == 0 else "modified",
            lineage_id="lineage",
            diff_text=f"patch {ordinal}",
            hunks=_hunk(base, target),
            base_blob_text=base,
            base_blob_available=True,
            blob_text=target,
            is_target=True,
        )],
    )


def test_apply_structured_patch_is_exact_and_fail_closed():
    hunks = [{
        "old_start": 2,
        "old_lines": 1,
        "new_start": 2,
        "new_lines": 2,
        "lines": [
            {"origin": "-", "content": "b\n"},
            {"origin": "+", "content": "B\n"},
            {"origin": "+", "content": "X\n"},
        ],
    }]
    assert apply_structured_patch("a\nb\nc\n", hunks) == "a\nB\nX\nc\n"
    with pytest.raises(ValueError, match="context mismatch"):
        apply_structured_patch("a\nwrong\nc\n", hunks)


def test_periodic_anchors_bound_complete_reconstruction_chains():
    commits = [
        _commit(0, "", "a\n"),
        _commit(1, "a\n", "b\n"),
        _commit(2, "b\n", "c\n"),
    ]
    stats = prepare_reconstruction_chains(commits, anchor_interval=2)
    changes = [commit.file_changes[0] for commit in commits]
    assert stats == {
        "targets_checked": 3,
        "targets_valid": 3,
        "targets_rejected": 0,
        "anchors": 2,
    }
    assert [change.is_anchor for change in changes] == [True, False, True]
    assert changes[0].base_blob_text == ""
    assert changes[1].base_blob_text == ""
    assert changes[2].base_blob_text == "b\n"

    history = [(i, change) for i, change in enumerate(changes)]
    assert reconstruction_chain(history, 1) == history[:2]
    assert reconstruction_chain(history, 2) == history[2:]


def test_invalid_patch_is_not_a_training_target():
    commit = _commit(0, "wrong\n", "gold\n")
    commit.file_changes[0].hunks = _hunk("expected\n", "gold\n")
    stats = prepare_reconstruction_chains([commit])
    change = commit.file_changes[0]
    assert stats["targets_rejected"] == 1
    assert not change.is_target
    assert not change.reconstruction_valid


def test_reconstruction_chain_revalidates_serialized_patches():
    commits = [_commit(0, "", "a\n"), _commit(1, "a\n", "b\n")]
    prepare_reconstruction_chains(commits, anchor_interval=8)
    history = [(i, commit.file_changes[0]) for i, commit in enumerate(commits)]
    commits[1].file_changes[0].hunks = _hunk("wrong\n", "b\n")

    with pytest.raises(ValueError, match="patch failed"):
        reconstruction_chain(history, 1)


def test_invalid_patch_cannot_pass_when_gold_is_empty():
    commit = _commit(0, "base\n", "")
    commit.file_changes[0].hunks = [{
        "old_start": 1,
        "old_lines": 1,
        "new_start": 1,
        "new_lines": 0,
        "lines": [{"origin": "-", "content": "wrong\n"}],
    }]
    stats = prepare_reconstruction_chains([commit])
    assert stats["targets_rejected"] == 1
    assert not commit.file_changes[0].is_target


def test_encoded_evidence_contains_join_key_and_only_anchor_base():
    commit = _commit(0, "base\n", "target\n")
    change = commit.file_changes[0]
    change.is_anchor = True
    text = render_file_change_evidence(commit, change)
    assert "commit-sha: sha-0" in text
    assert "commit-message: message 0" in text
    assert "base\n" in text
    change.is_anchor = False
    assert "base\n" not in render_file_change_evidence(commit, change)


def test_schema_v2_validation_rejects_legacy_and_incoherent_targets():
    record = {
        "schema_version": 2,
        "repo": "owner/repo",
        "sha": "a" * 40,
        "parent_sha": "b" * 40,
        "window_idx": 0,
        "ordinal": 0,
        "file_changes": [{
            "file_idx": 0,
            "path": "f.py",
            "old_path": "f.py",
            "change_type": "modified",
            "lineage_id": "lineage",
            "diff_text": "",
            "hunks": [],
            "base_blob_text": "base\n",
            "base_blob_available": True,
            "blob_text": "target\n",
            "is_anchor": True,
            "is_target": True,
            "reconstruction_valid": True,
        }],
    }
    require_record_schema(record)

    legacy = dict(record, schema_version=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        require_record_schema(legacy)

    incoherent = dict(record)
    incoherent["file_changes"] = [dict(record["file_changes"][0], is_target=False)]
    with pytest.raises(ValueError, match="flags must agree"):
        require_record_schema(incoherent)


def test_git_walker_uses_first_parent_and_preserves_rename_replay(tmp_path):
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=tmp_path, check=True,
            text=True, capture_output=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "f.txt").write_text("root\n")
    git("add", "f.txt")
    git("commit", "-qm", "root")
    main_branch = git("branch", "--show-current")

    git("checkout", "-qb", "feature")
    (tmp_path / "feature.txt").write_text("feature\n")
    git("add", "feature.txt")
    git("commit", "-qm", "feature-only")
    feature_sha = git("rev-parse", "HEAD")

    git("checkout", "-q", main_branch)
    (tmp_path / "f.txt").write_text("main\n")
    git("commit", "-qam", "main-change")
    git("merge", "--no-ff", "-qm", "merge-feature", "feature")
    git("mv", "f.txt", "renamed.txt")
    git("commit", "-qm", "rename")

    rows = list(walk_repo_commits_oldest_first(str(tmp_path), "owner/repo"))
    shas = [row[0] for row in rows]
    assert feature_sha not in shas
    assert [row[2] for row in rows] == [
        "root", "main-change", "merge-feature", "rename",
    ]
    for index, (sha, parent_sha, _message, _timestamp, files) in enumerate(rows):
        assert sha
        if index:
            assert parent_sha == rows[index - 1][0]
        for change in files:
            if change.base_blob_text is not None and change.blob_text is not None:
                assert apply_structured_patch(
                    change.base_blob_text, change.hunks,
                ) == change.blob_text

    rename = rows[-1][4][0]
    assert rename.change_type == "renamed"
    assert rename.old_path == "f.txt"
    assert rename.path == "renamed.txt"

    capped = list(walk_repo_commits_oldest_first(
        str(tmp_path), "owner/repo", max_walked=2,
    ))
    assert [row[2] for row in capped] == ["merge-feature", "rename"]
