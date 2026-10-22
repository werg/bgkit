"""Git history knowledge-retrieval dataset wrapper.

Adds git-history-specific functionality on top of Phase2QADataset:
- Per-sample ``question_type`` from metadata
- Grouping by repo for commit-chain assembly
- ``get_chain_samples(repo_id, max_commits)`` for L1 cross-commit compression

The metadata parquet is expected to contain ``question_type`` and ``repo_path``
columns (written by the git QA mmap converter).  ``commit_sha`` is used for
temporal ordering within a repo chain.
"""

from __future__ import annotations

from collections import defaultdict

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample


class GitHistoryDataset(Phase2QADataset):
    """Git commit history knowledge-retrieval dataset.

    Extends the generic loader with:

    * ``question_type(idx)`` -- returns the QA question type for a sample
      (e.g. ``factual_recall``, ``rationale``, ``diff_grounded``, etc.).
    * ``repo_indices`` -- mapping from repo identifier to list of dataset
      indices belonging to that repo.
    * ``get_chain_samples(repo_id, max_commits)`` -- returns a list of
      ``QASample`` objects from the same repo, ordered by their position in
      the dataset, capped at *max_commits*.  Used by L1 cross-commit
      compression training to assemble commit chains.
    """

    # Columns we need from the parquet beyond the base set.
    _EXTRA_COLUMNS = ("question_type", "repo_path", "commit_sha")

    def __init__(self, data_dir: str, **kwargs):
        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name="git_history",
            metadata_columns=columns,
            **kwargs,
        )

        # Build repo -> [valid_idx, ...] mapping.
        self._repo_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            repo_id = str(
                meta.get("repo_path", meta.get("document_id", meta.get("id", "")))
            )
            self._repo_to_indices[repo_id].append(idx)

    # ------------------------------------------------------------------
    # Per-sample accessors
    # ------------------------------------------------------------------

    def question_type(self, idx: int) -> str:
        """Return the question type for sample *idx*.

        Valid types (from ``generate_git_qa.py``): ``factual_recall``,
        ``rationale``, ``diff_grounded``, ``cross_commit``, ``temporal``,
        ``state_recall``.  Falls back to ``"unknown"`` when the column is
        absent.
        """
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        return str(meta.get("question_type", "unknown"))

    # ------------------------------------------------------------------
    # Repo grouping
    # ------------------------------------------------------------------

    @property
    def repo_indices(self) -> dict[str, list[int]]:
        """Mapping from repo identifier to list of dataset indices."""
        return dict(self._repo_to_indices)

    @property
    def repo_ids(self) -> list[str]:
        """Sorted list of unique repo identifiers in this dataset."""
        return sorted(self._repo_to_indices.keys())

    def get_chain_samples(
        self,
        repo_id: str,
        max_commits: int = 32,
    ) -> list[QASample]:
        """Return up to *max_commits* ``QASample`` objects from *repo_id*.

        Samples are returned in dataset order (which mirrors the commit
        extraction order from ``generate_git_qa.py``).  When multiple QA
        pairs exist per commit, all are included but the total is capped at
        *max_commits* samples.

        This method is designed for L1 cross-commit compression where the
        trainer needs multiple commit contexts from the same repo packed
        together for joint encoding.

        Args:
            repo_id: Repository identifier (e.g. ``"owner/repo"``).
            max_commits: Maximum number of samples to return.

        Returns:
            List of ``QASample`` objects, possibly empty if *repo_id* is
            not found.
        """
        indices = self._repo_to_indices.get(repo_id, [])
        # Indices are already in insertion order (ascending dataset index).
        chain_indices = indices[:max_commits]
        return [self[idx] for idx in chain_indices]

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]

        # Inject question_type and repo_path into sample metadata.
        sample.metadata["question_type"] = str(meta.get("question_type", "unknown"))
        sample.metadata["repo_path"] = str(
            meta.get("repo_path", meta.get("document_id", ""))
        )
        sample.metadata["commit_sha"] = str(meta.get("commit_sha", ""))

        # Add question_type as a tag.
        qt = sample.metadata["question_type"]
        if qt != "unknown" and qt not in sample.tags:
            sample.tags.append(qt)

        return sample
