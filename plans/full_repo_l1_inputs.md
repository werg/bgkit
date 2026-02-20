# Full-Repo L1 Input Context

**Status:** Deferred — implement after first training run validates the basic pipeline.

## Problem

When L1 is enabled, `DescriptionSubset` and `StructuralSubset` gather input files
from `_repo_commit_groups`, which is built from `_joined_indices` — the set of files
that successfully joined with their respective target datasets (descriptions or
structural data). Files that lack a target are excluded from L1 input entirely.

This means the encoder only sees a *subset* of the repo during L1 training. At
inference time, the encoder must compress *all* files in a repo. The training
distribution doesn't match the inference distribution.

## Goal

L1 input should include **all tokenized files** for a `(repo_path, commit_sha)`,
not just files that have description/structural targets. The *target* selection
(what the decoder must reconstruct) stays unchanged — it still requires a
joined target.

## Current Architecture

### Data flow (L1 path)

```
MmapTokenDataset          MmapDescriptionDataset
  (all tokenized files)     (files with descriptions)
        |                          |
        +--- key join -------------+
        |
  _joined_indices: [(tok_idx, desc_vi), ...]
        |
  enable_l1() builds _repo_commit_groups from _joined_indices
        |
  __getitem__ L1 path:
    1. Look up (repo_path, commit_sha) for this sample
    2. Get repo_indices from _repo_commit_groups  <-- ONLY JOINED FILES
    3. _gather_l1_files reads token_ds[joined_indices[ri][0]]
    4. Target: repo-level description (or file-level fallback)
```

### Key files

| File | Role |
|------|------|
| `src/bgkit/data/datasets/compression_dataset.py` | `DescriptionSubset`, `StructuralSubset`, `_gather_l1_files()` |
| `src/bgkit/data/datasets/mmap_token_dataset.py` | `MmapTokenDataset` — source tokens + metadata |

### Key constants

```python
DEFAULT_MAX_FILES_PER_REPO = 32    # compression_dataset.py:33
DEFAULT_MAX_TOKENS_PER_FILE = 8192 # compression_dataset.py:34
```

## Implementation

### 1. Build a full-repo file index in `enable_l1()`

In both `DescriptionSubset.enable_l1()` and `StructuralSubset.enable_l1()`, build a
**second** grouping that maps `(repo_path, commit_sha)` to **all** token dataset
chunk indices for that repo — not just the ones in `_joined_indices`.

```python
# In enable_l1(), after building _repo_commit_groups:

# Build full-repo file groups from the token dataset directly
self._all_repo_files: dict[tuple[str, str], list[int]] = {}
chunk_file_idx = self._token_ds.chunk_file_indices
meta = self._token_ds.get_metadata_table()
repo_paths = meta.column("repo_path").to_pylist()
commit_shas = meta.column("commit_sha").to_pylist()

for tok_idx in range(len(self._token_ds)):
    file_idx = int(chunk_file_idx[tok_idx])
    key = (repo_paths[file_idx], commit_shas[file_idx])
    if key not in self._all_repo_files:
        self._all_repo_files[key] = []
    self._all_repo_files[key].append(tok_idx)
```

### 2. Write a new `_gather_l1_all_files()` helper

The current `_gather_l1_files()` takes `repo_indices` into `joined_indices` and
reads `joined_indices[ri][0]` to get the token index. The new function takes
raw token dataset indices directly.

```python
def _gather_l1_all_files(
    tok_indices: list[int],
    token_ds,
    ice_ds,
    max_files: int = DEFAULT_MAX_FILES_PER_REPO,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_FILE,
    skip_extensions: frozenset[str] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
    """Gather repo files for L1 from the full token dataset.

    Unlike _gather_l1_files(), this reads directly from token dataset indices
    (not joined indices), so it includes files without description/structural
    targets.
    """
    file_ids_list = []
    file_masks_list = []
    file_ice_list = []
    has_ice = ice_ds is not None

    for tok_idx in tok_indices:
        if len(file_ids_list) >= max_files:
            break

        inner = token_ds[tok_idx]
        rids = inner["token_ids"]

        # Optional: skip known binary/non-text extensions
        if skip_extensions:
            fp = inner.get("file_path", "")
            ext = PurePosixPath(fp).suffix.lower()
            if ext in skip_extensions:
                continue

        # Cap token length
        if rids.size(0) > max_tokens:
            rids = rids[:max_tokens]

        file_ids_list.append(rids)
        file_masks_list.append(torch.ones(rids.size(0), dtype=torch.bool))

        if has_ice and tok_idx < len(ice_ds):
            ice_vals = ice_ds[tok_idx]["ce_values"]
            if ice_vals.size(0) > max_tokens:
                ice_vals = ice_vals[:max_tokens]
            file_ice_list.append(ice_vals)

    ice_result = file_ice_list if has_ice and file_ice_list else None
    return file_ids_list, file_masks_list, ice_result
```

### 3. Define skip extensions

Binary and non-text file extensions that made it through tokenization but
shouldn't be fed to the encoder as meaningful content. The token converter
doesn't filter by extension, so some of these may exist in the mmap data.

```python
# At module level in compression_dataset.py
_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    # Fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Compiled / binary
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".o", ".a",
    ".class", ".jar", ".war", ".ear",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    # Data blobs
    ".bin", ".dat", ".pickle", ".pkl", ".npy", ".npz",
    # Minified
    ".min.js", ".min.css",
})
```

Note: `.min.js`/`.min.css` won't match via `PurePosixPath.suffix` (which returns
`.js`/`.css`). Either check `fp.endswith(".min.js")` separately or accept that
minified files get included. Minified files tokenize poorly but won't break
anything — just noisy context.

### 4. Update L1 `__getitem__` paths

In `DescriptionSubset.__getitem__` (L1 branch, currently line ~381-433):

```python
# Replace:
repo_indices = (
    self._repo_commit_groups.get(group_key, [idx])
    if self._repo_commit_groups else [idx]
)
file_ids_list, file_masks_list, ice_result = _gather_l1_files(
    repo_indices, self._joined_indices, self._token_ds, self._ice_ds,
    max_files=self._max_files_per_repo,
    max_tokens=self._max_tokens_per_file,
)

# With:
all_tok_indices = self._all_repo_files.get(group_key, [])
if not all_tok_indices:
    # Fallback: use joined files only (shouldn't happen)
    all_tok_indices = [self._joined_indices[idx][0]]
file_ids_list, file_masks_list, ice_result = _gather_l1_all_files(
    all_tok_indices, self._token_ds, self._ice_ds,
    max_files=self._max_files_per_repo,
    max_tokens=self._max_tokens_per_file,
    skip_extensions=_SKIP_EXTENSIONS,
)
```

Same change in `StructuralSubset.__getitem__` (L1 branch, currently line ~598-613).

### 5. Update L1 cached length recomputation

In `enable_l1()`, the length recomputation currently sums joined files per group.
Update it to sum from `_all_repo_files` instead:

```python
# In the length recomputation loop:
for i, (tok_idx, _) in enumerate(self._joined_indices):
    file_idx = int(chunk_file_idx[tok_idx])
    key = (repo_paths[file_idx], commit_shas[file_idx])
    all_tok_indices = self._all_repo_files.get(key, [])
    n_files = min(len(all_tok_indices), self._max_files_per_repo)
    total_tokens = 0
    for gtok_idx in all_tok_indices[:n_files]:
        total_tokens += min(
            int(self._token_ds.lengths[gtok_idx]),
            self._max_tokens_per_file,
        )
    self._cached_lengths[i] = total_tokens + self._max_overhead
```

Note: this is an approximation because it doesn't account for `skip_extensions`
filtering (we don't have file paths available cheaply during length computation).
The lengths will slightly overestimate, which is fine — it means batches will be
slightly smaller than optimal, not OOM.

### 6. Keep `_repo_commit_groups` for now

Don't remove `_repo_commit_groups`. It's still needed to determine *which indices
in the dataset belong to which repo* for the purpose of deduplication — multiple
joined indices in the same repo should share L1 context. The new
`_all_repo_files` is an additional index, not a replacement.

## Testing

### Unit tests

1. **Test `_gather_l1_all_files` directly:**
   - Mock token dataset with 5 files, verify all 5 returned
   - Verify `max_files` cap works
   - Verify `max_tokens` truncation works
   - Verify `skip_extensions` filtering works (add a `.png` file, confirm skipped)

2. **Test `DescriptionSubset` L1 path with partial descriptions:**
   - Create a token dataset with 10 files in one repo
   - Create a description dataset with only 5 of those 10 files
   - Call `enable_l1()`, then `__getitem__`
   - Verify the returned `RepoCompressionSample.file_token_ids` has ~10 files
     (minus skipped extensions), not just 5

3. **Same test for `StructuralSubset`.**

4. **Test cached length recomputation:**
   - Verify `token_length()` after `enable_l1()` reflects all-repo-files sum,
     not just joined-files sum

### Integration test (manual)

Run a short training loop (100 steps) with a small dataset where some repos have
files without descriptions. Verify via logging that L1 samples contain more files
than the joined count.

## Risks

- **Memory:** More files per L1 sample = more tokens per batch. The `max_files`
  and `max_tokens` caps mitigate this, but if repos have many tokenized files,
  batches will be larger. Monitor GPU memory during first training run.

- **Noisy context:** Including low-quality files (config, lockfiles, etc.) as
  encoder input may hurt more than help if the encoder wastes capacity on them.
  If training metrics degrade, consider more aggressive extension filtering or
  a file-importance ranking (e.g., prefer `.py`/`.ts`/`.rs` over `.json`/`.yaml`).

- **Length estimation mismatch:** Cached lengths won't account for
  `skip_extensions` filtering (see step 5). This means some batches will have
  fewer tokens than the sampler expected. Harmless but slightly wasteful.

## Non-goals

- **Including files not in the token dataset at all.** If a file was excluded by
  `convert_tokens_to_npy.py` (e.g., truly binary files detected by pygit2
  `blob.is_binary`), it won't appear in `MmapTokenDataset` and we don't try to
  resurrect it. The token dataset is our universe of files.

- **Changing the L0 path.** L0 (single-file compression) continues to use only
  joined files. This change only affects the L1 input context.

- **Changing target selection.** The description/structural target for each
  sample is still determined by the join. We're only changing which files the
  encoder sees as input.
