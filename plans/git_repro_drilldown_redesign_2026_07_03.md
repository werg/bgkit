# Git-commit-repro → pure recursive drill-down redesign (+ shared Phase-2 drill architecture)

**Date:** 2026-07-03
**Status:** design locked, one open decision (§7), pending implementation
**Supersedes:** the current `browse + bgkit` trajectory for `git_commit_repro`

## 1. Why we're rebuilding

The current run trained a design where the compressed representations were **decorative** — the L1 backbone moved 0.019% over 340 steps, `l1.head` changed exactly 0.0, and the loss was flat because every training target was solvable *without reading the reps*:

1. **Navigation shortcut** — `browse` responses emit a **plaintext child-listing** (`render_browse_response`, `browse_tree.py:192-206`) next to the tree rep, so the next node ID is copyable from text; the rep is never needed. `∂(nav loss)/∂(rep) ≈ 0`.
2. **No discrimination** — the walk is a single gold path with **no distractor targets** (`build_file_reconstruction_trajectory`, `commit_repro.py:465-506`); navigation isn't even a choice.
3. **Reconstruction shortcut** — gold is predictable autoregressively from its own prefix, so drill survivors get little gradient.

`browse` itself is a **QA-framework leftover** (topic hierarchies for KILT/PubMedQA) reused verbatim for git-repro, where it's redundant with the drill.

## 2. Target architecture (locked)

**Tree / caching**
- Deep commit-history tree (`root → repo → repo/wK → c16 → c4 → commit-leaf`, `commit_repro.py:275-416`), encoded **live once per repo** (full-backprop, whole tree trainable) with a **generic** query-agnostic compression prompt, shared across that repo's target files. (Full-backprop through the whole tree is the central goal — §7.)
- Per-sample, recompute **only the top-level node** = L1 over the cached second-level survivors, with a **task-reformulated compression prompt**: *"find the diffs that touched `<filename>` up to the commit: `<commit message>`."* Query-aware entry point and first drill step.

**Drill-down (multi-path, depth-first, no backtrack markers)**
- A file is touched by many commits → find a **capped number** of the touching diffs → **multiple paths** through the tree, each ending at a diff-leaf that touched the file. Emitted as a single **depth-first** linear sequence (no explicit backtrack tokens).
- **0–4 distractor branches per sample**, each splitting off at a **random point** in the walk (a wrong sibling subtree the model must not follow).
- **Single `drill` op** — no `browse`, no plaintext child-listing. Child identity lives only in the reps.

**Losses**
- **Navigation discrimination** — at each branch, pick the correct child among {gold child(ren) + distractors}. This is the signal that finally makes the tree reps load-bearing (it's unsolvable without reading them). **Primary rep-training signal.**
- **Reconstruction** — **normal autoregressive** CE over the full gold file, generated from the collected drill survivors (no separate untrained prefix; already the current behavior, `_encode_decode_group:4659-4692`). Not masked.

**Shared code** — the drill-down trajectory + recursive-encode + discrimination loss live in one reusable module, inherited by KILT/PubMedQA and future runs.

## 3. What already exists (reuse, do not rebuild)

- **Touching-set**: `commit_repro.build_per_file_index` (`commit_repro.py:435-447`) already yields the complete oldest-first list of every commit touching each file. The current builder computes it then discards all but K-preceding of one target. **No new git extraction.**
- **Offline generic cache**: `scripts/precompute_l1_tree.py` caches per-node L1-output survivors keyed `(dataset, node_id)`, every node, **already query-agnostic** (`:22-24`, `:281-287`). Read via `SurvivorBlockCache.get/has/node_ids` (`l0_cache.py:65-208`). This is exactly the "deep tree cached with generic prompt" substrate.
- **Recursive encode**: `_run_recursive_l1`/`_recursive_l1_forward` (`kr_kb_trainer.py:3815-3833`) take `(children_reps_l1in, q_emb, ratio)`. Per-sample query emb `_recursive_query_emb` (`:3451`) and generic emb `_recursive_general_prompt_emb` (`:4410`) both exist. Cache-read of a node's cached children: `_encode_path_node` (`:3687-3700`).
- **Multi-drill batching**: `_encode_decode_group` already batches multiple drill turns per sample (`:4625-4646`).
- **Normal-AR reconstruction**: the answer turn decodes the full gold blob from spliced survivors with no separate file prefix — already what we want.

## 4. What changes

### 4a. Data-prep — `commit_repro.py` + `scripts/build_commit_repro_trajectories.py`
- Rewrite `build_file_reconstruction_trajectory` (`commit_repro.py:465-506`) from single-walk to **multi-path DFS**: take the capped touching-commit set, emit a depth-first sequence of `drill` turns collecting each leaf's `file_change_id`, then a single `answer(gold_blob)`. Drop all `browse` turns.
- Extend `WalkStep` (`:455-463`) + the builder loop (`build_commit_repro_trajectories.py:116-171`) to collect the **capped touching set** (from `build_per_file_index`) rather than one-target-per-commit.
- **New: branch distractors.** Author 0–4 wrong sibling subtrees at random branch points in the DFS. This is a genuinely new mechanism — distinct from the existing within-commit *file* distractors (`_sample_distractors`, `:3029-3102`), which stay for the (separate) relevance loss if ever re-enabled.
- New cap constant `MAX_TOUCHING_DIFFS` (replaces the single-target `K`-preceding slice).

### 4b. Trajectory template — `src/bgkit/data/bgkit_tool_template.py`
- Add a single `drill(child_id)` turn type carrying (a) the compression-prompt splice and (b) the distractor sibling IDs for that branch. Loss on the drill-target choice.
- Remove the plaintext child-listing path for recursive-drill datasets (the `browse` response text).

### 4c. Trainer — `src/bgkit/training/phase2/kr_kb_trainer.py`
- **Sever the browse-turn coupling**: `_repo_group_key` (`:3904-3916`) and the assert in `_forward_backward_per_repo` (`:5774-5778`) require the first turn to be a `browse`. Derive the shared-tree root from the sample's repo/window id instead.
- **Keep the live per-repo full-backprop encode** (`_compute_shared_repo_tree:4473-4546`, `full_backprop_per_repo: true`, `_l1_tree_cache=None` at `:918`). The whole tree stays trainable — full-backprop is the goal (§7). No cache adoption. The tree is encoded once per repo and shared across that repo's target files; gradient flows through the entire tree.
- **Per-sample head**: reuse `_encode_path_node`'s cache-read (`:3687-3700`) + `_run_recursive_l1` (`:3815`) over the head's cached second-level children, driven by `_recursive_query_emb` (task prompt) instead of `_recursive_general_prompt_emb`.

### 4d. Loss — `kr_kb_trainer.py`
- **Navigation discrimination loss**: at each branch, a softmax over {gold child + distractor siblings} scored from the reps (no text to copy). This is the new rep-training signal. (Cleanest as softmax-CE over the option set; alternative per-child binary noted for discussion.)
- **Reconstruction**: unchanged normal-AR gold CE.

### 4e. Shared module — new `src/bgkit/models/recursive_l1.py` (or `training/phase2/recursive_drill.py`)
- Factor the tree-head → child-drill → L0 recursion + the DFS drill-down trajectory primitives + the discrimination loss out of `KRKBTrainer` into one module, so KILT/PubMedQA/future datasets reuse the identical pure-drill architecture; only corpus-specific **tree construction** (commit-DAG vs DBpedia/MeSH) stays dataset-local.

## 4f. SHARED drill-down architecture (the reusable core — fixes ALL tree/drill-down runs)

**Mandate:** this must be shared code that fixes every current and future tree/drill-down run (git-repro, KILT, PubMedQA, NarrativeQA, …), not git-repro-specific.

**Key simplification found in the template:** the `bgkit` turn already implements the drill op correctly — a supervised tool call `bgkit(ids=[node], query=…)` (loss=`turn.loss`) followed by a sentinel with **no text side-channel** (`bgkit_tool_template.py:466-475`): child IDs travel only through the spliced survivor embeddings (ID-pinning). So the drill is already ID-pinned, plaintext-free, supervised. **`browse` is the only plaintext leak** (`render_browse_response`). The redesign = *remove browse, navigate by chained `bgkit` drills.* The discrimination (predict the right child ID) then can't be solved by copying — the ID exists only in the parent's survivors.

**Shared abstraction — `bgkit/data/drilldown.py` (new, dataset-agnostic):**
- `build_drilldown_trajectory(tree, head_node_id, target_leaf_ids, task_query, gold_answer, *, n_distractors, rng) -> list[TrajectoryTurn]`:
  - Emit a **depth-first** traversal over the union of paths from `head_node_id` to each `target_leaf_id`. One `bgkit` **drill turn per visited node** (loss=True on drills that advance toward a target).
  - **First drill = the head/top node**, marked `is_head=True` in args → trainer encodes it live with `task_query` (per-sample). All deeper drills carry no query → trainer presents the **shared generic-encoded tree** node reps.
  - **0–4 distractor drills** at random branch points: drill a wrong sibling subtree, `loss=False` (seen as negative/exploration context; the model is not trained to emit them). Sampled from the tree's siblings — dataset-agnostic mechanism, tree-specific pool.
  - Final `answer(gold_answer)` turn, loss=True, normal AR.
- The turn type stays `bgkit` (reuse the template's ID-pinned splice + `bgkit_call_spans` for the discrimination-token span). A tiny `is_head` arg selects task-query vs shared-tree-rep at the trainer. **No new template turn kind** — just stop emitting `browse`.

**Dataset provides only:** `(tree, head_node_id, target_leaf_ids, task_query, gold_answer)`. git-repro: commit-DAG tree, window node as head, the capped touching-diff leaves, the "find diffs that touched `<file>` up to `<msg>`" query, the file blob. KILT/PubMedQA: their category/MeSH tree, root as head, the gold article/passage leaves, the question as query, the answer.

**Trainer shared path** (factor out of `KRKBTrainer` into `recursive_l1.py` where practical): head-node live encode with task query + shared generic tree encode (`_compute_shared_repo_tree`, full-backprop) + drill navigation over node reps + the discrimination loss over `bgkit_call_spans` (already the supervised ID tokens, now un-shortcutable) + normal-AR answer.

**Discrimination loss:** the existing CE over the `bgkit` tool-call ID tokens (`bgkit_call_spans`) already *is* the "predict the right child" signal once browse plaintext is gone. Distractor drills (loss=False) provide contrastive context. No separate softmax head needed initially; a dedicated child-choice softmax over {gold + distractor sibling reps} is a later enhancement if CE-on-ID-tokens proves weak.

## 5. Data / cache rebuild sequence
1. (unchanged) `extract_commit_repro.py` → commit JSONL.
2. (unchanged) `build_commit_repro_tree.py` → tree forest + sidecar.
3. **(new builder)** `build_commit_repro_trajectories.py` → multi-path DFS trajectories with distractors + capped touching-set.
4. `precompute_l1_tree.py` → generic per-node L1 cache (already generic; ensure git-repro consumes it).
5. Retrain from the Phase-1 checkpoint (fresh — current weights learned to ignore the reps).

## 6. Phased implementation
1. **Shared module skeleton** (`recursive_l1.py`) + move the recursive-encode/head/discrimination primitives.
2. **Template**: single `drill` turn, drop plaintext.
3. **Data-prep**: multi-path DFS builder + distractors + cap.
4. **Trainer wiring**: sever browse coupling, per-sample task-prompt head, cache adoption, discrimination loss.
5. **Config** + integration test (`tests/integration/test_phase2_kb_e2e.py` analog for drill-down).
6. **Rebuild data** + **retrain** + validate (the survivor-ablation gap on navigation tokens should now be **large**).

## 7. RESOLVED — full-backprop through the live tree (the central goal)

**Decision: full-backprop.** Full-backprop through the *entire* tree is the **central goal** of this training run. git-repro is deliberately the regime where full-backprop is still feasible — the point is to prove it out here before moving to trees too big for it, where cached / path-selective becomes necessary. So we **keep the current live per-repo encode** (`_compute_shared_repo_tree`, `kr_kb_trainer.py:4473-4546`, `full_backprop_per_repo: true`) with the whole tree trainable, and we do **not** adopt the offline cache in this run.

**"Pre-compute and share across targets" = the per-repo encode-once**, not frozen caching: the deep tree is encoded once per repo per step (generic prompt) and shared across that repo's target files (the current Option-A per-repo path), with full-backprop flowing through the whole tree. Only the **top node is per-target** — recomputed per file with the task-reformulated prompt over the shared tree's 2nd-level survivors.

The offline cache (`precompute_l1_tree.py`) and path-selective mode stay reserved for the **later big-tree regime** and are out of scope here. This run is the feasible-size full-backprop proof.
