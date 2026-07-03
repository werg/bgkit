# Shared tree-encode with child-ID injection (all tree/drill-down runs)

**Date:** 2026-07-03
**Mandate:** the child-ID-in-rep fix MUST be a single shared primitive reused by every current + future tree-based training path — impossible to forget or diverge.

## The bug (both paths wrong, differently)

A tree node's rep must encode **which children it has, by ID**, so the decoder can read a child ID out of a parent's rep and choose where to drill. Today:
- **Live full-backprop** (`kr_kb_trainer._encode_subtree`, :3527-3543): concatenates only `l1_auto_reproduce(child.l1out)` — children's *survivors*, **no IDs**.
- **Offline cache** (`encoder.encode_node`, :474, via `precompute_l1_tree.py`): injects the node's **OWN** id as a 1-token prompt (:502-505) — not the *children's* IDs interleaved. Also wrong for navigation.
- **Cached QA** (`_cached_tree_node_survivor`, `_shared_tree_head_survivor`): same as live — survivors only.

Result: nav_gap ≈ 0 (measured) — the decoder can't read child IDs, so it navigates by the guessable positional ID format instead. (Separately fixed by opaque BIP-39 IDs.)

## The shared primitive — `src/bgkit/models/recursive_l1.py` (new)

```
encode_tree_node(encoder, tokenizer, children_ids, children_l1outs,
                 query_emb, ratio, *, selection_mode="threshold",
                 project=True) -> (proj_or_None, l1out)
```
Single source of truth for encoding ONE tree node from its children:
1. For each child `i`: `id_emb = embed_tokens(tokenizer.encode(f" {children_ids[i]}"))` (L1-input space, **un-bridged**); `surv = encoder.l1_auto_reproduce(children_l1outs[i])` (bridge L1-output→L1-input; for leaves the caller passes L0 survivors already bridged via `l0.auto_reproduce`).
2. Interleave per child: `[id_emb | surv]`. Build `pinned` bool mask (True over `id_emb` rows, False over `surv`) and the single-segment `cu`.
3. `encoder.run_l1_and_project(interleaved, cu, ratio, prompt_embeddings_l1=query_emb, pinned_positions_l1=pinned, selection_mode_l1=selection_mode)` — the existing `pinned_positions_l1` keeps the IDs through the head so they land in the node's output rep.
4. Return `(proj_out, l1_out)` (or `(None, l1_out)` when `project=False`, for the offline cache which stores L1-output only).

`embed_tokens = encoder.l0.backbone.get_input_embeddings()`. The child-id embeddings **carry gradient** into `embed_tokens` (desired — the model learns the ID token embeddings). No node's-own-id prompt — the *parent* supplies each child's id, so it's redundant.

## Migrate ALL callers to it (the "must be reused" guarantee)

| path | file:fn | change |
|---|---|---|
| live full-backprop, interior | `_encode_subtree` :3527-3543 | build `children_ids`/`children_l1outs` from `node.children`, call `encode_tree_node(project=True)` |
| live full-backprop, leaf | `_encode_leaf_subtree` :3549 | children = the leaf-tag's articles; ids = **file names**; call it |
| cached QA node | `_cached_tree_node_survivor` | same call, children from cache |
| cached QA head | `_shared_tree_head_survivor` | same call, task query |
| offline cache build | `precompute_l1_tree.py` (`encode_node`) | replace `encode_node` with `encode_tree_node(project=False)`; **delete/deprecate `encode_node`'s own-id-prompt path** |

After this, `encode_node` has no callers → delete it (or make it a thin deprecated shim raising a pointer to `encode_tree_node`). `_run_recursive_l1`/`_recursive_l1_forward` either become the internals of `encode_tree_node` or call it.

## IDs (dataset-local construction, shared utility)

- Shared `src/bgkit/data/opaque_ids.py`: `bip39_id(hash_input, n_words=2)` + `single_token_bip39_wordlist(tokenizers)` (filter BIP-39 to words that are 1 token in Qwen3.5 + Falcon). Deterministic (sha256).
- git-repro (`commit_repro.py`): interior node ids = `bip39_id(sha)`; leaf article ids = **file name**. mmap doc-ids follow.
- KILT/PubMedQA (future): their tree builders use the same `opaque_ids` helper where navigation matters. Because the *encode* primitive is shared, any tree that carries ids in its nodes gets injection for free.

## Rollout
1. Build `encode_tree_node` + `opaque_ids` + unit tests (shared).
2. Migrate the 5 callers; delete `encode_node` own-id path.
3. git-repro id scheme (BIP-39 + file names) + tests.
4. **[gated on go-ahead]** data regen (tree + trajectories + mmap) + relaunch, then re-measure nav_gap (expect it to jump).

## Why this satisfies the mandate
Every tree node in the codebase is encoded by exactly one function. A new tree dataset can't accidentally skip ID injection — there's no other way to encode a node. The id *scheme* is dataset-local, but the id *plumbing into the rep* is shared and mandatory.
