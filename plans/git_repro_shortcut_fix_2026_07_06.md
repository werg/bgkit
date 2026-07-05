# git-repro decorative-reps shortcut — fix plan (2026-07-06)

**Status:** items #1 (ablation eval) + #3 (shared-tree 0-node fix) DONE + committed
(`c3a09a6`, `80df1ac`). Item #2 (close the shortcut) is a **design decision** —
this doc lays out the options for your go-ahead. I did NOT blindly reverse the
retrieval-task design overnight.

## The problem (proven by live ablation probe + 4 subagents)
Training loss plateaus ~1.5 but eval IMPROVES — because the model learns a
**shortcut**: `recon_gap = 0.0` and `nav_gap = 0.0` (zeroing the compressed tree
survivors changes NEITHER reconstruction NOR navigation CE). The compressed reps
are **decorative**. recon_gap was +0.016 early and decayed to 0 over 3000 steps.

## The 5 shortcuts (ranked; shortcut-hunt subagent, file:line in its report)
1. **Reconstruction leak (dominant).** Mode mix (0.10 full / 0.30 no_drill / 0.60
   truncated): 90% of samples never put the target diff in the reps (no_drill = only
   the whole-window head rep; truncated cut *above* the retrieval leaf), yet the gold
   is the *exact file blob* — unrecoverable from 0.05-retention reps. → model learns
   pure AR reconstruction, ignores reps GLOBALLY. `drilldown.py` DEFAULT_MODE_WEIGHTS,
   `_build_no_drill`/`_build_truncated`.
2. **AR-predictability.** Even full-mode recon is teacher-forced next-token CE; code
   is self-predictable so reps add ~nothing (full-mode recon_gap only ~0.02).
3. **Nav leak.** Nav loss = plain token-CE over IDs that are a deterministic
   `(repo,sha)` function, frozen in the .arrow across epochs, sharing a plaintext
   `{repo}/` prefix → memorized over 5000 steps. No rep-based discrimination head
   (planned §4d, never built); distractors are `loss=False` (decorative).
4. **Leaf ID plaintext-copyable.** `file_change_leaf_id = {commit_node_id}/{path}` —
   both parts already in context (`commit_repro.py:280`).
5. Distractors decorative (`loss=False`; subsumed by #3).

## Fix options for RECON (pick one — this is the design call)
- **(A) Revert recon to full-drill.** Set mode weights so recon-bearing trajectories
  drill to the diff (target reachable in the reps). Simple, proven mechanism, LOW risk.
  BUT recon_gap only ~0.02 (AR-predictability ceiling) and it abandons the
  retrieve-from-higher-reps idea. Change: `DEFAULT_MODE_WEIGHTS` / `build_file_drilldown_trajectory` mode_weights → (1.0,0,0); regen trajectories.
- **(B) Cloze recon (RECOMMENDED — preserves your retrieval intent).** Keep the modes,
  but change the SCORED target: mask spans in the gold that are rep-exclusive (appear
  only in a retrieved/compressed diff, NOT in the head query prefix) and score only
  those. Forces the model to RETRIEVE the masked content from the compressed reps →
  recon_gap can't be AR-shortcut. More work: identify rep-exclusive gold spans at
  trajectory-build time (diff-added lines not in the query), emit a cloze mask; loss
  over cloze positions only. Needs a build-script change + a decode-loss-mask change +
  GPU verification.

## Fix for NAV (do both, alongside the recon fix)
- **Rep-based discrimination loss** (the real fix): at each branch, softmax over
  {gold child rep, distractor-sibling reps} scored FROM the parent rep (ties the
  currently-`loss=False` distractors into the loss). Then nav can't be a memorized
  string. New loss term; GPU verification needed.
- **Opaque + non-memorizable leaf IDs:** make `file_change_leaf_id` opaque
  (`bip39_id('fc|'+commit_node_id+'|'+path+'|'+file_idx)`) AND re-salt interior ids
  per-epoch with a nonce so `(repo,position)` no longer predicts the id. Data change +
  the tree/splice keys must be re-derived consistently (encode_tree_node pins them).

## Recommendation
**(B) cloze recon + nav discrimination + opaque/re-salted ids.** It preserves the
retrieval framing you chose, and is the only path to a STRONG recon_gap/nav_gap
(vs (A)'s ~0.02 ceiling). It's more work and needs a data regen + GPU verification,
so it warrants your sign-off rather than an overnight blind implementation.

## What's running overnight (measurement + the clear fixes)
Resumed from step 3210 with #1 (eval now logs `eval/ablation/{zeroed,noise,neither}/…`
+ `eval/recon_gap` + `eval/nav_gap` every 500 steps) and #3 (shared-tree floor — reps
now PRESENT; watch `per_repo_shared_tree_nodes` >0 and `phase2_kb_shared_tree_collapsed`
warns → ~0). The model stays shortcut-poisoned (recon fix not in) so expect
recon_gap/nav_gap ≈ 0 in the eval — that's the baseline the real fix must move. When
you pick (A) or (B), it wants a FRESH restart from summarization (poisoned model won't
un-learn the shortcut by resuming).
