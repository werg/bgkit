# L1 cross-section interaction rework (2026-06-11)

## Problem (confirmed in code)
The L0/L1 split rebuild (2026-05-03) left **L1 with no cross-section interaction** — defeating
the "Interaction" in Knowledge Interaction Transformer:
- `encoder.forward` runs L1 with `content_cu_seqlens = l0_out.survivor_cu_seqlens` (`encoder.py:364`)
  — the **per-section** boundaries. Each section is compressed in its own varlen segment again;
  sections never attend to each other in the encoder. They only merge at the **decoder** (`group_cu`).
- L1 is built `with_prompt=False` (`encoder.py:479,537`) → no `prompt_separator_embedding`, and
  `encoder.forward` passes **no prompt** to L1. (Phase 2 KB passes a query to L1 but `with_prompt=False`
  makes it a no-op — same bug, second site.)
- No section separators anywhere.

## Goal
L1 = the global cross-section interaction layer. Per sample, merge all its sections' L0 survivors
into ONE L1 segment with **full bidirectional attention** across sections AND the compression prompt,
with **section-separator embeddings** between sections. L0 stays per-section (local compression first).
This matches Phase 2 KB's intent (merge a query's articles at L1).

## Design decisions (user-approved 2026-06-11)
- **L1 prompt**: the SAME compression prompt as L0, prepended ONCE at the start of the merged L1
  segment (not per-section). L0 keeps it per-section.
- **Re-training**: keep the live 8192 run going (banks L0/decoder); restart with migration once
  implemented + tested. New L1 params init fresh; L1 re-adapts to the cross-section regime.
- **Separators are context-only**: they attend but are NOT survivor candidates (excluded from the
  head's selection via the `content_selectable_mask`, same mechanism as the prompt).

## Implementation (additive / backward-compatible — old path stays when new args are None)

1. **`level_compressor.py`**
   - `_build_combined_pack(... , content_selectable_mask=None)`: the content portion of
     `content_position_mask` uses `content_selectable_mask` (default all-True) so separator positions
     can be excluded from selection. Also honored in the no-prompt branch.
   - `LevelCompressor.forward(... , content_selectable_mask=None)`: thread it to `_build_combined_pack`
     and the no-prompt branch's `content_pos_mask`.

2. **`encoder.py`**
   - Build L1 `with_prompt=True` (both `_build_*` constructors): gives L1 `prompt_separator_embedding`.
   - Add `self.section_separator_embedding = nn.Parameter(zeros(hidden))` (small init).
   - `forward(... , content_group_cu_seqlens=None, prompt_embeddings_l1=None, prompt_cu_seqlens_l1=None,
     prompt_position_ids_l1=None)`:
     - If `content_group_cu_seqlens is None` → CURRENT per-section L1 (backward compat).
     - Else → re-segment: for each sample, concat its sections' bridged L0 survivors with
       `section_separator_embedding` between them → per-sample `content`, `content_cu` (per-sample),
       `content_position_ids` (per-sample restart), `content_selectable_mask` (False at separators).
       Run L1 with `prompt_embeddings_l1` (one prompt per sample) + the selectable mask.
     - L1 output is now per-sample; projection + `enc_out.survivor_cu_seqlens` are per-sample.

3. **`summarization_round_robin.py::_encode_batch`**
   - Build `content_group_cu_seqlens` from `group_doc_counts` (cumsum → section-index boundaries).
   - Build L1 prompt = `comp_prompt_ids[g]` once per sample (vs per-section for L0).
   - Pass both to `encoder.forward`. `group_cu` becomes `enc_out.survivor_cu_seqlens` directly
     (L1 already per-sample); drop the per-doc→per-group re-sum.
   - L1 survivorship aux losses now per-sample (was per-section) — verify cu plumbing.

4. **Migration** (`encoder.py::from_pretrained_with_state_dict`): init `section_separator_embedding`
   and `l1.prompt_separator_embedding` if absent. L1 backbone weights carry over.

5. **Phase 2 consistency**: L1 `with_prompt=True` fixes the no-op query there automatically.

6. **Tests**: unit-test re-segmentation (sections merge, separators inserted + excluded from selection,
   prompt prepended once, L1 cu per-sample); confirm old path unchanged when `content_group_cu_seqlens=None`.

## Compute note
Merging ~5 sections (~5k survivors) into one L1 segment → L1 attention ~5× vs per-section, but it's the
correct computation and shrinks as the curriculum ratio descends.
