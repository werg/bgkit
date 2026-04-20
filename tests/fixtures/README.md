# Parity Fixtures for FA4-Varlen Migration

**Status (2026-04-20): retained** — these fixtures captured padded-mode reference
outputs BEFORE the FA4-varlen migration landed. The packed rewrite is now the
only code path (fallbacks deleted), so the fixtures serve as **frozen regression
references** against future rewrites. Keep them in the tree.

These `.pt` files capture padded-mode reference outputs from the pre-migration
`(B, L)` + `attention_mask` code. Wave 1 and Wave 2 parity tests load them
and compare the same computation after the varlen rewrite (packed `(N,)` +
`cu_seqlens`) using `torch.testing.assert_close`.

| File | Contents | How Wave tests use it |
|---|---|---|
| `encoder_reference.pt` | BgKITEncoder forward (no compression + compressed). Inputs: content_ids, attention_mask, input_embeddings. Outputs: survivor_embeddings, survivor_mask, base_raw, logits_for_op. | Wave 1 encoder test: run packed forward, unpack to padded, compare. |
| `decoder_reference.pt` | ReconstructionDecoder.forward_with_single_splice. Inputs: token_ids, token_attention_mask, survivor_embeddings, survivor_attention_mask, splice positions. Outputs: loss (scalar), hidden_states, loss_mask. | Wave 1 decoder test: run varlen splice forward, compare loss and hidden states. |
| `deltanet_reference.pt` | chunk_gated_delta_rule: batched padded call AND per-sample sequential calls. Inputs: (q,k,v,g,beta,scale,valid_lengths). Outputs: delta_out, recurrent_state per sample. | Wave 1 DeltaNet test: run single packed varlen call, compare against per-sample sequential reference. |
| `survivorship_losses_reference.pt` | All survivorship loss functions (ratio, decisiveness, min_survivors, qa_position, utility_grad_bce). Inputs: base_raw, logits_for_op, valid_mask, etc. Outputs: per-loss scalars. | Wave 2 losses test: same inputs, same outputs expected from varlen-adapted loss functions. |
| `phase2_losses_reference.pt` | Phase 2 per-article L0 and per-trajectory L1 aux losses (ratio, decisiveness, relevance, min_survivors). Inputs: L0/L1 logit lists + masks. Outputs: per-component loss scalars. | Wave 2 Phase 2 losses test: packed L0/L1 pass, unpack and compare each component. |
| `step3_smoke_microbatch.pt` | One full Step 3 forward+backward pass (untrained encoder+decoder). Inputs: content_ids, attention_mask. Outputs: loss scalar, grad_norm, survivor embeddings. | Verification §5: smoke-test the packed Step 3 pipeline end-to-end. |

All fixtures use `torch.manual_seed(17)`, B=4, L≤128, D=1024, bf16 weights,
fp32 reference outputs where noted. Load with `torch.load(..., weights_only=False)`.
