"""Verify Stage A memory fits in DGX Spark's 128 GB unified budget.

Stage A is the most demanding phase2_kb stage on paper:

- Live L0 (no cache) — every bgkit call does a full L0 forward over every
  article in the referenced leaf tag, followed by a full L1 forward.
- L0 LoRA and L1 LoRA both trainable; decoder fully fine-tuned.
- Activation checkpointing on (default), but still pays for two encoder
  forwards (L0 + L1) plus a long-context decoder pass per step.

This script reports two things:

1. **Static budget**: parameter counts × dtype bytes for weights, activations
   estimated analytically from the plan's contexts (100 articles × doc length
   for L0; ~10K L1 positions; ~6K decoder positions), optimizer states for
   the trainable subset. Runs on CPU; no GPU required.

2. **Live verification** (``--live``): construct the full model exactly the
   way the trainer does, run one ``_run_live_l0`` + ``_run_l1_batch`` +
   decoder backward cycle on a synthetic sample, and report
   ``torch.cuda.max_memory_allocated()`` / ``max_memory_reserved``. Skips
   gracefully if no GPU is available.

Usage::

    # Static estimate (no GPU needed):
    .venv/bin/python scripts/verify_stage_a_memory.py
    .venv/bin/python scripts/verify_stage_a_memory.py --articles 150 --batch-size 2

    # Live forward/backward (requires GPU — run inside training container):
    docker compose -f docker/docker-compose.yaml run --rm train-phase2-kb-stage-a \\
        python scripts/verify_stage_a_memory.py --live

    # Worst-case live check (Stage C wikipedia profile, no activation ckpt):
    docker compose -f docker/docker-compose.yaml run --rm train-phase2-kb-stage-a \\
        python scripts/verify_stage_a_memory.py --live --ac-off \\
            --doc-length 2048 --l0-retention 0.05 --articles 100
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
from pathlib import Path

# Project import bootstrap
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


BUDGET_GB = 128.0  # DGX Spark unified memory


@dataclasses.dataclass
class MemoryReport:
    section: str
    gb: float
    notes: str = ""


def _gb(bytes_: int | float) -> float:
    return float(bytes_) / (1024**3)


# ---------------------------------------------------------------------------
# Static estimate
# ---------------------------------------------------------------------------


def static_estimate(
    *,
    articles_per_bgkit: int,
    doc_length: int,
    l0_retention: float,
    decoder_context: int,
    lora_rank: int,
    activation_checkpointing: bool,
    batch_size: int = 1,
    bgkit_turns_per_sample: int = 4,
) -> list[MemoryReport]:
    """Analytic memory estimate for one Stage A training step.

    Assumes bf16 activations and weights, fp32 optimizer state (Adam m, v),
    and the 0.8B Qwen3.5 encoder + decoder from the KB-scale plan.
    """
    # Qwen3.5-0.8B: 1.14B params. Hidden 1024, 24 layers, ~7 lora-targeted
    # projections per layer. Source: CLAUDE.md and bidirectional_qwen35.py.
    encoder_params = 1_140_000_000
    decoder_params = 1_140_000_000
    hidden_dim = 1024
    n_layers = 24
    n_lora_targets = 7  # q/k/v/o + gate/up/down
    # LoRA adapter params per wrapper: rank * (in + out). For the 7 targets
    # on Qwen3.5 that's 4*(rank*1024*2) + 3*(rank*1024*2 + rank*3072*2) ish.
    # A single 1024→1024 target costs rank * (1024 + 1024) = 2048 * rank.
    # gate/up/down go 1024↔3072 so they cost rank * (1024 + 3072) = 4096 * rank.
    per_layer_lora = 4 * (2048 * lora_rank) + 3 * (4096 * lora_rank)
    lora_params_per_level = per_layer_lora * n_layers

    ice_params = 500_000  # ~3-layer MLP, negligible
    topic_params = 0  # optional; not enabled in Stage A by default

    reports: list[MemoryReport] = []

    # -------- Weights (bf16, frozen + trainable both on device) --------
    frozen_base_bytes = (encoder_params + decoder_params + ice_params) * 2
    # Decoder is fully trainable in Stage A. We still store it in bf16 for
    # forward/activations, with fp32 master weights held in the optimizer.
    reports.append(MemoryReport(
        "Encoder base + decoder weights (bf16)",
        _gb(frozen_base_bytes),
        "Encoder base frozen; decoder trainable, held in bf16 for forward.",
    ))

    lora_total_params = 2 * lora_params_per_level  # L0 + L1
    reports.append(MemoryReport(
        "LoRA adapters L0 + L1 (bf16)",
        _gb(lora_total_params * 2),
        f"rank={lora_rank}, {n_layers} layers × {n_lora_targets} targets.",
    ))

    # -------- Optimizer state (Adam m, v in fp32 over trainable params) --------
    # Trainable set:
    #   - decoder: 1.14B params
    #   - lora L0 + L1: lora_total_params
    #   - topic embeddings: 0
    # Each trainable param holds: 1x fp32 master (4 bytes) + Adam m (4) + v (4)
    # = 12 bytes on top of the already-counted bf16 weight. bitsandbytes 8bit
    # optimizer cuts m/v to 1 byte each but we account for the worst case.
    trainable_params = decoder_params + lora_total_params + topic_params
    opt_bytes = trainable_params * 12
    reports.append(MemoryReport(
        "Optimizer state (fp32 master + Adam m/v)",
        _gb(opt_bytes),
        f"AdamW on {trainable_params/1e6:.0f}M trainable params.",
    ))

    # -------- Gradients (bf16 over trainable) --------
    grad_bytes = trainable_params * 2
    reports.append(MemoryReport(
        "Gradients (bf16 over trainable)",
        _gb(grad_bytes),
        "Held during backward; freed after optimizer step.",
    ))

    # -------- Activations --------
    # Per-layer activation rule-of-thumb for a transformer block:
    # residual (B*L*D) + SDPA QKV projections (~3× residual) + MLP intermediate
    # (~3× residual because gate/up/down go 1024↔3072) + norm/residual buffers.
    # ≈ 8× (B*L*D) bf16 bytes per layer. This is pessimistic for SDPA which
    # reuses workspace, but matches observed PyTorch behavior under autograd.
    def _transformer_activation(batch: int, seq: int, layers: int) -> int:
        per_layer = batch * seq * hidden_dim * 2  # residual stream
        scale_per_layer = 8  # see comment above
        if activation_checkpointing:
            # Saved activations ≈ 1 layer boundary × (fwd + scratch) + current block.
            return per_layer * scale_per_layer * 2  # boundary + in-flight block
        # Full stack: every layer keeps its activations in the autograd graph.
        return per_layer * scale_per_layer * layers

    # L0 forward: (articles_per_bgkit × doc_length × hidden) per sample.
    # With cross-sample batching on the decoder side, multiple samples'
    # L0 forwards run sequentially (they don't all live at once — autograd
    # retains the L0 graph until L1 + decoder backward finishes, though).
    B_l0 = articles_per_bgkit
    L_l0 = doc_length
    # Both L0 and L1 graphs are retained across samples in the batch until the
    # final backward, so the peak scales with batch_size.
    l0_act_bytes = _transformer_activation(B_l0, L_l0, n_layers) * batch_size
    reports.append(MemoryReport(
        "L0 encoder activations",
        _gb(l0_act_bytes),
        f"{B_l0} articles × {L_l0} tokens × batch={batch_size}, "
        f"activation_checkpointing={activation_checkpointing}",
    ))

    # L1 forward: one big sequence of (articles_per_bgkit × ceil(L0 * l0_retention)) + query.
    l1_seq_len = max(1, int(articles_per_bgkit * L_l0 * l0_retention)) + 256
    l1_act_bytes = (
        _transformer_activation(1, l1_seq_len, n_layers)
        * bgkit_turns_per_sample
        * batch_size
    )
    reports.append(MemoryReport(
        f"L1 encoder activations (×{bgkit_turns_per_sample} turns × batch {batch_size})",
        _gb(l1_act_bytes),
        f"{l1_seq_len} positions/turn, bf16, "
        f"activation_checkpointing={activation_checkpointing}",
    ))

    # Decoder activations: long context (~5-6K positions per sample).
    B_dec = batch_size
    dec_act_bytes = _transformer_activation(B_dec, decoder_context, n_layers)
    reports.append(MemoryReport(
        "Decoder activations",
        _gb(dec_act_bytes),
        f"{decoder_context} tokens × batch={batch_size}, "
        f"activation_checkpointing={activation_checkpointing}",
    ))

    # Attention softmax workspace + cuBLAS scratch. Rough 5% overhead.
    scratch_bytes = int(sum(r.gb for r in reports) * 1024**3 * 0.05)
    reports.append(MemoryReport(
        "Scratch / cuBLAS / workspace",
        _gb(scratch_bytes),
        "Rough 5% overhead on everything above.",
    ))

    return reports


def print_table(title: str, reports: list[MemoryReport]) -> float:
    print(f"\n=== {title} ===")
    total = 0.0
    name_w = max(len(r.section) for r in reports)
    for r in reports:
        total += r.gb
        print(f"  {r.section:<{name_w}}  {r.gb:7.2f} GB   {r.notes}")
    print(f"  {'TOTAL':<{name_w}}  {total:7.2f} GB")
    print(f"  (budget: {BUDGET_GB:.0f} GB unified; headroom: "
          f"{BUDGET_GB - total:+.2f} GB)")
    return total


# ---------------------------------------------------------------------------
# Live verification
# ---------------------------------------------------------------------------


def live_verify(
    *,
    articles_per_bgkit: int,
    doc_length: int,
    decoder_context: int,
    lora_rank: int,
    activation_checkpointing: bool,
) -> None:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM

    from bgkit.models.decoder import EmbeddingSegment, ReconstructionDecoder, TokenSegment
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.models.lora_encoder import DEFAULT_LORA_TARGETS, LoRARouter
    from bgkit.utils.attention_backend import resolve_attention_implementation
    from bgkit.utils.packing import position_ids_from_cu

    if not torch.cuda.is_available():
        print("\n[live] CUDA unavailable — skipping live verification.")
        return
    device = torch.device("cuda")

    print("\n[live] Loading Qwen3.5-0.8B encoder + decoder on GPU...")
    backbone_name = "Qwen/Qwen3.5-0.8B-Base"
    decoder_name = "Qwen/Qwen3.5-0.8B"

    # Construct the encoder exactly the same way the trainer does, minus the
    # Phase 1 state dict load. Random-init LoRA adapters (B=0 so identity at
    # init) and random-init flag embeddings are fine for a memory check —
    # we're measuring parameter/activation shapes, not numerics.
    encoder = BgKITEncoder.from_pretrained(
        backbone_name,
        hidden_dim=1024,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attention_implementation(),
    ).to(device)
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=1024)
    decoder.train()

    if activation_checkpointing and hasattr(decoder_backbone, "gradient_checkpointing_enable"):
        try:
            decoder_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            decoder_backbone.gradient_checkpointing_enable()

    # Install LoRA adapters (both levels)
    router = LoRARouter.install(
        encoder,
        target_names=DEFAULT_LORA_TARGETS,
        levels={"l0": lora_rank, "l1": lora_rank},
        alpha=2.0 * lora_rank,
        dropout=0.0,
    )
    LoRARouter.bind(router)
    encoder.requires_grad_(False)
    router.set_level_trainable("l0", True)
    router.set_level_trainable("l1", True)

    # Optimizer over decoder + LoRA params
    trainable_params: list[nn.Parameter] = []
    trainable_params += [p for p in decoder.parameters() if p.requires_grad]
    trainable_params += router.adapter_parameters("l0")
    trainable_params += router.adapter_parameters("l1")
    optim = torch.optim.AdamW(trainable_params, lr=1e-4)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    weights_alloc = torch.cuda.memory_allocated(device)
    print(f"[live] After model + optimizer construction: "
          f"{_gb(weights_alloc):.2f} GB allocated")

    # Synthetic L0 forward — packed FA4 varlen form.
    # Build a flat (N_content, D) buffer with per-article cu_seqlens.
    n_articles = articles_per_bgkit
    article_len = doc_length
    token_embeds = encoder.compressor.backbone.get_input_embeddings()
    vocab_size = token_embeds.num_embeddings

    # All articles have the same length in this synthetic test; construct
    # flat content buffer and per-article cu_seqlens.
    input_ids_flat = torch.randint(0, vocab_size, (n_articles * article_len,), device=device)
    content_flat = token_embeds(input_ids_flat)  # (N_content, D)
    lengths_l0 = [article_len] * n_articles
    l0_cu = torch.zeros(n_articles + 1, dtype=torch.int32, device=device)
    l0_cu[1:] = torch.tensor(lengths_l0, dtype=torch.int32, device=device).cumsum(0)
    l0_pos = position_ids_from_cu(l0_cu, int(content_flat.shape[0]))

    # L0 retention ratio — survivorship head will select this fraction.
    l0_retention = 0.20

    print(f"[live] L0 forward: B={n_articles}, L={article_len}, "
          f"retention={l0_retention} ...")
    with router.active("l0"):
        if activation_checkpointing:
            from torch.utils.checkpoint import checkpoint

            def _run_l0(c, cu, pos, _enc=encoder):
                return _enc(
                    content_embeddings=c,
                    content_cu_seqlens=cu,
                    content_position_ids=pos,
                    target_ratio=l0_retention,
                    level="l0",
                )

            l0_out = checkpoint(
                _run_l0, content_flat, l0_cu, l0_pos,
                use_reentrant=False,
            )
        else:
            l0_out = encoder(
                content_embeddings=content_flat,
                content_cu_seqlens=l0_cu,
                content_position_ids=l0_pos,
                target_ratio=l0_retention,
                level="l0",
            )

    # L0 survivors — flat (N_survivors, D) with per-article boundaries in
    # survivor_cu_seqlens.  Treat all articles' survivors as a single L1
    # content sequence (one "bgkit turn").
    l0_surv_flat = l0_out.survivor_embeddings  # (N_survivors, D)
    k_total = int(l0_surv_flat.shape[0])

    # L1 content: all L0 survivors packed as one sample.
    l1_cu = torch.tensor([0, k_total], dtype=torch.int32, device=device)
    l1_pos = position_ids_from_cu(l1_cu, k_total)

    # Synthetic per-token pinned flags: pin the first 8 positions (article IDs).
    l1_pinned = torch.zeros(k_total, dtype=torch.bool, device=device)
    l1_pinned[:min(8, k_total)] = True

    # Synthetic query (32 tokens).
    query_len = 32
    query_ids_flat = torch.randint(0, vocab_size, (query_len,), device=device)
    query_emb = token_embeds(query_ids_flat)  # (N_query, D)
    query_cu = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    query_pos = position_ids_from_cu(query_cu, query_len)

    l1_retention = 0.50
    print(f"[live] L1 forward: K_total={k_total}, query_len={query_len} ...")
    with router.active("l1"):
        if activation_checkpointing:
            from torch.utils.checkpoint import checkpoint

            def _run_l1(c, c_cu, c_pos, qe, q_cu, q_pos, pp, _enc=encoder):
                return _enc(
                    content_embeddings=c,
                    content_cu_seqlens=c_cu,
                    content_position_ids=c_pos,
                    prompt_embeddings=qe,
                    prompt_cu_seqlens=q_cu,
                    prompt_position_ids=q_pos,
                    pinned_positions=pp,
                    target_ratio=l1_retention,
                    level="l1",
                )

            l1_out = checkpoint(
                _run_l1, l0_surv_flat, l1_cu, l1_pos,
                query_emb, query_cu, query_pos, l1_pinned,
                use_reentrant=False,
            )
        else:
            l1_out = encoder(
                content_embeddings=l0_surv_flat,
                content_cu_seqlens=l1_cu,
                content_position_ids=l1_pos,
                prompt_embeddings=query_emb,
                prompt_cu_seqlens=query_cu,
                prompt_position_ids=query_pos,
                pinned_positions=l1_pinned,
                target_ratio=l1_retention,
                level="l1",
            )

    l1_survivors = l1_out.survivor_embeddings  # (K1, D) flat

    # Decoder forward: synthetic token context of decoder_context length with
    # l1_survivors spliced in the middle via the interleaved segment API.
    dec_embed = decoder_backbone.get_input_embeddings()
    vocab_dec = dec_embed.weight.shape[0]
    mid = decoder_context // 2

    pre_ids = torch.randint(0, vocab_dec, (1, mid), device=device)
    post_ids = torch.randint(0, vocab_dec, (1, decoder_context - mid - 1), device=device)

    # Use the interleaved-segment decoder API (same as KRKBTrainer).
    segments = [
        TokenSegment(
            token_ids=pre_ids,
            loss_mask=torch.ones(1, mid, dtype=torch.bool, device=device),
        ),
        EmbeddingSegment(
            embeddings=l1_survivors.unsqueeze(0),  # (1, K1, D)
        ),
        TokenSegment(
            token_ids=post_ids,
            loss_mask=torch.ones(1, decoder_context - mid - 1, dtype=torch.bool, device=device),
        ),
    ]
    total_len = mid + int(l1_survivors.shape[0]) + (decoder_context - mid - 1)
    print(f"[live] Decoder forward: L={total_len} ...")
    loss = decoder.forward_interleaved_with_loss(segments)
    print(f"[live] Decoder loss: {loss.item():.4f}")

    print("[live] Backward ...")
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(f"\n[live] Peak memory allocated: {_gb(peak_alloc):.2f} GB")
    print(f"[live] Peak memory reserved:  {_gb(peak_reserved):.2f} GB")
    print(f"[live] Budget:                {BUDGET_GB:.0f} GB unified")
    headroom = BUDGET_GB - _gb(peak_reserved)
    status = "OK" if headroom > 0 else "OVER BUDGET"
    print(f"[live] Headroom (vs reserved): {headroom:+.2f} GB  [{status}]")

    # Cleanup
    del encoder, decoder, decoder_backbone, optim
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=100,
                        help="Articles per bgkit call (fan-out cap; default 100)")
    parser.add_argument("--doc-length", type=int, default=500,
                        help=(
                            "Average tokens per article for L0 input. Stage A "
                            "averages PubMedQA/NarrativeQA/git/memory ≈ 500 "
                            "tokens. For Stage C Wikipedia scale, pass 2048."
                        ))
    parser.add_argument("--l0-retention", type=float, default=0.20,
                        help=(
                            "L0 retention ratio (Stage A datasets default 0.20; "
                            "memory dataset uses 0.40; Wikipedia uses 0.05)"
                        ))
    parser.add_argument("--decoder-context", type=int, default=6000,
                        help="Approximate decoder token context per step")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Samples per training step (defaults to 1)")
    parser.add_argument("--bgkit-turns", type=int, default=4,
                        help="Avg bgkit tool calls per trajectory (default 4)")
    parser.add_argument("--ac-off", action="store_true",
                        help="Disable activation checkpointing in estimates/live")
    parser.add_argument("--live", action="store_true",
                        help="Also run a live forward/backward on GPU")
    args = parser.parse_args()

    activation_checkpointing = not args.ac_off

    print("Phase 2 KB Stage A — memory verification")
    print(f"  articles_per_bgkit = {args.articles}")
    print(f"  doc_length         = {args.doc_length}")
    print(f"  l0_retention       = {args.l0_retention}")
    print(f"  decoder_context    = {args.decoder_context}")
    print(f"  lora_rank          = {args.lora_rank}")
    print(f"  batch_size         = {args.batch_size}")
    print(f"  bgkit_turns        = {args.bgkit_turns}")
    print(f"  activation_ckpt    = {activation_checkpointing}")

    reports = static_estimate(
        articles_per_bgkit=args.articles,
        doc_length=args.doc_length,
        l0_retention=args.l0_retention,
        decoder_context=args.decoder_context,
        lora_rank=args.lora_rank,
        activation_checkpointing=activation_checkpointing,
        batch_size=args.batch_size,
        bgkit_turns_per_sample=args.bgkit_turns,
    )
    total = print_table("Static estimate", reports)

    if total > BUDGET_GB:
        print("\n[WARN] Static estimate exceeds budget. Consider:")
        print("  - Lower --articles or --doc-length")
        print("  - Enable activation checkpointing (already on by default)")
        print("  - Switch optimizer to bitsandbytes 8bit (adamw8bit)")
        print("  - Reduce decoder context via trajectory pruning")

    if args.live:
        try:
            live_verify(
                articles_per_bgkit=args.articles,
                doc_length=args.doc_length,
                decoder_context=args.decoder_context,
                lora_rank=args.lora_rank,
                activation_checkpointing=activation_checkpointing,
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n[live] Failed: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
