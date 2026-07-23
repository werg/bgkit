#!/usr/bin/env python
"""Decisive leak-vs-noise probes for ``forward_interleaved_packed``.

The ref-vs-batched comparison in scripts/test_decode_batching_parity.py can
not distinguish a residual cross-sample LEAK from bf16 kernel-dispatch noise
(reference = single-segment kernels, batched = multi-segment varlen kernels;
their accumulation orders differ, so grads disagree at the percent level even
with perfect isolation). The two probes here are exact and noise-immune:

1. CROSS-GRAD: packed forward over [A, B]; backward ONLY loss_B. If samples
   are isolated, ``surv_A.grad`` is structurally zero (masked/skipped tiles),
   and ``surv_B.grad`` from loss_A is zero by causality. Any nonzero = leak,
   with magnitude = leak size. No comparison against another run — zero noise.

2. CONTENT-SWAP: packed forward over [A, B] vs [A', B] where A' has the same
   SHAPES but re-randomized content. Kernel dispatch is identical, so
   loss_B must be BIT-IDENTICAL across the two runs iff B is isolated from
   A's content (deterministic forward).

Both probes run in the production-mirrored configuration (deltanet class
patch for qwen35, falcon_h1 runtime patch via ReconstructionDecoder init,
scale-matched survivors, falcon multiplier folding — same as the parity
harness) for eval AND train mode.

Run in the container (trainer stopped):
    python scripts/diag_decode_leak.py --only qwen35
    python scripts/diag_decode_leak.py --only falcon_h1
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from bgkit.models.decoder import (
    EmbeddingSegment,
    ReconstructionDecoder,
    TokenSegment,
    normalize_decoder_family,
)
from bgkit.utils.attention_backend import (
    resolve_decoder_attention_implementation,
)
from bgkit.utils.deltanet_patch import (
    patch_fused_rms_norm_gated_for_sm121,
    patch_gated_delta_rule_numerics,
)

# (prefix_no_loss, n_survivors, gold_loss_tokens) — mid-chunk boundaries.
FILE_SHAPES = [(48, 8, 220), (130, 16, 40), (30, 4, 480), (75, 12, 96)]


def build_decoder(name: str, family: str, device: torch.device) -> ReconstructionDecoder:
    fam = normalize_decoder_family(family)
    if fam == "qwen35":
        patch_gated_delta_rule_numerics()
        patch_fused_rms_norm_gated_for_sm121()
    attn = resolve_decoder_attention_implementation("auto", decoder_family=fam)
    backbone = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    ).to(device)
    if fam == "qwen35":
        patch_gated_delta_rule_numerics(model=backbone)
    if fam == "falcon_h1":
        cfg = backbone.config
        emb = backbone.get_input_embeddings()
        head = backbone.lm_head
        if head.weight.data_ptr() == emb.weight.data_ptr():
            head.weight = torch.nn.Parameter(emb.weight.detach().clone())
        with torch.no_grad():
            emb.weight.mul_(cfg.embedding_multiplier)
            head.weight.mul_(cfg.lm_head_multiplier)
    hidden = backbone.get_input_embeddings().weight.shape[1]
    return ReconstructionDecoder(backbone, hidden_dim=hidden, decoder_family=fam)


def make_files(vocab, hidden, device, gen, surv_scale, shapes=FILE_SHAPES):
    files, survs = [], []
    for pfx, ns, gold in shapes:
        prefix = TokenSegment(
            token_ids=torch.randint(0, vocab, (pfx,), generator=gen, device=device),
            loss_mask=torch.zeros(pfx, dtype=torch.bool, device=device),
        )
        surv = (
            torch.randn(ns, hidden, generator=gen, device=device, dtype=torch.bfloat16)
            * surv_scale
        ).detach().requires_grad_(True)
        gold_seg = TokenSegment(
            token_ids=torch.randint(0, vocab, (gold,), generator=gen, device=device),
            loss_mask=torch.ones(gold, dtype=torch.bool, device=device),
        )
        files.append([prefix, EmbeddingSegment(embeddings=surv), gold_seg])
        survs.append(surv)
    return files, survs


def cross_grad_probe(dec, files, survs, tag: str) -> None:
    """Backward each file's loss ALONE; report cross-sample survivor grads."""
    losses = dec.forward_interleaved_packed(files)
    for i in range(len(files)):
        for s in survs:
            s.grad = None
        losses_i = dec.forward_interleaved_packed(files)
        losses_i[i].backward()
        row = []
        for j, s in enumerate(survs):
            g = s.grad
            gmax = 0.0 if g is None else float(g.detach().abs().max())
            row.append(f"g[surv_{j}]={gmax:.3e}")
        cross = [
            float(survs[j].grad.detach().abs().max())
            for j in range(len(survs))
            if j != i and survs[j].grad is not None
        ]
        cross_max = max(cross) if cross else 0.0
        verdict = "ISOLATED" if cross_max == 0.0 else f"LEAK({cross_max:.3e})"
        print(f"    [{tag}] dloss_{i}/dsurv_*: {' '.join(row)} -> {verdict}")
    del losses


@torch.no_grad()
def content_swap_probe(dec, files, vocab, hidden, device, surv_scale, tag: str) -> None:
    """Re-randomize earlier files' content; later files' losses must not move."""
    base = [x.detach().float().item() for x in dec.forward_interleaved_packed(files)]
    # Determinism check: same inputs twice.
    rep = [x.detach().float().item() for x in dec.forward_interleaved_packed(files)]
    rep_d = max(abs(a - b) for a, b in zip(base, rep, strict=True))
    print(f"    [{tag}] repeat-determinism max|dl|={rep_d:.3e}")
    gen2 = torch.Generator(device=device).manual_seed(999)
    swapped, _ = make_files(vocab, hidden, device, gen2, surv_scale)
    for k in range(len(files) - 1):
        # Replace file k with new content (same shapes); files k+1.. unchanged.
        mixed = list(files)
        mixed[k] = swapped[k]
        out = [x.detach().float().item() for x in dec.forward_interleaved_packed(mixed)]
        diffs = [abs(out[j] - base[j]) for j in range(len(files))]
        later = max(diffs[k + 1:]) if k + 1 < len(files) else 0.0
        verdict = "ISOLATED" if later == 0.0 else f"LEAK({later:.3e})"
        print(
            f"    [{tag}] swap file {k}: later-file max|dl|={later:.3e} "
            f"(own |dl|={diffs[k]:.3e}) -> {verdict}"
        )


def run_family(name: str, family: str, device: torch.device) -> None:
    fam = normalize_decoder_family(family)
    print(f"\n=== {fam} ({name}) ===")
    dec = build_decoder(name, fam, device)
    vocab = dec._get_inner_model_and_head()[1].weight.shape[0]
    hidden = dec.hidden_dim
    emb = dec._get_inner_model_and_head()[0].get_input_embeddings()
    surv_scale = float(emb.weight.detach().float().std())
    gen = torch.Generator(device=device).manual_seed(1234)
    files, survs = make_files(vocab, hidden, device, gen, surv_scale)
    for mode in ("eval", "train"):
        (dec.eval if mode == "eval" else dec.train)()
        cross_grad_probe(dec, files, survs, f"{fam}/{mode}")
        content_swap_probe(dec, files, vocab, hidden, device, surv_scale, f"{fam}/{mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--falcon", default="tiiuae/Falcon-H1-Tiny-90M-Instruct")
    ap.add_argument("--only", choices=["qwen35", "falcon_h1"], default=None)
    args = ap.parse_args()
    import transformers

    print(f"transformers={transformers.__version__} torch={torch.__version__}")
    device = torch.device("cuda")
    if args.only in (None, "qwen35"):
        run_family(args.qwen, "qwen35", device)
    if args.only in (None, "falcon_h1"):
        run_family(args.falcon, "falcon_h1", device)


if __name__ == "__main__":
    main()
