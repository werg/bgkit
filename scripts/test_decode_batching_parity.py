#!/usr/bin/env python
"""GPU bit-parity gate for ``ReconstructionDecoder.forward_interleaved_packed``.

The decode-batching change (commit 041451e) replaces the per-file decode loop in
``KRKBTrainer._encode_decode_group`` with ONE packed FA4-varlen forward, then
computes CE per file on each file's isolated hidden slice. The claim is that this
is BIT-EXACT vs calling ``forward_interleaved_with_loss`` per file — the packed
``cu_seqlens`` path must isolate attention AND reset the DeltaNet/Mamba recurrent
state at every file boundary, so ``hidden[file_slice]`` equals an independent
per-file forward.

This gate verifies that on the REAL decoder (needs the FA4 varlen kernels, so it
runs in the container on GPU). It checks BOTH:
  * loss parity — per-file batched loss == per-file loop loss, and
  * gradient parity — d(sum losses)/d(survivor embeddings) matches,
in EVAL and TRAIN mode, for each requested decoder family.

Do NOT enable ``per_repo_sample_group_size > 1`` in a run until this prints
``ALL PARITY CHECKS PASS``.

Usage (in the container, trainer stopped):
    python scripts/test_decode_batching_parity.py \
        --qwen Qwen/Qwen3.5-0.8B --falcon tiiuae/Falcon-H1-Tiny-90M-Instruct
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

# Deterministic file shapes (prefix_no_loss_tokens, survivor_embeds, gold_loss_tokens).
# MID-CHUNK: totals 276/186/514/183 — segment boundaries fall mid-64-chunk, the
# realistic case. ALIGNED: totals 256/192/512/192 — every boundary is a multiple
# of 64, so it lands on an FLA DeltaNet chunk boundary. Comparing the two isolates
# whether a residual Qwen divergence is an FLA varlen mid-chunk-boundary artifact.
FILE_SHAPES_MIDCHUNK = [
    (48, 8, 220),
    (130, 16, 40),
    (30, 4, 480),
    (75, 12, 96),
]
FILE_SHAPES_ALIGNED = [
    (48, 8, 200),   # 256
    (120, 8, 64),   # 192
    (32, 16, 464),  # 512
    (60, 4, 128),   # 192
]
FILE_SHAPES = FILE_SHAPES_MIDCHUNK
LOSS_ATOL = 2e-3   # bf16 CE reduction tolerance
GRAD_RTOL = 3e-2
GRAD_ATOL = 2e-3


def build_decoder(name: str, family: str, device: torch.device) -> ReconstructionDecoder:
    fam = normalize_decoder_family(family)
    attn = resolve_decoder_attention_implementation("auto", decoder_family=fam)
    backbone = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    ).to(device)
    hidden = backbone.get_input_embeddings().weight.shape[1]
    dec = ReconstructionDecoder(backbone, hidden_dim=hidden, decoder_family=fam)
    return dec


def make_files(vocab, hidden, device, gen, shapes=FILE_SHAPES):
    """One list of segments per file: [prefix tokens (no loss) | survivor
    embeddings (grad leaf) | gold tokens (loss)] — mirrors a drill+decode."""
    files, survs = [], []
    for pfx, ns, gold in shapes:
        prefix = TokenSegment(
            token_ids=torch.randint(0, vocab, (pfx,), generator=gen, device=device),
            loss_mask=torch.zeros(pfx, dtype=torch.bool, device=device),
        )
        surv = torch.randn(
            ns, hidden, generator=gen, device=device, dtype=torch.bfloat16,
        ).requires_grad_(True)
        gold_seg = TokenSegment(
            token_ids=torch.randint(0, vocab, (gold,), generator=gen, device=device),
            loss_mask=torch.ones(gold, dtype=torch.bool, device=device),
        )
        files.append([prefix, EmbeddingSegment(embeddings=surv), gold_seg])
        survs.append(surv)
    return files, survs


def _clone_grad_leaf(files, survs):
    """Fresh grad leaves so ref/batched backward don't share/accumulate grads."""
    new_files, new_survs = [], []
    for segs, s in zip(files, survs, strict=True):
        s2 = s.detach().clone().requires_grad_(True)
        segs2 = [
            EmbeddingSegment(embeddings=s2) if isinstance(seg, EmbeddingSegment) else seg
            for seg in segs
        ]
        new_files.append(segs2)
        new_survs.append(s2)
    return new_files, new_survs


def run_family(name: str, family: str, device: torch.device, shapes=FILE_SHAPES) -> bool:
    tot = [sum(s) for s in shapes]
    print(f"\n=== {family} ({name}) — file totals {tot}, cu boundaries "
          f"{[sum(tot[:i+1]) for i in range(len(tot))]} ===")
    dec = build_decoder(name, family, device)
    vocab = dec._get_inner_model_and_head()[1].weight.shape[0]
    hidden = dec.hidden_dim
    gen = torch.Generator(device=device).manual_seed(1234)
    files, survs = make_files(vocab, hidden, device, gen, shapes)

    ok = True
    for mode in ("eval", "train"):
        (dec.eval if mode == "eval" else dec.train)()

        # Reference: per-file forward + per-file backward on fresh leaves.
        rf, rsurv = _clone_grad_leaf(files, survs)
        ref_losses = [dec.forward_interleaved_with_loss(f) for f in rf]
        torch.stack(ref_losses).sum().backward()
        ref_l = [x.detach().float().item() for x in ref_losses]
        ref_g = [s.grad.detach().float().clone() for s in rsurv]

        # Batched: one packed forward, per-file losses, one backward.
        bf, bsurv = _clone_grad_leaf(files, survs)
        bat_losses = dec.forward_interleaved_packed(bf)
        torch.stack([o for o in bat_losses]).sum().backward()
        bat_l = [x.detach().float().item() for x in bat_losses]
        bat_g = [s.grad.detach().float().clone() for s in bsurv]

        for i in range(len(files)):
            ld = abs(ref_l[i] - bat_l[i])
            gd = (ref_g[i] - bat_g[i]).abs()
            gmax = float(gd.max())
            gok = torch.allclose(bat_g[i], ref_g[i], rtol=GRAD_RTOL, atol=GRAD_ATOL)
            lok = ld < LOSS_ATOL
            ok = ok and gok and lok
            print(
                f"  [{mode}] file {i}: loss ref={ref_l[i]:.6f} bat={bat_l[i]:.6f} "
                f"|dl|={ld:.2e} {'ok' if lok else 'FAIL'} | grad max|d|={gmax:.2e} "
                f"{'ok' if gok else 'FAIL'}"
            )
    print(f"  {family}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--falcon", default="tiiuae/Falcon-H1-Tiny-90M-Instruct")
    ap.add_argument("--only", choices=["qwen35", "falcon_h1"], default=None)
    ap.add_argument("--shapes", choices=["midchunk", "aligned", "both"], default="both")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA — the FA4 varlen path is not exercised on CPU.")

    shape_sets = []
    if args.shapes in ("midchunk", "both"):
        shape_sets.append(("midchunk", FILE_SHAPES_MIDCHUNK))
    if args.shapes in ("aligned", "both"):
        shape_sets.append(("aligned", FILE_SHAPES_ALIGNED))

    results = {}
    for tag, shapes in shape_sets:
        if args.only in (None, "qwen35"):
            results[f"qwen35/{tag}"] = run_family(args.qwen, "qwen35", device, shapes)
        if args.only in (None, "falcon_h1"):
            results[f"falcon_h1/{tag}"] = run_family(
                args.falcon, "falcon_h1", device, shapes)

    print("\n" + "=" * 40)
    all_ok = all(results.values())
    for fam, r in results.items():
        print(f"  {fam}: {'PASS' if r else 'FAIL'}")
    print("ALL PARITY CHECKS PASS" if all_ok else "PARITY FAILED — do NOT enable batching")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
