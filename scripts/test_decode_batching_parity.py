#!/usr/bin/env python
"""GPU isolation + parity gate for ``ReconstructionDecoder.forward_interleaved_packed``.

The decode-batching change replaces the per-file decode loop in
``KRKBTrainer._encode_decode_group`` with ONE packed FA4-varlen forward, then
computes CE per file on each file's isolated hidden slice. The claim: the packed
path isolates attention AND resets every stateful mixer (Qwen3.5 DeltaNet
delta-rule + short-conv; Falcon-H1 Mamba conv + chunked scan) at every file
boundary, so each file's loss/gradients equal an independent per-file forward.

Verification methodology (2026-07 revision)
-------------------------------------------
A packed multi-segment forward and a per-file forward run DIFFERENT kernel
dispatches (varlen multi-segment vs single-segment / dense). Their bf16
accumulation orders differ, so hidden states — and especially gradients —
disagree at the few-percent level per element after 24-48 layers EVEN WITH
MATHEMATICALLY PERFECT isolation. An element-wise ``allclose`` on grads can
therefore never distinguish "leak" from "kernel noise". This gate instead
checks isolation with two EXACT probes that have zero kernel-noise
sensitivity, plus noise-calibrated reference-agreement checks:

1. CROSS-GRAD (exact, primary): in the packed forward, backward file i's loss
   ALONE — every other file's survivor grad must be EXACTLY zero. Any leak
   path (attention, conv window, recurrent state) would produce nonzero
   cross-grads. Tolerance: 0.0, bitwise.
2. CONTENT-SWAP (exact, primary): re-randomize file k's content (same
   shapes → identical kernel dispatch); files k+1.. must produce
   BIT-IDENTICAL losses up to the measured run-to-run determinism floor
   (~1e-6; hard cap 1e-5). Any forward value-dependence on a predecessor's
   content — however it leaks — moves later losses (the pre-fix Falcon
   mid-chunk Mamba state-passing leak measured 5e-3..2.4e-2 here).
3. LOSS AGREEMENT (noise-calibrated): per-file batched loss vs per-file loop
   loss, |dl| <= LOSS_ATOL + LOSS_RTOL * ref.
4. GRAD AGREEMENT (noise-calibrated): d(sum losses)/d(survivors): relative
   L2 error, cosine similarity, and max element deviation vs the ref grad's
   absmax scale.

Runs on the REAL decoders in the production configuration (deltanet
class-patch for qwen35, Falcon-H1 runtime patch, FA4/FA2 varlen kernels) in
EVAL and TRAIN mode, for each requested decoder family.

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
from bgkit.utils.deltanet_patch import (
    patch_fused_rms_norm_gated_for_sm121,
    patch_gated_delta_rule_numerics,
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

# --- Exact isolation probes (primary; zero kernel-noise sensitivity) --------
# Cross-grads are structurally zero in an isolated graph — bitwise check.
CROSS_GRAD_MAX = 0.0
# Content-swap: later files must be bit-identical up to the run-to-run
# determinism floor (measured in-run, ~1e-6 from one nondeterministic op);
# hard cap well below any leak (the pre-fix Falcon mid-chunk Mamba
# state-passing leak measured 5e-3..2.4e-2 on this probe).
CONTENT_SWAP_MAX = 1e-5

# --- Reference-agreement checks (noise-calibrated; secondary) ---------------
# Ref (per-file, single-segment/dense kernels) vs batched (multi-segment
# varlen kernels) differ by bf16 accumulation-order noise only — isolation
# itself is certified by the exact probes above. Calibrated 2026-07-23
# against measured post-fix noise on both families (worst observed: loss
# rel 1.6e-3; grad relL2 ~ few %, max element ~11% of the grad's absmax):
LOSS_ATOL = 2e-3
LOSS_RTOL = 3e-3
GRAD_REL_L2_MAX = 0.10   # ||g_bat - g_ref||2 / ||g_ref||2
GRAD_COS_MIN = 0.99      # cosine(g_bat, g_ref)
GRAD_ATOL = 5e-3         # max element |d| <= GRAD_ATOL + GRAD_SCALE_RTOL*absmax(ref)
GRAD_SCALE_RTOL = 0.15


def build_decoder(name: str, family: str, device: torch.device) -> ReconstructionDecoder:
    fam = normalize_decoder_family(family)
    if fam == "qwen35":
        # Mirror production (scripts/train.py): the DeltaNet numerics/packed
        # patch is class-patched BEFORE model construction so every layer
        # instance carries the gate clamp + packed-context cu/seq_idx
        # injection — the config this gate must certify. Idempotent.
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
        # Fold the config's embedding_multiplier / lm_head_multiplier into
        # the weights. The interleaved decode path feeds raw embed_tokens
        # rows and raw lm_head — production behavior, to which the TRAINED
        # falcon decoder adapts. With PRETRAINED weights that raw path runs
        # ~9x/~13x off-scale (CE ~300), a chaotic regime where deterministic
        # bf16 varlen-vs-dense kernel noise alone reaches ~1e-1 on the loss
        # and drowns the leak signal these tolerances are calibrated for.
        # Folding is weight-space-equivalent to the stock model (exactly the
        # operating point a trained checkpoint restores) and touches neither
        # the production code path nor the tolerances. lm_head may be tied
        # to embed_tokens — untie before scaling the two differently.
        cfg = backbone.config
        emb = backbone.get_input_embeddings()
        head = backbone.lm_head
        if head.weight.data_ptr() == emb.weight.data_ptr():
            head.weight = torch.nn.Parameter(emb.weight.detach().clone())
        with torch.no_grad():
            emb.weight.mul_(cfg.embedding_multiplier)
            head.weight.mul_(cfg.lm_head_multiplier)
    hidden = backbone.get_input_embeddings().weight.shape[1]
    dec = ReconstructionDecoder(backbone, hidden_dim=hidden, decoder_family=fam)
    return dec


def make_files(vocab, hidden, device, gen, shapes=FILE_SHAPES, surv_scale=1.0):
    """One list of segments per file: [prefix tokens (no loss) | survivor
    embeddings (grad leaf) | gold tokens (loss)] — mirrors a drill+decode.

    ``surv_scale`` scales the random survivor embeddings to the decoder's
    input-embedding RMS so the model runs at a realistic operating point.
    Unscaled N(0,1) survivors are ~50x out-of-distribution; on Falcon-H1
    (whose decode path feeds raw ``embed_tokens`` rows and raw ``lm_head``,
    without the config's embedding/lm_head multipliers) that drove CE to
    ~300 where deterministic bf16 varlen-vs-dense kernel noise alone is
    ~1e-1 absolute on the loss — drowning the leak signal these absolute
    tolerances are calibrated for. Scale-matching restores a sane operating
    point; it does NOT relax any tolerance."""
    files, survs = [], []
    for pfx, ns, gold in shapes:
        prefix = TokenSegment(
            token_ids=torch.randint(0, vocab, (pfx,), generator=gen, device=device),
            loss_mask=torch.zeros(pfx, dtype=torch.bool, device=device),
        )
        surv = (
            torch.randn(
                ns, hidden, generator=gen, device=device, dtype=torch.bfloat16,
            )
            * surv_scale
        ).detach().requires_grad_(True)
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


def _check_cross_grads(dec, files, survs, mode: str) -> bool:
    """Exact probe 1: d loss_i / d surv_{j!=i} must be bitwise zero."""
    ok = True
    for i in range(len(files)):
        for s in survs:
            s.grad = None
        losses = dec.forward_interleaved_packed(files)
        losses[i].backward()
        cross = [
            float(survs[j].grad.detach().abs().max())
            for j in range(len(survs))
            if j != i and survs[j].grad is not None
        ]
        cmax = max(cross) if cross else 0.0
        this_ok = cmax <= CROSS_GRAD_MAX
        ok = ok and this_ok
        print(
            f"  [{mode}] cross-grad loss_{i}: max|d/dsurv_other|={cmax:.3e} "
            f"{'ok' if this_ok else 'FAIL'}"
        )
    for s in survs:
        s.grad = None
    return ok


@torch.no_grad()
def _check_content_swap(dec, files, vocab, hidden, device, shapes, surv_scale,
                        mode: str) -> bool:
    """Exact probe 2: predecessors' content must not move later files' losses."""
    base = [x.detach().float().item() for x in dec.forward_interleaved_packed(files)]
    rep = [x.detach().float().item() for x in dec.forward_interleaved_packed(files)]
    det_floor = max(abs(a - b) for a, b in zip(base, rep, strict=True))
    swap_gen = torch.Generator(device=device).manual_seed(999)
    swapped, _ = make_files(vocab, hidden, device, swap_gen, shapes, surv_scale)
    ok = True
    for k in range(len(files) - 1):
        mixed = list(files)
        mixed[k] = swapped[k]
        out = [x.detach().float().item() for x in dec.forward_interleaved_packed(mixed)]
        later = max(abs(out[j] - base[j]) for j in range(k + 1, len(files)))
        this_ok = later <= max(CONTENT_SWAP_MAX, det_floor)
        ok = ok and this_ok
        print(
            f"  [{mode}] content-swap file {k}: later-file max|dl|={later:.3e} "
            f"(determinism floor {det_floor:.1e}) {'ok' if this_ok else 'FAIL'}"
        )
    return ok


def run_family(name: str, family: str, device: torch.device, shapes=FILE_SHAPES) -> bool:
    tot = [sum(s) for s in shapes]
    print(f"\n=== {family} ({name}) — file totals {tot}, cu boundaries "
          f"{[sum(tot[:i+1]) for i in range(len(tot))]} ===")
    dec = build_decoder(name, family, device)
    vocab = dec._get_inner_model_and_head()[1].weight.shape[0]
    hidden = dec.hidden_dim
    emb = dec._get_inner_model_and_head()[0].get_input_embeddings()
    surv_scale = float(emb.weight.detach().float().std())
    print(f"  survivor scale (embed-table std): {surv_scale:.5f}")
    gen = torch.Generator(device=device).manual_seed(1234)
    files, survs = make_files(vocab, hidden, device, gen, shapes, surv_scale)

    ok = True
    for mode in ("eval", "train"):
        (dec.eval if mode == "eval" else dec.train)()

        # --- Exact isolation probes (primary) ---------------------------
        ok = _check_cross_grads(dec, files, survs, mode) and ok
        ok = _check_content_swap(
            dec, files, vocab, hidden, device, shapes, surv_scale, mode,
        ) and ok

        # --- Reference agreement (noise-calibrated, secondary) ----------
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
            gref_absmax = float(ref_g[i].abs().max())
            rel_l2 = float(
                (ref_g[i] - bat_g[i]).norm() / ref_g[i].norm().clamp_min(1e-12)
            )
            cos = float(
                torch.nn.functional.cosine_similarity(
                    ref_g[i].reshape(1, -1), bat_g[i].reshape(1, -1),
                )
            )
            lok = ld <= LOSS_ATOL + LOSS_RTOL * abs(ref_l[i])
            gok = (
                rel_l2 <= GRAD_REL_L2_MAX
                and cos >= GRAD_COS_MIN
                and gmax <= GRAD_ATOL + GRAD_SCALE_RTOL * gref_absmax
            )
            ok = ok and gok and lok
            print(
                f"  [{mode}] file {i}: loss ref={ref_l[i]:.6f} bat={bat_l[i]:.6f} "
                f"|dl|={ld:.2e} {'ok' if lok else 'FAIL'} | grad relL2={rel_l2:.2e} "
                f"cos={cos:.5f} max|d|={gmax:.2e} absmax={gref_absmax:.2e} "
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
