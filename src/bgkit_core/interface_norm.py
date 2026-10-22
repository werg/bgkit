"""THE encoder→decoder interface contract, shared by bgkit (Qwen3.5 lineage)
and bgkit2 (LFM2.5 reset). One implementation: two copies of a CONTRACT class
is the failure feedback_representation_interface_contracts names.

--- Original bgkit narrative (2026-08-31 → 2026-09-05), kept verbatim ---
The contract between the encoder's output and the decoder's input.

WHY THIS EXISTS. Measured 2026-08-31 on the same 128 eval documents, asking
whether a document can be retrieved from its own survivors:

                          raw cosine   corpus-whitened
    Phase-1 base reps        0.898          0.953
    widenet v8 reps          0.031          0.953

The information was in v8's reps the whole time. What Phase-2 training did
was bury it: 99.0% of every survivor's energy became ONE vector, identical
across the entire corpus (cosine 1.00000 to the corpus mean), leaving the
document-specific part at ~1% of the norm. The decoder's first operation on
a spliced embedding is an RMSNorm, which divides by a magnitude that shared
vector owns, so the content entered attention at ~1% amplitude.

Nothing in the stack could have removed it. Qwen3.5 uses RMSNorm end to end
-- scale only, never a mean -- and the projection block's own output rescale
targeted the ENCODER-space norm of its input, a quantity with no relationship
to the decoder's embedding distribution.

WHAT THIS DOES. Standardise per channel against slowly-moving statistics, then
rescale to the decoder's embedding norm. Two consequences, and the second is
the point:

1. The corpus-constant component is subtracted, so what reaches the decoder is
   the part that differs between documents.
2. A constant output stops being reachable. Inflating a shared direction no
   longer changes what the decoder sees, so no gradient can be served by
   doing it -- the collapse is not merely corrected downstream, it stops
   paying.

Statistics are an EMA, NOT the current batch. Gradients therefore flow only
through the numerator, there is no train/eval discrepancy to reconcile, and a
batch whose composition happens to be skewed cannot move what the decoder
reads. See ``feedback_representation_interface_contracts``: this class of
defect -- shape-correct, distribution-wrong, silent -- has recurred often
enough to be worth a named boundary rather than a per-site fix.

THE AFFINE IS OFF BY DEFAULT, AND THAT IS MEASURED, NOT STYLISTIC. The first
arm ran with a learnable gain and shift, on the reasoning that the lineage's
reps operate at ~27x the embedding norm so 1x is not obviously the right
target. Watching it for 65 steps settled the question: the emitted
norm ratio climbed 1.30 -> 6.52 monotonically while shared_frac went 0.26 ->
0.94. A trainable shift IS a corpus-constant, re-added after the very
standardisation that removed one, and a trainable gain restores the free
scale direction the contract exists to close. Both muting routes reopen
through the affine. The decoder's own input projection can absorb any fixed
per-channel scaling, so nothing expressive is lost by pinning it.

CORRECTION (same day, from the relaunch's own data): pinning the affine did
NOT change the trajectory. With the affine off and momentum 0.1 the emitted
norm ratio followed the same path (1.55 1.19 1.13 ... 2.41 4.11 1.77 2.21
3.72) and shared_frac rose to ~0.93 just as it had with the affine on. So
neither hypothesis below was the cause. The pinning stands as hygiene -- it
does close a route in principle -- but it did not fix what was observed, and
the run killed on that diagnosis was killed on a reading the data refutes.

What this class of normalisation CAN do is remove a corpus-constant, and it
does: mean_cos_to_corpus stayed at 0.01-0.26 where v8 reached 1.00000. What
it CANNOT do is raise rank. A per-channel affine cannot turn k identical rows
into k different ones, so a within-document collapse passes straight through.

THE MOMENTUM HAS TO TRACK THE ENCODER. The same 65 steps saw
``l0_base_raw_std`` go 0.32 -> 1.96, so a reference with a ~100-update
half-life is stale by the time it is used and the standardiser divides by a
number that is too small -- which inflates the output on its own, affine or
not. Match the momentum to how fast the upstream is actually moving.

--- bgkit2 notes ---
The contract between the encoder's output and the decoder's input.

Copied from the old lineage (``bgkit.models.interface_norm``), where it was
added on 2026-08-31 after the encoder's per-document representation collapsed
to a corpus-constant that owned 99% of every vector's energy. Here it is not
opt-in: every projection output passes through it from step 0.

Per-channel standardisation against an EMA reference (never the current
batch, so gradients flow only through the numerator and train/eval read the
same statistics), then a FIXED rescale to the decoder's mean embedding row
norm. The affine is pinned: a trainable shift is a corpus constant re-added
after the standardisation that removed one, and a trainable gain restores the
free scale direction the contract exists to close. The decoder's own input
projections absorb any fixed per-channel scaling, so nothing expressive is
lost.

What it cannot do is raise rank: k identical rows stay identical. Rank and
retrieval are measured separately (``bgkit2.probes``).

``prenorm`` (2026-09-05): the EMA reference is only a contract if its INPUT is
bounded. Standardisation makes the loss invariant to the projection's raw
scale, so that scale is a null direction that Adam random-walks and then
compounds: P1's running_mean grew 12 -> 68 -> 406 -> 1386 (steps 1000/2000/
4000/5000, running_var to 9e4) and P7c's to 2883. A reference that lags a
runaway by even one momentum window standardises by stale numbers, which is
where the step-5000 spike (loss 2.7 -> 7.9, grad 75) and the eval that read
7.2 nats on SQuAD came from. With ``prenorm`` every row is RMS-normalised to
unit scale (parameter-free, in fp32) BEFORE the statistics are taken, so the
reference lives in a bounded space by construction and the raw scale can
drift without consequence.

``match_moments`` (2026-09-06): per-channel standardisation to UNIT variance
puts the reps in an isotropic space, but the decoder's token-embedding
manifold is not isotropic (LFM2.5-350M: per-channel std 0.024 on average
with a 0.21-norm shared mean, 26% of the row norm). The only fixed points of
a whitening contract are whitened distributions, so the projection can never
place a rep ON the manifold, the frozen decoder reads every rep off-manifold,
and a rank-32 decoder adapter cannot invert a 1024-wide per-channel rescale.
That is why, in every regime of the lineage, one rep per token (x1) still
reconstructs at ~2.5 nats against 0.1 for the token embeddings themselves.
With ``match_moments`` the standardised rep is re-coloured with the
decoder's own per-channel mean and std (fixed buffers taken from
``embed_tokens``, never learned): ``out = mu_emb + z * sigma_emb``. The
projection now has an exact fixed point at the token embeddings, the
scale/shift runaways stay closed, and the reps land in the space the
decoder was trained to read.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderInterfaceNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        target_row_norm: float = 1.0,
        momentum: float = 0.01,
        eps: float = 1e-5,
        row_norm: bool = False,
        prenorm: bool = False,
        match_moments: bool = False,
        var_floor_frac: float = 0.01,
        rms_only: bool = False,
        affine: bool = False,  # old bgkit callers pass it; refused (see narrative)
    ) -> None:
        super().__init__()
        if affine:
            raise ValueError(
                "a learnable affine on the interface is not a contract; pinned by design"
            )
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1]; got {momentum}")
        if not target_row_norm > 0:
            raise ValueError(f"target_row_norm must be > 0; got {target_row_norm}")
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        # HARD scale constraint: after the EMA mean subtraction, every row is
        # rescaled to exactly target_row_norm. The EMA-variance version is a
        # soft contract the encoder can outrun -- P4/P7 pushed the emitted
        # norm to 2.4x the reference within a few hundred steps and the
        # gradients exploded. With row_norm the scale direction is closed.
        self.row_norm = bool(row_norm)
        self.prenorm = bool(prenorm)
        self.match_moments = bool(match_moments)
        # Per-channel standardisation amplifies a low-variance channel by
        # 1/sqrt(var): with an absolute eps of 1e-5 a channel the projection
        # holds nearly constant is blown up 100-300x, which is a free scale
        # direction the encoder can exploit (P15: emitted norm 2.9x the
        # embedding norm, gradient norms >100). Floor every channel's variance
        # at var_floor_frac x the mean channel variance: amplification is
        # bounded by 1/sqrt(var_floor_frac) BY CONSTRUCTION.
        self.var_floor_frac = float(var_floor_frac)
        # RMS-ONLY contract (2026-09-08): every row rescaled to exactly
        # target_row_norm and NOTHING else — no running statistics, no
        # per-channel affine, no re-colouring. Measured on P19a4: the
        # standardise-then-recolour path leaves a constant offset of 0.56x the
        # embedding norm on every rep (half the Zipfian-vs-uniform mean
        # mismatch of match_moments, half the standardisation removing any
        # constant the projection emits), capping cos at ~0.75 while the
        # nearest-neighbour token identity is 0.987. True embeddings pass this
        # contract at cos 1.000; scale is bounded by construction.
        self.rms_only = bool(rms_only)
        # Target per-channel moments (the decoder's embedding table); set by
        # set_target_moments. Identity moments until then.
        self.register_buffer("target_mean", torch.zeros(dim))
        self.register_buffer("target_std", torch.ones(dim))
        self.register_buffer("gain", torch.tensor(float(target_row_norm) / (dim**0.5)))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        # The first update SETS the statistics instead of blending into the
        # (0, 1) initialisation, and runs in eval mode too, so an uncalibrated
        # module never standardises by numbers that describe nothing.
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @property
    def target_row_norm(self) -> float:
        return float(self.gain.item()) * (self.dim**0.5)

    @torch.no_grad()
    def set_target_moments(self, table: torch.Tensor) -> None:
        """Per-channel mean/std of the decoder's embedding table (V, dim)."""
        t = table.detach().float()
        if t.dim() != 2 or t.shape[1] != self.dim:
            raise ValueError(f"expected (V, {self.dim}); got {tuple(t.shape)}")
        self.target_mean.copy_(t.mean(dim=0))
        self.target_std.copy_(t.std(dim=0, unbiased=False).clamp_min(1e-6))

    @torch.no_grad()
    def set_target_row_norm(self, value: float) -> None:
        if not value > 0:
            raise ValueError(f"target_row_norm must be > 0; got {value}")
        self.gain.fill_(float(value) / (self.dim**0.5))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.shape[0] < 2:
            return
        mean = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=False)
        if int(self.num_updates.item()) == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
        else:
            m = self.momentum
            self.running_mean.mul_(1.0 - m).add_(mean, alpha=m)
            self.running_var.mul_(1.0 - m).add_(var, alpha=m)
        self.num_updates.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (N, dim) of VALID rows only — never pass padding rows."""
        if x.numel() == 0:
            return x
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}; got {tuple(x.shape)}")
        if self.rms_only:
            xf = x.float()
            eps = max(self.eps, 0.05 * float(self.target_row_norm))
            return (xf / (xf.norm(dim=-1, keepdim=True) + eps) * self.target_row_norm).to(x.dtype)
        if self.prenorm:
            xf = x.float()
            x = (xf * (self.dim**0.5 / (xf.norm(dim=-1, keepdim=True) + self.eps))).to(x.dtype)
        if self.training or int(self.num_updates.item()) == 0:
            self._update(x)
        mean = self.running_mean.to(x.dtype)
        if self.row_norm:
            # fp32 and a LARGE epsilon relative to the target norm: the
            # gradient of x/|x| is ~1/|x|, and a centred rep near zero in bf16
            # produced the non-finite steps P7b skipped (15 in 100 steps).
            xc = (x - mean).float()
            eps = max(self.eps, 0.05 * float(self.target_row_norm))
            xc = xc / (xc.norm(dim=-1, keepdim=True) + eps)
            return (xc * self.target_row_norm).to(x.dtype)
        var = self.running_var.to(torch.float32)
        if self.var_floor_frac > 0:
            var = var.clamp_min(self.var_floor_frac * var.mean())
        std = (var + self.eps).sqrt().to(x.dtype)
        if self.match_moments:
            z = (x - mean).float() / std.float()
            return (self.target_mean + z * self.target_std).to(x.dtype)
        return (x - mean) / std * self.gain.to(x.dtype)

    @torch.no_grad()
    def reseed_if_reference_out_of_contract(self) -> bool:
        """Under ``prenorm`` the per-channel variances average to at most 1
        (unit-RMS rows). A loaded reference above that was computed from
        un-normalised inputs (a checkpoint saved before the flag); re-seed it
        from the next batch instead of standardising by numbers that describe
        a different space. Returns True when a re-seed was scheduled."""
        if self.rms_only or not self.prenorm or int(self.num_updates.item()) == 0:
            return False
        if float(self.running_var.float().mean().item()) <= 2.0:
            return False
        self.num_updates.zero_()
        return True

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, target_row_norm={self.target_row_norm:.4f}, "
            f"momentum={self.momentum}, row_norm={self.row_norm}, prenorm={self.prenorm}, "
            f"match_moments={self.match_moments}"
        )


@torch.no_grad()
def rep_stats(
    reps: torch.Tensor, rep_mask: torch.Tensor, embed_row_norm: float
) -> dict[str, float]:
    """Cheap health numbers for a batch of reps (B, K, D) + mask.

    norm_ratio      mean row norm / decoder embedding row norm (the norm guard)
    eff_rank_within mean per-document participation ratio of the CENTRED rows
                    (~1 means every survivor of a document is the same vector)
    shared_frac     fraction of a document's energy in its own mean direction
    cos_to_corpus   mean cosine between each document's mean vector and the
                    batch mean (1.0 = one vector for every document)
    """
    out: dict[str, float] = {}
    valid = rep_mask.bool()
    if valid.sum() == 0:
        return out
    flat = reps[valid].float()
    out["norm_ratio"] = float(flat.norm(dim=-1).mean().item() / max(embed_row_norm, 1e-8))
    doc_means, ranks, shared = [], [], []
    for b in range(reps.shape[0]):
        x = reps[b][valid[b]].float()
        if x.shape[0] == 0:
            continue
        mu = x.mean(dim=0)
        doc_means.append(mu)
        if x.shape[0] >= 2:
            xc = x - mu
            s = torch.linalg.svdvals(xc)
            p = s.pow(2)
            ranks.append(float((p.sum() ** 2 / (p.pow(2).sum() + 1e-12)).item()))
            total = x.pow(2).sum()
            proj = (x @ (mu / (mu.norm() + 1e-8))).pow(2).sum()
            shared.append(float((proj / (total + 1e-12)).item()))
    if ranks:
        out["eff_rank_within"] = sum(ranks) / len(ranks)
        out["shared_frac"] = sum(shared) / len(shared)
    if len(doc_means) >= 2:
        dm = torch.stack(doc_means)
        corpus = dm.mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(dm, corpus[None, :], dim=-1)
        out["cos_to_corpus"] = float(cos.mean().item())
    return out
