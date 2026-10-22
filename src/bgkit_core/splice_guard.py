"""Splice guard (old lineage ``decoder._rep_rank_stats`` / ``_maybe_guard_
spliced_rep_norm``, 2026-08-31): thresholds on what the decoder receives, and a
cross-payload EMA of the payload MEAN so a shared constant that grows slowly
across runs is caught (a per-process reference re-anchored every start and let
a 484 -> 70985 runaway hide). Two muting routes, two signatures: a growing
constant shows up as rank collapse / shared_frac -> 1; a vanishing gain shows
up on the norm band (effective rank is scale-invariant)."""

from __future__ import annotations

import torch


class SpliceGuard:
    def __init__(
        self,
        norm_lo: float = 0.5,
        norm_hi: float = 2.0,
        eff_rank_lo: float = 2.0,
        shared_frac_hi: float = 0.95,
        cos_ema_hi: float = 0.99,
        momentum: float = 0.02,
        eff_rank_min_reps: float = 8.0,
    ) -> None:
        self.norm_lo, self.norm_hi = float(norm_lo), float(norm_hi)
        self.eff_rank_min_reps = float(eff_rank_min_reps)
        self.eff_rank_lo = float(eff_rank_lo)
        self.shared_frac_hi = float(shared_frac_hi)
        self.cos_ema_hi = float(cos_ema_hi)
        self.momentum = float(momentum)
        self.ema_mean: torch.Tensor | None = None

    @torch.no_grad()
    def observe(self, reps: torch.Tensor, rep_mask: torch.Tensor, stats: dict) -> dict:
        """Update the payload-mean EMA and return {name: value} of every
        violated threshold (empty when the payload is inside the contract)."""
        rows = reps[rep_mask].float()
        out: dict = {}
        if rows.numel() == 0:
            return out
        mean = rows.mean(0)
        if self.ema_mean is None:
            self.ema_mean = mean.clone()
        cos_ema = float(torch.nn.functional.cosine_similarity(mean, self.ema_mean, dim=0))
        self.ema_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        stats["cos_to_corpus_ema"] = cos_ema
        nr = stats.get("norm_ratio")
        if nr is not None and not (self.norm_lo <= nr <= self.norm_hi):
            out["norm_ratio"] = nr
        er = stats.get("eff_rank_within")
        # A payload of k reps per document cannot have a within-document
        # participation ratio above k: under a wide ratio band a 128x batch on
        # short documents carries 1-3 reps, and the rank check would fire on
        # every such batch (P19j, 2026-09-09). Judge rank only where it can
        # be read: at least eff_rank_min_reps reps per document on average.
        per_doc = rep_mask.sum(dim=-1).float()
        k_mean = float(per_doc[per_doc > 0].mean()) if bool((per_doc > 0).any()) else 0.0
        if er is not None and er < self.eff_rank_lo and k_mean >= self.eff_rank_min_reps:
            out["eff_rank_within"] = er
        sf = stats.get("shared_frac")
        if sf is not None and sf > self.shared_frac_hi:
            out["shared_frac"] = sf
        if cos_ema > self.cos_ema_hi and sf is not None and sf > 0.5:
            out["cos_to_corpus_ema"] = cos_ema
        return out

    def state_dict(self) -> dict:
        return {"ema_mean": None if self.ema_mean is None else self.ema_mean.cpu()}

    def load_state_dict(self, state: dict) -> None:
        v = state.get("ema_mean")
        self.ema_mean = None if v is None else v.clone()
