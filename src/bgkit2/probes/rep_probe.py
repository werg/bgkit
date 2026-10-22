"""Do the reps distinguish one document from another? No decoder involved.

Per document, split its rep rows into disjoint parity halves, mean-pool each
half, L2-normalise, and ask whether half A retrieves half B among M documents
(top-1; chance 1/M). A representation that carries document-specific content
retrieves itself; one that emits the same summary for every document scores
at chance. Reported raw and after corpus whitening: a large raw->whitened gap
means the content is present but buried under a shared direction (the
2026-08 failure mode). Also the across-document participation ratio.
"""

from __future__ import annotations

import torch


def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if x.shape[0] < 4:
        return None
    return x[0::2].float().mean(dim=0), x[1::2].float().mean(dim=0)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _top1(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = _unit(a) @ _unit(b).T
    return float((sim.argmax(dim=-1) == torch.arange(a.shape[0], device=a.device)).float().mean())


def _participation_ratio(x: torch.Tensor) -> float:
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc.float())
    p = s.pow(2)
    return float((p.sum() ** 2 / (p.pow(2).sum() + 1e-12)).item())


@torch.no_grad()
def rep_distinguishability(per_doc_reps: list[torch.Tensor]) -> dict[str, float]:
    """``per_doc_reps``: one (k_i, D) tensor per document (valid rows only)."""
    pairs = [h for h in (_halves(r) for r in per_doc_reps) if h is not None]
    if len(pairs) < 2:
        return {"n_docs": float(len(pairs))}
    a = torch.stack([p[0] for p in pairs])
    b = torch.stack([p[1] for p in pairs])
    out = {
        "n_docs": float(a.shape[0]),
        "chance_top1": 1.0 / a.shape[0],
        "top1_raw": _top1(a, b),
        "eff_rank_across": _participation_ratio(torch.cat([a, b])),
    }
    # Corpus whitening from the pooled halves (ridge-regularised).
    pooled = torch.cat([a, b]).float()
    mu = pooled.mean(dim=0, keepdim=True)
    xc = pooled - mu
    cov = xc.T @ xc / max(xc.shape[0] - 1, 1)
    d = cov.shape[0]
    cov = cov + torch.eye(d, device=cov.device) * (1e-3 * cov.diagonal().mean() + 1e-8)
    evals, evecs = torch.linalg.eigh(cov)
    w = evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt()) @ evecs.T
    out["top1_whitened"] = _top1((a - mu) @ w, (b - mu) @ w)
    return out
