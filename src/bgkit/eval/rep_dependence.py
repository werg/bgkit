"""Per-token rep dependence: does the decoder actually consult the reps?

WHY THIS IS NOT A POOLED NUMBER. A pooled CE gap is dominated by tokens that
never needed the reps in the first place. Under teacher forcing most target
tokens are predictable from the preceding target tokens alone — syntax,
discourse structure, entities already introduced — so a model can drive pooled
CE down a long way without ever consulting the compressed context. Measured on
the Phase-2 replay task 2026-08-29: CE fell 2.42 -> 2.17 while the rep gap
SHRANK 0.039 -> 0.025. The pooled number moved the wrong way relative to the
thing it was supposed to indicate.

So the primary statistic here is the DISTRIBUTION of the per-token gap, not its
mean. If reps are load-bearing for a minority of content-introducing tokens,
that shows up as a heavy right tail with a mean near zero — which is exactly
the case a pooled metric cannot distinguish from "reps are useless".

Assumption-free by construction: no notion of "content word" is required for
the primary read. The optional source-overlap split is reported alongside as a
secondary view, clearly labelled as heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def per_token_ce(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ce, positions)`` for next-token prediction at masked sites.

    ``ce[i]`` is the cross-entropy in nats for the token at ``positions[i]``,
    predicted from the hidden state one position earlier.

    Computed in position chunks and via ``logsumexp - target_logit`` rather
    than materialising ``(S, V)`` logits: the vocabulary is ~150k, so a 2k-token
    sequence in fp32 would be ~1.2 GB of logits for a number that is one scalar
    per position.
    """
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]
        token_ids = token_ids[0]
        loss_mask = loss_mask[0]
    # Shift: hidden at t predicts token at t+1.
    h = hidden_states[:-1]
    tgt = token_ids[1:]
    valid = loss_mask[1:].to(torch.bool)
    idx = valid.nonzero().flatten()
    if idx.numel() == 0:
        empty = torch.zeros(0, device=hidden_states.device)
        return empty, empty.long()

    ces: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, idx.numel(), chunk):
            sel = idx[s : s + chunk]
            logits = lm_head(h[sel]).float()
            lse = torch.logsumexp(logits, dim=-1)
            tgt_logit = logits.gather(-1, tgt[sel].unsqueeze(-1)).squeeze(-1)
            ces.append(lse - tgt_logit)
    return torch.cat(ces), idx


@dataclass
class RepDependenceStats:
    """Distributional summary of the per-token gap ``ce_zeroed - ce_reps``."""

    n_tokens: int = 0
    ce_reps_mean: float = 0.0
    ce_zeroed_mean: float = 0.0
    gap_mean: float = 0.0
    gap_median: float = 0.0
    gap_p90: float = 0.0
    gap_p99: float = 0.0
    gap_top_decile_mean: float = 0.0
    frac_gap_over_0p5: float = 0.0
    frac_gap_over_2p0: float = 0.0
    frac_gap_negative: float = 0.0
    extra: dict = field(default_factory=dict)

    def render(self) -> str:
        pct = lambda v: f"{100 * v:5.1f}%"  # noqa: E731
        return (
            f"tokens scored          {self.n_tokens}\n"
            f"CE with reps           {self.ce_reps_mean:.4f} nats\n"
            f"CE with reps zeroed    {self.ce_zeroed_mean:.4f} nats\n"
            f"\n"
            f"PER-TOKEN GAP (ce_zeroed - ce_reps), the distribution:\n"
            f"  mean                 {self.gap_mean:+.4f}\n"
            f"  median               {self.gap_median:+.4f}\n"
            f"  p90                  {self.gap_p90:+.4f}\n"
            f"  p99                  {self.gap_p99:+.4f}\n"
            f"  top-decile mean      {self.gap_top_decile_mean:+.4f}\n"
            f"  frac gap > 0.5 nats  {pct(self.frac_gap_over_0p5)}\n"
            f"  frac gap > 2.0 nats  {pct(self.frac_gap_over_2p0)}\n"
            f"  frac gap < 0         {pct(self.frac_gap_negative)}\n"
        )


def summarize_gap(
    ce_reps: torch.Tensor, ce_zeroed: torch.Tensor,
) -> RepDependenceStats:
    """Distributional read on the per-token gap.

    A near-zero MEAN with a heavy right tail means the reps are load-bearing
    for a minority of tokens; a near-zero mean with a thin tail means they are
    not being used at all. Both give the same pooled number, which is why the
    pooled number alone has been actively misleading in this project.
    """
    if ce_reps.numel() == 0:
        return RepDependenceStats()
    gap = (ce_zeroed - ce_reps).float()
    q = torch.quantile(
        gap, torch.tensor([0.5, 0.9, 0.99], device=gap.device, dtype=gap.dtype),
    )
    k = max(1, gap.numel() // 10)
    top = torch.topk(gap, k).values
    return RepDependenceStats(
        n_tokens=int(gap.numel()),
        ce_reps_mean=float(ce_reps.mean()),
        ce_zeroed_mean=float(ce_zeroed.mean()),
        gap_mean=float(gap.mean()),
        gap_median=float(q[0]),
        gap_p90=float(q[1]),
        gap_p99=float(q[2]),
        gap_top_decile_mean=float(top.mean()),
        frac_gap_over_0p5=float((gap > 0.5).float().mean()),
        frac_gap_over_2p0=float((gap > 2.0).float().mean()),
        frac_gap_negative=float((gap < 0).float().mean()),
    )


def split_by_source_overlap(
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    common_ids: set[int] | None = None,
) -> torch.Tensor:
    """Heuristic mask: target tokens whose id also occurs in the source.

    SECONDARY EVIDENCE ONLY, and stated as such wherever it is reported. A
    token appearing in the source is a weak proxy for "this token had to come
    from the source" — function words appear in both, and a paraphrased content
    word appears in neither. ``common_ids`` lets the caller exclude the most
    frequent vocabulary so the split is not swamped by stopwords.
    """
    src = set(int(t) for t in source_ids.flatten().tolist())
    if common_ids:
        src -= common_ids
    keep = torch.tensor(
        [int(t) in src for t in target_ids.flatten().tolist()],
        dtype=torch.bool, device=target_ids.device,
    )
    return keep
