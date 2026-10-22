"""P8: the decoder-space adapter of the decoder scale-up (plans/decoder_scaleup_8b_2026_09_25.md).

The trained S2 encoder emits reps in the LFM2.5-350M decoder's embedding space (post projection,
post interface norm), and so do the cached repo-tree nodes. A larger decoder (LFM2.5-8B-A1B,
hidden 2048, a different vocabulary) reads them through this adapter:

    y = contract( L x + MLP(x) )

* ``L`` (1024 -> 2048, linear) is initialised in CLOSED FORM: the least-squares map from the
  350M embedding table to the 8B embedding table over the token STRINGS the two vocabularies
  share (their ids differ). A rep that sits near the 350M embedding of a token therefore lands
  near the 8B embedding of the same string from step 0 -- the alignment the old anchor loss
  bought, without position-pairing encoder and decoder tokens.
* ``MLP`` (1024 -> 4096 -> 2048, GELU) is a residual correction, zero-initialised at its output.
* ``contract`` is a ``DecoderInterfaceNorm`` targeting the 8B table's row norm and moments,
  the same interface contract the encoder keeps toward its own decoder.

The frozen encoder is never touched: every rep path (text slots, summary slots, compaction
spines, cached tree nodes) goes through the one adapter at splice time.
"""

from __future__ import annotations

import torch
from torch import nn

from .interface_norm import DecoderInterfaceNorm


class DecoderSpaceAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 2,
        momentum: float = 0.01,
        prenorm: bool = True,
        match_moments: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        hid = int(out_dim * hidden_mult)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, out_dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.contract = DecoderInterfaceNorm(
            out_dim, momentum=momentum, prenorm=prenorm, match_moments=match_moments
        )

    def forward(self, reps: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``reps`` (..., in_dim) -> (..., out_dim); rows where ``mask`` is False stay zero (the
        contract's statistics see valid rows only)."""
        shape = reps.shape[:-1]
        x = reps.reshape(-1, self.in_dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            y = self.linear(x) + self.mlp(x)
            out = torch.zeros(x.shape[0], self.out_dim, dtype=y.dtype, device=y.device)
            valid = (
                mask.reshape(-1).bool() if mask is not None
                else torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            )
            # an exactly-zero rep carries nothing (the zeroed ablation arm): it stays zero
            valid = valid & (x.abs().amax(dim=-1) > 0)
            if bool(valid.any()):
                out[valid] = self.contract(y[valid])
        return out.reshape(*shape, self.out_dim).to(reps.dtype)

    @torch.no_grad()
    def init_from_tables(
        self, src_table: torch.Tensor, dst_table: torch.Tensor, pairs: list[tuple[int, int]],
        ridge: float = 1e-3,
    ) -> dict[str, float]:
        """Closed-form ``linear``: ridge least squares ``dst[j] ~ W src[i] + b`` over ``pairs``
        (src id, dst id). Returns the fit's relative residual and cosine on the pairs."""
        i = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        j = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        X = src_table[i].double()
        Y = dst_table[j].double()
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        A = Xc.T @ Xc + ridge * Xc.shape[0] * torch.eye(Xc.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(A, Xc.T @ Yc)  # (in, out)
        b = my - mx @ W
        self.linear.weight.copy_(W.T.float())
        self.linear.bias.copy_(b.float())
        pred = X @ W + b
        rel = float(((pred - Y).norm() / Y.norm()).item())
        cos = float(torch.nn.functional.cosine_similarity(pred, Y, dim=-1).mean().item())
        return {"pairs": float(len(pairs)), "rel_residual": rel, "mean_cos": cos}


def shared_token_pairs(src_tok, dst_tok) -> list[tuple[int, int]]:
    """(src id, dst id) for every token string both vocabularies contain."""
    dv = dst_tok.get_vocab()
    return sorted((int(i), int(dv[s])) for s, i in src_tok.get_vocab().items() if s in dv)
