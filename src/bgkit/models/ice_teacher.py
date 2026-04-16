"""ICETeacher: frozen importance-scoring wrapper used to distill survivorship.

Loads a trained ICE CNN checkpoint (from the pre-survivorship-head era) plus
a reference to the encoder's own input-embedding table (ICE was trained on
Qwen3.5-0.8B-Base embeddings, which is the same backbone BgKITEncoder uses,
so no separate embedding model is needed). Produces per-position importance
scores for a batch of content token ids, then derives a top-k teacher mask
at the configured target compression ratio.

The student (SurvivorshipHead) is trained to match this mask via BCE during
the early phase of Phase-1 Step-3 so it gets discriminative per-position
signal from day one instead of collapsing to the aggregate-ratio solution
(all probs near target, none above 0.5 → empty survivor set).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from bgkit.models.ice import ICE


class ICETeacher(nn.Module):
    def __init__(
        self,
        ice_checkpoint_path: str | Path,
        embed_tokens: nn.Embedding,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 192,
        num_layers: int = 3,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.ice = ICE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
        )
        ckpt_path = Path(ice_checkpoint_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "model.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.ice.load_state_dict(state)
        self.ice.eval()
        for p in self.ice.parameters():
            p.requires_grad_(False)
        self._embed_tokens = embed_tokens
        self._unloaded = False

    @property
    def is_loaded(self) -> bool:
        return not self._unloaded

    def unload(self) -> None:
        """Free ICE model weights from device.

        After this call, ``score()`` and ``teacher_mask()`` raise
        ``RuntimeError``. Idempotent — second call is a no-op. The trained
        downstream model has zero runtime ICE dependency, so calling this
        once BCE warmup ends is safe.

        Properly drops the ICE submodule from self._modules (not just the
        Python attr) so its parameters are released. Clears CUDA cache if
        available.
        """
        if self._unloaded:
            return
        # Drop from _modules registry so parameters are no longer tracked.
        if "ice" in self._modules:
            del self._modules["ice"]
        # Replace the attribute with None via __dict__ to avoid re-registering.
        self.__dict__["ice"] = None
        self._unloaded = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_loaded(self) -> None:
        if self._unloaded:
            raise RuntimeError(
                "ICE unloaded; do not call after warmup. Reference moments "
                "for moment-match are pre-computed offline by "
                "scripts/probe_ice_distribution.py."
            )

    @torch.no_grad()
    def score(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position ICE importance score (predicted CE under Qwen3.5-0.8B-Base).

        Args:
            token_ids: (B, L) content token ids.
            attention_mask: (B, L) 1 for real positions, 0 for padding.
        Returns:
            (B, L) float32 scores with padded positions set to -inf so they
            will never be selected by top-k.
        """
        self._check_loaded()
        embs = self._embed_tokens(token_ids).float()
        scores = self.ice(embs)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        return scores

    @torch.no_grad()
    def teacher_mask(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ratio: float,
    ) -> torch.Tensor:
        """Top-k teacher mask per sequence at the given ratio.

        Args:
            token_ids: (B, L).
            attention_mask: (B, L).
            target_ratio: fraction of valid positions to keep (e.g. 0.1).
        Returns:
            (B, L) float32 mask in {0.0, 1.0}. Exactly
            ``ceil(target_ratio * valid_len)`` positions are 1 per sequence.
        """
        self._check_loaded()
        scores = self.score(token_ids, attention_mask)
        valid_lens = attention_mask.sum(dim=1).clamp(min=1)
        k_per_seq = torch.ceil(valid_lens.float() * target_ratio).long().clamp(min=1)
        mask = torch.zeros_like(scores, dtype=torch.float32)
        # Per-row top-k (k varies) — gather via argsort on descending scores.
        order = scores.argsort(dim=1, descending=True)
        k_max = int(k_per_seq.max().item())
        topk_idx = order[:, :k_max]
        # Positional index grid for comparison with per-row k.
        col = torch.arange(k_max, device=scores.device).unsqueeze(0)
        keep = col < k_per_seq.unsqueeze(1)
        rows = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand_as(topk_idx)
        mask[rows[keep], topk_idx[keep]] = 1.0
        return mask
