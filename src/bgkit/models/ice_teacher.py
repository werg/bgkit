"""ICETeacher: frozen importance-scoring wrapper used to distill survivorship.

Loads a trained ICE CNN checkpoint (from the pre-survivorship-head era) plus
a reference to the encoder's own input-embedding table (ICE was trained on
Qwen3.5-0.8B-Base embeddings, which is the same backbone BgKITEncoder uses,
so no separate embedding model is needed). Produces per-position importance
scores for a packed batch of content token ids, then derives a top-k teacher
mask at the configured target compression ratio.

The student (SurvivorshipHead) is trained to match this mask via BCE during
the early phase of Phase-1 Step-3 so it gets discriminative per-position
signal from day one instead of collapsing to the aggregate-ratio solution
(all probs near target, none above 0.5 → empty survivor set).

Packed signatures
-----------------
``score(token_ids, cu_seqlens)`` and ``teacher_mask(token_ids, cu_seqlens, ratio)``
both consume a flat ``(N,)`` token-id tensor and a ``(B+1,)`` int32
``cu_seqlens`` boundary tensor (FA4 varlen convention) and return flat
``(N,)`` float32 tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from bgkit.models.ice import ICE
from bgkit.utils.packing import lengths_from_cu


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
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position ICE importance score (predicted CE under Qwen3.5-0.8B-Base).

        Runs the ICE CNN per segment (kernel convolutions do not cross segment
        boundaries), then concatenates back into a flat ``(N,)`` tensor.

        Args:
            token_ids: ``(N,)`` packed content token ids (int64).
            cu_seqlens: ``(B+1,)`` int32 cumulative segment boundaries.
        Returns:
            ``(N,)`` float32 scores.
        """
        self._check_loaded()
        if token_ids.ndim != 1:
            raise ValueError(
                f"ICETeacher.score expects packed (N,) token_ids; got shape "
                f"{tuple(token_ids.shape)}",
            )
        lengths = lengths_from_cu(cu_seqlens.to(torch.int64))
        # Per-segment forward. ICE is a tiny 1D CNN and ICE.forward expects
        # (batch, seq_len, input_dim), so we unsqueeze batch=1 per segment.
        scores_list: list[torch.Tensor] = []
        offsets = cu_seqlens.to(torch.int64).tolist()
        for i, seg_len in enumerate(lengths.tolist()):
            if seg_len == 0:
                continue
            start = offsets[i]
            end = offsets[i + 1]
            seg_ids = token_ids[start:end]
            embs = self._embed_tokens(seg_ids).float().unsqueeze(0)  # (1, L, D)
            seg_scores = self.ice(embs).squeeze(0)  # (L,)
            scores_list.append(seg_scores)
        if not scores_list:
            return torch.zeros(0, dtype=torch.float32, device=token_ids.device)
        return torch.cat(scores_list, dim=0)

    @torch.no_grad()
    def teacher_mask(
        self,
        token_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        target_ratio: float,
    ) -> torch.Tensor:
        """Top-k teacher mask per segment at the given ratio (packed).

        Args:
            token_ids: ``(N,)`` packed content token ids.
            cu_seqlens: ``(B+1,)`` int32 cumulative segment boundaries.
            target_ratio: fraction of valid positions to keep (e.g. 0.1).
        Returns:
            ``(N,)`` float32 mask in {0.0, 1.0}. Exactly
            ``ceil(target_ratio * L_i)`` positions are 1 per segment ``i``
            (clamped to at least 1).
        """
        self._check_loaded()
        if token_ids.ndim != 1:
            raise ValueError(
                f"ICETeacher.teacher_mask expects packed (N,) token_ids; "
                f"got shape {tuple(token_ids.shape)}",
            )
        scores = self.score(token_ids, cu_seqlens)
        mask = torch.zeros_like(scores, dtype=torch.float32)
        lengths = lengths_from_cu(cu_seqlens.to(torch.int64))
        offsets = cu_seqlens.to(torch.int64).tolist()
        for i, seg_len in enumerate(lengths.tolist()):
            if seg_len == 0:
                continue
            start = offsets[i]
            end = offsets[i + 1]
            seg_scores = scores[start:end]
            k = max(1, int(-(-seg_len * target_ratio // 1)))  # ceil
            # Clamp k within segment length.
            k = min(k, seg_len)
            top = torch.topk(seg_scores, k, largest=True).indices
            mask[start:end].index_fill_(0, top, 1.0)
        return mask
