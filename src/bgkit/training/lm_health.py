"""Held-out plain-text language health for the decoder (2026-08-25).

WHY THIS EXISTS. The Phase-2 wide-net runs destroyed the decoder's language
model without a single in-distribution metric noticing. Measured after the
fact by plain next-token cross-entropy on held-out FineWeb-Edu:

    stock Qwen3.5-0.8B                    PPL   15.1
    summarization base (step 51945)       PPL   33.2   <- healthy start
    wide-net v6   (2629 steps)            PPL  670.7
    wide-net v7   (+999 steps at 6x)      PPL 2585.2   <- still descending

Throughout, the runs' own eval loss, token accuracy and exact match looked
merely mediocre, and a zeroed-rep ablation still showed the reps to be
load-bearing (fileneedle EM 0.385 -> 0.026), so nothing flagged a problem.
The damage is diffuse — swapping the pristine embedding back in makes
perplexity WORSE (1125 vs 611), because the backbone co-adapted to the
rotated token geometry — so no weight-norm guard on a single tensor would
have caught it either.

The cheap, format-free instrument that WOULD have caught it in the first
hundred steps is this one: mean CE over a fixed slice of ordinary text.
It needs no chat template, no instruction-following and no task ability, so
it isolates "is this still a language model" from "is it good at our task".

Wire-up: trainers call :func:`lm_health_metrics` from ``evaluate()`` and get
``eval/lm_health/{ce,ppl}`` back. Gate with ``training.lm_health_samples``
(0 disables). Watch for drift from the FIRST eval's value, not an absolute
threshold — the healthy starting point differs per lineage.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import structlog
import torch

logger = structlog.get_logger()


def load_health_chunks(
    tokens_dir: str | Path,
    *,
    n_docs: int = 32,
    seq_len: int = 1024,
) -> list[torch.Tensor]:
    """Fixed slice of held-out text as ``(seq_len,)`` token tensors.

    Drawn from the LAST shard so it stays clearly disjoint from anything the
    training corpora sample, and deterministic so the metric is comparable
    across steps and across runs.
    """
    import pyarrow.parquet as pq

    shards = sorted(Path(tokens_dir).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.parquet under {tokens_dir}")
    rows = pq.read_table(shards[-1], columns=["token_ids"]).to_pylist()
    chunks: list[torch.Tensor] = []
    for row in rows:
        arr = np.asarray(row["token_ids"], dtype=np.int64)
        if arr.size >= seq_len:
            chunks.append(torch.from_numpy(arr[:seq_len]))
        if len(chunks) >= n_docs:
            break
    return chunks


@torch.no_grad()
def lm_health_metrics(
    decoder,
    chunks: list[torch.Tensor],
    device: torch.device,
    *,
    prefix: str = "eval/lm_health",
) -> dict[str, float]:
    """Mean next-token CE (and perplexity) of ``decoder`` on ``chunks``.

    MUST go through the decoder's own supported forward. bgkit's in-training
    decoders are patched for FA4 varlen PACKED attention and have no padded
    fallback, so a bare ``model(input_ids=(B, L))`` hits the packed path with
    no ``cu_seqlens`` and triggers a device-side assert — which poisons the
    CUDA context and kills the whole run, not just the metric. That is
    exactly what happened on the first lrprobe launch (2026-08-25), so this
    routes plain text through ``forward_interleaved_with_loss`` as a single
    all-loss ``TokenSegment`` — the same primitive the trainers already use
    every step. A ``try/except`` cannot save you here: by the time Python
    sees the error the context is already dead, so the call has to be
    correct, not merely guarded.

    Falls back to a bare HF call only for objects with no interleaved API
    (the standalone probes, which load weights into a stock HF model).
    """
    if not chunks:
        return {}
    interleaved = getattr(decoder, "forward_interleaved_with_loss", None)
    model = getattr(decoder, "backbone", decoder)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tok = 0
    try:
        for ids in chunks:
            ids = ids.to(device)
            if interleaved is not None:
                from bgkit.models.decoder import TokenSegment

                mask = torch.ones_like(ids, dtype=torch.bool)
                out = interleaved([TokenSegment(ids, loss_mask=mask)])
                loss = out.loss if hasattr(out, "loss") else out
                n = int(ids.shape[-1]) - 1
                total_nll += float(loss.item()) * n
                total_tok += n
            else:
                batched = ids.unsqueeze(0)
                logits = model(input_ids=batched).logits[:, :-1].float()
                targets = batched[:, 1:]
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(nll.item())
                total_tok += int(targets.numel())
    finally:
        if was_training:
            model.train()
    if total_tok == 0:
        return {}
    ce = total_nll / total_tok
    return {f"{prefix}/ce": ce, f"{prefix}/ppl": math.exp(min(ce, 20.0))}


def load_decoder_tensors(root: str | Path, family: str = "qwen35") -> dict:
    """Decoder state dict from any on-disk checkpoint layout bgkit ships.

    Three have accumulated: the Phase-2 trainers' joint ``model.pt`` keyed
    ``decoders.<family>.backbone.*``, BlobSFTTrainer's ``decoder.backbone.*``,
    and the summarization base's per-model ``decoder_qwen.pt``. A probe that
    knows only one of them silently loads NOTHING and reports a fake result,
    so the prefix search is explicit and failure is loud.
    """
    import torch

    root = Path(root)
    joint, solo = root / "model.pt", root / "decoder_qwen.pt"
    src = joint if joint.exists() else solo
    if not src.exists():
        raise FileNotFoundError(f"no model.pt or decoder_qwen.pt under {root}")
    sd = torch.load(str(src), map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    for prefix in (f"decoders.{family}.backbone.", "decoder.backbone.", "backbone.", ""):
        cand = (
            {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if prefix else dict(sd)
        )
        if "model.embed_tokens.weight" in cand:
            return cand
    raise ValueError(f"cannot locate decoder tensors in {src}")
