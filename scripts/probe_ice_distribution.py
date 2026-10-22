#!/usr/bin/env python
"""Probe ICE score distribution: compute reference (skew, excess_kurt) once, save.

The survivorship-head pivot uses ``moment_match_loss`` to anchor the base head's
output distribution shape to ICE's. Rather than calling ICE every step, we run
ICE ONCE offline against a representative sample and save the standardized 3rd
+ 4th moments of its scores. Training loads two floats and never touches ICE
again post-warmup.

Output: ``$DATA_DIR/diagnostics/ice_reference_moments.json`` containing
``{"skew": float, "excess_kurt": float, "n_positions": int}``.

Sanity check: if standardized moments are near (0, 0), raise an error — that
would mean ICE scores are already approximately Gaussian after standardization,
which is suspicious for a causal-LM entropy score and likely indicates a
loading bug. Investigate the ICE checkpoint before proceeding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("probe_ice_distribution")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ice-checkpoint",
        type=Path,
        required=True,
        help="ICE checkpoint directory or model.pt path.",
    )
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        required=True,
        help="MmapTokenDataset directory to draw sequences from.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HF backbone for the input embedding table (must match ICE training).",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=1000,
        help="Number of sequences to sample.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Max sequence length per sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to $DATA_DIR/diagnostics/ice_reference_moments.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    import torch
    from transformers import AutoConfig

    from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
    from bgkit.env import get_data_dir
    from bgkit.models.ice_teacher import ICETeacher

    output_path = args.output
    if output_path is None:
        output_path = get_data_dir() / "diagnostics" / "ice_reference_moments.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s output=%s", device, output_path)

    # Load ONLY the embedding table (vocab × hidden) from the same backbone
    # ICE was trained on. Using safetensors directly avoids loading the full
    # 1.1B-param backbone just to pull embed_tokens.
    logger.info("loading_embedding_table model=%s", args.backbone)
    config = AutoConfig.from_pretrained(args.backbone, trust_remote_code=True)
    # Config shapes may live on the outer or inner (language_model) config.
    text_config = getattr(config, "text_config", config)
    vocab_size = getattr(text_config, "vocab_size", None) or config.vocab_size
    hidden_size = getattr(text_config, "hidden_size", None) or config.hidden_size
    embed = torch.nn.Embedding(vocab_size, hidden_size)

    # Pull embedding weights via safetensors without instantiating the full model.
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=args.backbone,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    # Locate the shard containing embed_tokens; Qwen3.5 typically names it
    # `model.embed_tokens.weight` or `language_model.model.embed_tokens.weight`.
    import json
    import glob
    embed_keys = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",  # Qwen3.5 multimodal layout
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    embed_weight = None
    index_path = Path(local_dir) / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        wmap = index.get("weight_map", {})
        for key in embed_keys:
            if key in wmap:
                shard = Path(local_dir) / wmap[key]
                loaded = load_file(str(shard))
                embed_weight = loaded[key]
                break
    if embed_weight is None:
        # Single-file model.
        for sfile in glob.glob(str(Path(local_dir) / "*.safetensors")):
            loaded = load_file(sfile)
            for key in embed_keys:
                if key in loaded:
                    embed_weight = loaded[key]
                    break
            if embed_weight is not None:
                break
    if embed_weight is None:
        logger.error("could_not_find_embedding_weights in %s", local_dir)
        return 5
    with torch.no_grad():
        embed.weight.copy_(embed_weight)
    embed.to(device)

    logger.info("loading_ice path=%s", args.ice_checkpoint)
    teacher = ICETeacher(
        args.ice_checkpoint, embed,
        input_dim=hidden_size,
    ).to(device)
    teacher.eval()

    logger.info("opening_mmap dir=%s", args.mmap_dir)
    dataset = MmapTokenDataset(str(args.mmap_dir), max_seq_len=args.max_seq_len)

    if len(dataset) == 0:
        logger.error("dataset_empty dir=%s", args.mmap_dir)
        return 2

    rng = torch.Generator().manual_seed(args.seed)
    n = min(args.n_sequences, len(dataset))
    indices = torch.randperm(len(dataset), generator=rng)[:n].tolist()

    # Aggregate sums for true global standardization.
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0

    # First pass: compute global mean + std.
    logger.info("first_pass n=%d", n)
    flat_scores: list[torch.Tensor] = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        token_ids = (
            sample["token_ids"].unsqueeze(0)
            if isinstance(sample, dict)
            else sample.unsqueeze(0)
        ).to(device)
        attn_mask = torch.ones_like(token_ids)
        scores = teacher.score(token_ids, attn_mask)  # (1, L)
        valid = attn_mask.bool()
        s = scores[valid]
        s = s[torch.isfinite(s)]
        if s.numel() == 0:
            continue
        s_cpu = s.float().cpu()
        flat_scores.append(s_cpu)
        total_count += int(s_cpu.numel())
        total_sum += float(s_cpu.sum().item())
        total_sumsq += float((s_cpu * s_cpu).sum().item())
        if (i + 1) % 100 == 0:
            logger.info("first_pass_progress %d/%d", i + 1, n)

    if total_count < 100:
        logger.error("too_few_valid_positions n=%d", total_count)
        return 3

    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean * mean, 1e-12)
    std = var ** 0.5
    logger.info("global mean=%.6f std=%.6f n=%d", mean, std, total_count)

    # Second pass: accumulate 3rd + 4th central moments using running sums.
    sum3 = 0.0
    sum4 = 0.0
    for chunk in flat_scores:
        z = (chunk - mean) / std
        sum3 += float((z ** 3).sum().item())
        sum4 += float((z ** 4).sum().item())

    skew = sum3 / total_count
    excess_kurt = sum4 / total_count - 3.0
    logger.info("standardized skew=%.6f excess_kurt=%.6f", skew, excess_kurt)

    if abs(skew) < 0.1 and abs(excess_kurt) < 0.2:
        logger.error(
            "near-Gaussian standardized moments (skew=%.4f, excess_kurt=%.4f); "
            "this is suspicious for a causal-LM entropy score. Suspect a "
            "loading bug in the ICE checkpoint before proceeding.",
            skew, excess_kurt,
        )
        return 4

    payload = {
        "skew": skew,
        "excess_kurt": excess_kurt,
        "n_positions": total_count,
        "ice_checkpoint": str(args.ice_checkpoint),
        "backbone": args.backbone,
        "n_sequences": n,
        "max_seq_len": args.max_seq_len,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote_reference_moments path=%s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
