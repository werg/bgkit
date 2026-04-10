"""Phase 3 distillation trainer: SWE-bench trajectory distillation.

Distills teacher agent trajectories (from 480B/70B/Sonnet) into the student
(0.8B + BgKIT), using 3 context sources:
1. Compressed filesystem (L0/L1 from BgKIT encoder)
2. Git history (commit chains)
3. Prior agentic sessions (ordered by base_commit)

Training input: [BgKIT context frames | issue description | filtered teacher trajectory]
Loss: CE on trajectory tokens (tool calls, code edits, reasoning)
~30% examples without BgKIT injection (baseline preservation)

Cache directory layout (under ``bgkit_cache_dir``):
- ``filesystem/manifest.parquet`` — keyed by (repo, base_commit), ``path`` column
  points to per-(repo, commit) ``survivors.npy`` files
- ``git_history/manifest.parquet`` — keyed by repo, ``path`` column points to
  per-repo commit-chain ``survivors.npy`` files
- ``prior_sessions/manifest.parquet`` — keyed by repo, one row per trajectory
  ordered by base_commit, ``path`` column points to ``survivors.npy``
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bgkit.data.datasets.swe_trajectory_dataset import SWETrajectoryDataset
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.training.base_trainer import BaseTrainer

logger = structlog.get_logger()


def _load_manifest(parquet_path: Path) -> list[dict]:
    """Load a manifest.parquet file and return rows as list of dicts."""
    import pyarrow.parquet as pq

    if not parquet_path.exists():
        return []
    return pq.read_table(parquet_path).to_pylist()


def _load_survivors_from_path(npy_path: str) -> torch.Tensor | None:
    """Load a survivors.npy file as a torch tensor, returning None on error."""
    path = Path(npy_path)
    if not path.exists():
        return None
    arr = np.load(str(path))
    if arr.ndim == 1:
        arr = arr[:, None]
    return torch.from_numpy(arr.copy())


class _ContextSourceCache:
    """Index for one context source (filesystem, git_history, or prior_sessions).

    The manifest.parquet is loaded once and indexed for fast lookup by the
    relevant key(s).
    """

    def __init__(self, manifest_path: Path, key_columns: list[str]):
        self._key_columns = key_columns
        self._index: dict[tuple[str, ...], list[dict]] = {}
        rows = _load_manifest(manifest_path)
        for row in rows:
            key = tuple(str(row.get(col, "")) for col in key_columns)
            self._index.setdefault(key, []).append(row)
        logger.info(
            "context_source_loaded",
            path=str(manifest_path),
            keys=len(self._index),
            rows=len(rows),
        )

    def __len__(self) -> int:
        return len(self._index)

    def lookup(self, *key_values: str) -> list[dict]:
        """Return manifest rows matching the key, or empty list."""
        return self._index.get(tuple(key_values), [])


class DistillationTrainer(BaseTrainer):
    """Phase 3 distillation from SWE-bench teacher trajectories.

    Steps 2-3: progressively larger teacher models.
    Step 3 extends to Qwen3.5-35B target (like KRStep5Trainer).
    """

    _log_every = 5

    def __init__(self, cfg):
        super().__init__(cfg)
        self._no_injection_fraction = float(
            cfg.training.get("no_injection_fraction", 0.30),
        )
        self._bgkit_cache_dir = cfg.training.get("bgkit_cache_dir")

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load decoder (student model)
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        decoder_backbone.to(self.device)
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_backbone.get_input_embeddings().weight.shape[1],
        )
        self.decoder.train()

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True)

        # Load BgKIT encoder from Phase 2 checkpoint (if configured)
        self.encoder = None
        self.ice = None
        phase2_ckpt = self.cfg.training.get("phase2_checkpoint")
        if phase2_ckpt:
            self._load_encoder(str(phase2_ckpt))

        # Load multi-source context caches
        self._fs_cache: _ContextSourceCache | None = None
        self._git_cache: _ContextSourceCache | None = None
        self._session_cache: _ContextSourceCache | None = None

        if self._bgkit_cache_dir:
            cache_root = Path(str(self._bgkit_cache_dir))

            # Source 1: compressed filesystem — keyed by (repo, base_commit)
            fs_manifest = cache_root / "filesystem" / "manifest.parquet"
            if fs_manifest.exists():
                self._fs_cache = _ContextSourceCache(
                    fs_manifest, key_columns=["repo", "base_commit"],
                )

            # Source 2: git history — keyed by repo
            git_manifest = cache_root / "git_history" / "manifest.parquet"
            if git_manifest.exists():
                self._git_cache = _ContextSourceCache(
                    git_manifest, key_columns=["repo"],
                )

            # Source 3: prior sessions — keyed by repo
            session_manifest = cache_root / "prior_sessions" / "manifest.parquet"
            if session_manifest.exists():
                self._session_cache = _ContextSourceCache(
                    session_manifest, key_columns=["repo"],
                )

            sources = sum(
                c is not None
                for c in [self._fs_cache, self._git_cache, self._session_cache]
            )
            logger.info("phase3_context_sources", available=sources, cache_dir=str(cache_root))

        # Build model container
        self.model = nn.ModuleDict({"decoder": self.decoder})
        if self.encoder is not None:
            self.model["encoder"] = self.encoder
        self.model.to(self.device)

        # Load trajectory dataset
        data_dir = str(self.cfg.data.get("trajectory_dir", "data/trajectories"))
        self.dataset = SWETrajectoryDataset(
            data_dir,
            tokenizer=self.tokenizer,
            max_trajectory_tokens=int(self.cfg.data.get("max_trajectory_tokens", 32768)),
            max_issue_tokens=int(self.cfg.data.get("max_issue_tokens", 2048)),
            require_resolved=True,
        )

        total = len(self.dataset)
        eval_size = min(max(1, int(total * 0.05)), 200)
        train_size = total - eval_size
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, eval_size], generator=generator,
        )

        batch_size = int(self.cfg.get("batch_size", 1))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate,
        )

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(
            [{"params": params}],
            default_lr=float(self.cfg.training.get("lr", 1.0e-4)),
        )

    def _load_encoder(self, checkpoint_path: str) -> None:
        """Load BgKIT encoder from Phase 2 checkpoint."""
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.models.ice import ICE
        from bgkit.training.checkpointing import load_checkpoint

        logger.info("phase3_loading_encoder", checkpoint=checkpoint_path)
        _metadata, state_dicts = load_checkpoint(Path(checkpoint_path))
        model_state = state_dicts.get("model", {})

        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items() if k.startswith("encoder.")
        }
        if encoder_state:
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                self.cfg.model.get("encoder", {}).get(
                    "backbone_name", "Qwen/Qwen3.5-0.8B-Base",
                ),
                encoder_state,
                hidden_dim=1024,
            )
            self.encoder.to(self.device).eval()
            self.encoder.requires_grad_(False)

        ice_state = {
            k.replace("ice.", "", 1): v
            for k, v in model_state.items() if k.startswith("ice.")
        }
        if ice_state:
            self.ice = ICE(input_dim=1024, hidden_dim=128, num_layers=3)
            self.ice.load_state_dict(ice_state, strict=False)
            self.ice.to(self.device).eval()
            self.ice.requires_grad_(False)

    def _collate(self, batch: list[dict]) -> dict:
        """Collate trajectory samples."""
        from bgkit.data.collators import pad_and_collate

        result = {
            "instance_ids": [s["instance_id"] for s in batch],
            "repos": [s["repo"] for s in batch],
            "base_commits": [s["base_commit"] for s in batch],
            "issue_texts": [s["issue_text"] for s in batch],
        }

        if "trajectory_token_ids" in batch[0]:
            traj_ids, traj_mask = pad_and_collate(
                [s["trajectory_token_ids"] for s in batch],
            )
            result["trajectory_token_ids"] = traj_ids
            result["trajectory_attention_mask"] = traj_mask

        if "issue_token_ids" in batch[0]:
            issue_ids, issue_mask = pad_and_collate(
                [s["issue_token_ids"] for s in batch],
            )
            result["issue_token_ids"] = issue_ids
            result["issue_attention_mask"] = issue_mask

        return result

    def _load_source_survivors(
        self,
        cache: _ContextSourceCache | None,
        *key_values: str,
        base_commit_filter: str | None = None,
    ) -> torch.Tensor | None:
        """Load survivors from a context source cache for one sample.

        Args:
            cache: The context source cache to query.
            *key_values: Key values for the cache lookup (e.g. repo, base_commit).
            base_commit_filter: For prior_sessions, only include rows whose
                base_commit sorts before this value (earlier trajectories).

        Returns:
            Concatenated survivors tensor or None if unavailable.
        """
        if cache is None:
            return None
        rows = cache.lookup(*key_values)
        if not rows:
            return None

        # For prior sessions: filter to trajectories earlier than current base_commit
        if base_commit_filter is not None:
            rows = [r for r in rows if str(r.get("base_commit", "")) < base_commit_filter]
            if not rows:
                return None

        parts = []
        for row in rows:
            path = row.get("path")
            if not path:
                continue
            survivors = _load_survivors_from_path(str(path))
            if survivors is not None and survivors.numel() > 0:
                parts.append(survivors)

        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def _get_bgkit_context(
        self, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Assemble multi-source BgKIT context for the batch.

        For each sample, concatenates survivors from up to 3 sources:
        1. Compressed filesystem at (repo, base_commit)
        2. Git history commit chains for repo
        3. Prior agentic session embeddings for repo (ordered before base_commit)

        Returns:
            (context, mask) tensors padded across the batch, or None if no
            context is available for any sample.
        """
        repos = batch.get("repos", [])
        base_commits = batch.get("base_commits", [])
        if not repos:
            return None

        batch_size = len(repos)
        per_sample: list[torch.Tensor] = []
        any_found = False

        for i in range(batch_size):
            repo = str(repos[i])
            commit = str(base_commits[i]) if i < len(base_commits) else ""

            sources: list[torch.Tensor] = []

            # Source 1: compressed filesystem survivors at (repo, base_commit)
            fs_survivors = self._load_source_survivors(self._fs_cache, repo, commit)
            if fs_survivors is not None:
                sources.append(fs_survivors)

            # Source 2: git history commit chains for repo
            git_survivors = self._load_source_survivors(self._git_cache, repo)
            if git_survivors is not None:
                sources.append(git_survivors)

            # Source 3: prior session survivors (only those before current base_commit)
            session_survivors = self._load_source_survivors(
                self._session_cache, repo, base_commit_filter=commit,
            )
            if session_survivors is not None:
                sources.append(session_survivors)

            if sources:
                combined = torch.cat(sources, dim=0)
                per_sample.append(combined)
                any_found = True
            else:
                # Placeholder — will be masked out
                per_sample.append(torch.empty(0))

        if not any_found:
            return None

        # Determine hidden dim from a non-empty sample
        hidden_dim = 0
        for s in per_sample:
            if s.numel() > 0:
                hidden_dim = s.size(-1)
                break
        if hidden_dim == 0:
            return None

        # Pad to batch
        max_len = max((s.size(0) if s.numel() > 0 else 0) for s in per_sample)
        if max_len == 0:
            return None

        padded = torch.zeros(batch_size, max_len, hidden_dim)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, s in enumerate(per_sample):
            if s.numel() > 0:
                length = s.size(0)
                # Ensure 2D
                if s.ndim == 1:
                    s = s.unsqueeze(-1)
                padded[i, :length] = s[:length]
                mask[i, :length] = True

        return padded.to(self.device), mask.to(self.device)

    def _forward_backward(self, batch) -> dict[str, float]:
        """Distillation forward pass with multi-source context and issue tokens.

        Assembles decoder input as [BgKIT context | issue tokens | trajectory tokens].
        Loss is computed only on the trajectory token portion.
        """
        inject = random.random() > self._no_injection_fraction

        trajectory_ids = batch["trajectory_token_ids"].to(self.device)
        trajectory_mask = batch["trajectory_attention_mask"].to(self.device)
        batch_size = trajectory_ids.size(0)

        # Get issue token embeddings
        has_issue = "issue_token_ids" in batch
        embed_layer = self.decoder.backbone.get_input_embeddings()

        if has_issue:
            issue_ids = batch["issue_token_ids"].to(self.device)
            issue_mask = batch["issue_attention_mask"].to(self.device)
            issue_len = issue_ids.size(1)
        else:
            issue_len = 0

        # Determine BgKIT context
        bgkit_context = self._get_bgkit_context(batch) if inject else None

        # Build target: [issue_tokens | trajectory_tokens]
        # Loss mask: 0 on issue tokens, 1 on trajectory tokens
        if has_issue:
            target_ids = torch.cat([issue_ids, trajectory_ids], dim=1)
            target_mask = torch.cat([issue_mask, trajectory_mask], dim=1)
            # Loss only on trajectory portion
            issue_loss_zeros = torch.zeros_like(issue_mask, dtype=torch.bool)
            trajectory_loss_ones = trajectory_mask.bool()
            loss_mask = torch.cat([issue_loss_zeros, trajectory_loss_ones], dim=1)
        else:
            target_ids = trajectory_ids
            target_mask = trajectory_mask
            loss_mask = trajectory_mask.bool()

        context_sources = 0
        if bgkit_context is not None:
            context_embeds, context_mask = bgkit_context
            context_sources = int(context_mask.any(dim=-1).sum().item())
            loss = self.decoder.forward_with_loss(
                context_embeds,
                target_ids,
                target_mask,
                context_mask,
                loss_mask=loss_mask,
            )
        else:
            # No BgKIT context: use a single-token empty context
            empty_context = embed_layer(trajectory_ids[:, :1])
            empty_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
            loss = self.decoder.forward_with_loss(
                empty_context,
                target_ids,
                target_mask,
                empty_mask,
                loss_mask=loss_mask,
            )

        (loss / self._accum_steps).backward()
        return {
            "loss": loss.detach(),
            "injected": float(inject),
            "context_sources": float(context_sources),
            "issue_tokens": float(issue_len),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss_with_ctx = 0.0
        total_loss_no_ctx = 0.0
        batches_with_ctx = 0
        batches_no_ctx = 0

        embed_layer = self.decoder.backbone.get_input_embeddings()

        for batch in self.eval_dataloader:
            trajectory_ids = batch["trajectory_token_ids"].to(self.device)
            trajectory_mask = batch["trajectory_attention_mask"].to(self.device)
            batch_size = trajectory_ids.size(0)

            has_issue = "issue_token_ids" in batch
            if has_issue:
                issue_ids = batch["issue_token_ids"].to(self.device)
                issue_mask = batch["issue_attention_mask"].to(self.device)
                target_ids = torch.cat([issue_ids, trajectory_ids], dim=1)
                target_mask = torch.cat([issue_mask, trajectory_mask], dim=1)
                issue_loss_zeros = torch.zeros_like(issue_mask, dtype=torch.bool)
                trajectory_loss_ones = trajectory_mask.bool()
                loss_mask = torch.cat([issue_loss_zeros, trajectory_loss_ones], dim=1)
            else:
                target_ids = trajectory_ids
                target_mask = trajectory_mask
                loss_mask = trajectory_mask.bool()

            # Eval with context
            bgkit_context = self._get_bgkit_context(batch)
            if bgkit_context is not None:
                context_embeds, context_mask = bgkit_context
                loss_ctx = self.decoder.forward_with_loss(
                    context_embeds, target_ids, target_mask, context_mask,
                    loss_mask=loss_mask,
                )
                total_loss_with_ctx += loss_ctx.item()
                batches_with_ctx += 1
            else:
                # Eval without context
                empty_context = embed_layer(trajectory_ids[:, :1])
                empty_mask = torch.ones(
                    batch_size, 1, dtype=torch.bool, device=self.device,
                )
                loss_no = self.decoder.forward_with_loss(
                    empty_context, target_ids, target_mask, empty_mask,
                    loss_mask=loss_mask,
                )
                total_loss_no_ctx += loss_no.item()
                batches_no_ctx += 1

        self.model.train()

        total_batches = batches_with_ctx + batches_no_ctx
        total_loss = total_loss_with_ctx + total_loss_no_ctx

        metrics = {
            "eval/loss": total_loss / max(total_batches, 1),
        }
        if batches_with_ctx > 0:
            metrics["eval/loss_with_context"] = total_loss_with_ctx / batches_with_ctx
        if batches_no_ctx > 0:
            metrics["eval/loss_no_context"] = total_loss_no_ctx / batches_no_ctx
        metrics["eval/context_coverage"] = (
            batches_with_ctx / max(total_batches, 1)
        )
        return metrics
