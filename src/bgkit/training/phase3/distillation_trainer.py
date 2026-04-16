"""Phase 3 distillation trainer: SWE-bench trajectory distillation.

Distills teacher agent trajectories (from 480B/70B/Sonnet) into the student
(0.8B + BgKIT), using 3 context sources:
1. Compressed filesystem (L0/L1 from BgKIT encoder)
2. Git history (commit chains)
3. Prior agentic sessions (ordered by base_commit)

Training input: [issue description | bgkit tool-response slot | filtered teacher trajectory]
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
from bgkit.utils.attention_backend import resolve_attention_implementation

logger = structlog.get_logger()

_DISTILL_BGKIT_SENTINEL = "<<<BGKIT_DISTILL_CONTEXT_0d4e61f9>>>"


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

    The student/target is Qwen3.5-0.8B throughout — the same decoder used
    by every other phase of bgkit. Teacher trajectories come from larger
    external models (SWE-bench OpenHands, Llama-70B, etc.) but we never
    train a larger target in-house.
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
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # Load decoder (student model)
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=attention_impl,
        )
        decoder_backbone.to(self.device)
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_backbone.get_input_embeddings().weight.shape[1],
        )
        self.decoder.train()

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True)
        self._distill_bgkit_prefix_ids = torch.tensor(
            self.tokenizer.encode(
                "<tool_call>bgkit</tool_call>\n<tool_response>\n",
                add_special_tokens=False,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._distill_bgkit_sentinel_ids = torch.tensor(
            self.tokenizer.encode(_DISTILL_BGKIT_SENTINEL, add_special_tokens=False),
            dtype=torch.long,
            device=self.device,
        )
        self._distill_bgkit_suffix_ids = torch.tensor(
            self.tokenizer.encode("\n</tool_response>\n", add_special_tokens=False),
            dtype=torch.long,
            device=self.device,
        )

        # Load BgKIT encoder from Phase 2 checkpoint (if configured)
        self.encoder = None
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

    def _build_decoder_targets(
        self,
        trajectory_ids: torch.Tensor,
        trajectory_mask: torch.Tensor,
        issue_ids: torch.Tensor | None = None,
        issue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a decoder target batch with a fixed BgKIT tool-response slot."""
        batch_size = trajectory_ids.size(0)
        target_rows: list[torch.Tensor] = []
        loss_rows: list[torch.Tensor] = []
        splice_starts: list[int] = []
        splice_lens: list[int] = []

        for b in range(batch_size):
            issue = torch.empty(0, dtype=torch.long, device=self.device)
            if issue_ids is not None and issue_mask is not None:
                issue_len = int(issue_mask[b].sum().item())
                issue = issue_ids[b, :issue_len]
            traj_len = int(trajectory_mask[b].sum().item())
            traj = trajectory_ids[b, :traj_len]

            prefix = self._distill_bgkit_prefix_ids
            sentinel = self._distill_bgkit_sentinel_ids
            suffix = self._distill_bgkit_suffix_ids
            seq = torch.cat([issue, prefix, sentinel, suffix, traj], dim=0)
            loss = torch.cat([
                torch.zeros(
                    issue.size(0) + prefix.size(0) + sentinel.size(0) + suffix.size(0),
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.ones(traj.size(0), dtype=torch.bool, device=self.device),
            ], dim=0)
            target_rows.append(seq)
            loss_rows.append(loss)
            splice_starts.append(int(issue.size(0) + prefix.size(0)))
            splice_lens.append(int(sentinel.size(0)))

        max_len = max(int(row.size(0)) for row in target_rows)
        target_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        target_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for b, (seq, loss) in enumerate(zip(target_rows, loss_rows, strict=True)):
            slen = int(seq.size(0))
            target_ids[b, :slen] = seq
            target_mask[b, :slen] = True
            loss_mask[b, :slen] = loss

        return (
            target_ids,
            target_mask,
            loss_mask,
            torch.tensor(splice_starts, dtype=torch.long, device=self.device),
            torch.tensor(splice_lens, dtype=torch.long, device=self.device),
        )

    def _forward_backward(self, batch) -> dict[str, float]:
        """Distillation forward pass with multi-source context and issue tokens.

        Assembles decoder input as
        [issue tokens | bgkit tool-response slot | trajectory tokens].
        Loss is computed only on the trajectory token portion.
        """
        inject = random.random() > self._no_injection_fraction

        trajectory_ids = batch["trajectory_token_ids"].to(self.device)
        trajectory_mask = batch["trajectory_attention_mask"].to(self.device)
        batch_size = trajectory_ids.size(0)

        has_issue = "issue_token_ids" in batch

        if has_issue:
            issue_ids = batch["issue_token_ids"].to(self.device)
            issue_mask = batch["issue_attention_mask"].to(self.device)
            issue_len = issue_ids.size(1)
        else:
            issue_len = 0

        # Determine BgKIT context
        bgkit_context = self._get_bgkit_context(batch) if inject else None

        target_ids, target_mask, loss_mask, splice_starts, splice_lens = (
            self._build_decoder_targets(
                trajectory_ids,
                trajectory_mask,
                issue_ids if has_issue else None,
                issue_mask if has_issue else None,
            )
        )

        context_sources = 0
        if bgkit_context is not None:
            context_embeds, context_mask = bgkit_context
            context_sources = int(context_mask.any(dim=-1).sum().item())
            loss = self.decoder.forward_with_single_splice(
                survivor_embeddings=context_embeds,
                survivor_attention_mask=context_mask,
                token_ids=target_ids,
                token_attention_mask=target_mask,
                splice_starts=splice_starts,
                splice_lengths=splice_lens,
                loss_mask=loss_mask,
            )
        else:
            # Preserve the same in-sequence geometry even when context is absent.
            empty_context = torch.zeros(
                batch_size, 1, self.decoder.hidden_dim,
                dtype=self.decoder.backbone.get_input_embeddings().weight.dtype,
                device=self.device,
            )
            empty_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
            loss = self.decoder.forward_with_single_splice(
                survivor_embeddings=empty_context,
                survivor_attention_mask=empty_mask,
                token_ids=target_ids,
                token_attention_mask=target_mask,
                splice_starts=splice_starts,
                splice_lengths=splice_lens,
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

        for batch in self.eval_dataloader:
            trajectory_ids = batch["trajectory_token_ids"].to(self.device)
            trajectory_mask = batch["trajectory_attention_mask"].to(self.device)
            batch_size = trajectory_ids.size(0)

            has_issue = "issue_token_ids" in batch
            if has_issue:
                issue_ids = batch["issue_token_ids"].to(self.device)
                issue_mask = batch["issue_attention_mask"].to(self.device)
            else:
                issue_ids = None
                issue_mask = None
            target_ids, target_mask, loss_mask, splice_starts, splice_lens = (
                self._build_decoder_targets(
                    trajectory_ids, trajectory_mask, issue_ids, issue_mask,
                )
            )

            # Eval with context
            bgkit_context = self._get_bgkit_context(batch)
            if bgkit_context is not None:
                context_embeds, context_mask = bgkit_context
                loss_ctx = self.decoder.forward_with_single_splice(
                    survivor_embeddings=context_embeds,
                    survivor_attention_mask=context_mask,
                    token_ids=target_ids,
                    token_attention_mask=target_mask,
                    splice_starts=splice_starts,
                    splice_lengths=splice_lens,
                    loss_mask=loss_mask,
                )
                total_loss_with_ctx += loss_ctx.item()
                batches_with_ctx += 1
            else:
                empty_context = torch.zeros(
                    batch_size, 1, self.decoder.hidden_dim,
                    dtype=self.decoder.backbone.get_input_embeddings().weight.dtype,
                    device=self.device,
                )
                empty_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
                loss_no = self.decoder.forward_with_single_splice(
                    survivor_embeddings=empty_context,
                    survivor_attention_mask=empty_mask,
                    token_ids=target_ids,
                    token_attention_mask=target_mask,
                    splice_starts=splice_starts,
                    splice_lengths=splice_lens,
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
