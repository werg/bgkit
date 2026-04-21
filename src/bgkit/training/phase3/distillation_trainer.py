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
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

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


def _make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Build cumulative sequence lengths tensor from a list of lengths."""
    t = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0, out=t[1:])
    return t


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

        max_batch_tokens = int(self.cfg.training.get("max_batch_tokens", 16384))
        # Eval defaults to 2× train budget (no backward → lower peak at
        # same budget). Overridable via training.max_batch_tokens_eval.
        max_batch_tokens_eval = self._resolve_eval_batch_budget(
            self.cfg.training, max_batch_tokens,
        )
        seed = int(self.cfg.get("seed", 42))

        # Build per-sample token lengths (trajectory + issue)
        def _sample_length(idx: int) -> int:
            sample = self.dataset[idx]
            traj_len = (
                int(sample["trajectory_token_ids"].size(0))
                if "trajectory_token_ids" in sample else 0
            )
            issue_len = (
                int(sample["issue_token_ids"].size(0))
                if "issue_token_ids" in sample else 0
            )
            # Add prefix/suffix token overhead
            extra = (
                self._distill_bgkit_prefix_ids.size(0)
                + self._distill_bgkit_suffix_ids.size(0)
            )
            return traj_len + issue_len + extra

        train_indices = list(self.train_dataset.indices)
        eval_indices = list(self.eval_dataset.indices)

        train_lengths = [_sample_length(i) for i in train_indices]
        eval_lengths = [_sample_length(i) for i in eval_indices]

        # Stash for live-tunable budget rebuild (see BaseTrainer._handle_max_batch_tokens)
        self._train_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = self._collate
        self._num_workers = 0
        self._pin_memory = False
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval

        self.train_sampler = PackedTokenBudgetSampler(
            dataset=self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
        )
        eval_sampler = PackedTokenBudgetSampler(
            dataset=self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
            seed=seed,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=0,
            collate_fn=self._collate,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
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
        """Collate trajectory samples into packed format.

        Produces flat tensors with cu_seqlens for trajectory and issue sequences.
        No padding tokens; segmentation lives in cu_seqlens.
        """
        result = {
            "instance_ids": [s["instance_id"] for s in batch],
            "repos": [s["repo"] for s in batch],
            "base_commits": [s["base_commit"] for s in batch],
            "issue_texts": [s["issue_text"] for s in batch],
        }

        if "trajectory_token_ids" in batch[0]:
            traj_seqs = [s["trajectory_token_ids"] for s in batch]
            traj_lengths = [int(t.size(0)) for t in traj_seqs]
            traj_cu = _make_cu_seqlens(traj_lengths)
            traj_total = int(traj_cu[-1])
            result["trajectory_token_ids"] = torch.cat(traj_seqs, dim=0)
            result["trajectory_cu_seqlens"] = traj_cu
            result["trajectory_position_ids"] = position_ids_from_cu(traj_cu, traj_total)
            result["trajectory_max_seqlen"] = max(traj_lengths) if traj_lengths else 0

        if "issue_token_ids" in batch[0]:
            issue_seqs = [s["issue_token_ids"] for s in batch]
            issue_lengths = [int(t.size(0)) for t in issue_seqs]
            issue_cu = _make_cu_seqlens(issue_lengths)
            issue_total = int(issue_cu[-1])
            result["issue_token_ids"] = torch.cat(issue_seqs, dim=0)
            result["issue_cu_seqlens"] = issue_cu
            result["issue_position_ids"] = position_ids_from_cu(issue_cu, issue_total)
            result["issue_max_seqlen"] = max(issue_lengths) if issue_lengths else 0

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
        """Assemble multi-source BgKIT context for the batch — packed format.

        For each sample, concatenates survivors from up to 3 sources:
        1. Compressed filesystem at (repo, base_commit)
        2. Git history commit chains for repo
        3. Prior agentic session embeddings for repo (ordered before base_commit)

        Returns:
            ``(flat_embeddings, survivor_cu_seqlens)`` where
            ``flat_embeddings`` is ``(K_total, D)`` float and
            ``survivor_cu_seqlens`` is ``(B+1,)`` int32, or None if no
            context is available for any sample.
        """
        repos = batch.get("repos", [])
        base_commits = batch.get("base_commits", [])
        if not repos:
            return None

        batch_size = len(repos)
        per_sample: list[torch.Tensor | None] = []
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
                per_sample.append(None)

        if not any_found:
            return None

        # Determine hidden dim from a non-None sample
        hidden_dim = 0
        for s in per_sample:
            if s is not None and s.numel() > 0:
                hidden_dim = s.size(-1)
                break
        if hidden_dim == 0:
            return None

        # Build flat survivors + cu_seqlens; empty samples contribute 0 survivors
        flat_parts: list[torch.Tensor] = []
        lengths: list[int] = []

        dummy = torch.zeros(0, hidden_dim)
        for s in per_sample:
            if s is not None and s.numel() > 0:
                # Ensure 2D (K_i, D)
                if s.ndim == 1:
                    s = s.unsqueeze(-1).expand(-1, hidden_dim)
                flat_parts.append(s)
                lengths.append(s.size(0))
            else:
                flat_parts.append(dummy)
                lengths.append(0)

        flat_embeddings = torch.cat(flat_parts, dim=0).to(self.device)  # (K_total, D)
        survivor_cu = _make_cu_seqlens(lengths).to(self.device)  # (B+1,) int32

        return flat_embeddings, survivor_cu

    def _build_decoder_inputs(
        self,
        batch: dict,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int, torch.Tensor]]]:
        """Build per-sample prefix_ids, suffix_ids, and loss_mask parts for the decoder.

        ``forward_with_single_splice`` lays out each sample as
        ``[prefix_i | survivors_i | suffix_i]``.  We map:

        - ``prefix_ids[i]`` = ``[issue_i | bgkit_tool_prefix]``
        - survivors inserted here as raw embeddings (no placeholder token)
        - ``suffix_ids[i]`` = ``[bgkit_tool_suffix | trajectory_i]``

        Loss is computed only on trajectory tokens (the bgkit wrapper tokens
        inside the suffix are masked out).

        Returns
        -------
        prefix_ids : list[Tensor]
            Length-B list of 1-D int64 tensors.
        suffix_ids : list[Tensor]
            Length-B list of 1-D int64 tensors.
        loss_mask_parts : list[tuple[int, int, Tensor]]
            Per-sample ``(pre_len, suf_len, lm_suf)`` tuples consumed by
            :meth:`_make_flat_loss_mask`.
        """
        has_issue = "issue_token_ids" in batch
        has_traj = "trajectory_token_ids" in batch

        batch_size = len(batch["repos"])
        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        loss_mask_parts: list[tuple[int, int, torch.Tensor]] = []

        bgkit_suffix_len = int(self._distill_bgkit_suffix_ids.size(0))

        issue_cu = batch["issue_cu_seqlens"].tolist() if has_issue else None
        traj_cu = batch["trajectory_cu_seqlens"].tolist() if has_traj else None

        prefix_token = self._distill_bgkit_prefix_ids  # (P_pre,)
        suffix_token = self._distill_bgkit_suffix_ids  # (P_suf,)

        for b in range(batch_size):
            # Issue tokens for this sample (may be empty)
            if has_issue and issue_cu is not None:
                i_start, i_end = int(issue_cu[b]), int(issue_cu[b + 1])
                issue_i = batch["issue_token_ids"][i_start:i_end].to(self.device)
            else:
                issue_i = torch.empty(0, dtype=torch.long, device=self.device)

            # Trajectory tokens for this sample
            if has_traj and traj_cu is not None:
                t_start, t_end = int(traj_cu[b]), int(traj_cu[b + 1])
                traj_i = batch["trajectory_token_ids"][t_start:t_end].to(self.device)
            else:
                traj_i = torch.empty(0, dtype=torch.long, device=self.device)

            # prefix: [issue | bgkit_prefix] — survivors are inserted after this
            pre = torch.cat([issue_i, prefix_token], dim=0)
            # suffix: [bgkit_suffix | traj] — follows the survivors
            suf = torch.cat([suffix_token, traj_i], dim=0)

            prefix_ids.append(pre)
            suffix_ids.append(suf)

            # Loss mask for suffix: False for bgkit wrapper tokens, True for traj
            pre_len = pre.size(0)
            suf_len = suf.size(0)
            lm_suf = torch.zeros(suf_len, dtype=torch.bool, device=self.device)
            lm_suf[bgkit_suffix_len:] = True  # trajectory positions only
            loss_mask_parts.append((pre_len, suf_len, lm_suf))

        return prefix_ids, suffix_ids, loss_mask_parts

    def _make_flat_loss_mask(
        self,
        loss_mask_parts: list[tuple[int, int, torch.Tensor]],
        survivor_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Build the flat (N_total,) loss mask for forward_with_single_splice.

        forward_with_single_splice lays out each sample as
        [prefix_i | survivors_i | suffix_i].  This helper mirrors that layout
        and marks only the trajectory sub-tokens inside each suffix as True.

        Parameters
        ----------
        loss_mask_parts:
            Per-sample ``(pre_len, suf_len, lm_suf)`` tuples from
            ``_build_decoder_inputs``.
        survivor_cu_seqlens:
            ``(B+1,)`` int32 survivor counts, or a zeros tensor when context
            is absent.

        Returns
        -------
        Tensor
            ``(N_total,)`` bool.
        """
        surv_counts = (survivor_cu_seqlens[1:] - survivor_cu_seqlens[:-1]).tolist()
        all_parts: list[torch.Tensor] = []
        for b, (pre_len, _suf_len, lm_suf) in enumerate(loss_mask_parts):
            k_i = int(surv_counts[b])
            pre_zeros = torch.zeros(pre_len, dtype=torch.bool, device=self.device)
            surv_zeros = torch.zeros(k_i, dtype=torch.bool, device=self.device)
            all_parts.extend([pre_zeros, surv_zeros, lm_suf])
        return torch.cat(all_parts, dim=0)

    def _forward_backward(self, batch) -> dict[str, float]:
        """Distillation forward pass with multi-source context and issue tokens.

        Assembles decoder input as
        [issue tokens | bgkit tool-response slot | trajectory tokens].
        Loss is computed only on the trajectory token portion.
        """
        inject = random.random() > self._no_injection_fraction

        batch_size = len(batch["repos"])
        issue_len = int(batch["issue_cu_seqlens"][-1]) if "issue_cu_seqlens" in batch else 0

        # Determine BgKIT context (packed)
        bgkit_context = self._get_bgkit_context(batch) if inject else None

        prefix_ids, suffix_ids, loss_mask_parts = self._build_decoder_inputs(batch)

        if bgkit_context is not None:
            flat_embeddings, survivor_cu = bgkit_context
            context_sources = int(
                (survivor_cu[1:] - survivor_cu[:-1]).gt(0).sum().item()
            )
        else:
            # Empty survivors: zero-length for every sample
            zero_lengths = [0] * batch_size
            survivor_cu = _make_cu_seqlens(zero_lengths).to(self.device)
            embed_dtype = self.decoder.backbone.get_input_embeddings().weight.dtype
            flat_embeddings = torch.zeros(0, self.decoder.hidden_dim, dtype=embed_dtype,
                                          device=self.device)
            context_sources = 0

        flat_loss_mask = self._make_flat_loss_mask(loss_mask_parts, survivor_cu)

        loss = self.decoder.forward_with_single_splice(
            survivor_embeddings=flat_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=flat_loss_mask,
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
            batch_size = len(batch["repos"])

            prefix_ids, suffix_ids, loss_mask_parts = self._build_decoder_inputs(batch)

            # Eval with context
            bgkit_context = self._get_bgkit_context(batch)
            if bgkit_context is not None:
                flat_embeddings, survivor_cu = bgkit_context
                flat_loss_mask = self._make_flat_loss_mask(loss_mask_parts, survivor_cu)
                loss_ctx = self.decoder.forward_with_single_splice(
                    survivor_embeddings=flat_embeddings,
                    survivor_cu_seqlens=survivor_cu,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    loss_mask=flat_loss_mask,
                )
                total_loss_with_ctx += loss_ctx.item()
                batches_with_ctx += 1
            else:
                zero_lengths = [0] * batch_size
                survivor_cu = _make_cu_seqlens(zero_lengths).to(self.device)
                embed_dtype = self.decoder.backbone.get_input_embeddings().weight.dtype
                flat_embeddings = torch.zeros(
                    0, self.decoder.hidden_dim, dtype=embed_dtype, device=self.device,
                )
                flat_loss_mask = self._make_flat_loss_mask(loss_mask_parts, survivor_cu)
                loss_no = self.decoder.forward_with_single_splice(
                    survivor_embeddings=flat_embeddings,
                    survivor_cu_seqlens=survivor_cu,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                    loss_mask=flat_loss_mask,
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
