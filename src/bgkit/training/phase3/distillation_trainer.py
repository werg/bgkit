"""Phase 3 distillation trainer: SWE-bench trajectory distillation.

Imitates external teacher-agent trajectories in the student (0.8B + BgKIT).
Compressed filesystem context is implemented end-to-end. Optional git-history
and prior-session manifests are supported by the consumer but disabled until
their producers are supplied; prior sessions require real timestamps.

Training input: [issue description | bgkit tool-response slot | filtered teacher trajectory]
Loss: CE on trajectory tokens (tool calls, code edits, reasoning)
~30% examples without BgKIT injection (baseline preservation)

Cache directory layout (under ``bgkit_cache_dir``):
- ``filesystem/manifest.parquet`` — keyed by (repo, base_commit), ``path`` column
  points to per-(repo, commit) ``survivors.npy`` files
- ``git_history/manifest.parquet`` — keyed by (repo, base_commit), ``path``
  column points to history truncated at that commit
- ``prior_sessions/manifest.parquet`` — keyed by repo, one row per trajectory
  with ``timestamp``/``trajectory_timestamp`` and a ``path`` column
"""

from __future__ import annotations

import random
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bgkit.data.datasets.swe_trajectory_dataset import SWETrajectoryDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder, normalize_decoder_family
from bgkit.training.base_trainer import BaseTrainer
from bgkit.utils.attention_backend import resolve_decoder_attention_implementation
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


def _timestamp_seconds(value) -> float | None:
    """Normalize Unix or ISO-8601 timestamps; return None when unavailable."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            UTC,
        ).timestamp()
    except ValueError:
        return None


class _SurvivorTensorCache:
    """Bounded process-local cache that removes repeated ``np.load`` I/O."""

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._bytes = 0

    def get(self, path: str) -> torch.Tensor | None:
        cached = self._items.pop(path, None)
        if cached is not None:
            self._items[path] = cached
            return cached
        tensor = _load_survivors_from_path(path)
        if tensor is None:
            return None
        size = tensor.numel() * tensor.element_size()
        if self.max_bytes > 0 and size <= self.max_bytes:
            while self._items and self._bytes + size > self.max_bytes:
                _old_path, old = self._items.popitem(last=False)
                self._bytes -= old.numel() * old.element_size()
            self._items[path] = tensor
            self._bytes += size
        return tensor


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
        self._manifest_dir = manifest_path.parent
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

    def resolve_path(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = self._manifest_dir / path
        return str(path)


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
        cache_gib = float(cfg.training.get("survivor_cache_gib", 2.0))
        self._survivor_cache = _SurvivorTensorCache(int(cache_gib * 1024**3))

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load decoder (student model)
        decoder_cfg = self.cfg.model.decoder
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
        )
        decoder_name = decoder_cfg.backbone_name
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=decoder_attention_impl,
        )
        decoder_backbone.to(self.device)
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_backbone.get_input_embeddings().weight.shape[1],
            decoder_family=decoder_family,
        )
        self.decoder.set_lm_ce_impl(
            self.cfg.training.get(
                "decoder_ce_impl",
                self.cfg.compute.get("decoder_ce_impl", None),
            )
        )
        self.decoder.set_lm_ce_strict(
            self.cfg.training.get(
                "decoder_ce_strict",
                self.cfg.compute.get("decoder_ce_strict", None),
            )
        )
        logger.info(
            "phase3_decoder_ce_impl_selected",
            impl=self.decoder.lm_ce_impl,
            strict=self.decoder.lm_ce_strict,
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
            self._load_encoder(self._resolve_phase2_checkpoint(str(phase2_ckpt)))

        # Load multi-source context caches
        self._fs_cache: _ContextSourceCache | None = None
        self._git_cache: _ContextSourceCache | None = None
        self._session_cache: _ContextSourceCache | None = None

        enabled_sources = [
            str(v) for v in self.cfg.training.get(
                "context_sources", ["filesystem"],
            )
        ]
        valid_sources = {"filesystem", "git_history", "prior_sessions"}
        unknown_sources = set(enabled_sources) - valid_sources
        if unknown_sources:
            raise ValueError(f"Unknown Phase-3 context sources: {sorted(unknown_sources)}")

        if enabled_sources and not self._bgkit_cache_dir:
            raise ValueError(
                "Phase 3 context_sources are enabled but bgkit_cache_dir is unset"
            )
        if self._bgkit_cache_dir:
            cache_root = Path(str(self._bgkit_cache_dir))

            # Source 1: compressed filesystem — keyed by (repo, base_commit)
            fs_manifest = cache_root / "filesystem" / "manifest.parquet"
            # Compatibility with the pre-2026-07 encoder output. New writes are
            # always canonicalized under filesystem/.
            legacy_fs_manifest = cache_root / "manifest.parquet"
            if not fs_manifest.exists() and legacy_fs_manifest.exists():
                fs_manifest = legacy_fs_manifest
                logger.warning(
                    "phase3_legacy_filesystem_manifest",
                    path=str(fs_manifest),
                    hint="re-run encode_swe_repos.py to migrate to filesystem/",
                )
            if "filesystem" in enabled_sources and fs_manifest.exists():
                self._fs_cache = _ContextSourceCache(
                    fs_manifest, key_columns=["repo", "base_commit"],
                )

            # Source 2: git history — keyed by repo
            git_manifest = cache_root / "git_history" / "manifest.parquet"
            if "git_history" in enabled_sources and git_manifest.exists():
                self._git_cache = _ContextSourceCache(
                    git_manifest, key_columns=["repo", "base_commit"],
                )

            # Source 3: prior sessions — keyed by repo
            session_manifest = cache_root / "prior_sessions" / "manifest.parquet"
            if "prior_sessions" in enabled_sources and session_manifest.exists():
                self._session_cache = _ContextSourceCache(
                    session_manifest, key_columns=["repo"],
                )

            loaded_by_name = {
                "filesystem": self._fs_cache,
                "git_history": self._git_cache,
                "prior_sessions": self._session_cache,
            }
            missing_sources = [
                source for source in enabled_sources
                if loaded_by_name[source] is None
            ]
            if missing_sources:
                raise FileNotFoundError(
                    f"Enabled Phase-3 context manifests are missing under "
                    f"{cache_root}: {missing_sources}"
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
        # Eval defaults to 2x train budget (no backward -> lower peak at
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

    def _resolve_phase2_checkpoint(self, configured: str) -> str:
        """Resolve ``auto`` to the best completed Stage-B Phase-2 checkpoint."""
        if configured != "auto":
            return configured

        from bgkit.training.checkpoint_registry import CheckpointRegistry

        checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
        registry = CheckpointRegistry(checkpoint_dir)
        registry.backfill(checkpoint_dir)
        requested_run = self.cfg.training.get("phase2_run_name", None)
        candidates = []
        for entry in registry.list_entries(phase="phase2_kb", status="completed"):
            if not entry.on_disk:
                continue
            if requested_run is not None and entry.run_name != str(requested_run):
                continue
            snapshot = entry.config_snapshot or {}
            if str(snapshot.get("stage", "")).upper() != "B":
                continue
            metrics = entry.metrics or {}
            loss = metrics.get("eval/loss", metrics.get("eval/eval/loss"))
            if loss is not None:
                candidates.append((float(loss), -int(entry.step), entry))
        if not candidates:
            scope = f" for run {requested_run!r}" if requested_run else ""
            raise ValueError(
                "phase2_checkpoint=auto found no completed Stage-B phase2_kb "
                f"checkpoint with eval/loss{scope}"
            )
        entry = min(candidates, key=lambda item: (item[0], item[1]))[2]
        return str(checkpoint_dir / entry.name)

    def _load_encoder(self, checkpoint_path: str) -> None:
        """Load BgKIT encoder from Phase 2 checkpoint."""
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.training.checkpointing import load_checkpoint

        logger.info("phase3_loading_encoder", checkpoint=checkpoint_path)
        _metadata, state_dicts = load_checkpoint(Path(checkpoint_path))
        model_state = state_dicts.get("model", {}) or {}
        encoder_state = state_dicts.get("encoder") or {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items() if k.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                f"Phase-2 checkpoint {checkpoint_path} contains no encoder state"
            )
        if encoder_state:
            ctrl_src = self.cfg.model.get("threshold_controller", {})
            default_ratio = 0.10
            threshold_controller_cfg = {
                "init_theta": float(
                    ctrl_src.get("init_theta", 1.0 - 2.0 * default_ratio),
                ),
                "lr": float(ctrl_src.get("lr", 0.02)),
                "momentum": float(ctrl_src.get("momentum", 0.0)),
                "clamp": float(ctrl_src.get("clamp", 0.99)),
                "anchor_ratios": list(ctrl_src.get("anchor_ratios", [])) or None,
                "ratio_space": str(ctrl_src.get("ratio_space", "log")),
                "init_target_ratio": default_ratio,
                "default_query_ratio": default_ratio,
            }
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                self.cfg.model.get("encoder", {}).get(
                    "backbone_name", "Qwen/Qwen3.5-0.8B-Base",
                ),
                encoder_state,
                hidden_dim=int(
                    self.cfg.model.get("encoder", {}).get("hidden_dim", 1024),
                ),
                threshold_controller_cfg=threshold_controller_cfg,
            )
            self.encoder.to(self.device).eval()
            self.encoder.requires_grad_(False)
            self.register_checkpoint_source("phase2_encoder", checkpoint_path)

    def _collate(self, batch: list[dict]) -> dict:
        """Collate trajectory samples into packed format.

        Produces flat tensors with cu_seqlens for trajectory and issue sequences.
        No padding tokens; segmentation lives in cu_seqlens.
        """
        result = {
            "instance_ids": [s["instance_id"] for s in batch],
            "repos": [s["repo"] for s in batch],
            "base_commits": [s["base_commit"] for s in batch],
            "trajectory_timestamps": [s.get("trajectory_timestamp") for s in batch],
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
        before_timestamp: float | None = None,
    ) -> torch.Tensor | None:
        """Load survivors from a context source cache for one sample.

        Args:
            cache: The context source cache to query.
            *key_values: Key values for the cache lookup (e.g. repo, base_commit).
            before_timestamp: For prior sessions, include only rows with a
                real timestamp strictly before the current trajectory.

        Returns:
            Concatenated survivors tensor or None if unavailable.
        """
        if cache is None:
            return None
        rows = cache.lookup(*key_values)
        if not rows:
            return None

        if before_timestamp is not None:
            filtered = []
            for row in rows:
                row_ts = _timestamp_seconds(
                    row.get("trajectory_timestamp", row.get("timestamp")),
                )
                if row_ts is not None and row_ts < before_timestamp:
                    filtered.append(row)
            rows = filtered
            if not rows:
                return None

        parts = []
        for row in rows:
            path = row.get("path")
            if not path:
                continue
            resolved = cache.resolve_path(str(path))
            tensor_cache = getattr(self, "_survivor_cache", None)
            survivors = (
                tensor_cache.get(resolved)
                if tensor_cache is not None
                else _load_survivors_from_path(resolved)
            )
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
        trajectory_timestamps = batch.get("trajectory_timestamps", [])
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
            git_key_columns = getattr(self._git_cache, "_key_columns", [])
            git_survivors = self._load_source_survivors(
                self._git_cache,
                *(
                    (repo, commit)
                    if "base_commit" in git_key_columns
                    else (repo,)
                ),
            )
            if git_survivors is not None:
                sources.append(git_survivors)

            # Source 3: prior session survivors (only those before current base_commit)
            current_ts = _timestamp_seconds(
                trajectory_timestamps[i]
                if i < len(trajectory_timestamps) else None,
            )
            session_survivors = (
                self._load_source_survivors(
                    self._session_cache, repo, before_timestamp=current_ts,
                )
                if current_ts is not None else None
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

        decoder_dtype = self.decoder.backbone.get_input_embeddings().weight.dtype
        flat_embeddings = torch.cat(flat_parts, dim=0).to(
            device=self.device, dtype=decoder_dtype,
        )  # (K_total, D)
        survivor_cu = _make_cu_seqlens(lengths).to(self.device)  # (B+1,) int32

        return flat_embeddings, survivor_cu

    def _build_decoder_inputs(
        self,
        batch: dict,
        *,
        include_bgkit_slot: bool = True,
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

        bgkit_suffix_len = (
            int(self._distill_bgkit_suffix_ids.size(0))
            if include_bgkit_slot else 0
        )

        issue_cu = batch["issue_cu_seqlens"].tolist() if has_issue else None
        traj_cu = batch["trajectory_cu_seqlens"].tolist() if has_traj else None

        if include_bgkit_slot:
            prefix_token = self._distill_bgkit_prefix_ids
            suffix_token = self._distill_bgkit_suffix_ids
        else:
            prefix_token = torch.empty(0, dtype=torch.long, device=self.device)
            suffix_token = torch.empty(0, dtype=torch.long, device=self.device)

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

        prefix_ids, suffix_ids, loss_mask_parts = self._build_decoder_inputs(
            batch, include_bgkit_slot=inject,
        )

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
        total_batches = 0
        samples_with_ctx = 0
        total_samples = 0

        for batch in self.eval_dataloader:
            batch_size = len(batch["repos"])
            total_samples += batch_size

            # Paired ablation: evaluate the same examples twice. The no-context
            # arm omits both survivor vectors and the synthetic BgKIT wrapper,
            # matching the training-time baseline-preservation examples.
            prefix_ctx, suffix_ctx, mask_parts_ctx = self._build_decoder_inputs(
                batch, include_bgkit_slot=True,
            )
            prefix_no, suffix_no, mask_parts_no = self._build_decoder_inputs(
                batch, include_bgkit_slot=False,
            )

            bgkit_context = self._get_bgkit_context(batch)
            if bgkit_context is not None:
                flat_embeddings, survivor_cu = bgkit_context
                samples_with_ctx += int(
                    (survivor_cu[1:] - survivor_cu[:-1]).gt(0).sum().item()
                )
            else:
                embed_dtype = self.decoder.backbone.get_input_embeddings().weight.dtype
                flat_embeddings = torch.zeros(
                    0, self.decoder.hidden_dim, dtype=embed_dtype, device=self.device,
                )
                survivor_cu = _make_cu_seqlens([0] * batch_size).to(self.device)

            ctx_mask = self._make_flat_loss_mask(mask_parts_ctx, survivor_cu)
            loss_ctx = self.decoder.forward_with_single_splice(
                survivor_embeddings=flat_embeddings,
                survivor_cu_seqlens=survivor_cu,
                prefix_ids=prefix_ctx,
                suffix_ids=suffix_ctx,
                loss_mask=ctx_mask,
            )

            zero_cu = _make_cu_seqlens([0] * batch_size).to(self.device)
            embed_dtype = self.decoder.backbone.get_input_embeddings().weight.dtype
            empty_embeddings = torch.zeros(
                0, self.decoder.hidden_dim, dtype=embed_dtype, device=self.device,
            )
            no_mask = self._make_flat_loss_mask(mask_parts_no, zero_cu)
            loss_no = self.decoder.forward_with_single_splice(
                survivor_embeddings=empty_embeddings,
                survivor_cu_seqlens=zero_cu,
                prefix_ids=prefix_no,
                suffix_ids=suffix_no,
                loss_mask=no_mask,
            )
            total_loss_with_ctx += loss_ctx.item()
            total_loss_no_ctx += loss_no.item()
            total_batches += 1

        self.model.train()

        metrics = {
            "eval/loss": total_loss_with_ctx / max(total_batches, 1),
            "eval/loss_with_context": total_loss_with_ctx / max(total_batches, 1),
            "eval/loss_no_context": total_loss_no_ctx / max(total_batches, 1),
            "eval/context_delta": (
                total_loss_no_ctx - total_loss_with_ctx
            ) / max(total_batches, 1),
            "eval/context_coverage": samples_with_ctx / max(total_samples, 1),
        }
        return metrics
