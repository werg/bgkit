"""bgkit2 on the project's BaseTrainer.

Everything operational is inherited and NOT reimplemented: the CUDA
memory-fraction cap and peer-CUDA guard (scripts/train.py), the adaptive
cache flush and system-abort threshold, the step watchdog and faulthandler
self-dump, graceful SIGTERM saves, the checkpoint registry / NVMe fast dir /
HDD archive, auto-resume with model fingerprinting, live config, and the
rep-dependence-reported check. The numerics live in :class:`BgKIT2Core`.

Hydra wiring: ``training.phase: bgkit2``; the bgkit2 dataclass config sits
under ``training.bgkit2`` (see configs/training/bgkit2.yaml).
"""

from __future__ import annotations

import random
from typing import ClassVar

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from bgkit.training.base_trainer import BaseTrainer

from ..data.autoencode import Collator
from ..data.templates import Templates, set_no_think_template
from ..models.loading import load_tokenizer
from .config import Config, _validate
from .core import BgKIT2Core

try:  # project structlog wrapper
    from bgkit.utils.logging import get_logger

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover
    import structlog

    logger = structlog.get_logger(__name__)



def _release_host_arenas() -> None:
    """Return freed glibc heap pages to the OS. Setup stages large host buffers (checkpoint
    state dicts, dataset tables); once freed they stay mapped in malloc arenas, and on unified
    memory every page of RSS is a page CUDA cannot allocate -- worse, forked dataloader workers
    inherit them (2026-09-26: 20 GB Shared_Dirty in the 8B-A trainer and its worker)."""
    import contextlib
    import ctypes
    import gc

    gc.collect()
    with contextlib.suppress(OSError):  # non-glibc host
        ctypes.CDLL("libc.so.6").malloc_trim(0)

class _Pair(nn.Module):
    """Container so ``self.model`` names what this trainer owns."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def bgkit2_config_from_hydra(cfg) -> Config:
    sub = cfg.training.get("bgkit2", None)
    base = OmegaConf.structured(Config)
    if sub is not None:
        base = OmegaConf.merge(base, sub)
    OmegaConf.resolve(base)
    out: Config = OmegaConf.to_object(base)  # type: ignore[assignment]
    _validate(out)
    return out


class BgKIT2Trainer(BgKIT2Core, BaseTrainer):
    SPLICES_REPS: ClassVar[bool] = True
    _use_device_prefetcher = False
    _log_every = 10

    # Live tuning via ${CHECKPOINT_DIR}/control.json (review item 8: every old
    # compression trainer exposed its curriculum; bgkit2 exposed nothing, so a
    # stalling descent could only be answered by a restart). Put run-specific
    # knobs in the RUN-NAME block — the `bgkit2` phase block is inherited by
    # every bgkit2 run (feedback_control_json_per_phase_namespace).
    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "anchor_weight": "_live_anchor_weight",
        "anchor_floor": "_live_anchor_floor",
        "kd_weight": "_live_kd_weight",
        "ce_weight": "_live_ce_weight",
        "selection_weight": "_live_selection_weight",
        "selection_every": "_live_selection_every",
        "kd_every": "_live_kd_every",
        "p_random_selection": "_live_p_random_selection",
        "head_scale_anchor_weight": "_live_head_scale_anchor_weight",
        "ratio_rate_log2_per_step": "_live_ratio_rate",
        "ratio_rate_power": "_live_ratio_rate_power",
        "ratio_mult_min": "_live_ratio_mult_min",
        "mastery_target_gap": "_live_mastery_target_gap",
        "ratio_ceiling_steps": "_live_ratio_ceiling_steps",
    }

    def __init__(self, cfg) -> None:
        BaseTrainer.__init__(self, cfg)
        self.step_cfg = cfg.training
        self.cfg2 = bgkit2_config_from_hydra(cfg)
        self.device = torch.device("cpu")
        self.amp_dtype = torch.float32

    @property
    def current_step(self) -> int:
        return int(self.global_step)

    # ------------------------------------------------------------------
    def setup(self) -> None:
        cfg2 = self.cfg2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and cfg2.run.dtype == "bfloat16")
            else torch.float32
        )
        seed = int(self.cfg.get("seed", 42))
        self.tok = load_tokenizer(cfg2.model.decoder_id)
        if cfg2.model.decoder_no_think:
            set_no_think_template(self.tok)
        # two-tokenizer runs: the encoder's (context) tokenizer; None = self.tok serves both
        self.ctx_tok = (
            load_tokenizer(cfg2.model.ctx_tokenizer_id) if cfg2.model.ctx_tokenizer_id else None
        )
        self.templates = Templates.from_tokenizer(self.tok, style=cfg2.data.prompt_style)
        if self.ctx_tok is not None:
            # Losses that pair encoder position i with decoder token i are undefined when the
            # vocabularies differ: refuse them instead of training on silently misaligned ids.
            bad = [n for n, on in (
                ("anchor.weight", float(cfg2.anchor.weight) > 0),
                ("model.rep_token_residual", bool(cfg2.model.rep_token_residual)),
                ("model.rep_token_residual_only", bool(cfg2.model.rep_token_residual_only)),
                ("kd.selection_weight", float(cfg2.kd.selection_weight) > 0),
            ) if on]
            if bad:
                raise ValueError(
                    f"model.ctx_tokenizer_id is set (encoder and decoder tokenizers differ); "
                    f"these position-paired options must be off: {bad}")
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        enc_src = cfg2.model.init_encoder_from
        if enc_src:
            self._load_encoder_init(enc_src)
        if cfg2.model.init_head_from:
            self._load_head_init(cfg2.model.init_head_from)
        if cfg2.model.init_decoder_from:
            self._load_decoder_init(cfg2.model.init_decoder_from)
        layers = self.finish_models()
        self.teacher = self.build_teacher()
        if self.teacher is not None:
            logger.info("bgkit2_kd_teacher_loaded", path=cfg2.kd.teacher_checkpoint)
        gc_default = bool(self.step_cfg.get("gradient_checkpointing", True))
        # Separate switches: the decoder's activations dominate (P19b OOM'd at
        # 73 GB with both off on the shared unified pool); a frozen encoder's
        # recompute buys little and costs a full extra forward.
        if bool(self.step_cfg.get("gradient_checkpointing_encoder", gc_default)):
            self.encoder.backbone.gradient_checkpointing_enable()
        if bool(self.step_cfg.get("gradient_checkpointing_decoder", gc_default)):
            self.decoder.gradient_checkpointing_enable()
        self.model = _Pair(self.encoder, self.decoder)
        self.model.train()
        self.register_checkpoint_source("encoder", enc_src or None)
        self.register_checkpoint_source("decoder", None)
        self.register_startup_extra("encoder_id", cfg2.model.encoder_id)
        self.register_startup_extra("decoder_id", cfg2.model.decoder_id)
        self.register_startup_extra("trainable_backbone_layers", str(layers))
        self.register_startup_extra("prompt_style", cfg2.data.prompt_style)
        self.register_startup_extra("task_stores", str(list(cfg2.data.task_stores)))

        self.collate = Collator(
            self.templates,
            enc_pad_id=None if self.ctx_tok is None else self.chat_renderer().enc_pad_id,
        )
        self.train_ds, self.eval_ds, self.task_eval = self.build_datasets(seed)
        self.tree_eval_ds = None
        self.tree_dataloader = None
        self._tree_iter = None
        if cfg2.data.repo_tree_store:
            from transformers import AutoTokenizer

            from ..data.repo_tree import RepoTreeDataset, collate_trees

            self.enc_tok = AutoTokenizer.from_pretrained(cfg2.model.encoder_id)
            tree_kw = dict(
                max_files=cfg2.data.repo_tree_max_files,
                file_tokens=cfg2.data.repo_tree_file_tokens,
                # PATH WALK: one item per step of the walk through the ENCODED tree
                walk_depth=int(cfg2.multilevel.max_depth) if cfg2.multilevel.walk else 0,
                # node encodings come from the cache: do not build file tensors
                # -- unless small trees are encoded LIVE through the encoder
                # adapter (multilevel.live_tree_max_files)
                load_files=(not cfg2.data.repo_tree_root_cache
                            or int(cfg2.multilevel.live_tree_max_files) > 0),
            )
            tree_train = RepoTreeDataset(cfg2.data.repo_tree_store, "train", **tree_kw)
            self.tree_train_ds = tree_train
            self.tree_eval_ds = RepoTreeDataset(cfg2.data.repo_tree_store, "eval", **tree_kw)
            every = int(cfg2.data.repo_tree_every)
            if every > 0:
                # Stage D: trees INTERLEAVED with the corpus/task mix every k steps.
                self.tree_dataloader = DataLoader(
                    tree_train, batch_size=cfg2.data.batch_size, shuffle=True,
                    collate_fn=collate_trees, num_workers=0, drop_last=True,
                )
            elif every == 0:
                # Stage C: the training set IS the trees.
                self.train_ds = tree_train
                self.collate = collate_trees
            # every < 0: trees are EVAL-ONLY (the agentic run trains the walk
            # inside agent episodes; bare tree batches taught it to skip tools)
            self.tree_eval_only = every < 0
            logger.info(
                "repo_tree_datasets",
                train=len(tree_train), eval=len(self.tree_eval_ds),
                max_files=cfg2.data.repo_tree_max_files, every=every,
            )
            if cfg2.data.repo_tree_root_cache:
                self.load_tree_root_cache(cfg2.data.repo_tree_root_cache)
                # a PARTIAL cache is fine: train and evaluate on the repos it holds
                cached = self.tree_node_cache.repos
                n_tr, n_ev = len(tree_train), len(self.tree_eval_ds)
                if any(r["repo"] not in cached for r in tree_train.rows) or any(
                    r["repo"] not in cached for r in self.tree_eval_ds.rows
                ):
                    logger.warning(
                        "repo_tree_restricted_to_cached_repos",
                        train=f"{n_tr}->{tree_train.restrict_to(cached)}",
                        eval=f"{n_ev}->{self.tree_eval_ds.restrict_to(cached)}",
                        cached_repos=len(cached),
                    )
        nw = int(cfg2.data.num_workers)
        _release_host_arenas()  # before the workers fork, so they do not inherit freed arenas
        if int(cfg2.data.length_bucket_window) > 0:
            from ..data.samplers import LengthBucketBatchSampler

            sampler = LengthBucketBatchSampler(
                len(self.train_ds),
                cfg2.data.batch_size,
                self.train_ds.length_hint,
                window_batches=int(cfg2.data.length_bucket_window),
                seed=seed,
                max_tokens=int(cfg2.data.max_batch_tokens) or None,
            )
            self.train_dataloader = DataLoader(
                self.train_ds,
                batch_sampler=sampler,
                collate_fn=self.collate,
                num_workers=nw,
                persistent_workers=nw > 0,
                prefetch_factor=4 if nw > 0 else None,
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_ds,
                batch_size=cfg2.data.batch_size,
                shuffle=False,  # each index is its own random draw; order is already i.i.d.
                collate_fn=self.collate,
                num_workers=nw,
                persistent_workers=nw > 0,
                prefetch_factor=4 if nw > 0 else None,
                drop_last=True,
            )
        self.eval_dataloader = None  # evaluation is owned by BgKIT2Core

        self._build_multilevel_teacher()
        groups = self.param_groups()
        self.optimizer = self._create_optimizer(groups, float(self.step_cfg.lr))
        self.ratio_schedule = self.build_ratio_schedule(seed)
        self.select_rng = random.Random(seed + 1)
        logger.info(
            "bgkit2_setup_complete",
            trainable_params=int(sum(p.numel() for g in groups for p in g["params"])),
            param_groups=[g["name"] for g in groups],
            trainable_backbone_layers=layers,
            train_samples=len(self.train_ds),
            task_stores=list(self.task_eval.keys()),
        )

    def _build_multilevel_teacher(self) -> None:
        """Stage B needs a FROZEN SNAPSHOT of the encoder as its target.

        Not the live encoder under no_grad: that target drifts as the student
        trains, and the student can then satisfy the loss by moving both sides
        together. bgkit1 Step 4.7 loaded a checkpoint for the same reason.
        """
        from pathlib import Path

        from bgkit.training.checkpointing import load_checkpoint

        m = self.cfg2.multilevel
        self.encoder_teacher = None
        if not m.enabled:
            return
        src = m.teacher_checkpoint or self.cfg2.model.init_encoder_from
        if not src:
            raise ValueError("multilevel.enabled needs teacher_checkpoint or init_encoder_from")
        teacher = self.build_encoder()
        _meta, state = load_checkpoint(Path(src))
        enc_state = state.get("encoder")
        if not enc_state:
            raise ValueError(f"{src} carries no encoder state")
        self.load_encoder_state(teacher, enc_state)
        teacher.interface_norm.set_target_row_norm(self.decoder.embed_row_norm)
        teacher.interface_norm.set_target_moments(self.decoder.embed_tokens.weight)
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        self.encoder_teacher = teacher.to(self.device).eval()
        logger.info(
            "multilevel_teacher_loaded",
            path=src,
            teacher_ratio=float(m.teacher_ratio),
            params=int(sum(p.numel() for p in teacher.parameters())),
        )

    def _load_encoder_init(self, path: str) -> None:
        from pathlib import Path

        from bgkit.training.checkpointing import load_checkpoint

        _meta, state = load_checkpoint(Path(path))
        enc_state = state.get("encoder")
        if not enc_state:
            raise ValueError(f"{path} carries no encoder state")
        drop = tuple(self.cfg2.model.init_drop_prefixes)
        tolerated = self.load_encoder_state(self.encoder, enc_state, drop_prefixes=drop)
        logger.info(
            "bgkit2_encoder_warm_started",
            path=path,
            absent_buffers=[k for k in tolerated if not k.startswith(drop)] if drop else tolerated,
            dropped_prefixes=list(drop),
        )
        self._reseed_interface_if_needed(path)

    def _load_head_init(self, path: str) -> None:
        from pathlib import Path

        from bgkit.training.checkpointing import load_checkpoint

        _meta, state = load_checkpoint(Path(path))
        enc_state = state.get("encoder") or {}
        n = self.load_head_state(self.encoder, enc_state)
        logger.info("bgkit2_head_warm_started", path=path, tensors=n)

    def _load_decoder_init(self, path: str) -> None:
        from pathlib import Path

        from bgkit.training.checkpointing import load_checkpoint

        _meta, state = load_checkpoint(Path(path))
        dec_state = state.get("decoder")
        if not dec_state:
            raise ValueError(f"{path} carries no decoder state")
        self.load_decoder_state(dec_state, source=f"warm start {path}")
        logger.info("bgkit2_decoder_warm_started", path=path, tensors=len(dec_state))

    def _migrate_state_dict(self, name: str, state: dict) -> dict:
        """Resume: an encoder state saved without an adapter this run installs
        (encoder_mode adapter, compaction spine) loads by the same key mapping
        as a warm start -- peft's ``base_layer`` rename, and adapter copies
        seeded from the LOADED base -- instead of a partial load that would
        leave the wrapped base weights at their pretrained values."""
        if name == "encoder":
            return self._remap_adapter_keys(self.encoder, state)
        return state

    def _reseed_interface_if_needed(self, source: str) -> None:
        if any([n.reseed_if_reference_out_of_contract() for n in self.encoder.interface_norms()]):
            logger.warning(
                "interface_reference_reseeded",
                source=source,
                reason="prenorm on; loaded running_var describes un-normalised inputs",
            )

    def _post_weight_load_hook(self) -> None:
        super()._post_weight_load_hook()
        self._reseed_interface_if_needed(str(getattr(self, "_last_checkpoint_path", "")))

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _dynamic_ckpt_managed_models(self) -> list[tuple[str, nn.Module]]:
        # Flush-only: the encoder forward drives HF layers directly and the
        # mode-flip machinery has not been validated on LFM2 blocks.
        return []

    def _clip_gradients(self, max_norm: float) -> float:
        """Skip a step whose gradients are non-finite instead of raising.

        P4 and P7 died on a single NaN gradient after hundreds of healthy
        steps. Zero the grads (the optimizer step then only applies momentum)
        and count; raise only if it keeps happening."""
        gn = super()._clip_gradients(max_norm)
        if not (gn == gn and abs(gn) != float("inf")):
            self._nonfinite_steps = getattr(self, "_nonfinite_steps", 0) + 1
            self._nonfinite_streak = getattr(self, "_nonfinite_streak", 0) + 1
            logger.warning(
                "nonfinite_grad_skipped",
                step=int(self.global_step),
                total=self._nonfinite_steps,
                streak=self._nonfinite_streak,
            )
            if self._nonfinite_streak > 20:
                raise RuntimeError("non-finite gradients on 20 consecutive steps")
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0
        self._nonfinite_streak = 0
        return gn

    def _apply_encoder_freeze(self) -> bool:
        """optim.encoder_freeze_steps: encoder params take no gradient before
        that step (AdamW skips params without a grad, so no decay either);
        restored exactly once at the boundary. Returns whether frozen."""
        n = int(self.cfg2.optim.encoder_freeze_steps)
        if n <= 0:
            return False
        frozen = int(self.global_step) < n
        state = getattr(self, "_encoder_frozen", None)
        if state is None:
            # remember which encoder params are trainable by policy
            self._encoder_trainable = [p for p in self.encoder.parameters() if p.requires_grad]
            state = not frozen  # force the first transition to apply
        if frozen != state:
            for p in self._encoder_trainable:
                p.requires_grad_(not frozen)
            logger.info(
                "encoder_freeze_phase",
                step=int(self.global_step),
                frozen=frozen,
                params=len(self._encoder_trainable),
                until=n,
            )
            self._encoder_frozen = frozen
        return frozen

    def _next_tree_batch(self):
        if self._tree_iter is None:
            self._tree_iter = iter(self.tree_dataloader)
        try:
            return next(self._tree_iter)
        except StopIteration:
            self._tree_iter = iter(self.tree_dataloader)
            return next(self._tree_iter)

    def _forward_backward(self, batch) -> dict[str, float]:
        self.encoder.train()
        self.decoder.train()
        self._apply_encoder_freeze()
        every = int(self.cfg2.data.repo_tree_every)
        if self.tree_dataloader is not None and every > 0 and int(self.global_step) % every == 0:
            batch = self._next_tree_batch()  # a tree step in place of a corpus step
        total, stats = self.training_loss(batch)
        (total / max(1, int(self._accum_steps))).backward()
        sel = self.utility_selection_backward()
        if sel is not None:
            stats["loss/selection"] = sel
            stats["utility_teacher_rate"] = float(getattr(self, "_utility_teacher_rate", 0.0))
        pc = self.prompt_contrast_backward()
        if pc is not None:
            stats.update(pc)
        return stats

    @torch.no_grad()
    def evaluate_tree(self, max_items: int = 512, max_gen: int = 128, ds=None,
                      tag: str = "tree", max_walks: int = 96) -> dict[str, float]:
        """Tree eval on the repo-disjoint split, by what the model PRODUCES as
        well as by CE. Teacher-forced CE against two floors (no slot = the
        honest one; zeroed reps are out of distribution and ~1.8 nats worse),
        greedy generation scored exactly, and -- for the PATH WALK -- step
        accuracy by depth plus the full greedy walk from the root to a file."""
        ds = self.tree_eval_ds if ds is None else ds
        self.encoder.eval()
        self.decoder.eval()
        walk = int(getattr(ds, "walk_depth", 0)) > 0
        tot = tot0 = totn = 0.0
        n_tok = 0
        n_gen = 0
        em = same_dir = same_base = in_repo = 0.0
        by_depth: dict[int, list[float]] = {}
        dec = getattr(self.tok, "decode", None)

        def prompt_ids(t):
            q_text, a_text = self.tree_prompt(t)
            q = self.tok(q_text, add_special_tokens=False)["input_ids"]
            a = self.tok(a_text, add_special_tokens=False)["input_ids"]
            return (
                [torch.tensor(q, dtype=torch.long, device=self.device)],
                [torch.tensor(a, dtype=torch.long, device=self.device)],
                a,
            )

        def generate(pre, reps, mask, n_new: int) -> tuple[list[int], str]:
            empty = [torch.zeros(0, dtype=torch.long, device=self.device)]
            with self.autocast():
                prompt = self.decoder.build_batch(
                    pre, reps, mask, empty, pad_side="left", loss_on_suffix=False
                )
                gen = self.decoder.generate(prompt, max_new_tokens=n_new)
            ids = gen[0].tolist()
            text = dec(ids, skip_special_tokens=True) if dec is not None else ""
            return ids, text

        # Rows are grouped by repo: "the first N rows" is a handful of repos
        # (P31's first 128 rows covered 10 of its 19 held-out repos, P30's 17).
        # Take an even stride over the whole split; cached nodes make rows cheap.
        n_all = len(ds)
        k = min(max_items, n_all)
        picks = sorted({round(j * (n_all - 1) / max(k - 1, 1)) for j in range(k)})
        gen_every = max(1, len(picks) // max_gen)
        for pi, i in enumerate(picks):
            t = ds[i]
            r, mk = self.tree_root(t)
            pre, suf, a = prompt_ids(t)
            reps, mask = r[None], mk[None]
            with self.autocast():
                ce = float(self.decoder.forward(self.decoder.build_batch(pre, reps, mask, suf)))
                ce0 = float(self.decoder.forward(
                    self.decoder.build_batch(pre, torch.zeros_like(reps), mask, suf)
                ))
                cen = float(self.decoder.forward(self.decoder.build_batch(pre, None, None, suf)))
            tot += ce * len(a)
            tot0 += ce0 * len(a)
            totn += cen * len(a)
            n_tok += len(a)
            if pi % gen_every or n_gen >= max_gen:
                continue
            ids, text = generate(pre, reps, mask, len(a) + 8)
            n_gen += 1
            hit = float(ids[: len(a)] == list(a))
            em += hit
            if walk:
                by_depth.setdefault(0 if not t.node else t.node.count("/") + 1, []).append(hit)
            if dec is None or walk:
                continue
            got = text.split()[0] if text.split() else ""
            want = t.answer
            same_dir += float(got.rsplit("/", 1)[0] == want.rsplit("/", 1)[0] and "/" in got)
            same_base += float(got.rsplit("/", 1)[-1] == want.rsplit("/", 1)[-1])
            in_repo += float(got in set(t.paths))
        n_tok = max(n_tok, 1)
        g = max(n_gen, 1)
        out = {
            "eval/tree/nav_ce": tot / n_tok,
            "eval/tree/nav_ce_zeroed": tot0 / n_tok,
            "eval/tree/nav_gain_nats": (tot0 - tot) / n_tok,
            "eval/tree/nav_ce_noslot": totn / n_tok,
            "eval/tree/nav_gain_vs_noslot": (totn - tot) / n_tok,
            "eval/tree/gen_rows": float(n_gen),
        }
        if not walk:
            out.update({
                "eval/tree/gen_path_exact": em / g,
                "eval/tree/gen_same_dir": same_dir / g,
                "eval/tree/gen_same_basename": same_base / g,
                "eval/tree/gen_is_a_repo_path": in_repo / g,
            })
        else:
            out["eval/tree/walk_step_exact"] = em / g
            for d, hits in sorted(by_depth.items()):
                out[f"eval/tree/walk_step_exact_depth{d}"] = sum(hits) / len(hits)
            if dec is not None:
                out.update(self._evaluate_full_walks(ds, generate, max_walks=max_walks))
        if tag != "tree":
            out = {k.replace("eval/tree/", f"eval/{tag}/", 1): v for k, v in out.items()}
        return out

    def _evaluate_tree_train_slice(self) -> dict[str, float]:
        """The same tree eval on a slice of TRAINING repos (eval/tree_train/*).
        A repo's cached encoding is a fixed input asked about thousands of
        times: the adapter can use it as an ID for a memorised file list. Only
        the held-out repos show READING; the gap to this slice is the measure
        of remembering (P30: train CE 0.5 vs held-out 3.18)."""
        train = getattr(self, "tree_train_ds", None)
        if train is None:
            return {}
        return self.evaluate_tree(max_items=128, max_gen=64, ds=train, tag="tree_train")

    def _evaluate_full_walks(self, ds, generate, max_walks: int) -> dict[str, float]:
        """GREEDY walks from the root: the model's own choice names the next
        node. A walk succeeds iff it ends on the answer file; it fails as soon
        as it names something that is not a child of the current node."""
        from ..models.multilevel import node_children

        seen: set[int] = set()
        questions: list[int] = []
        for ri, _node, _t in ds.steps:
            if ri not in seen:
                seen.add(ri)
                questions.append(ri)
        stride = max(1, len(questions) // max_walks)
        questions = questions[::stride][:max_walks]
        ok = invalid = 0.0
        lens: list[int] = []
        kids_cache: dict[str, dict[str, set[str]]] = {}
        for wi, ri in enumerate(questions):
            if wi and wi % 64 == 0:  # standalone evals (scripts/bgkit2_eval_walk.py) need progress
                logger.info("tree_eval_walk_progress", done=wi, of=len(questions), exact=ok / wi)
            row = ds.rows[ri]
            all_paths, _ = ds._files(row)
            paths = list(all_paths[: ds.max_files])
            kids = kids_cache.setdefault(row["repo"], node_children(paths, ds.walk_depth))
            node, n_steps, result = "", 0, None
            while n_steps < 12 and result is None:
                n_steps += 1
                t = type("WalkItem", (), {"repo": row["repo"], "node": node,
                                          "question": row["question"], "answer": ""})()
                q_text, _ = self.tree_prompt(t)
                q = self.tok(q_text, add_special_tokens=False)["input_ids"]
                pre = [torch.tensor(q, dtype=torch.long, device=self.device)]
                r, mk = self.tree_root(t)
                _ids, text = generate(pre, r[None], mk[None], 40)
                got = text.split("\n", 1)[0].strip()
                if got not in kids.get(node, set()):
                    result = "invalid"
                elif got.endswith("/"):
                    node = f"{node}/{got[:-1]}" if node else got[:-1]
                else:
                    result = "ok" if (f"{node}/{got}" if node else got) == row["answer"] else "wrong"
            ok += float(result == "ok")
            invalid += float(result in ("invalid", None))
            lens.append(n_steps)
        n = max(len(questions), 1)
        return {
            "eval/tree/walk_full_exact": ok / n,
            "eval/tree/walk_invalid_step": invalid / n,
            "eval/tree/walk_mean_steps": sum(lens) / n,
            "eval/tree/walk_questions": float(len(questions)),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        tree_ds = getattr(self, "tree_eval_ds", None)
        if (
            tree_ds is not None
            and getattr(self, "tree_dataloader", None) is None
            and not getattr(self, "tree_eval_only", False)
        ):
            # Stage C: the trees ARE the run; nothing else to evaluate.
            metrics = self.evaluate_tree()
            metrics["eval/loss"] = metrics["eval/tree/nav_ce"]
            metrics.update(self._evaluate_tree_train_slice())
            if self.cfg2.multilevel.prompted_readout:
                # CONTROL: the same read-out pass WITHOUT the question. If it
                # does as well, the prompt is inert and the gain is just the
                # extra trained pass.
                self.readout_prompt_mode = "none"
                try:
                    metrics.update(self.evaluate_tree(tag="tree_noprompt"))
                finally:
                    self.readout_prompt_mode = "question"
            return metrics
        metrics = self.evaluate_all()
        if tree_ds is not None:
            # Stage D: trees are interleaved with the task mix, so the eval
            # reports BOTH -- the tree gain and every task metric the mix is
            # supposed to preserve. eval/loss stays the task eval's (the
            # pruning metric and mastery logic key on the task side).
            metrics.update(self.evaluate_tree())
            metrics.update(self._evaluate_tree_train_slice())
        self._maybe_release_ratio_hold(metrics)
        self._update_mastery(metrics)
        return metrics

    # -- live-config handlers (validated; each logs old -> new) --------------
    def _live_set(self, key: str, obj, attr: str, val, lo=None, hi=None, kind=float) -> None:
        if not isinstance(val, (int, float)):
            logger.warning("live_config_type_error", key=key, value=val, expected="numeric")
            return
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            logger.warning("live_config_range_error", key=key, value=val, lo=lo, hi=hi)
            return
        old = getattr(obj, attr)
        setattr(obj, attr, kind(val))
        logger.info("live_config_update", key=key, old=old, new=getattr(obj, attr))

    def _live_anchor_weight(self, v):
        self._live_set("anchor_weight", self.cfg2.anchor, "weight", v, lo=0.0)

    def _live_anchor_floor(self, v):
        self._live_set("anchor_floor", self.cfg2.anchor, "floor", v, lo=0.0)

    def _live_kd_weight(self, v):
        self._live_set("kd_weight", self.cfg2.kd, "weight", v, lo=0.0)

    def _live_ce_weight(self, v):
        self._live_set("ce_weight", self.cfg2.kd, "ce_weight", v, lo=0.0)

    def _live_selection_weight(self, v):
        self._live_set("selection_weight", self.cfg2.kd, "selection_weight", v, lo=0.0)

    def _live_kd_every(self, v):
        self._live_set("kd_every", self.cfg2.kd, "kd_every", v, lo=1, kind=int)

    def _live_selection_every(self, v):
        self._live_set("selection_every", self.cfg2.kd, "selection_every", v, lo=1, kind=int)

    def _live_p_random_selection(self, v):
        self._live_set("p_random_selection", self.cfg2.ratio, "p_random_selection", v, 0.0, 1.0)

    def _live_head_scale_anchor_weight(self, v):
        self._live_set("head_scale_anchor_weight", self.cfg2.kd, "head_scale_anchor_weight", v, 0.0)

    def _live_ratio_rate(self, v):
        self._live_set(
            "ratio_rate_log2_per_step", self.ratio_schedule, "rate_log2_per_step", v, 0.0
        )

    def _live_ratio_rate_power(self, v):
        self._live_set("ratio_rate_power", self.ratio_schedule, "rate_power", v, 0.0, 2.0)

    def _live_ratio_mult_min(self, v):
        self._live_set("ratio_mult_min", self.ratio_schedule, "mult_min", v, 0.0, 1.0)

    def _live_mastery_target_gap(self, v):
        self._live_set("mastery_target_gap", self.cfg2.ratio, "mastery_target_gap", v, 0.0)

    def _live_ratio_ceiling_steps(self, v):
        self._live_set("ratio_ceiling_steps", self.ratio_schedule, "ceiling_steps", v, 0, kind=int)

    def _post_optimizer_step(self, step: int) -> None:
        super()._post_optimizer_step(step)
        self.ratio_schedule.advance(int(step))

    def _update_mastery(self, metrics: dict[str, float]) -> None:
        r = self.cfg2.ratio
        if not r.adaptive:
            return
        fam = r.mastery_family
        k_floor, k_full = f"eval/xfloor/{fam}/loss", f"eval/full_text/{fam}/loss"
        if k_floor not in metrics or k_full not in metrics:
            logger.warning("ratio_mastery_metric_missing", family=fam)
            return
        gap = float(metrics[k_floor]) - float(metrics[k_full])
        mult = 1.0 if gap <= float(r.mastery_target_gap) else float(r.mastery_target_gap) / gap
        mult = self.ratio_schedule.set_mastery(mult)
        step = int(self.global_step)
        logger.info(
            "ratio_mastery",
            step=step,
            floor=self.ratio_schedule.floor(step),
            gap=gap,
            target_gap=r.mastery_target_gap,
            multiplier=mult,
            steps_per_doubling=self.ratio_schedule.steps_per_doubling(step),
            progress=self.ratio_schedule.progress,
        )

    def _maybe_release_ratio_hold(self, metrics: dict[str, float]) -> None:
        r = self.cfg2.ratio
        step = int(self.global_step)
        if not r.hold_gate_metric or step >= self.ratio_schedule.hold_steps:
            return
        if r.hold_gate_metric not in metrics:
            logger.warning("ratio_hold_gate_metric_missing", metric=r.hold_gate_metric)
            return
        value = float(metrics[r.hold_gate_metric])
        ref = float(metrics[r.hold_gate_reference]) if r.hold_gate_reference else 0.0
        gap = value - ref
        if gap <= float(r.hold_gate_max_gap) and self.ratio_schedule.release_hold(step):
            logger.warning(
                "ratio_hold_released",
                step=step,
                metric=r.hold_gate_metric,
                value=value,
                reference=ref,
                gap=gap,
                max_gap=r.hold_gate_max_gap,
            )
        else:
            logger.info("ratio_hold_gate_open", step=step, gap=gap, max_gap=r.hold_gate_max_gap)

    # ------------------------------------------------------------------
    def checkpoint_models(self) -> dict[str, nn.Module]:
        if self.cfg2.model.decoder_save_trainable_only:
            from ..models.decoder import TrainableDecoderState

            view = self.__dict__.get("_trainable_decoder_view")
            if view is None or view.decoder is not self.decoder:
                view = self.__dict__["_trainable_decoder_view"] = TrainableDecoderState(self.decoder)
            return {"encoder": self.encoder, "decoder": view}
        return {"encoder": self.encoder, "decoder": self.decoder}

    def _build_training_state(self, *args, **kwargs) -> dict:
        state = super()._build_training_state(*args, **kwargs)
        state["bgkit2_ratio_schedule"] = self.ratio_schedule.state_dict()
        guard = getattr(self, "splice_guard", None)
        if guard is not None and guard.ema_mean is not None:
            state["bgkit2_splice_guard_ema"] = guard.ema_mean.cpu().tolist()
        # JSON-friendly: (version, [ints...], gauss_next) — restored as tuples.
        v, vec, g = self.select_rng.getstate()
        state["bgkit2_select_rng"] = [v, [int(x) for x in vec], g]
        return state

    def _restore_training_state(self, training_state: dict) -> None:
        ema = training_state.get("bgkit2_splice_guard_ema")
        if ema is not None:
            from bgkit_core.splice_guard import SpliceGuard

            self.splice_guard = SpliceGuard()
            self.splice_guard.ema_mean = torch.tensor(ema, device=self.device)
        rs = training_state.get("bgkit2_ratio_schedule")
        if rs is not None:
            self.ratio_schedule.load_state_dict(rs)
        sr = training_state.get("bgkit2_select_rng")
        if sr is not None:
            v, vec, g = sr
            self.select_rng.setstate((int(v), tuple(int(x) for x in vec), g))
