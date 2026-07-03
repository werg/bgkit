"""Round-robin Qwen+Falcon summarization training.

Holds both decoders simultaneously, routes each microbatch through one
family (round-robin or weighted-random), and co-evolves the encoder +
both projection_blocks + both decoders in a single training run.

Architecture decisions:

- The encoder's source tokens (Qwen-tokenized) are family-agnostic. Per
  microbatch, the trainer swaps ``encoder.set_active_decoder_family`` and
  runs through the chosen decoder.
- Two-level packing on the encoder side: each sample is a "group" of
  source documents. Each doc is its own attention segment
  (``content_cu_seqlens`` marks per-doc boundaries). Survivors are then
  aggregated per GROUP for the decoder splice — the decoder sees one
  concatenated survivor span per sample.
- Chat template is built per-microbatch using the existing
  ``tokenize_with_sentinel`` helper. The "content" passed in is the
  family-specific target summary tokens; the helper returns a full
  prefix+content+suffix tokenization with loss-masked content span.
  We then extract ``prefix_ids = full[:splice]`` and
  ``suffix_ids = full[splice+splice_len:]`` for
  ``decoder.forward_with_single_splice``.
- Loss is decoder CE on the target span only.
- Curriculum: linear ratio ramp ``start → end`` over ``ramp_steps``,
  live-tunable via ``control.json``. Easy-start defaults
  (ratio_start=0.9) per the realign-run lesson.
- Checkpoint: ``encoder.pt``, ``decoder_qwen.pt``, ``decoder_falcon.pt``,
  ``optimizer_state_by_name.pt``. Decoder LoRA not used here.

Forward-carry note: the mixin-style decoder setup is meant to be
re-usable across Phase 2 KR and Phase 3 distillation. When those
trainers are built, factor out the dual-decoder setup + checkpoint
handling into ``MultiDecoderTrainerMixin``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.chat_template import (
    TOOL_CONFIGS,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.summarization_dataset import SummarizationGroupDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import (
    CheckpointMetadata,
    load_checkpoint,
)
from bgkit.training.compression_curriculum import (
    CURRICULUM_LIVE_CONFIG_FIELDS,
    CompressionCurriculumMixin,
    linear_ratio,
)
from bgkit.training.gradient_utils import (
    maybe_enable_decoder_gradient_checkpointing,
    maybe_enable_gradient_checkpointing,
)
from bgkit.utils.attention_backend import (
    resolve_attention_implementation,
    resolve_decoder_attention_implementation,
)
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


# Each summarization dataset name in the data layout maps to the variant
# bank + TOOL_CONFIGS entry used to build per-sample chat templates.
_DATASET_TO_BANK: dict[str, str] = {
    "multi_news": "multi_news_summarization",
    "arxiv_s2orc": "scientific_abstract",
    "pmc_oa_md": "scientific_abstract",
}


class _FlatIndexDataset:
    """Minimal Dataset adapter exposing a list of flat indices.

    ``PackedTokenBudgetSampler`` calls ``__len__`` to size epochs and
    treats batch contents as opaque — our collator turns each flat idx
    into the real ``(dataset_name, row_idx)`` lookup.
    """

    def __init__(self, indices: np.ndarray):
        self._indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int) -> int:
        return int(self._indices[i])


def _build_chat_inputs_for_sample(
    tokenizer,
    variant: dict[str, str],
    config,
    *,
    group_id: str,
    target_ids: np.ndarray,
    device: torch.device,
    encoder_tokenizer=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (prefix_ids, suffix_ids, loss_mask) for one summarization sample.

    Uses the existing sentinel-based chat template helper with the target
    tokens passed in as ``content_token_ids`` — semantically, for
    summarization the "content the assistant produces" IS the target
    summary. The helper returns full token IDs + a loss mask covering
    the content span. We then split full ids into
    ``prefix = full[:bgkit_splice_start]`` and
    ``suffix = full[bgkit_splice_start + bgkit_splice_len:]`` so the
    decoder's single-splice forward can insert the survivor embeddings
    at the right spot.
    """
    target_t = torch.as_tensor(target_ids, dtype=torch.long)
    out = tokenize_with_sentinel(
        tokenizer,
        variant,
        config,
        file_path=group_id,
        language="markdown",
        content_token_ids=target_t,
        encoder_tokenizer=encoder_tokenizer,
    )
    token_ids: torch.Tensor = out["token_ids"]
    loss_mask: torch.Tensor = out["loss_mask"]
    splice_start = int(out["bgkit_splice_start"].item())
    splice_len = int(out["bgkit_splice_len"].item())
    prefix = token_ids[:splice_start]
    # Decoder forward_with_single_splice expects the suffix to contain
    # all post-splice tokens (template tail + content + post-content tail).
    # The full sequence layout is:
    #     [prefix | <sentinel splice region> | response_prefix_tail | content | template_tail]
    # tokenize_with_sentinel layout maps that to:
    #     full = prefix_ids + content_ids + suffix_ids  (assistant content REPLACES sentinel)
    #     bgkit_splice_start = position in prefix_ids of the sentinel
    # So the "tail" after the splice region in OUR sequence is everything
    # from splice_start+splice_len onward. Note: tokenize_with_sentinel
    # writes the content (= target summary) AT a later position than the
    # sentinel — it concatenates prefix_ids + content_ids + suffix_ids
    # where content_ids replaces the sentinel-containing region.
    # The suffix below contains the response_prefix tokens + target +
    # template_tail; loss mask preserves the per-token target mask the
    # helper built.
    suffix = token_ids[splice_start + splice_len:]
    suffix_mask = loss_mask[splice_start + splice_len:].to(torch.bool)
    # The ChatML-wrapped compression prompt the decoder's bgkit tool-call
    # advertises (its `prompt` argument) — fed to the encoder so the
    # compression is actually conditioned on it (query-aware), instead of the
    # encoder seeing an empty prompt. Tokenized with the ENCODER tokenizer.
    compression_prompt_ids = out["compression_prompt_ids"].to(torch.long)
    return (
        prefix.to(device),
        suffix.to(device),
        suffix_mask.to(device),
        compression_prompt_ids.to(device),
    )


class SummarizationRoundRobinTrainer(CompressionCurriculumMixin, BaseTrainer):
    """Round-robin Qwen+Falcon decoder training on summarization corpora.

    Inherits closed-loop survivorship control (dual-ascent θ + aux losses)
    from :class:`CompressionCurriculumMixin` — without it the encoder's
    keep-rate drifts off the curriculum target with nothing to correct it
    (the 2026-06-08 over-compression regression).
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        **CURRICULUM_LIVE_CONFIG_FIELDS,
        "qwen_decoder_prob": "_qwen_decoder_prob",
    }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto"),
        )

        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        # --- Resolve + load encoder state dict FIRST so the threshold cfg
        #     can be derived from the saved anchor grid. ---
        step1_checkpoint = self._resolve_step1_checkpoint()
        if step1_checkpoint is None:
            raise ValueError(
                "summarization_round_robin requires step1_checkpoint "
                "(merged ckpt with encoder + both projections + both decoders).",
            )
        logger.info("loading_step1_checkpoint", path=step1_checkpoint)
        _meta, state_dicts = load_checkpoint(Path(step1_checkpoint))
        if "encoder" not in state_dicts:
            raise ValueError(f"step1 ckpt missing 'encoder': {step1_checkpoint}")

        # --- Derive threshold cfg from the saved anchor grid ---
        model_cfg = self.cfg.model
        ctrl_src = tcfg.get("model", {}).get(
            "threshold_controller", model_cfg.get("threshold_controller", {}),
        )
        anchor_t = state_dicts["encoder"].get("l0.threshold.anchor_ratios")
        anchors = (
            anchor_t.tolist()
            if anchor_t is not None
            else (list(ctrl_src.get("anchor_ratios", [])) or None)
        )
        threshold_cfg = {
            "init_theta": float(ctrl_src.get("init_theta", 0.0)),
            "lr": float(ctrl_src.get("lr", 0.05)),
            "momentum": float(ctrl_src.get("momentum", 0.0)),
            "clamp": float(ctrl_src.get("clamp", 1.5)),
            "anchor_ratios": anchors,
            "ratio_space": str(ctrl_src.get("ratio_space", "log")),
            "init_target_ratio": float(tcfg.get("target_ratio_start", 0.9)),
            "default_query_ratio": float(tcfg.get("target_ratio_start", 0.9)),
            "kernel_bandwidth": (
                float(ctrl_src["kernel_bandwidth"])
                if ctrl_src.get("kernel_bandwidth") is not None
                else None
            ),
        }

        # --- Build encoder ---
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name,
            state_dicts.pop("encoder"),
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            threshold_controller_cfg=threshold_cfg,
        )
        self.encoder.to(device)
        self.encoder.requires_grad_(True)
        self.encoder.train()
        maybe_enable_gradient_checkpointing(self.encoder.l0.backbone, self.cfg)
        if getattr(self.encoder, "l1", None) is not None:
            maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)

        # --- Build BOTH decoders ---
        qwen_cfg = tcfg.get("decoder_qwen", {})
        falcon_cfg = tcfg.get("decoder_falcon", {})
        if not qwen_cfg or not falcon_cfg:
            raise ValueError(
                "summarization_round_robin requires training.decoder_qwen + "
                "training.decoder_falcon config blocks.",
            )
        qwen_name = qwen_cfg["backbone_name"]
        falcon_name = falcon_cfg["backbone_name"]
        decoder_attention_qwen = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family="qwen35",
        )
        decoder_attention_falcon = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family="falcon_h1",
        )

        logger.info("loading_decoder", family="qwen35", model=qwen_name)
        qwen_backbone = AutoModelForCausalLM.from_pretrained(
            qwen_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=decoder_attention_qwen,
            device_map=device,
        )
        self.decoder_qwen = ReconstructionDecoder(
            qwen_backbone,
            hidden_dim=int(qwen_backbone.get_input_embeddings().weight.shape[1]),
            decoder_family="qwen35",
        )

        logger.info("loading_decoder", family="falcon_h1", model=falcon_name)
        falcon_backbone = AutoModelForCausalLM.from_pretrained(
            falcon_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=decoder_attention_falcon,
            device_map=device,
        )
        self.decoder_falcon = ReconstructionDecoder(
            falcon_backbone,
            hidden_dim=int(falcon_backbone.get_input_embeddings().weight.shape[1]),
            decoder_family="falcon_h1",
        )

        # Load per-family decoder weights from the merged step1 ckpt OR
        # per-family override paths in config.
        qwen_dec_source: str | None = None
        falcon_dec_source: str | None = None
        if "decoder_qwen" in state_dicts:
            self.decoder_qwen.load_state_dict(state_dicts.pop("decoder_qwen"))
            qwen_dec_source = step1_checkpoint
            logger.info("loaded_decoder_from_step1", family="qwen35")
        elif qwen_cfg.get("checkpoint"):
            ck = Path(qwen_cfg["checkpoint"])
            logger.info("loading_qwen_decoder_ckpt", path=str(ck))
            _, q_state = load_checkpoint(ck)
            sd = q_state.get("decoder_merged", q_state.get("decoder"))
            if sd is None:
                raise ValueError(
                    f"qwen decoder ckpt {ck} missing 'decoder'/'decoder_merged'",
                )
            self.decoder_qwen.load_state_dict(sd)
            qwen_dec_source = str(ck)
        if "decoder_falcon" in state_dicts:
            self.decoder_falcon.load_state_dict(state_dicts.pop("decoder_falcon"))
            falcon_dec_source = step1_checkpoint
            logger.info("loaded_decoder_from_step1", family="falcon_h1")
        elif falcon_cfg.get("checkpoint"):
            ck = Path(falcon_cfg["checkpoint"])
            logger.info("loading_falcon_decoder_ckpt", path=str(ck))
            _, f_state = load_checkpoint(ck)
            sd = f_state.get("decoder_merged", f_state.get("decoder"))
            if sd is None:
                raise ValueError(
                    f"falcon decoder ckpt {ck} missing 'decoder'/'decoder_merged'",
                )
            self.decoder_falcon.load_state_dict(sd)
            falcon_dec_source = str(ck)

        self.decoder_qwen.requires_grad_(True)
        self.decoder_qwen.train()
        self.decoder_falcon.requires_grad_(True)
        self.decoder_falcon.train()
        maybe_enable_decoder_gradient_checkpointing(self.decoder_qwen.backbone, self.cfg)
        maybe_enable_decoder_gradient_checkpointing(self.decoder_falcon.backbone, self.cfg)
        self.model = self.encoder

        # --- Tokenizers ---
        encoder_tokenizer_name = bgkit_cfg.get(
            "tokenizer_name", "Qwen/Qwen3.5-0.8B-Base",
        )
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_tokenizer_name, trust_remote_code=True,
        )
        self.tokenizer_qwen = AutoTokenizer.from_pretrained(
            qwen_name, trust_remote_code=True,
        )
        self.tokenizer_falcon = AutoTokenizer.from_pretrained(
            falcon_name, trust_remote_code=True,
        )

        # --- Variant banks ---
        data_cfg = tcfg.data
        variant_dir = Path(data_cfg.get("prompt_variants_dir", "configs/prompt_variants"))
        self._variant_banks: dict[str, list[dict]] = {}
        # Discover datasets from the data config below; for now, eagerly
        # load all banks we know about so per-sample lookups are O(1).
        for bank_key in set(_DATASET_TO_BANK.values()):
            bank_path = variant_dir / f"{bank_key}.json"
            if not bank_path.exists():
                raise FileNotFoundError(f"variant bank missing: {bank_path}")
            with open(bank_path) as f:
                self._variant_banks[bank_key] = json.load(f)

        # --- Datasets per family per dataset name ---
        dirs_cfg = data_cfg.get("summarization_dirs")
        if dirs_cfg is None:
            raise ValueError(
                "training.data.summarization_dirs required; map family → "
                "{dataset_name: processed_dir}.",
            )
        max_src = int(tcfg.get("max_total_source_tokens", 8192))
        max_tgt = int(tcfg.get("max_target_tokens", 1024))
        self._family_datasets: dict[str, dict[str, SummarizationGroupDataset]] = {}
        for family, per_dataset in dirs_cfg.items():
            self._family_datasets[family] = {}
            for ds_name, path in per_dataset.items():
                logger.info(
                    "loading_summarization_dir",
                    family=family, dataset=ds_name, path=path,
                )
                self._family_datasets[family][ds_name] = SummarizationGroupDataset(
                    path,
                    max_total_source_tokens=max_src,
                    max_target_tokens=max_tgt,
                )

        if set(self._family_datasets.keys()) != {"qwen35", "falcon_h1"}:
            raise ValueError(
                "summarization_dirs must have exactly 'qwen35' and 'falcon_h1' keys.",
            )
        q_keys = set(self._family_datasets["qwen35"].keys())
        f_keys = set(self._family_datasets["falcon_h1"].keys())
        if q_keys != f_keys:
            raise ValueError(
                f"qwen35 datasets {q_keys} != falcon_h1 datasets {f_keys}; "
                "need parallel coverage.",
            )
        if not q_keys:
            raise ValueError("no summarization datasets configured")
        self._dataset_names = sorted(q_keys)
        # Build group_id intersection per dataset_name. Parallel
        # Qwen / Falcon tokenizations can drop different examples under
        # per-doc length filtering (different tokenizer = different
        # token counts → different overflow boundary), so row counts and
        # row identities don't necessarily match. Intersect on group_id
        # to find samples present in BOTH families.
        flat: list[tuple[str, int, int]] = []  # (dataset_name, qwen_row, falcon_row)
        for ds_name in self._dataset_names:
            q_map = self._family_datasets["qwen35"][ds_name].group_id_to_row()
            f_map = self._family_datasets["falcon_h1"][ds_name].group_id_to_row()
            common = sorted(set(q_map.keys()) & set(f_map.keys()))
            n_q = len(self._family_datasets["qwen35"][ds_name])
            n_f = len(self._family_datasets["falcon_h1"][ds_name])
            logger.info(
                "dataset_intersection",
                dataset=ds_name,
                qwen_rows=n_q, falcon_rows=n_f, common=len(common),
            )
            for gid in common:
                flat.append((ds_name, q_map[gid], f_map[gid]))
        if not flat:
            raise ValueError("group_id intersection across families is empty")
        self._flat_index = flat
        n_total = len(flat)
        lengths = np.empty(n_total, dtype=np.int64)
        for flat_idx, (ds_name, q_row, _f_row) in enumerate(flat):
            lengths[flat_idx] = self._family_datasets["qwen35"][ds_name].lengths[q_row]
        self._flat_lengths = lengths

        # Train / eval split (random permutation, deterministic seed).
        max_eval = int(tcfg.get("max_eval_samples", 500))
        eval_n = min(max_eval, max(1, int(n_total * 0.05)))
        rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        perm = rng.permutation(n_total)
        eval_idx = perm[:eval_n]
        train_idx = perm[eval_n:]
        # If ``sort_samples_ascending: true``, present training samples in
        # ascending order of source length (smaller groups first). Compression
        # gives us headroom on the bigger groups later. Defaults true for the
        # safety-first first run.
        if bool(tcfg.get("sort_samples_ascending", True)):
            train_idx = train_idx[np.argsort(lengths[train_idx])]
            logger.info(
                "summarization_train_sorted_ascending",
                n=len(train_idx),
                shortest=int(lengths[train_idx[0]]) if len(train_idx) else 0,
                longest=int(lengths[train_idx[-1]]) if len(train_idx) else 0,
            )
        self._train_flat_idx = train_idx
        self._eval_flat_idx = eval_idx

        max_batch_tokens = int(tcfg.get("max_batch_tokens", 4096))
        max_batch_tokens_eval = int(tcfg.get("max_batch_tokens_eval", max_batch_tokens))
        num_workers = int(self.cfg.compute.get("num_workers", 0))
        pin_memory = bool(self.cfg.compute.get("pin_memory", False))

        self.train_dataset = _FlatIndexDataset(self._train_flat_idx)
        self.eval_dataset = _FlatIndexDataset(self._eval_flat_idx)
        train_lengths = lengths[self._train_flat_idx]
        eval_lengths = lengths[self._eval_flat_idx]

        # When sort_ascending is on, disable sampler shuffle so the
        # ascending order is preserved.
        sort_asc = bool(tcfg.get("sort_samples_ascending", True))
        # Sampler bucketing. ``quantile`` (the sampler default) groups samples
        # into length-quantile buckets and feeds them ONE BUCKET AT A TIME —
        # consecutive optimizer steps come from the same narrow length band.
        # This corpus is sharply BIMODAL (≈95% arxiv at the 8192 cap, ≈5%
        # multi_news at ~2k tokens), so quantile buckets cleanly separate
        # "long arxiv" from "short multi_news" and the run traverses them in
        # huge blocks: thousands of all-long steps, then thousands of all-short
        # steps (diagnosed 2026-06-18 — the "survivor cliff" at step ~19730 was
        # the sampler crossing from the arxiv bucket into the multi_news bucket,
        # NOT a compression failure). That makes the per-step length
        # distribution non-stationary and couples the step-indexed compression
        # curriculum to whichever length band the sampler happens to be in.
        # ``none`` = a single global shuffled pool, so every optimizer step sees
        # a random long/short mix (stationary survivor counts, curriculum
        # decoupled from length). Safe here: under FA4 packed/varlen the budget
        # is sum(L_i^2), and at max_batch_tokens=500 almost every sample is an
        # oversized singleton anyway, so bucketing buys little and costs the
        # block-wise non-stationarity above.
        bucket_mode = str(tcfg.get("sampler_bucket_mode", "none"))
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset, lengths=train_lengths,
            max_batch_tokens=max_batch_tokens, shuffle=not sort_asc,
            bucket_mode=bucket_mode,
            bucket_shuffle=not sort_asc,
            seed=int(self.cfg.get("seed", 42)),
        )
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset, lengths=eval_lengths,
            max_batch_tokens=max_batch_tokens_eval, shuffle=False,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_sampler=self.train_sampler,
            collate_fn=self._collate,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_sampler=eval_sampler,
            collate_fn=self._collate,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # --- Curriculum (live-tunable) ---
        # Per-level. L0 and L1 each compress, so end-to-end compression
        # is approximately ratio_l0 * ratio_l1 of input.
        self._target_ratio_start = float(tcfg.get("target_ratio_start", 0.9))
        self._target_ratio_end = float(tcfg.get("target_ratio_end", 0.10))
        self._target_ratio_ramp_steps = int(tcfg.get("target_ratio_ramp_steps", 20000))
        self._target_ratio_l1_start = float(
            tcfg.get("target_ratio_l1_start", self._target_ratio_start),
        )
        self._target_ratio_l1_end = float(
            tcfg.get("target_ratio_l1_end", self._target_ratio_end),
        )
        self._target_ratio_l1_ramp_steps = int(
            tcfg.get("target_ratio_l1_ramp_steps", self._target_ratio_ramp_steps),
        )
        # When to turn L1 on. None / 0 = on from step 0; positive N = L0-only
        # for the first N steps to let L0 stabilize, then add L1.
        self._l1_introduction_step = int(tcfg.get("l1_introduction_step", 0))
        self._qwen_decoder_prob = float(tcfg.get("qwen_decoder_prob", 0.5))
        self._microbatch_counter = 0

        # --- Closed-loop survivorship control (CompressionCurriculumMixin) ---
        # Dual-ascent θ holds the encoder keep-rate at the curriculum target;
        # min-survivors aux loss (per training.survivorship) guards the
        # per-sample tail. Without this the keep-rate drifts off-target.
        self._init_survivorship_state(
            surv_cfg=tcfg.get("survivorship"),
            ice_cfg=tcfg.get("ice_distillation"),
        )
        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 25) or 1
        )

        # --- Optimizer ---
        proj_q = list(self.encoder.projection_blocks["qwen35"].parameters())
        proj_f = list(self.encoder.projection_blocks["falcon_h1"].parameters())
        proj_ids = {id(p) for p in proj_q + proj_f}
        enc_other = [p for p in self.encoder.parameters() if id(p) not in proj_ids]
        dec_q = [p for p in self.decoder_qwen.parameters() if p.requires_grad]
        dec_f = [p for p in self.decoder_falcon.parameters() if p.requires_grad]
        base_lr = float(tcfg.lr)
        param_groups = [
            {"params": enc_other, "lr": base_lr, "base_lr": base_lr},
            {"params": proj_q, "lr": base_lr, "base_lr": base_lr},
            {"params": proj_f, "lr": base_lr, "base_lr": base_lr},
            {"params": dec_q, "lr": base_lr, "base_lr": base_lr},
            {"params": dec_f, "lr": base_lr, "base_lr": base_lr},
        ]
        self.optimizer = self._create_optimizer(
            param_groups, base_lr, exclude_from_muon=frozenset(),
        )

        # Banner registration — gives operator clear sight into all
        # checkpoint sources at launch.
        self.register_checkpoint_source("encoder", step1_checkpoint)
        self.register_checkpoint_source("decoder_qwen", qwen_dec_source)
        self.register_checkpoint_source("decoder_falcon", falcon_dec_source)
        self.register_startup_extra("n_train_samples", len(self._train_flat_idx))
        self.register_startup_extra("n_eval_samples", len(self._eval_flat_idx))
        self.register_startup_extra("datasets", ",".join(self._dataset_names))
        self.register_startup_extra("qwen_decoder_prob", self._qwen_decoder_prob)
        self.register_startup_extra("target_ratio_start", self._target_ratio_start)
        self.register_startup_extra("target_ratio_end", self._target_ratio_end)
        self.register_startup_extra("target_ratio_ramp_steps", self._target_ratio_ramp_steps)

        logger.info(
            "summarization_round_robin_setup",
            train_samples=len(self._train_flat_idx),
            eval_samples=len(self._eval_flat_idx),
            datasets=self._dataset_names,
            qwen_decoder_prob=self._qwen_decoder_prob,
        )

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_step1_checkpoint(self) -> str | None:
        src = self.cfg.get("step1_checkpoint", None)
        if src == "auto":
            ckpt_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                ckpt_dir,
                phase="phase1_summarization_round_robin",
                metric="eval/loss",
                label="step1_checkpoint",
            )
            src = str(resolved)
        return src

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def _collate(self, flat_indices: list[int]) -> dict:
        source_docs: list[list[np.ndarray]] = []
        targets_qwen: list[np.ndarray] = []
        targets_falcon: list[np.ndarray] = []
        dataset_names: list[str] = []
        group_ids: list[str] = []
        for flat_i in flat_indices:
            ds_name, q_idx, f_idx = self._flat_index[int(flat_i)]
            q_row = self._family_datasets["qwen35"][ds_name][q_idx]
            f_row = self._family_datasets["falcon_h1"][ds_name][f_idx]
            # Source from qwen-target dir (source tokens are identical
            # across families — same encoder tokenizer, same input text;
            # only the target column differs).
            source_docs.append(q_row["doc_token_ids"])
            targets_qwen.append(q_row["target_token_ids"])
            targets_falcon.append(f_row["target_token_ids"])
            dataset_names.append(ds_name)
            group_ids.append(q_row["group_id"])
        return {
            "source_docs": source_docs,
            "targets_qwen": targets_qwen,
            "targets_falcon": targets_falcon,
            "dataset_names": dataset_names,
            "group_ids": group_ids,
        }

    # ------------------------------------------------------------------
    # Decoder routing
    # ------------------------------------------------------------------

    def _pick_decoder_family(self) -> str:
        if abs(self._qwen_decoder_prob - 0.5) > 1e-6:
            return "qwen35" if random.random() < self._qwen_decoder_prob else "falcon_h1"
        family = "qwen35" if (self._microbatch_counter % 2 == 0) else "falcon_h1"
        self._microbatch_counter += 1
        return family

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _current_target_ratio(self) -> float:
        return linear_ratio(
            self.global_step,
            self._target_ratio_start,
            self._target_ratio_end,
            self._target_ratio_ramp_steps,
        )

    def _current_target_ratio_l1(self) -> float | None:
        """L1 ratio, or None if L1 not yet introduced."""
        if self.global_step < self._l1_introduction_step:
            return None
        return linear_ratio(
            self.global_step,
            self._target_ratio_l1_start,
            self._target_ratio_l1_end,
            self._target_ratio_l1_ramp_steps,
            introduction_step=self._l1_introduction_step,
        )

    # ------------------------------------------------------------------
    # Closed-loop survivorship control (per optimizer step)
    # ------------------------------------------------------------------

    def _post_optimizer_step(self, step: int) -> None:
        """Dual-ascent θ update for L0 (+ L1 when active), so the encoder's
        survivor keep-rate tracks the curriculum target instead of drifting."""
        levels = ("l0", "l1") if self._current_target_ratio_l1() is not None else ("l0",)
        self._run_dual_ascent(step, levels=levels)

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Surface θ / mean-rate (keep-rate) from the dual-ascent update."""
        self._inject_survivorship_metrics(metrics)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_batch(
        self, batch: dict, comp_prompt_ids: list[torch.Tensor] | None = None,
    ):
        """Two-level packed encoder forward.

        Each sample is a "group" of source docs. We pack each doc as its
        own attention segment (no cross-doc attention) via per-doc
        ``content_cu_seqlens``. The encoder selects survivors per doc;
        we then aggregate survivors per group (sum across that group's
        docs) for the decoder splice.
        """
        device = self.device
        flat: list[torch.Tensor] = []
        doc_lens: list[int] = []
        group_doc_counts: list[int] = []
        for docs in batch["source_docs"]:
            group_doc_counts.append(len(docs))
            for d in docs:
                t = torch.as_tensor(d, dtype=torch.long)
                flat.append(t)
                doc_lens.append(int(t.shape[0]))
        if flat:
            content_ids = torch.cat(flat).to(device)
        else:
            content_ids = torch.zeros(0, dtype=torch.long, device=device)
        cu_file = torch.zeros(len(doc_lens) + 1, dtype=torch.int32, device=device)
        if doc_lens:
            cu_file[1:] = torch.cumsum(
                torch.tensor(doc_lens, dtype=torch.int32, device=device), dim=0,
            )
        position_ids = position_ids_from_cu(cu_file, int(content_ids.shape[0]))
        embed = self.encoder.l0.backbone.get_input_embeddings()
        n_groups = len(batch["source_docs"])
        ratio = self._current_target_ratio()
        ratio_l1 = self._current_target_ratio_l1()

        # Query-aware compression: condition the encoder on the SAME compression
        # prompt the decoder's bgkit tool-call advertises. The encoder's prompt
        # fuse pairs ONE prompt segment per CONTENT segment (per doc), so each
        # group's compression prompt is replicated across its source docs. When
        # no prompt is supplied (legacy path), fall back to an empty prompt
        # (fuse is skipped for zero-length prompts).
        if comp_prompt_ids is not None and doc_lens:
            prompt_flat: list[torch.Tensor] = []
            prompt_lens: list[int] = []
            for g, n_docs in enumerate(group_doc_counts):
                cp = comp_prompt_ids[g].to(device=device, dtype=torch.long)
                for _ in range(n_docs):
                    prompt_flat.append(cp)
                    prompt_lens.append(int(cp.shape[0]))
            prompt_ids = torch.cat(prompt_flat).to(device)
            prompt_cu = torch.zeros(len(doc_lens) + 1, dtype=torch.int32, device=device)
            prompt_cu[1:] = torch.cumsum(
                torch.tensor(prompt_lens, dtype=torch.int32, device=device), dim=0,
            )
            prompt_position_ids = position_ids_from_cu(
                prompt_cu, int(prompt_ids.shape[0]),
            )
            prompt_embeddings = embed(prompt_ids)
        else:
            empty_prompt = torch.zeros(0, dtype=torch.long, device=device)
            prompt_cu = torch.zeros(n_groups + 1, dtype=torch.int32, device=device)
            prompt_position_ids = torch.zeros(0, dtype=torch.long, device=device)
            prompt_embeddings = embed(empty_prompt)

        # Section→sample grouping for cross-section L1: indices into cu_file
        # marking which L0 sections belong to the same sample. L1 merges each
        # sample's sections into ONE segment (separators between, prompt once)
        # so it attends ACROSS sections + prompt (the "Interaction" in bgKIT).
        content_group_cu = torch.zeros(n_groups + 1, dtype=torch.int32, device=device)
        content_group_cu[1:] = torch.cumsum(
            torch.tensor(group_doc_counts, dtype=torch.int32, device=device), dim=0,
        )
        # L1 prompt: the SAME compression prompt as L0, once per sample (vs L0's
        # per-section copy). Skipped (None) when no prompt is supplied.
        if comp_prompt_ids is not None and doc_lens:
            l1_prompt_flat = torch.cat([
                comp_prompt_ids[g].to(device=device, dtype=torch.long)
                for g in range(n_groups)
            ])
            l1_prompt_lens = [int(comp_prompt_ids[g].shape[0]) for g in range(n_groups)]
            l1_prompt_cu = torch.zeros(n_groups + 1, dtype=torch.int32, device=device)
            l1_prompt_cu[1:] = torch.cumsum(
                torch.tensor(l1_prompt_lens, dtype=torch.int32, device=device), dim=0,
            )
            l1_prompt_pos = position_ids_from_cu(l1_prompt_cu, int(l1_prompt_flat.shape[0]))
            l1_prompt_emb = embed(l1_prompt_flat)
        else:
            l1_prompt_emb = None
            l1_prompt_cu = None
            l1_prompt_pos = None

        enc_out = self.encoder(
            content_embeddings=embed(content_ids),
            content_cu_seqlens=cu_file,
            content_position_ids=position_ids,
            prompt_embeddings=prompt_embeddings,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio_l0=ratio,
            target_ratio_l1=ratio_l1,
            content_group_cu_seqlens=content_group_cu,
            prompt_embeddings_l1=l1_prompt_emb,
            prompt_cu_seqlens_l1=l1_prompt_cu,
            prompt_position_ids_l1=l1_prompt_pos,
        )
        # When L1 is active it already merged sections → survivor segments are
        # per-sample. When L1 is skipped (ratio_l1 None) segments are per-section
        # and must be re-summed into per-sample groups (legacy path).
        per_seg = (
            enc_out.survivor_cu_seqlens[1:] - enc_out.survivor_cu_seqlens[:-1]
        ).tolist()
        if len(per_seg) == n_groups:
            per_group = [int(x) for x in per_seg]
            group_cu = enc_out.survivor_cu_seqlens.to(torch.int32)
        else:
            per_group = []
            idx = 0
            for n in group_doc_counts:
                per_group.append(int(sum(per_seg[idx : idx + n])))
                idx += n
            group_cu = torch.zeros(n_groups + 1, dtype=torch.int32, device=device)
            group_cu[1:] = torch.cumsum(
                torch.tensor(per_group, dtype=torch.int32, device=device), dim=0,
            )
        # ``cu_file`` is the per-doc (L0 content) packed boundary tensor — the
        # content_cu_seqlens the survivorship aux losses use for per-sample
        # (min-survivors) reductions at L0.
        return enc_out, group_cu, per_group, ratio, ratio_l1, cu_file

    # ------------------------------------------------------------------
    # Chat input construction
    # ------------------------------------------------------------------

    def _build_chat_inputs(
        self, family: str, batch: dict,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        """Per-sample decoder chat inputs AND the matching encoder compression
        prompts, built from the SAME variant so the bgkit tool-call's advertised
        ``prompt`` argument is exactly what the encoder is conditioned on.
        Returns ``(prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids)`` —
        ``comp_prompt_ids[i]`` is the (encoder-tokenized) compression prompt for
        sample/group ``i``."""
        tokenizer = self.tokenizer_qwen if family == "qwen35" else self.tokenizer_falcon
        targets = batch["targets_qwen"] if family == "qwen35" else batch["targets_falcon"]
        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        suffix_masks: list[torch.Tensor] = []
        comp_prompt_ids: list[torch.Tensor] = []
        for sample_idx, (ds_name, group_id, tgt) in enumerate(
            zip(batch["dataset_names"], batch["group_ids"], targets, strict=True),
        ):
            bank_key = _DATASET_TO_BANK[ds_name]
            bank = self._variant_banks[bank_key]
            variant = bank[(self.global_step + sample_idx) % len(bank)]
            config = TOOL_CONFIGS[bank_key]
            pre, suf, mask, comp = _build_chat_inputs_for_sample(
                tokenizer, variant, config,
                group_id=group_id, target_ids=tgt, device=self.device,
                encoder_tokenizer=self.encoder_tokenizer,
            )
            prefix_ids.append(pre)
            suffix_ids.append(suf)
            suffix_masks.append(mask)
            comp_prompt_ids.append(comp)
        return prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        family = self._pick_decoder_family()
        self.encoder.set_active_decoder_family(family)
        decoder = self.decoder_qwen if family == "qwen35" else self.decoder_falcon

        # Decoder chat inputs + the matching encoder compression prompts from
        # the same variant, THEN encode query-aware (the compression prompt
        # conditions survivor selection — the bgkit tool-call's `prompt` arg).
        prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids = self._build_chat_inputs(
            family, batch,
        )
        enc_out, group_cu, per_group, ratio, ratio_l1, cu_file = self._encode_batch(
            batch, comp_prompt_ids,
        )
        survivors = enc_out.survivor_embeddings
        # Build full loss mask matching n_total = sum(prefix_lens + surv_lens + suffix_lens).
        # We only score the suffix's target span; prefix and survivor positions are zero.
        full_masks: list[torch.Tensor] = []
        for sample_i, (pre, sm, suf) in enumerate(
            zip(prefix_ids, suffix_masks, suffix_ids, strict=True),
        ):
            n_pre = pre.shape[0]
            n_surv = int(per_group[sample_i])
            zeros_pre = torch.zeros(n_pre, dtype=torch.bool, device=sm.device)
            zeros_surv = torch.zeros(n_surv, dtype=torch.bool, device=sm.device)
            full_masks.append(torch.cat([zeros_pre, zeros_surv, sm]))
            _ = suf  # used only by decoder; mask alignment relies on its length being equal to sm
        loss_mask = torch.cat(full_masks)
        out = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=group_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=loss_mask,
        )
        loss = out.loss if hasattr(out, "loss") else out

        # Closed-loop survivorship (CompressionCurriculumMixin): per-level aux
        # losses + dual-ascent accumulation. Must run BEFORE enc_out.release()
        # (reads l0/l1 base_raw + logits_for_op) and feed the single backward.
        aux_metrics: dict[str, float] = {}
        total_loss = loss
        l0_out = enc_out.l0
        if l0_out is not None and l0_out.logits_for_op is not None:
            l0_loss, m0 = self._survivorship_loss_for_level(
                l0_out, "l0", ratio,
                content_cu_seqlens=cu_file,
                diag_every_n=self._diagnostic_metrics_every_n_steps,
            )
            total_loss = total_loss + l0_loss
            aux_metrics.update(m0)
        elif l0_out is not None:
            self._accumulate_level_state(l0_out, "l0", ratio)
        if enc_out.l1 is not None and ratio_l1 is not None:
            l1_out = enc_out.l1
            l1_cu = l0_out.survivor_cu_seqlens if l0_out is not None else None
            if l1_out.logits_for_op is not None:
                l1_loss, m1 = self._survivorship_loss_for_level(
                    l1_out, "l1", ratio_l1,
                    content_cu_seqlens=l1_cu,
                    diag_every_n=self._diagnostic_metrics_every_n_steps,
                )
                total_loss = total_loss + l1_loss
                aux_metrics.update(m1)
            else:
                self._accumulate_level_state(l1_out, "l1", ratio_l1)

        (total_loss / self._accum_steps).backward()
        enc_out.release()
        metrics = {
            "loss": float(loss.detach().item()),
            f"loss_{family}": float(loss.detach().item()),
            "target_ratio_l0": ratio,
            "target_ratio_l1": float(ratio_l1) if ratio_l1 is not None else -1.0,
            "active_family_qwen": 1.0 if family == "qwen35" else 0.0,
            "mean_survivors_per_group": float(sum(per_group) / max(1, len(per_group))),
        }
        metrics.update(aux_metrics)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder.eval()
        self.decoder_qwen.eval()
        self.decoder_falcon.eval()
        totals = {"loss_qwen": 0.0, "n_qwen": 0, "loss_falcon": 0.0, "n_falcon": 0}
        batch_idx = 0
        for batch in self.eval_dataloader:
            family = "qwen35" if (batch_idx % 2 == 0) else "falcon_h1"
            batch_idx += 1
            self.encoder.set_active_decoder_family(family)
            decoder = self.decoder_qwen if family == "qwen35" else self.decoder_falcon
            prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids = self._build_chat_inputs(
                family, batch,
            )
            enc_out, group_cu, per_group, _, _, _ = self._encode_batch(batch, comp_prompt_ids)
            full_masks: list[torch.Tensor] = []
            for sample_i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
                n_pre = pre.shape[0]
                n_surv = int(per_group[sample_i])
                zeros_pre = torch.zeros(n_pre, dtype=torch.bool, device=sm.device)
                zeros_surv = torch.zeros(n_surv, dtype=torch.bool, device=sm.device)
                full_masks.append(torch.cat([zeros_pre, zeros_surv, sm]))
            out = decoder.forward_with_single_splice(
                survivor_embeddings=enc_out.survivor_embeddings,
                survivor_cu_seqlens=group_cu,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                loss_mask=torch.cat(full_masks),
            )
            loss_val = out.loss if hasattr(out, "loss") else out
            if family == "qwen35":
                totals["loss_qwen"] += float(loss_val.item())
                totals["n_qwen"] += 1
            else:
                totals["loss_falcon"] += float(loss_val.item())
                totals["n_falcon"] += 1
            enc_out.release()
        self.encoder.train()
        self.decoder_qwen.train()
        self.decoder_falcon.train()
        out_d = {
            "loss_qwen": totals["loss_qwen"] / max(1, totals["n_qwen"]),
            "loss_falcon": totals["loss_falcon"] / max(1, totals["n_falcon"]),
        }
        out_d["loss"] = 0.5 * (out_d["loss_qwen"] + out_d["loss_falcon"])
        return out_d

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
            optimizer_type=self._optimizer_type,
            run_name=self.cfg.get("run_name", None),
        )
        # Route through the base helper so the NVMe fast-dir + async HDD archive
        # apply here too (this override used to call save_checkpoint() directly,
        # writing to the slow HDD and bypassing NVMe — the 2026-06-10 bug).
        return self._write_checkpoint(
            checkpoint_dir,
            metadata,
            encoder=self.encoder.state_dict(),
            decoder_qwen=self.decoder_qwen.state_dict(),
            decoder_falcon=self.decoder_falcon.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )

    def _named_parameters_for_optimizer(self):
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.decoder_qwen.named_parameters():
            yield f"decoder_qwen.{name}", param
        for name, param in self.decoder_falcon.named_parameters():
            yield f"decoder_falcon.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
        if "decoder_qwen" in state_dicts:
            self.decoder_qwen.load_state_dict(state_dicts["decoder_qwen"])
        if "decoder_falcon" in state_dicts:
            self.decoder_falcon.load_state_dict(state_dicts["decoder_falcon"])

    def trainable_parameters(self) -> list:
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder_qwen.parameters() if p.requires_grad]
        params += [p for p in self.decoder_falcon.parameters() if p.requires_grad]
        return params
