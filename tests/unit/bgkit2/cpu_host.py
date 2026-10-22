"""Test-only CPU host for :class:`BgKIT2Core` — the minimum loop around the
shared numerics (no checkpointing, no logging, no schedule of its own): the
production host is ``BgKIT2Trainer`` on ``BaseTrainer``. This replaces the
deleted standalone ``Trainer`` so the CPU tests drive the same code path."""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

from bgkit2.data.autoencode import Collator
from bgkit2.data.templates import Templates
from bgkit2.training.config import Config
from bgkit2.training.core import BgKIT2Core


class CpuHost(BgKIT2Core):
    def __init__(self, cfg: Config, tokenizer, encoder, decoder) -> None:
        self.cfg2 = cfg
        self.device = torch.device("cpu")
        self.amp_dtype = torch.float32
        self.tok = tokenizer
        self.templates = Templates.from_tokenizer(self.tok, style=cfg.data.prompt_style)
        self.encoder, self.decoder = encoder, decoder
        self.finish_models()
        self.opt = torch.optim.AdamW(self.param_groups(), weight_decay=cfg.optim.weight_decay)
        self.ratio_schedule = self.build_ratio_schedule(cfg.run.seed)
        self.select_rng = random.Random(cfg.run.seed + 1)
        self.step = 0
        self.collate = Collator(self.templates)
        self.train_ds, self.eval_ds, self.task_eval = self.build_datasets(cfg.run.seed)

    @property
    def current_step(self) -> int:
        return self.step

    def train(self, n_steps: int) -> list[dict[str, float]]:
        loader = DataLoader(
            self.train_ds, batch_size=self.cfg2.data.batch_size, collate_fn=self.collate
        )
        out = []
        for batch in loader:
            if len(out) >= n_steps:
                break
            self.encoder.train()
            self.decoder.train()
            total, stats = self.training_loss(batch)
            total.backward()
            sel = self.utility_selection_backward()
            if sel is not None:
                stats["loss/selection"] = sel
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            self.ratio_schedule.advance(self.step)
            out.append(stats)
        return out

    def evaluate(self) -> dict[str, float]:
        return self.evaluate_all()
