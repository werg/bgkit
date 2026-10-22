#!/usr/bin/env python
"""S1b memory probe: a few REAL training steps of an encoder-adapter run on its
LARGEST rows, reporting CUDA peak and host MemAvailable -- measured before the
relaunch, not guessed (user 2026-09-22: "we really do need to be careful about
memory").

Builds the trainer exactly as scripts/train.py does (Hydra, the experiment's
own config, BgKIT2Trainer.setup: warm starts, encoder adapter, gradient
checkpointing, teacher), then scans ``--scan`` training indices by length hint,
packs the longest into batches under the run's own ``data.max_batch_tokens``
(the sampler's rule: rows x longest hint), and runs ``--steps`` full
``_forward_backward`` + optimizer steps on them. Nothing is saved (the
checkpoint dir is a scratch dir; no save cadence is reached).

GUARDED (feedback: standalone GPU scripts must be guarded, 2026-09-17 host
loss): host-memory guard armed (abort under ``--host-floor-gb``), CUDA capped
at ``--cuda-fraction`` of the device, one JSON line of progress per step and a
background sampler of the MINIMUM MemAvailable seen during each step.

    docker compose -f docker/docker-compose.yaml run --rm probe-bgkit2-s1b-memory
    # or with arguments:
    ... run --rm probe-bgkit2-s1b-memory scripts/bgkit2_probe_s1b_memory.py \\
        --experiment bgkit2_s1b_showcase_encoder --steps 4 --scan 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


class _MinAvail:
    """Samples MemAvailable in the background; ``take()`` returns the minimum
    since the last call."""

    def __init__(self, read, interval: float = 0.25) -> None:
        self._read, self._interval = read, interval
        self._min: float | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            v = self._read()
            if v is not None and (self._min is None or v < self._min):
                self._min = v
            time.sleep(self._interval)

    def take(self) -> float | None:
        v, self._min = self._min, None
        return v

    def stop(self) -> None:
        self._stop.set()


def _batches(ds, scan: int, max_tokens: int, batch_size: int, n: int) -> list[list[int]]:
    """The ``n`` heaviest batches the run's sampler could form: the longest
    rows among ``scan`` indices (longest first), packed under the sampler's
    rule rows x longest <= max_tokens (and rows <= batch_size)."""
    hints = sorted(((int(ds.length_hint(i)), i) for i in range(min(scan, len(ds)))), reverse=True)
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for h, i in hints:
        m = max(cur_max, h)
        if cur and (len(cur) >= batch_size or (max_tokens and (len(cur) + 1) * m > max_tokens)):
            out.append(cur)
            if len(out) >= n:
                return out
            cur, m = [], h
        cur.append(i)
        cur_max = m
    if cur and len(out) < n:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="bgkit2_s1b_showcase_encoder")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--scan", type=int, default=20000, help="training indices scanned by length")
    ap.add_argument("--cuda-fraction", type=float, default=0.6)
    ap.add_argument("--host-floor-gb", type=float, default=12.0)
    ap.add_argument("--checkpoint-dir", default="/workspace/checkpoints/probe_s1b_memory")
    ap.add_argument("--override", action="append", default=[],
                    help="extra Hydra overrides, e.g. training.bgkit2.optim."
                         "encoder_grad_token_budget=4096")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from bgkit_core.host_memory_guard import arm, cap_cuda, mem_available_gb

    arm(float(args.host_floor_gb))
    cap_cuda(float(args.cuda_fraction))

    import torch
    from hydra import compose, initialize_config_dir

    from bgkit.utils.logging import setup_logging
    from bgkit2.training.bgkit2_trainer import BgKIT2Trainer

    setup_logging()
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
        cfg = compose(config_name="config", overrides=[
            f"+experiment={args.experiment}",
            "run_name=probe_s1b_memory",
            "resume_checkpoint=none",
            f"checkpoint_dir={args.checkpoint_dir}",
            "wandb.enabled=false",
            "training.eval_every=1000000",
            "training.save_every=1000000",
            *args.override,
        ])
    t0 = time.time()
    tr = BgKIT2Trainer(cfg)
    tr.setup()
    print(json.dumps({"event": "setup_done", "secs": round(time.time() - t0, 1),
                      "cuda_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                      "host_avail_gb": round(mem_available_gb() or -1, 1),
                      "param_groups": [g.get("name") for g in tr.optimizer.param_groups]}),
          flush=True)
    d = tr.cfg2.data
    picks = _batches(tr.train_ds, int(args.scan), int(d.max_batch_tokens), int(d.batch_size),
                     int(args.steps))
    sampler = _MinAvail(mem_available_gb)
    rows = []
    try:
        for step, idx in enumerate(picks):
            samples = [x for x in (tr.train_ds[i] for i in idx) if x is not None]
            if not samples:
                continue
            batch = tr.collate(samples)
            n_extra = 0 if getattr(batch, "extra_input_ids", None) is None else int(
                batch.extra_input_ids.shape[0])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sampler.take()
            t = time.time()
            stats = tr._forward_backward(batch)
            tr.optimizer.step()
            tr.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            r = {
                "step": step,
                "rows": len(idx),
                "longest_hint": max(int(tr.train_ds.length_hint(i)) for i in idx),
                "extra_slots": n_extra,
                "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "host_avail_min_gb": round(sampler.take() or -1, 1),
                "secs": round(time.time() - t, 1),
                "loss": round(float(stats.get("loss", float("nan"))), 4),
                **{k: v for k, v in stats.items() if k.startswith("enc_grad/")},
            }
            rows.append(r)
            print(json.dumps(r), flush=True)
            tr.global_step += 1
    finally:
        sampler.stop()
    summary = {
        "event": "summary",
        "experiment": args.experiment,
        "cuda_peak_gb_max": max((r["cuda_peak_gb"] for r in rows), default=None),
        "host_avail_min_gb": min((r["host_avail_min_gb"] for r in rows), default=None),
        "steps": len(rows),
    }
    print(json.dumps(summary), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"steps": rows, **summary}, indent=1))


if __name__ == "__main__":
    main()
