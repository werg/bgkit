#!/usr/bin/env python
"""Validate task stores through the REAL trainer path (``TaskDataset``), not a
hand-rolled renderer.

Every data defect this project has shipped (impossible targets, tool-call
preambles, an un-enforced prefix cap, clairvoyant trajectories) was invisible in
the parquet and obvious in a decoded sample. This script pulls N samples per
store exactly as training would — same renderer, same caps, same framing bank —
and reports, per family: how many rows the loader had to SKIP (over-long prefix
/ unrenderable), prefix and target token statistics, slot counts and rep budgets
for multi-slot rows, and how often the target is empty or degenerate. With
``--show`` it prints decoded samples so a human can read them.

Exit code 1 when a store trips a hard check (no renderable rows, >20% skips,
empty targets), so it can gate a launch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.framing_bank import FramingBank
from bgkit2.data.tasks import ChatRenderer, TaskDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="*")
    ap.add_argument(
        "--from-experiment",
        help="validate every task_store of a hydra experiment (pre-launch check): "
        "container paths are mapped back to the host with --data-root",
    )
    ap.add_argument("--data-root", default="/home/werg/bgkit-data-nvme/bgkit2")
    ap.add_argument(
        "--scan-rows",
        type=int,
        default=200_000,
        help="rows walked IN ORDER for the consecutive over-long check (the loader's real "
        "failure mode); rendering is only paid on tool_slots rows",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=300)
    # Default None: with --from-experiment these are taken from the config so
    # the gate renders at the cap the TRAINER uses. Hard-coded defaults made
    # the 2026-09-16 scan judge P26 at 1536 against a run at 4096 and report a
    # store as 82% unusable when the trainer was not skipping those rows at
    # all. An explicit flag still wins.
    ap.add_argument("--max-ctx-tokens", type=int, default=None)
    ap.add_argument("--max-target-tokens", type=int, default=None)
    ap.add_argument("--max-prefix-tokens", type=int, default=None)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--model-root", default="/home/werg/bgkit-data-nvme/models_hf",
                    help="host dir for the experiment's /workspace/data/models/hf model paths")
    ap.add_argument("--bank", action="store_true", help="draw phrasings like training does")
    ap.add_argument("--show", type=int, default=0, help="decode and print this many samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stores = list(args.stores)
    if args.from_experiment:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).resolve().parent.parent / "configs")
        ):
            cfg = compose(config_name="config", overrides=[f"+experiment={args.from_experiment}"])

        def find(node, key):
            if hasattr(node, "keys"):
                if key in node:
                    return node[key]
                for v in node.values():
                    got = find(v, key)
                    if got is not None:
                        return got
            return None

        paths = list(find(cfg, "task_stores") or [])
        weights = list(find(cfg, "task_weights") or [])
        if weights and len(weights) != len(paths):
            print(f"CONFIG ERROR: {len(paths)} task_stores but {len(weights)} task_weights")
            sys.exit(1)
        if weights and abs(sum(weights) - 1.0) > 1e-3:
            print(f"CONFIG ERROR: task_weights sum to {sum(weights):.4f}")
            sys.exit(1)
        missing = []
        eval_only = [str(p) for p in (find(cfg, "eval_only_stores") or [])]
        for p in [*paths, *eval_only]:
            host = str(p).replace("/workspace/bgkit2_data", args.data_root)
            if not Path(host).exists():
                missing.append(host)
            elif str(p) not in eval_only:  # eval-only stores have no train split
                stores.append(host)
        if missing:
            print("MISSING STORES:\n  " + "\n  ".join(missing))
            sys.exit(1)
        # Per-store prefix overrides, so the gate renders each store at the cap
        # the TRAINER will use for it. Without this the gate reports skips on a
        # store the run does not actually trim (read_replay_v2 at 7168), which
        # is the same gate-vs-run divergence fixed earlier today, reintroduced
        # by the new key.
        by_store = find(cfg, "task_max_prefix_tokens_by_store") or {}
        prefix_by_store = {str(k): int(v) for k, v in dict(by_store).items()}
        for flag, key in (
            ("max_ctx_tokens", "task_max_ctx_tokens"),
            ("max_target_tokens", "task_max_target_tokens"),
            ("max_prefix_tokens", "task_max_prefix_tokens"),
        ):
            if getattr(args, flag) is None:
                found = find(cfg, key)
                if found is not None:
                    setattr(args, flag, int(found))
        print(
            f"{args.from_experiment}: {len(paths)} task stores (+{len(eval_only)} eval-only), "
            "weights sum 1.0, all present | caps ctx {} target {} prefix {}".format(
                args.max_ctx_tokens, args.max_target_tokens, args.max_prefix_tokens
            )
        )
    if not stores:
        sys.exit("nothing to validate")
    prefix_by_store = locals().get("prefix_by_store") or {}
    # Fall back to the historical defaults for a bare store list.
    args.max_ctx_tokens = args.max_ctx_tokens or 8192
    args.max_target_tokens = args.max_target_tokens or 640
    args.max_prefix_tokens = args.max_prefix_tokens or 1536

    if args.from_experiment:
        # The run's OWN renderer: its decoder tokenizer (+ the encoder's on two-tokenizer runs)
        # and chat template mode (model.decoder_no_think), built by the trainer's helpers --
        # the 350M default missed a render failure only the 8B's no-think template produces
        # (8B-A relaunch, 2026-09-26).
        from bgkit2.training.standalone import StandaloneModels, experiment_config

        ecfg = experiment_config(args.from_experiment)
        m = ecfg.model
        for k in ("decoder_id", "ctx_tokenizer_id", "rep_space_decoder_id"):
            v = getattr(m, k)
            if v and str(v).startswith("/workspace/data/models/hf/"):
                setattr(m, k, str(v).replace("/workspace/data/models/hf", args.model_root))
        core = StandaloneModels(ecfg, "cpu")
        core.load_tokenizers()
        renderer = core.chat_renderer()
        tok = core.tok
        print(f"renderer: decoder {m.decoder_id} | encoder tokenizer {m.ctx_tokenizer_id} | "
              f"no_think {m.decoder_no_think}")
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        renderer = ChatRenderer.from_tokenizer(tok)
    failures: list[str] = []
    for store in stores:
        name = Path(store).stem
        try:
            ds = TaskDataset(
                store,
                renderer,
                split=args.split,
                max_ctx_tokens=args.max_ctx_tokens,
                max_target_tokens=args.max_target_tokens,
                max_prefix_tokens=prefix_by_store.get(name, args.max_prefix_tokens),
                seed=args.seed,
                bank=FramingBank(args.split, 0.5, 0.2) if args.bank else None,
            )
        except Exception as exc:
            print(f"{name}: LOAD FAILED {type(exc).__name__}: {exc}")
            failures.append(name)
            continue
        n = min(args.samples, ds.n)
        step = max(1, ds.n // n)
        pre: list[int] = []
        tgt: list[int] = []
        ctx: list[int] = []
        slots: Counter = Counter()
        empty_target = 0
        errors: Counter = Counter()
        shown = 0
        for k in range(n):
            try:
                s = ds[k * step]
            except Exception as exc:
                errors[type(exc).__name__] += 1
                continue
            p = int(s.prefix_ids.shape[0]) + int(s.instr_ids.shape[0])
            pre.append(p)
            tgt.append(int(s.target_ids.shape[0]))
            if int(s.target_ids.shape[0]) == 0:
                empty_target += 1
            n_slots = 1 + (len(s.extra_ctx) if getattr(s, "extra_ctx", None) else 0)
            if getattr(s, "ctx_ids", None) is not None and int(s.ctx_ids.shape[0]):
                ctx.append(int(s.ctx_ids.shape[0]))
                slots[n_slots] += 1
            else:
                slots[0] += 1
            if shown < args.show:
                shown += 1
                print(f"\n----- {name} sample {k * step} (prefix {p} tok, target {tgt[-1]} tok)")
                print("PREFIX:", tok.decode(s.prefix_ids.tolist())[-700:])
                if getattr(s, "ctx_ids", None) is not None and int(s.ctx_ids.shape[0]):
                    print("SLOT-0 CTX:", tok.decode(s.ctx_ids.tolist())[:300].replace("\n", " | "))
                print("TARGET:", tok.decode(s.target_ids.tolist())[:400])

        # The loader's failure mode is not "what fraction is bad" but "how many
        # bad IN A ROW": it hops over an over-long tool_slots row at most 8
        # times and then raises. Over-long rows cluster (consecutive turns of
        # one long episode), so a strided sample can report 0 skips while a
        # run of 14 waits in the store — that killed P26 at step 2340.
        worst_run = 0
        scan_errors: dict[str, int] = {}
        # No guard on row 0's framing: agent_tool_slots interleaves 'chat'
        # (turn-0) and 'tool_slots' rows and happens to start with 'chat', so a
        # row-0 guard would have skipped the scan on the very store that killed
        # P26. Non-slot rows cost nothing here — they just reset the run.
        run = 0
        if ds.n:
            for i in range(min(ds.n, args.scan_rows)):
                r = ds.row(i)
                if r["framing"] != "tool_slots":
                    run = 0
                    continue
                try:
                    meta = (
                        r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
                    )
                    segs = renderer.render_slots(r["instruction"], meta.get("tools"))
                    too_long = sum(int(x.shape[0]) for x in segs) > (
                        args.max_prefix_tokens or 10**9
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    # a row that will not render is as unusable as an over-long
                    # one, but anything else (NameError, ImportError...) is a
                    # bug in THIS script and must not be reported as bad data
                    scan_errors[type(exc).__name__] = scan_errors.get(type(exc).__name__, 0) + 1
                    too_long = True
                run = run + 1 if too_long else 0
                worst_run = max(worst_run, run)

        def stat(v: list[int]) -> str:
            if not v:
                return "-"
            v = sorted(v)
            return f"p50 {v[len(v) // 2]} p90 {v[int(len(v) * 0.9)]} max {v[-1]}"

        skipped = ds.skipped_long_prefix + ds.skipped_unrenderable
        ok = len(pre)
        print(
            f"{name}: rows {ds.n} families {ds.families} | sampled {ok}/{n} "
            f"| skips long-prefix {ds.skipped_long_prefix} unrenderable {ds.skipped_unrenderable} "
            f"| prefix {stat(pre)} | target {stat(tgt)} | ctx {stat(ctx)} "
            f"| slots {dict(sorted(slots.items()))}"
            + (f" | worst consecutive over-long run {worst_run}" if worst_run else "")
            + (f" | scan errors {scan_errors}" if scan_errors else "")
            + (f" | ERRORS {dict(errors)}" if errors else "")
        )
        if worst_run >= 8:
            # Not fatal since 2026-09-16: the loader jumps to a random row
            # after 4 sequential steps, so a clustered bad region no longer
            # exhausts its budget. Still worth printing — a long run means the
            # store wastes draws and samples its good rows out of order.
            print(f"  WARNING {name}: {worst_run} over-long rows in a row (loader jumps past them)")
        if ok == 0:
            failures.append(f"{name}: no renderable sample")
        elif ok < 0.8 * n:
            failures.append(f"{name}: only {ok}/{n} draws usable")
        elif empty_target:
            failures.append(f"{name}: {empty_target}/{ok} empty targets")
        elif skipped > 0.2 * n:
            failures.append(f"{name}: {skipped} skips over {n} draws (cap too low?)")
        elif ok and statistics.median(tgt) == 0:
            failures.append(f"{name}: median target length 0")
    if failures:
        print("\nFAILED: " + "; ".join(failures))
        sys.exit(1)
    print("\nall stores OK")


if __name__ == "__main__":
    main()
