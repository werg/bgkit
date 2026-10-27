#!/usr/bin/env python
"""LaTeX result tables for the bgkit paper (paper/tables/*.tex), generated from the eval outputs so
no number is typed by hand: the S2-final showcase sweep, SWE-bench Lite localisation and the
offline web-research evals, plus the full-context reference column (the original decoder reading the
whole context, scored on the same items) and the prompt/selection ablation of an earlier checkpoint.

  .venv/bin/python scripts/bgkit2_paper_tables.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

B = Path("/home/werg/bgkit-data-nvme/bgkit2/eval")
OUT = Path(__file__).resolve().parents[1] / "paper/tables"

SWEEP = [  # bench, metric, display name, n
    ("babilong_16k", "acc", "BABILong 16k", 100),
    ("babilong_4k", "acc", "BABILong 4k", 100),
    ("ruler_4096", "acc", "RULER 4k", 100),
    ("quality", "acc", "QuALITY", 100),
    ("narrativeqa", "judge_acc", "NarrativeQA", 60),
    ("hotpot", "judge_acc", "HotpotQA", 100),
    ("musique", "judge_acc", "MuSiQue", 100),
    ("longmemeval_s", "judge_acc", "LongMemEval-S", 60),
    ("locomo", "judge_acc", "LoCoMo", 100),
]


# the full-context run used a superset of the sweep's items; the column is rescored on the sweep's
# own items. NarrativeQA shares too few items (14 of 60) and LongMemEval-S does not fit the decoder.
FULL_RUN = "p34_final_full"
FULL_MIN_SHARED = 50
ABLATION = B / "ablations_p26_step23999.json"


# text-only baselines run after the sweep (scripts/bgkit2_bench.py arms of the same name, each read
# by the trained decoder and by the original one, ``base_<name>``), one file per benchmark:
# s2_baselines/<name>_<bench>.rows.jsonl
BASELINES = [("retrieval", "BM25"), ("lingua", "LongLLMLingua"), ("summary", "summary")]
# further runs over the same items: the kept-position tokens (bgkit's own selection, token
# embeddings instead of the encoder's vectors) and P19g (a decoder adapter trained on text in the slot)
EXTRA_RUNS = ["keptok", "textslot", "textfull"]  # textfull: the trained decoder, whole context
# the summary baseline's cells: the (benchmark, budget) pairs the original summary run generated
# (QuALITY: 60 and 56 of its 100 items). A later, unfinished generation pass left part of other
# budgets; those are not cells of the tables.
SUMMARY_CELLS = {"babilong_4k": {"512", "1024"}, "ruler_4096": {"512", "1024"},
                 "quality": {"512", "2048"}, "hotpot": {"256", "512"}, "musique": {"512", "1024"}}
# every LLM-judge score is read from one judge pass over all runs (same judge, same prompt):
# <JUDGED>/<run>/<stem>.judged.jsonl; falls back to the run's own judged file
JUDGED = B / "showcase/judged_qwen36_nvfp4"


def load(bench: str) -> dict:
    return json.loads((B / "showcase/s2_final" / f"{bench}.json").read_text())["summary"]


def _ids(path: Path) -> set[str]:
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def per_item(run: str, stem: str, metric: str) -> dict[str, dict[str, float]]:
    """arm -> item id -> score, from a run's rows (judged rows for LLM-judge benchmarks)."""
    if metric == "judge_acc":
        path = JUDGED / run / f"{stem}.judged.jsonl"
        if not path.exists():
            path = B / "showcase" / run / f"{stem}.judged.jsonl"
    else:
        path = B / "showcase" / run / f"{stem}.rows.jsonl"
    if not path.exists():
        return {}
    key = "judge" if metric == "judge_acc" else metric
    out: dict[str, dict[str, float]] = {}
    for r in map(json.loads, filter(str.strip, path.read_text().splitlines())):
        out.setdefault(r["arm"], {})[r["id"]] = float(r[key])
    return out


def bench_arms(bench: str, metric: str) -> dict[str, dict[str, float]]:
    """Every arm of a benchmark: the sweep's, then the baselines run after it."""
    arms = per_item("s2_final", bench, metric)
    summary_budgets = SUMMARY_CELLS.get(bench, set())
    for name in [n for n, _ in BASELINES] + EXTRA_RUNS:
        for arm, vals in per_item("s2_baselines", f"{name}_{bench}", metric).items():
            if arm.removeprefix("base_").startswith("summary@") and arm.split("@")[1] not in summary_budgets:
                continue
            arms.setdefault(arm, {}).update(vals)
    return arms


def best_reader(arms: dict, name: str, budget: int) -> dict[str, float] | None:
    """A text baseline at a budget, read by whichever decoder (trained or original) scores higher."""
    cands = [arms[k] for k in (f"{name}@{budget}", f"base_{name}@{budget}") if k in arms]
    return max(cands, key=lambda v: sum(v.values()) / len(v)) if cands else None


def mean(v: dict[str, float] | None) -> float | None:
    return sum(v.values()) / len(v) if v else None


def budgets_of(arms: dict) -> list[int]:
    return sorted({int(k.split("@")[1]) for k in arms if k.startswith("bgkit@")})


def full_items(bench: str, metric: str) -> dict[str, float] | None:
    """The original decoder (adapter off) reading the whole context, on the sweep's items."""
    ids = _ids(B / "showcase/s2_final" / f"{bench}.rows.jsonl")
    full = per_item(FULL_RUN, bench, metric).get("base_full@full", {})
    shared = {i: v for i, v in full.items() if i in ids}
    return shared if len(shared) >= FULL_MIN_SHARED else None


def full_context(bench: str, metric: str) -> float | None:
    return mean(full_items(bench, metric))


def paired_ci(a: dict[str, float], b: dict[str, float], n_boot: int = 10_000,
              seed: int = 0) -> tuple[float, float, float, int]:
    """Mean of a - b over the shared items and its 95% paired-bootstrap interval."""
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    return sum(diffs) / len(diffs), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], len(ids)


def f(v, bold=False) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


SWEEP_COLS = ["bgkit", "text", "base"] + [n for n, _ in BASELINES]


def sweep_cells(arms: dict, bu: int) -> dict[str, float | None]:
    vals = {a: mean(arms.get(f"{a}@{bu}")) for a in ("bgkit", "text", "base")}
    vals.update({n: mean(best_reader(arms, n, bu)) for n, _ in BASELINES})
    return vals


def sweep_table() -> str:
    rows = []
    for bench, metric, name, n in SWEEP:
        arms = bench_arms(bench, metric)
        for j, bu in enumerate(budgets_of(arms)):
            vals = sweep_cells(arms, bu)
            best = max(v for v in vals.values() if v is not None)
            cells = [f(vals[a], bold=(vals[a] == best)) for a in SWEEP_COLS]
            label = f"{name} ({n})" if j == 0 else ""
            if j == 1:  # second row: mean context length, so the compression ratio can be read off
                label = f"\\quad {load(bench)[f'bgkit@{bu}']['ctx_tokens'] / 1000:.1f}k tokens"
            full = f(full_context(bench, metric)) if j == 0 else ""
            rows.append(f"{label} & {bu:,} & " + " & ".join(cells) + f" & {full}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = " & ".join(["bgkit", "text", "base"] + [lbl for _, lbl in BASELINES])
    return "\n".join([
        r"\begin{tabular}{ll" + "r" * len(SWEEP_COLS) + "|r}",
        r"\toprule",
        f"Benchmark ($n$) & Budget & {head} & \\shortstack{{base,\\\\full context}} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def _ci(a, b) -> str:
    m, lo, hi, n = paired_ci(a, b)
    cell = f"${m:+.2f}$ \\scriptsize[${lo:+.2f}$, ${hi:+.2f}$]"
    return f"\\textbf{{{cell}}}" if lo > 0 or hi < 0 else cell


def stats_table() -> str:
    """bgkit minus each comparison arm, per benchmark and budget, with 95% paired-bootstrap
    intervals; bold where the interval excludes zero. Text baselines: the better reader, as in
    the sweep table."""
    rows = []
    sig = {k: [0, 0] for k in ("text", *(n for n, _ in BASELINES), "full")}
    for bench, metric, name, n in SWEEP:
        arms = bench_arms(bench, metric)
        full = full_items(bench, metric)
        for j, bu in enumerate(budgets_of(arms)):
            bg = arms[f"bgkit@{bu}"]
            comps = {"text": arms.get(f"text@{bu}"),
                     **{nm: best_reader(arms, nm, bu) for nm, _ in BASELINES}, "full": full}
            cells = []
            for k, other in comps.items():
                if not other:
                    cells.append("--")
                    continue
                m, lo, hi, _ = paired_ci(bg, other)
                sig[k][1] += 1
                sig[k][0] += lo > 0
                cells.append(_ci(bg, other))
            rows.append(f"{name if j == 0 else ''} & {bu:,} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    STATS.update({k: tuple(v) for k, v in sig.items()})
    head = " & ".join(["text"] + [lbl for _, lbl in BASELINES] + ["full context"])
    return "\n".join([
        r"\begin{tabular}{ll" + "c" * (len(BASELINES) + 2) + "}",
        r"\toprule",
        f"Benchmark & Budget & \\multicolumn{{{len(BASELINES) + 2}}}{{c}}{{bgkit minus}} \\\\",
        f"\\cmidrule(lr){{3-{len(BASELINES) + 4}}} & & {head} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


STATS: dict[str, tuple[int, int]] = {}


def budget_figure() -> str:
    """pgfplots group plot: score against budget per benchmark, one line per arm (text baselines
    with their better reader), the full-context score as a dashed line."""
    styles = {"bgkit": "blue!70!black, mark=*, thick", "text": "gray, mark=o",
              "retrieval": "orange!80!black, mark=triangle*", "lingua": "teal, mark=diamond*",
              "summary": "purple!70, mark=square*"}
    legend = {"bgkit": "bgkit", "text": "truncated text", "retrieval": "BM25",
              "lingua": "LongLLMLingua", "summary": "summary (35B)"}
    full_style = "black, dashed"
    # the legend lists every arm once, whether or not the first panel has it
    legend_lines = [f"\\addlegendimage{{{st}}}\n\\addlegendentry{{{legend[k]}}}" for k, st in styles.items()]
    legend_lines.append(f"\\addlegendimage{{{full_style}}}\n\\addlegendentry{{original decoder, full context}}")
    plots = []
    for i, (bench, metric, name, _) in enumerate(SWEEP):
        arms = bench_arms(bench, metric)
        bus = budgets_of(arms)
        lines = list(legend_lines) if i == 0 else []
        for k in styles:
            pts = [(bu, sweep_cells(arms, bu)[k]) for bu in bus]
            pts = [(x, y) for x, y in pts if y is not None]
            if pts:
                coords = " ".join(f"({x},{y:.3f})" for x, y in pts)
                lines.append(f"\\addplot[{styles[k]}, forget plot] coordinates {{{coords}}};")
        fc = full_context(bench, metric)
        if fc is not None:
            lines.append(f"\\addplot[{full_style}, forget plot, domain={bus[0]}:{bus[-1]}, samples=2] {{{fc:.3f}}};")
        ticks = ",".join(map(str, bus))
        legend_opts = ("legend to name=budgetlegend, legend columns=3, "
                       "legend style={font=\\footnotesize, draw=none}, " if i == 0 else "")
        plots.append(f"\\nextgroupplot[{legend_opts}title={{{name}}}, xtick={{{ticks}}}, "
                     f"xticklabels={{{','.join(f'{b // 1024}k' if b >= 1024 else str(b) for b in bus)}}}]\n"
                     + "\n".join(lines))
    return "\n".join([
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=3 by 3, horizontal sep=12mm, vertical sep=14mm},",
        r"  width=0.36\linewidth, height=0.3\linewidth, xmode=log, log basis x=2, ymin=0,",
        r"  title style={font=\small}, tick label style={font=\scriptsize},",
        r"  every axis plot/.append style={mark size=1.5pt},",
        r"  yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}]",
        *plots,
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\par\smallskip\ref{budgetlegend}",
    ])


def format_table() -> str:
    """Same selection and positions, different contents: the encoder's vectors against the tokens at
    the kept positions (better of the two decoders), per benchmark at its smallest budget; with
    the question-aware token pruner and truncated text at the same budget for scale."""
    rows = []
    for bench, metric, name, n in SWEEP:
        arms = bench_arms(bench, metric)
        bu = budgets_of(arms)[0]
        bg = arms[f"bgkit@{bu}"]
        tok = best_reader(arms, "bgkit_tokens", bu)
        if not tok:
            continue
        lingua = best_reader(arms, "lingua", bu)
        m, lo, hi, _ = paired_ci(bg, tok)
        diff = f"${m:+.2f}$ \\scriptsize[${lo:+.2f}$, ${hi:+.2f}$]"
        if lo > 0 or hi < 0:
            diff = f"\\textbf{{{diff}}}"
        ratio = load(bench)[f"bgkit@{bu}"]["ctx_tokens"] / bu
        rows.append(f"{name} & {bu:,} & {ratio:.0f}$\\times$ & {f(mean(bg))} & {f(mean(tok))} & "
                    f"{diff} & {f(mean(lingua))} & {f(mean(arms.get(f'text@{bu}')))} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrcrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{bgkit selection} & & \\",
        r"\cmidrule(lr){4-6}",
        r"Benchmark & Budget & Ratio & vectors & tokens & difference & LongLLMLingua & text \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


TEXT_ONLY = ["text", "base"] + [n for n, _ in BASELINES]
TEXT_LABEL = {"text": "truncated text", "base": "truncated text (original decoder)",
              "retrieval": "BM25", "lingua": "LongLLMLingua", "summary": "summary"}


def efficiency_table() -> str:
    """bgkit at the smallest budget against the best text-only arm at the largest budget of the
    same benchmark: positions per context token when bgkit matches or beats text at k times the
    budget. The paired interval is on the same items."""
    rows = []
    for bench, metric, name, n in SWEEP:
        arms = bench_arms(bench, metric)
        bus = budgets_of(arms)
        lo_b, hi_b = bus[0], bus[-1]
        bg = arms[f"bgkit@{lo_b}"]
        cands = {k: best_reader(arms, k, hi_b) if k in dict(BASELINES) else arms.get(f"{k}@{hi_b}")
                 for k in TEXT_ONLY}
        cands = {k: v for k, v in cands.items() if v}
        best_k = max(cands, key=lambda k: mean(cands[k]))
        m, lo, hi, _ = paired_ci(bg, cands[best_k])
        diff = f"${m:+.2f}$ \\scriptsize[${lo:+.2f}$, ${hi:+.2f}$]"
        if lo > 0 or hi < 0:
            diff = f"\\textbf{{{diff}}}"
        rows.append(f"{name} & {lo_b:,} & {f(mean(bg))} & {hi_b:,} & {TEXT_LABEL[best_k]} & "
                    f"{f(mean(cands[best_k]))} & {hi_b // lo_b}$\\times$ & {diff} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrlrrc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{bgkit} & \multicolumn{3}{c}{best text-only arm} & & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Benchmark & budget & score & budget & arm & score & positions & difference \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


COST = Path(__file__).resolve().parents[1] / "results/cost/cost.json"  # scripts/bgkit2_cost.py


def cost_table() -> str:
    """End-to-end cost of one question: the original decoder prefilling the whole context against
    the encoder pass plus a decoder prefill over the slots (released model, bf16, one GPU)."""
    rec = json.loads(COST.read_text())
    rows = []
    for r in rec["rows"]:
        arm = "full context" if r["arm"] == "full" else f"bgkit, {r['budget']:,}"
        ttft = r["enc_s"] + r["prefill_s"]
        enc = f"{r['enc_s']:.2f}" if r["arm"] != "full" else "--"
        length = f"{r['length']:,}" if r["arm"] == "full" else ""
        rows.append(f"{length} & {arm} & {r['positions']:,} & {enc} & {r['prefill_s']:.2f} & "
                    f"{ttft:.2f} & {r['per_token_s'] * 1000:.1f} & {r['peak_gb']:.2f} & "
                    f"{r['flops'] / 1e12:.0f} \\\\")
        if r["arm"] != "full" and r is not rec["rows"][-1] and rec["rows"][rec["rows"].index(r) + 1]["arm"] == "full":
            rows.append(r"\midrule")
    return "\n".join([
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Context & Arm & \shortstack{Decoder\\positions} & \shortstack{Encoder\\s} & "
        r"\shortstack{Prefill\\s} & \shortstack{First\\token s} & \shortstack{ms per\\token} & "
        r"\shortstack{Memory\\GB} & TFLOP \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def full_table() -> str:
    """Against the whole context: bgkit at the smallest budget, the SAME trained decoder reading the
    whole untruncated text (so only the input format differs), and the original decoder reading it."""
    rows = []
    for bench, metric, name, n in SWEEP:
        arms = bench_arms(bench, metric)
        tf = arms.get("text_full@full")
        if not tf or len(tf) < FULL_MIN_SHARED:
            continue
        bu = budgets_of(arms)[0]
        bg = arms[f"bgkit@{bu}"]
        cells = []
        for other in (tf, full_items(bench, metric)):
            if not other:
                cells.append("--")
                continue
            m, lo, hi, _ = paired_ci(bg, other)
            c = f"${m:+.2f}$ \\scriptsize[${lo:+.2f}$, ${hi:+.2f}$]"
            cells.append(f"\\textbf{{{c}}}" if lo > 0 or hi < 0 else c)
        ctx = load(bench)[f"bgkit@{bu}"]["ctx_tokens"]
        rows.append(f"{name} & {ctx / 1000:.1f}k & {bu:,} & {f(mean(bg))} & {f(mean(tf))} & "
                    f"{f(full_context(bench, metric))} & {cells[0]} & {cells[1]} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrcc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{bgkit} & \multicolumn{2}{c}{whole context} & \multicolumn{2}{c}{bgkit minus} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Benchmark & Context & budget & score & same decoder & original & same decoder & original \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def memory_agent_line() -> str:
    out = []
    for bench, name in (("longmemeval_s", "LongMemEval-S"), ("locomo", "LoCoMo")):
        v = load(bench).get("agent_memory@agent", {}).get("judge_acc")
        out.append(f"{v:.2f} on {name}")
    return " and ".join(out)


def swe_table() -> str:
    arms = json.loads((B / "swebench_loc_19999/summary.json").read_text())["arms"]
    order = [("compressed_ov", "bgkit + overview"), ("compressed", "bgkit"),
             ("full_text_ov", "text + overview"), ("full_text", "text")]
    rows = []
    for k, name in order:
        a = arms[k]
        loop = a["stop"].get("loop", 0) / a["n"]
        hits = round(a["answer_hit"] * a["n"])
        rows.append(f"{name} & {a['answer_hit']:.3f} ({hits}) & {a['read_gold_any']:.3f} & {a['answered']:.2f} & "
                    f"{a['turns']:.1f} & {loop:.2f} & {a['decoder_tokens'] / 1000:.1f}k \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arm & \shortstack{Gold file\\named} & \shortstack{Gold file\\opened} & Answered & Turns & Looped & \shortstack{Decoder\\tokens} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def web_table() -> str:
    rows = []
    for b, name in (("web_hotpot", "HotpotQA (pool)"), ("web_musique", "MuSiQue (pool)"),
                    ("browsecomp_plus", "BrowseComp-Plus")):
        s = json.loads((B / f"web_19999/{b}.json").read_text())["summary"]
        for j, (k, arm) in enumerate((("bgkit", "bgkit"), ("full_text", "text"))):
            a = s[k]
            loop = a["stop"].get("loop", 0) / a["n"]
            label = name if j == 0 else ""
            rows.append(f"{label} & {arm} & {a['judge_acc']:.2f} & {a['em']:.2f} & {a['turns']:.1f} & "
                        f"{loop:.2f} & {a['decoder_tokens'] / 1000:.1f}k \\\\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    return "\n".join([
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Benchmark & Arm & Judge acc. & EM & Turns & Looped & \shortstack{Decoder\\tokens} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def ablation_table() -> str:
    """Generative EM at 64x and teacher-forced answer loss at 4x/64x under four encoder conditions."""
    arms = json.loads(ABLATION.read_text())["arms"]
    order = [("learned", "question prompt, learned selection"), ("random", "question prompt, random selection"),
             ("noprompt", "no prompt"), ("wrongprompt", "prompt for a different question")]
    stores = [("qa_squad", "SQuAD"), ("qa_multineedle", "Multi-needle"), ("babi_filler_v3", "bAbI-in-filler")]
    rows = []
    for k, name in order:
        cells = []
        for st, _ in stores:
            m = arms[k][st]
            cells.append(f"{m['gen_x64/exact_match']:.2f} & {m['x4/loss']:.2f} & {m['x64/loss']:.2f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    ref = arms["learned"]
    rows.append(r"\midrule")
    rows.append("no context & " + " & ".join(
        f"-- & \\multicolumn{{2}}{{c}}{{{ref[st]['noctx/loss']:.2f}}}" for st, _ in stores) + r" \\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{nm}}}" for _, nm in stores)
    sub = " & ".join(["EM & $\\mathcal{L}_{4\\times}$ & $\\mathcal{L}_{64\\times}$"] * len(stores))
    return "\n".join([
        r"\begin{tabular}{l" + "rrr" * len(stores) + "}",
        r"\toprule",
        f" & {head} \\\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        f"Encoder condition & {sub} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def ratio_table() -> str:
    """Decoder loss on the gold answer by compression ratio (learned arm), against no context."""
    arm = json.loads(ABLATION.read_text())["arms"]["learned"]
    stores = [("qa_squad", "SQuAD"), ("qa_multineedle", "Multi-needle"), ("babi_filler_v3", "bAbI-in-filler")]
    rows = []
    for st, name in stores:
        m = arm[st]
        cells = " & ".join(f"{m[f'x{r}/loss']:.2f}" for r in (4, 16, 64, 128))
        rows.append(f"{name} & {cells} & {m['noctx/loss']:.2f} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrr|r}",
        r"\toprule",
        r"Task & 4$\times$ & 16$\times$ & 64$\times$ & 128$\times$ & no context \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.tex").write_text(sweep_table() + "\n")
    (OUT / "swe.tex").write_text(swe_table() + "\n")
    (OUT / "web.tex").write_text(web_table() + "\n")
    (OUT / "memory_agent.tex").write_text(memory_agent_line() + "\n")
    (OUT / "ablation.tex").write_text(ablation_table() + "\n")
    (OUT / "ratio.tex").write_text(ratio_table() + "\n")
    (OUT / "stats.tex").write_text(stats_table() + "\n")
    (OUT / "budget_fig.tex").write_text(budget_figure() + "\n")
    (OUT / "format.tex").write_text(format_table() + "\n")
    (OUT / "efficiency.tex").write_text(efficiency_table() + "\n")
    (OUT / "cost.tex").write_text(cost_table() + "\n")
    (OUT / "full.tex").write_text(full_table() + "\n")
    print(f"cells where bgkit is above the arm (interval excludes zero), of cells run: {STATS}")
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
