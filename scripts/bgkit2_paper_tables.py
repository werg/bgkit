#!/usr/bin/env python
"""LaTeX result tables for the bgkit paper (paper/tables/*.tex), generated from the eval outputs so
no number is typed by hand: the S2-final showcase sweep, SWE-bench Lite localisation and the
offline web-research evals, plus the full-context reference column (the original decoder reading the
whole context, scored on the same items) and the prompt/selection ablation of an earlier checkpoint.

  .venv/bin/python scripts/bgkit2_paper_tables.py
"""

from __future__ import annotations

import json
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


def load(bench: str) -> dict:
    return json.loads((B / "showcase/s2_final" / f"{bench}.json").read_text())["summary"]


def _ids(path: Path) -> set[str]:
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def full_context(bench: str, metric: str) -> float | None:
    """The original decoder (adapter off) reading the whole context, on the sweep's items."""
    d = B / "showcase" / FULL_RUN
    rows_path = d / (f"{bench}.judged.jsonl" if metric == "judge_acc" else f"{bench}.rows.jsonl")
    if not rows_path.exists():
        return None
    ids = _ids(B / "showcase/s2_final" / f"{bench}.rows.jsonl")
    key = "judge" if metric == "judge_acc" else metric
    vals = [float(r[key]) for r in map(json.loads, rows_path.read_text().splitlines())
            if r["arm"] == "base_full@full" and r["id"] in ids]
    return sum(vals) / len(vals) if len(vals) >= FULL_MIN_SHARED else None


def f(v, bold=False) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def sweep_table() -> str:
    rows = []
    for bench, metric, name, n in SWEEP:
        s = load(bench)
        budgets = sorted({int(k.split("@")[1]) for k in s if k.split("@")[1].isdigit()})
        for j, bu in enumerate(budgets):
            vals = {a: s.get(f"{a}@{bu}", {}).get(metric) for a in ("bgkit", "text", "base", "summary")}
            best = max(v for v in vals.values() if v is not None)
            cells = [f(vals[a], bold=(vals[a] == best)) for a in ("bgkit", "text", "base", "summary")]
            label = f"{name} ({n})" if j == 0 else ""
            if j == 1:  # second row: mean context length, so the compression ratio can be read off
                label = f"\\quad {s[f'bgkit@{bu}']['ctx_tokens'] / 1000:.1f}k tokens"
            full = f(full_context(bench, metric)) if j == 0 else ""
            rows.append(f"{label} & {bu:,} & " + " & ".join(cells) + f" & {full}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    return "\n".join([
        r"\begin{tabular}{llrrrr|r}",
        r"\toprule",
        r"Benchmark ($n$) & Budget & bgkit & text & base & summary & \shortstack{base,\\full context} \\",
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
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
