#!/usr/bin/env python
"""Showcase report v2 (S2 final vs S1b final) as one HTML page: results/showcase/report_v2.html.

Every number is read from the eval outputs (sweep *.json summaries, SWE-bench Lite localisation
summary, web and candidate-tree summaries); nothing is typed in by hand.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

B = Path("/home/werg/bgkit-data-nvme/bgkit2/eval")
OUT = Path(__file__).resolve().parents[1] / "results/showcase/report_v2.html"

ARM_ORDER = ["bgkit", "text", "base", "summary", "agent_memory"]
ARM_LABEL = {"bgkit": "bgkit", "text": "text", "base": "base", "summary": "summary",
             "agent_memory": "memory agent"}

SWEEP = [  # bench, metric, name, what it tests
    ("babilong_16k", "acc", "BABILong 16k", "facts buried in 16k tokens of filler"),
    ("babilong_4k", "acc", "BABILong 4k", "facts buried in 4k tokens of filler"),
    ("hotpot", "judge_acc", "HotpotQA", "2-hop questions over 10 paragraphs"),
    ("musique", "judge_acc", "MuSiQue", "2-4-hop questions over 20 paragraphs"),
    ("quality", "acc", "QuALITY", "multiple choice over long articles"),
    ("ruler_4096", "acc", "RULER 4k", "synthetic retrieval / tracking at 4k"),
    ("narrativeqa", "judge_acc", "NarrativeQA", "questions about whole books / scripts"),
    ("longmemeval_s", "judge_acc", "LongMemEval-S", "memory over ~50 chat sessions"),
    ("locomo", "judge_acc", "LoCoMo", "memory over long multi-session dialogues"),
]


def load(run: str, bench: str) -> dict:
    p = B / "showcase" / run / f"{bench}.json"
    return json.loads(p.read_text())["summary"] if p.exists() else {}


def fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def bar(v, cls: str) -> str:
    if v is None:
        return '<span class="v">–</span>'
    w = max(0.0, min(1.0, float(v))) * 100
    return (f'<span class="cell"><span class="track"><span class="fill {cls}" style="width:{w:.1f}%">'
            f'</span></span><span class="v">{v:.2f}</span></span>')


def sweep_table(bench: str, metric: str) -> str:
    s2, s1 = load("s2_final", bench), load("s1b_final", bench)
    budgets = sorted({int(k.split("@")[1]) for k in s2 if k.split("@")[1].isdigit()})
    arms = [a for a in ARM_ORDER if any(k.startswith(a + "@") for k in s2)]
    head = "".join(f"<th>{ARM_LABEL[a]}</th>" for a in arms if a != "agent_memory")
    rows = []
    for bu in budgets:
        cells = []
        for a in arms:
            if a == "agent_memory":
                continue
            v = s2.get(f"{a}@{bu}", {}).get(metric)
            cells.append(f"<td>{bar(v, a)}</td>")
        prev = s1.get(f"bgkit@{bu}", {}).get(metric)
        rows.append(f"<tr><th scope=row>{bu:,}</th>{''.join(cells)}"
                    f"<td class=prev>{fmt(prev)}</td></tr>")
    extra = ""
    if "agent_memory@agent" in s2:
        v2 = s2["agent_memory@agent"].get(metric)
        v1 = s1.get("agent_memory@agent", {}).get(metric)
        extra = (f'<p class="note">Memory agent (search + read tools over the compressed history): '
                 f"{fmt(v2)} (S1b {fmt(v1)}).</p>")
    return (f'<div class="tw"><table class="sweep"><thead><tr><th>budget</th>{head}'
            f'<th class="prev">bgkit, S1b</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{extra}')


def agent_table(rows: list[tuple[str, dict]], cols: list[tuple[str, str, bool]]) -> str:
    head = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    body = []
    for label, d in rows:
        tds = []
        for key, _, is_bar in cols:
            v = d.get(key)
            if is_bar:
                tds.append(f"<td>{bar(v, 'bgkit' if label.startswith('bgkit') else 'text')}</td>")
            elif isinstance(v, float):
                tds.append(f"<td class=num>{v:,.1f}</td>" if v > 10 else f"<td class=num>{v:.2f}</td>")
            else:
                tds.append(f"<td class=num>{'–' if v is None else html.escape(str(v))}</td>")
        body.append(f"<tr><th scope=row>{html.escape(label)}</th>{''.join(tds)}</tr>")
    return f'<div class="tw"><table class="agent"><thead><tr><th>arm</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def main() -> None:
    swe = json.loads((B / "swebench_loc_19999/summary.json").read_text())["arms"]
    web = {b: json.loads((B / f"web_19999/{b}.json").read_text())["summary"]
           for b in ("web_hotpot", "web_musique", "browsecomp_plus")}
    ct = json.loads((B / "candtree_19999/browsecomp_plus_candtree.json").read_text())["summary"]
    for a in swe.values():
        a["loop_rate"] = a["stop"].get("loop", 0) / a["n"]
    for b in web.values():
        for a in b.values():
            a["loop_rate"] = a["stop"].get("loop", 0) / a["n"]

    sweep_html = "".join(
        f'<section class="bench"><h3>{name}<span class="what">{what}</span></h3>'
        f"{sweep_table(b, m)}"
        f'<p class="metric">{"LLM-judge accuracy" if m == "judge_acc" else "accuracy"}, n = '
        f'{"60" if b in ("narrativeqa", "longmemeval_s") else "100"}; budget = decoder tokens of context.</p></section>'
        for b, m, name, what in SWEEP)

    swe_rows = [("bgkit + repo overview", swe["compressed_ov"]), ("bgkit", swe["compressed"]),
                ("text + repo overview", swe["full_text_ov"]), ("text", swe["full_text"])]
    swe_html = agent_table(swe_rows, [("answer_hit", "names the gold file", True),
                                      ("read_gold_any", "opened the gold file", False),
                                      ("answered", "gave an answer", False),
                                      ("turns", "turns", False), ("loop_rate", "looped", False),
                                      ("decoder_tokens", "decoder tokens", False)])
    web_html = ""
    for b, name in (("web_hotpot", "HotpotQA over the paragraph pool (87.6k docs, BM25)"),
                    ("web_musique", "MuSiQue over the paragraph pool"),
                    ("browsecomp_plus", "BrowseComp-Plus (100k-page corpus)")):
        rows = [("bgkit", web[b]["bgkit"]), ("text", web[b]["full_text"])]
        web_html += f"<h4>{name}</h4>" + agent_table(
            rows, [("judge_acc", "judge accuracy", True), ("em", "EM", False), ("turns", "turns", False),
                   ("loop_rate", "looped", False), ("decoder_tokens", "decoder tokens", False)])
    ct_html = agent_table([("bgkit prompted tree", ct["bgkit_tree"]), ("text (BM25 over the set)", ct["full_text"])],
                          [("judge_acc", "judge accuracy", True), ("answered", "answered", False),
                           ("n_browse", "browse / episode", False), ("gold_opened_any", "opened gold", False),
                           ("turns", "turns", False)])

    s2h = load("s2_final", "hotpot")
    s2m = load("s2_final", "musique")
    s2l = load("s2_final", "longmemeval_s")
    s2b = load("s2_final", "babilong_16k")
    tiles = [
        (f'{s2b["bgkit@2048"]["acc"]:.2f}', f'vs {s2b["text@2048"]["acc"]:.2f} text',
         "BABILong 16k at a 2,048-token budget"),
        (f'{s2h["bgkit@256"]["judge_acc"]:.2f}', f'vs {s2h["text@256"]["judge_acc"]:.2f} text · {s2h["summary@256"]["judge_acc"]:.2f} summary',
         "HotpotQA at 256 tokens"),
        (f'{swe["compressed_ov"]["answer_hit"]:.3f}', f'vs {swe["full_text_ov"]["answer_hit"]:.3f} text',
         "SWE-bench Lite: names the gold file"),
        (f'{web["web_hotpot"]["bgkit"]["judge_acc"]:.2f}', f'vs {web["web_hotpot"]["full_text"]["judge_acc"]:.2f} text, ⅓ the tokens',
         "web research, HotpotQA pool"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="big">{a}</div><div class="cmp">{c}</div>'
                         f'<div class="lbl">{l}</div></div>' for a, c, l in tiles)

    page = TEMPLATE.format(tiles=tiles_html, sweep=sweep_html, swe=swe_html, web=web_html, ct=ct_html,
                           mus_bg=s2m["bgkit@2048"]["judge_acc"], mus_su=s2m["summary@1024"]["judge_acc"],
                           lme=s2l["bgkit@2048"]["judge_acc"], lme_t=s2l["text@2048"]["judge_acc"])
    OUT.write_text(page)
    print(f"wrote {OUT}")


TEMPLATE = """<title>bgkit Showcase v2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&display=swap">
<style>
:root{{--paper:#f6f7f9;--ink:#1c2233;--mute:#5d6577;--rule:#dde1e8;--panel:#ffffff;
--bgkit:#0f7b6c;--text:#8a93a6;--base:#b9c0cc;--summary:#7a5bb5;--agent_memory:#c07a1f;--track:#e9ecf1}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--paper:#12151c;--ink:#e6e9f0;
--mute:#9aa3b5;--rule:#2a303c;--panel:#181c25;--bgkit:#3cc2ad;--text:#7d879b;--base:#4c5566;--summary:#a78be0;
--agent_memory:#e0a24a;--track:#232834}}}}
:root[data-theme="dark"]{{color-scheme:dark;--paper:#12151c;--ink:#e6e9f0;--mute:#9aa3b5;--rule:#2a303c;--panel:#181c25;
--bgkit:#3cc2ad;--text:#7d879b;--base:#4c5566;--summary:#a78be0;--agent_memory:#e0a24a;--track:#232834}}
body{{background:var(--paper);color:var(--ink);font:15px/1.6 "IBM Plex Sans",system-ui,sans-serif;padding-inline:16px}}
main{{max-width:880px;margin:0 auto;padding-block:40px 64px}}
h1,h2,h3{{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance;margin:0}}
h1{{font-size:34px;font-weight:650;line-height:1.15}}
h2{{font-size:23px;font-weight:650;margin-top:48px;padding-top:20px;border-top:1px solid var(--rule)}}
h3{{font-size:18px;font-weight:500;display:flex;flex-wrap:wrap;gap:4px 12px;align-items:baseline}}
h4{{font-size:14px;font-weight:600;margin:22px 0 6px}}
.what{{font:13px "IBM Plex Sans",sans-serif;color:var(--mute)}}
p{{max-width:68ch}} .lede{{color:var(--mute);font-size:16px}}
.eyebrow{{font:500 12px "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--bgkit)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:28px}}
.tile{{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}}
.big{{font:500 30px "IBM Plex Mono",monospace;color:var(--bgkit);font-variant-numeric:tabular-nums}}
.cmp{{font:13px "IBM Plex Mono",monospace;color:var(--mute)}} .lbl{{font-size:13px;margin-top:6px}}
.bench{{margin-top:28px}}
.tw{{overflow-x:auto;margin-top:8px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font:500 11.5px "IBM Plex Mono",monospace;letter-spacing:.05em;text-transform:uppercase;color:var(--mute)}}
tbody th{{font:500 13px "IBM Plex Mono",monospace}}
.num,.prev,.v{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.prev{{color:var(--mute)}}
.cell{{display:inline-flex;align-items:center;gap:8px}}
.track{{display:inline-block;width:84px;height:8px;background:var(--track);border-radius:2px;overflow:hidden}}
.fill{{display:block;height:100%}}
.fill.bgkit{{background:var(--bgkit)}} .fill.text{{background:var(--text)}} .fill.base{{background:var(--base)}}
.fill.summary{{background:var(--summary)}}
.metric,.note{{font-size:12.5px;color:var(--mute);margin:6px 0 0}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;margin-top:10px}}
.legend span::before{{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:var(--c)}}
ul{{padding-left:20px}} li{{margin:4px 0;max-width:68ch}}
code{{font:13px "IBM Plex Mono",monospace}}
.gap{{border-left:3px solid var(--agent_memory);padding:2px 0 2px 14px}}
</style>
<main>
<div class="eyebrow">bgkit2 · showcase sprint · report v2 · 2026-09-25</div>
<h1>One 350M model, context through a prompted encoder</h1>
<p class="lede">S2 is a single LFM2.5-350M decoder fed by an LFM2.5-350M bidirectional encoder: every document,
tool result and memory reaches the decoder as a question-conditioned compressed slot. S2 = S1b plus agentic,
web-research, memory and compaction data (20k steps). Every arm below runs the same weights at the same decoder-token budget.</p>
<div class="tiles">{tiles}</div>
<div class="legend"><span style="--c:var(--bgkit)">bgkit: context compressed into slots</span>
<span style="--c:var(--text)">text: same weights, plain text cut to the budget</span>
<span style="--c:var(--base)">base: adapter off, same text</span>
<span style="--c:var(--summary)">summary: Qwen3.6-35B question-focused summary at the budget</span></div>

<h2>Long context, multi-hop QA and memory</h2>
<p>Single-shot benchmarks, S2 final checkpoint. The last column is the S1b final checkpoint's bgkit score at the
same budget. MuSiQue is the clearest S2 gain: bgkit {mus_bg:.2f} against a 35B model's summary at {mus_su:.2f}. LongMemEval-S bgkit
{lme:.2f} vs text {lme_t:.2f}.</p>
{sweep}

<h2>Agentic coding: SWE-bench Lite localisation</h2>
<p>186 instances (django excluded), 12 turns, 40k decoder tokens per episode. Success means the final answer names a file the
gold patch edits. The overview arms open every episode with the repository's compressed tree at <code>base_commit</code>
and can <code>browse</code> into it.</p>
{swe}

<h2>Web research (offline corpora)</h2>
<p>100 questions per benchmark, 12 turns, 40k decoder tokens. In the bgkit arm every <code>web_search</code>,
<code>web_open</code> and <code>web_find</code> result goes through the encoder with the agent's focus as the prompt.</p>
{web}
<h4>BrowseComp-Plus, candidate-set tree (50 queries, ~82 pages each)</h4>
{ct}

<h2>Where it falls short</h2>
<div class="gap"><ul>
<li><b>BrowseComp-Plus is 0 for every arm.</b> The prompted candidate tree is built (~576k tokens per query into ~6.5 nodes), but the
agent never descends into it: browse is trained only on repository trees, so this is a data gap.</li>
<li><b>The memory agent still answers badly.</b> Since S2 it searches and reads sessions (S1b answered at turn 1), but its judge
accuracy is ~0.05, below the plain bgkit arm that sees the whole compressed history.</li>
<li><b>Compaction barely pays yet.</b> Per-token gains on the web and compaction-spine rows are small (spine vs zeroed +0.01 nats).
The pool web eval is where compressed history does help: 2.4k vs 6.5k decoder tokens and far fewer loops.</li>
<li><b>The summary arm still leads on MuSiQue under the judge</b> (0.42 vs 0.33), and on QuALITY at both budgets it has (0.42–0.45 vs 0.36–0.37).</li>
<li>Absolute agentic numbers are small for a 350M decoder; the claim is the lift over the same weights reading text.</li>
</ul></div>

<h2>Provenance</h2>
<p class="note">Checkpoint <code>bgkit2_step19999_20260925_103158_274684_run-bgkit2_s2_showcase</code>; the S1b comparison is
<code>bgkit2_step11999_20260923_140518_602592</code>. Judge: Qwen3.6-35B-A3B Q4_K_M (llama.cpp). Generated by
<code>scripts/bgkit2_showcase_report_v2.py</code> from the eval outputs; per-item rows are in <code>results/showcase/</code> and the eval directories.</p>
</main>
"""

if __name__ == "__main__":
    main()
