# Results

Raw outputs behind the paper's tables (`scripts/bgkit2_paper_tables.py` reads the same files from
the evaluation directory).

| Path | Paper | Contents |
|---|---|---|
| `showcase/showcase/s2_final/` | Table 1 | The benchmark sweep of the released checkpoint: per benchmark a summary JSON (score per arm and budget), per-item rows (`*.rows.jsonl`, prediction and gold) and LLM-judge verdicts (`*.judged.jsonl`) |
| `showcase/showcase/p34_final_full/` | Table 1, last column | The original decoder (adapter off, `base_full`) reading the whole context; scored on the sweep's items |
| `showcase/swebench_loc_s2/` | Table 4 | SWE-bench Lite localisation (186 non-Django instances), four arms |
| `showcase/web_s2/` | Table 5 | Web research: HotpotQA and MuSiQue over pooled paragraphs, BrowseComp-Plus |

Arms: `bgkit` (compressed slots, question as encoder prompt), `text` (same decoder, middle-truncated
text), `base` (original decoder, same text), `summary` (question-focused summary by
Qwen3.6-35B-A3B), `*_full` (whole context). Keys are `<arm>@<budget>`.
