# Results

Raw outputs behind the paper's tables (`scripts/bgkit2_paper_tables.py` reads the same files from
the evaluation directory).

| Path | Paper | Contents |
|---|---|---|
| `showcase/showcase/s2_final/` | Tables 1, 2, 6, 10 | The benchmark sweep of the released checkpoint: per benchmark a summary JSON (score per arm and budget), per-item rows (`*.rows.jsonl`, prediction and gold) and LLM-judge verdicts (`*.judged.jsonl`) |
| `showcase/showcase/s2_baselines/` | Tables 1, 2, 6, 10 | Baselines run on the same items afterwards: `retrieval_*` (BM25), `lingua_*` (LongLLMLingua), `summary_*` (summaries read by both decoders), `keptok_*` (`bgkit_tokens`: token embeddings at the encoder's kept positions), `textslot_*` (P19g, an earlier text-in-slot adapter), `textfull_*` (`text_full`: the trained decoder reading the whole context; Table 3) |
| `showcase/showcase/judged_qwen36_nvfp4/s2_baselines/` | Tables 1, 2, 6, 10 | LLM-judge verdicts for those baselines (Qwen3.6-35B-A3B, NVFP4 build) |
| `showcase/showcase/p34_final_full/` | Tables 1 (last column), 3 | The original decoder (adapter off, `base_full`) reading the whole context; scored on the sweep's items |
| `cost/cost.json` | Table 5 | `scripts/bgkit2_cost.py`: time, memory and FLOPs of one question, full-context prefill against encoder + slots |
| `showcase/swebench_loc_s2/` | Table 8 | SWE-bench Lite localisation (186 non-Django instances), four arms |
| `showcase/web_s2/` | Table 9 | Web research: HotpotQA and MuSiQue over pooled paragraphs, BrowseComp-Plus |

Arms: `bgkit` (compressed slots, question as encoder prompt), `text` (same decoder, middle-truncated
text), `base` (original decoder, same text), `summary` (question-focused summary by
Qwen3.6-35B-A3B), `retrieval`, `lingua`, `bgkit_tokens`, `textslot` (see above), `*_full` (whole
context); a `base_` prefix means the original decoder read that arm's text. Keys are `<arm>@<budget>`.
