# BgKIT Benchmark Targets — researched 2026-08-20

Companion to `plans/capability_packaging_2026_08_20.md` §11. Four sections:
(1) the compression-literature comparison set, (2) long-context suites,
(3) agent memory / compaction, (4) repo-scale code. Every choice is screened
for **0.8B-decoder feasibility** — a benchmark where sub-1B models floor out
cannot produce a headline no matter how good the compressor is.

---

## 1. The compression-literature comparison set

The soft-prompt / context-compression literature (2023 – mid-2026) has
converged on a small de-facto standard evaluation axis. Frequency-ranked
across ~26 surveyed methods (Gist tokens, AutoCompressors, ICAE, RECOMP,
LLMLingua/LongLLMLingua/LLMLingua-2, Activation Beacon, LLoCO, SnapKV,
xRAG, UltraGist, QGC, IC-Former, CompAct, COCOM, 500xCompressor, PISCO,
LCIRC, OSCAR, ARC-Encoder, BenchPress, CLaRa, LCLM-2026, …):

| Benchmark | Used by | Metric | 0.8B feasible? |
|---|---|---|---|
| NQ / TriviaQA (RAG-QA over retrieved/gold passages) | ~11 methods (xRAG, COCOM, PISCO, OSCAR, CompAct, RECOMP, QGC, CLaRa, …) | EM / F1 | **Yes** — OSCAR ran Llama-3.2-1B; BenchPress ran Qwen3-0.6B |
| HotpotQA + 2WikiMQA + MuSiQue (multi-hop) | ~12 methods | EM / F1 | **Yes** — stresses cross-doc link preservation |
| LongBench (esp. LongBench-E QA slice: Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA) | ~9 methods | per-task F1/ROUGE | **Slice yes** (proven at 0.6B in BenchPress); full suite marginal |
| Reconstruction BLEU/ROUGE from compressed rep | ICAE, 500xCompressor, IC-Former, KVzip | BLEU / EM | **Yes** — we already train it; ICAE ref: median BLEU >0.98 @4x |
| Long-context LM perplexity (PG-19 / Pile domains) | AutoCompressors, Activation Beacon, LCIRC | PPL given compressed history | **Yes** — size-agnostic curves |
| Passkey / NIAH / RULER | Beacon, SnapKV, all KV-cache papers | retrieval acc | **Yes** at lengths within reach |
| SQuAD | 500xCompressor, ARC-Encoder, BenchPress | F1/EM | **Yes** — easiest; good for ratio sweeps |
| GSM8K CoT-prompt compression | LLMLingua line | EM | Marginal; different regime — optional |

**Key comparables (direct head-to-head numbers to beat):**

- **BenchPress** (Cornell, Oct 2025, [arXiv:2510.20797](https://arxiv.org/abs/2510.20797),
  [code](https://github.com/lil-lab/benchpress)) — **the** standardized
  compression suite; ratios {4,8,16,32,64,128}×; decoders include
  **Qwen3-0.6B** — sub-1B decoders are first-class. Cheapest credible
  comparison; their finding that bidirectional compression beats causal
  gist-tokens is architecture-validating for BgKIT.
- **xRAG** ([arXiv:2405.13792](https://arxiv.org/abs/2405.13792)) — ~175×,
  Mistral-7B: NQ 39.1 EM, TriviaQA 65.8, HotpotQA 34.1.
- **COCOM** ([arXiv:2407.09252](https://arxiv.org/abs/2407.09252)) — 4–128×
  context embeddings, NQ 0.554 EM @4×.
- **PISCO** ([arXiv:2501.16075](https://arxiv.org/abs/2501.16075)) — 16×
  at 0–3% loss, trained purely by sequence-level distillation (validates
  our self-distillation bet).
- **OSCAR** ([arXiv:2504.07109](https://arxiv.org/abs/2504.07109)) — 16×
  online compression at ≈no loss incl. a **Llama-3.2-1B generator**;
  closest small-decoder comparable.
- **500xCompressor** ([arXiv:2408.03094](https://arxiv.org/abs/2408.03094))
  — ratio-vs-quality curves 6×–480×; retains 62–73% of full-prompt QA.
- **QGC** ([ACL 2024](https://aclanthology.org/2024.acl-long.685/)) —
  closest prior on **query-conditioned** compression (NQ/TriviaQA/HotpotQA).
- **LCLM / E2E compression at scale (2026)**
  ([arXiv:2606.09659](https://arxiv.org/abs/2606.09659)) — Qwen3-Embedding-0.6B
  encoder + Qwen3-4B decoder; beats SnapKV/KVzip Pareto on RULER/LongBench.
  An encoder in exactly our size class — must-cite, must-compare.

**Exploitable gaps in this literature:** (i) almost nobody evaluates on
**code**; (ii) query-conditioned compression barely tested (QGC/OSCAR only),
and SCBench explicitly punishes query-dependent KV methods in query-hidden
multi-turn settings — BgKIT should publish both query-conditioned and
query-free numbers; (iii) sub-1B decoders unserved until late 2025 — a 0.8B
result is novel but now has direct comparables.

**Protocol convention to adopt:** ratio grid {4, 8, 16, 32, 64, 128}× plus
our aggressive tail; report EM/F1-vs-ratio curves and reconstruction
BLEU-vs-ratio; always include full-context ceiling, truncation floor, and an
LLMLingua-2 text-compression baseline at matched ratio.

---

## 2. Long-context suites

### Top targets, ranked

1. **BABILong** ([repo](https://github.com/booydar/babilong),
   [HF leaderboard](https://huggingface.co/spaces/RMT-team/babilong)) — **the
   headline benchmark.** The only major leaderboard where sub-1B models are
   *celebrated* (fine-tuned RMT/ARMT on GPT-2-137M and Mamba-130M top it at
   multi-million-token lengths); fine-tuning on the provided train split is
   in-paradigm; splits run to **10M tokens**; the paper documents RAG as
   ineffective; needle facts are low-entropy natural sentences (survive lossy
   compression) and filler is genuinely irrelevant (ideal for
   query-conditioned survivorship). **No soft-prompt compressor occupies the
   leaderboard niche.** Headline: "a 0.8B model answers over 1M–10M-token
   haystacks it cannot physically attend to, via 100× learned compression."
   Cheapest harness (HF datasets; also in lm-eval-harness PR #3256).
2. **RULER** ([NVIDIA/RULER](https://github.com/NVIDIA/RULER)) — **the
   credibility benchmark.** Standard, trivial harness, small models proven
   non-floor (EXAONE-1.2B ≈ 87 at 4K). Self-comparison design: Qwen3.5-0.8B
   full-context vs. bgkit-compressed per length — show the degradation curve
   flattening and >128K becoming reachable at all. Report QA +
   single/multi-query NIAH at 10–30×; disclose aggregation (CWE/FWE) and
   extreme-ratio exact-string losses honestly (UUID needles must survive
   verbatim — query-conditioned mode is the mitigation, and must be
   disclosed).
3. **LongBench v1 + ZeroSCROLLS (QMSum/SQuALITY)** — **the comparability
   benchmark.** Every compression baseline that matters has numbers here
   (LLoCO 30×, LongLLMLingua, Activation Beacon, LLMLingua-2); F1/ROUGE
   partial credit keeps a 0.8B measurable; QMSum + SQuALITY are the
   canonical query-focused summarization tasks — exactly the
   targeted-compression shape. Pitch: "at 30–100× we beat LLoCO's 30× with
   a decoder 9× smaller."
4. **HELMET** (RAG + LongQA + Summ categories;
   [site](https://princeton-nlp.github.io/HELMET/)) — the academic-frontier
   benchmark, 8K–128K controlled; a 2025 soft-token gist-compression
   precedent exists at only ~6× ([arXiv:2511.08128](https://arxiv.org/pdf/2511.08128))
   to leapfrog. Skip Cite/re-rank (small-model floor). Optional side-demo:
   LOFT text-retrieval as "bgkit-as-compressed-corpus-index" over a 1M-token
   corpus.
5. **Wide-net pair: self-built LogHub long-log QA + LongTableBench** —
   **No long-context log-QA benchmark exists as of mid-2026** (LogEval is
   short-input; LogQA is retriever+reader over LogHub). Building one from
   LogHub raw logs (10K–1M tokens, LogQA-style single-line answers) fills a
   real gap and is the cleanest demo of "compress the whole grep output,
   still find the needle" — it doubles as the Family-B eval suite.
   **LongTableBench** ([EMNLP'25](https://aclanthology.org/2025.findings-emnlp.638.pdf))
   — 7 formats incl. SQL/CSV/JSON, ≤128K, F1, 52 models — *published the
   claim that end-to-end LLMs beat compression-based paradigms*; a
   query-conditioned 10–50× result closing the gap on fact-retrieval and
   filtering tasks is a direct, quotable rebuttal.

### Trap benchmarks (do not headline at 0.8B)

- **LongBench v2** — 8–9B models score ~5 pts above 25% chance; a 0.8B sits
  at chance.
- **InfiniteBench realistic tasks** (En.QA/MC/Sum — 7B already near floor;
  Retrieve.KV is anti-compression by construction; synthetic PassKey@200K
  is the one usable slice).
- **Loong, AA-LCR, Oolong, MRCR/Graphwalks, NeedleBench-ATC, HoloBench
  aggregation, LOFT-SQL, LongProc** — reasoning-, aggregation-, order-, or
  output-length-bound: decoder-capacity-limited, and compression can't fix
  (or actively destroys) what they measure.
- **NoLiMa** — borderline high-risk/high-reward (semantic, non-lexical
  needle selection would showcase the query head), but 4B is the smallest
  model ever scored and the Adobe license is non-commercial. Pilot first.

### Cross-cutting protocol cautions

1. Always disclose query-conditioned vs. query-agnostic encode — several
   benchmarks become materially easier when the query is known at encode
   time (which is legitimate for the wide-net use case, but must be stated).
2. Prefer F1/ROUGE/EM-substring benchmarks; MC formats give chance floors
   that mask compression quality at 0.8B.
3. Benchmarks that require *every* token (aggregation, ATC, Oolong) are
   definitionally adversarial to compression — include one as an
   honest-limits section, never as a surprise.

---

## 3. Agent memory & compaction

**Structural insight:** the entire published compaction frontier (OpenHands
condenser, ACON, TRACE, Context-Folding, CompactionRL, AgentFold, Factory.ai)
compresses to *text* at ~10–100× and pays an LLM call per compaction. Nobody
publishes sub-token (embedding-level) compaction. BgKIT's differentiators:
compression below the text floor, no compaction-time LLM call, trainable
end-to-end. The catch: soft-prompt injection requires owning the decoder — so
every comparison holds the decoder fixed at Qwen3.5-0.8B and varies only the
context-management policy (which is also the scientifically clean design).

### Compaction targets, ranked

1. **LOCA-bench** ([arXiv:2602.07962](https://arxiv.org/pdf/2602.07962),
   [code](https://github.com/hkust-nlp/LOCA-bench)) — controllable,
   arbitrarily extendable context growth with fixed task semantics; built
   precisely for comparing context-management strategies. The headline plot:
   *success rate vs. live-window budget* — bgkit compaction vs. truncation
   vs. text summary, same 0.8B agent. Curves carry the claim, not absolute
   SOTA.
2. **AppWorld with ACON/TRACE baselines** —
   [ACON](https://arxiv.org/abs/2510.00615) (Microsoft, ICML 2026) published
   "+46% task performance for smaller LMs" from compaction, legitimizing
   small-agent runs; [TRACE](https://arxiv.org/html/2608.06503) defines the
   failure taxonomy (Pass² multi-run stability, state mislocalization,
   regressive exploration). **Adopt Pass² as a headline metric.**
3. **OpenHands condenser protocol** ([blog](https://www.openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents):
   54% vs 53% resolve at <½ per-turn cost) — use as *protocol template* but
   run on task pools where a fine-tuned 0.8B has nonzero success (SWE-smith
   synthetic bugs, SWE-Gym / R2E-Gym envs, Terminal-Bench easy split) — a
   0.8B is ~0% on real SWE-bench Verified.
4. **SWE Context Bench** ([arXiv:2602.08316](https://arxiv.org/html/2602.08316))
   — cross-task compressed-experience reuse; oracle summaries lift resolution
   26.3→34.3%; evaluates Mem0/LangMem/Supermemory as baselines. Bridges the
   memory and compaction stories.

Not winnable at 0.8B: Long-Horizon-Terminal-Bench (9.8M tokens,
frontier-only), full SWE-bench Verified leaderboards, τ-bench (context not
binding). Metrics to report everywhere: task success, tokens-per-resolved-
task, peak live-window tokens, per-turn cost curve, Pass². Factory.ai's
probe-based compression eval ([link](https://factory.ai/news/evaluating-compression))
independently validates our recall-probe design — copy the methodology.

### Memory targets, ranked

1. **LongMemEval** (S + Oracle; [arXiv:2410.10813](https://arxiv.org/abs/2410.10813),
   MIT harness) — best-instrumented; published 7–8B reader numbers
   (27.8–47.2%) make a 0.8B run defensible if the claim is
   *accuracy-per-injected-context-token at fixed reader*. Winnable at 0.8B:
   information extraction, single-session, knowledge updates; concede
   temporal reasoning + abstention.
2. **LoCoMo** ([arXiv:2402.17753](https://arxiv.org/abs/2402.17753)) —
   mandatory checkbox (Mem0 92.5, Zep 80.3 self-reported / 58.4 corrected,
   Letta 74.0, OpenAI 52.9), but fits in ~26k tokens and is widely
   criticized — report, don't headline. Use Mem0's open harness
   ([memory-benchmarks](https://github.com/mem0ai/memory-benchmarks)).
3. **BEAM** ([code](https://github.com/mohammadtavakoli78/BEAM), ICLR 2026)
   — 1M–10M-token conversations; full context impossible, text-memory
   systems spend ~7k tokens/query. Frame: constant-size compressed store vs.
   growing text store; target fact-tracking / preference / knowledge-update
   categories at 1M; concede 10M temporal/event-ordering.
4. **MemBench capacity axis** ([arXiv:2506.21605](https://arxiv.org/abs/2506.21605))
   + **MemoryAgentBench** AR/LRU splits ([arXiv:2507.05257](https://arxiv.org/abs/2507.05257))
   — "degradation as store grows" and ≥100k long-range understanding are
   tailor-made for a fixed-budget compressed store.

**Contamination caution:** MSC and PerLTQA are BgKIT Phase-2 *training*
corpora — strictly split or exclude from evals. Nearest latent-memory
competitors to beat: MemoryLLM (~16–20k retention horizon) and M+
([arXiv:2502.00592](https://arxiv.org/abs/2502.00592), ~150k horizon).

### Training-data windfall (Family A)

Public long-trajectory corpora directly usable for compaction SFT:

| Corpus | Size | Source |
|---|---|---|
| SWE-Zero (NVIDIA) | **318k** OpenHands trajectories | [HF](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) |
| Toucan | **1.5M** tool-agentic trajectories (~500 real MCP servers) | [HF/GitHub](https://github.com/TheAgentArk/Toucan) |
| SWE-rebench (Nebius) | 67k trajectories | [HF](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) |
| SWE-Hero (NVIDIA) | 34k trajectories | [HF](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) |
| SWE-smith | 26k trajectories (+ synthetic-bug envs) | [GitHub](https://github.com/SWE-bench/SWE-smith) |

This is orders of magnitude more compaction-SFT raw material than the
original OpenHands download script assumed.

---

## 4. Repo-scale code

**The validating prior:** "On the Effectiveness of Context Compression for
Repository-Level Tasks" ([arXiv:2604.13725](https://arxiv.org/html/2604.13725))
finds text-to-text compression collapses to no-context level at 7–12×, while
**text-to-vector latent compression exceeds full-context performance at every
ratio from 4× to 128×** (Qwen2.5-Coder-3B/7B, ComplexCodeEval) — interpreted
as learned filtering of repo noise. This is the closest published validation
of BgKIT's core bet and the natural beat-this target. Hard-compression
baseline: LongCodeZip ([arXiv:2510.00446](https://arxiv.org/abs/2510.00446),
5.6× ceiling). Product baseline: aider's tree-sitter repo map
([docs](https://aider.chat/docs/repomap.html)).

### Showcase benchmarks, ranked

1. **CrossCodeEval (+ CrossCodeLongEval, RepoEval)**
   ([arXiv:2310.11248](https://arxiv.org/abs/2310.11248)) — natively
   three-armed (no cross-file ctx / retrieved / oracle), cheap EM/ES, and the
   only family with **published sub-1B anchors**: Qwen2.5-Coder-0.5B = 24.6
   avg EM (CCEval) / 19.7 (CCLongEval) / 28.1 (RepoEval); 1.5B = 40.2/28.3/40.5
   ([Qwen2.5-Coder report](https://arxiv.org/abs/2409.12186)). Protocol:
   replace the 2,048-token BM25 arm with whole-repo bgkit embeddings at a
   smaller token-equivalent budget.
2. **Long Code Arena — project-level completion**
   ([arXiv:2406.11612](https://arxiv.org/abs/2406.11612)) — the only
   benchmark whose context axis reaches our regime (>768K-char "huge"
   bucket, 296 instances); line-category labels (inproject vs infile vs
   common) isolate exactly the ambient-repo signal. Cheap EM.
3. **RepoBench v1.1** ([arXiv:2306.03091](https://arxiv.org/abs/2306.03091))
   — 2k→16k context sweep → "bgkit embeddings vs. N raw tokens" trade-off
   curves; leakage-controlled at collection.
4. **SWE-QA + DeepCodeBench** (repo QA, no execution infra) —
   [SWE-QA](https://arxiv.org/abs/2509.14635) (720 Qs / 15 repos, judge
   scoring; frontier direct 51–62, RAG +8–15, agents +13–16 — deltas
   transfer directly to our no-ctx/RAG/bgkit design);
   [DeepCodeBench](https://www.qodo.ai/blog/deepcodebench-real-world-codebase-understanding-by-qa-benchmarking/)
   fact-recall scoring gives graded credit a 0.8B registers on and ships
   full-context/no-context conditions. SWE-QA-Pro adds decontaminated
   long-tail repos + SFT trajectories (also usable as training data).
5. **RepoQA** ([evalplus](https://github.com/evalplus/repoqa)) — secondary,
   as a compression-fidelity stress test (verbatim function from compressed
   embeddings); even 7B instruct models scatter 2–28%.

### Expectations at 0.8B

- Cross-file completion is the healthy zone (0.5B ≈ 20–28 EM, 1.5B ≈ 28–40 —
  Qwen3.5-0.8B lands between; the claim is the delta: bgkit ≥ BM25-RAG at
  equal decoder, both ≫ no-context which is single-digit).
- Repo QA: modest absolutes (~20–35 judged); position as context-condition
  deltas, not leaderboard absolutes.
- SWE-bench patch resolution: assume ~0% (smallest above-floor specialists
  are 7B). Defer to the D3/RL era; use SWE-bench-Live/Pro for hygiene if
  attempted.
- Leakage posture everywhere: within-decoder relative comparison
  (no-context / repo-map / BM25-RAG / bgkit, same decoder), optionally
  validated on a post-cutoff slice (SWE-bench-Live monthly issues,
  self-collected post-2026 repos).

---

## 5. Consolidated benchmark program

Tiered recommendation across all four research areas:

| Tier | Benchmark | Family | Why |
|---|---|---|---|
| **Headline** | BABILong (1M–10M) | B (+A) | sub-1B leaderboard, RAG fails there, niche unoccupied |
| **Headline** | LOCA-bench success-vs-window curves | A | built for compaction comparisons; curves not absolutes |
| **Credibility** | RULER (self-comparison per length) | B | standard, small-model-proven, honest-ablation friendly |
| **Comparability** | BenchPress + LongBench v1 (+ QMSum/SQuALITY) | B | direct numbers vs. LLoCO/LLMLingua/Beacon; Qwen3-0.6B precedent |
| **Comparability** | NQ/TriviaQA/HotpotQA RAG-QA @ matched ratios | B | the soft-compression literature's standard axis (xRAG/COCOM/PISCO/OSCAR) |
| **Wide-net** | Self-built LogHub long-log QA; LongTableBench | B | fills a real gap; rebuts a published anti-compression claim |
| **Memory** | LongMemEval (S+Oracle); BEAM-1M categories; LoCoMo checkbox | C | fixed-reader accuracy-per-injected-token framing |
| **Compaction** | AppWorld (ACON/TRACE protocol, Pass²); OpenHands-condenser protocol on SWE-smith/R2E pools | A | published small-LM compaction baselines to beat |
| **Repo** | CrossCodeEval family (0.5B anchors); Long Code Arena huge-bucket; SWE-QA/DeepCodeBench | D | three-armed by construction; cheap scoring |
| **Fidelity** | Reconstruction BLEU vs. ICAE/500xCompressor; RepoQA stress test | all | we already train this; ratio-vs-BLEU curves |
| **Honest limits** | RULER CWE/FWE, HoloBench aggregation, LongBench v2 footnote | — | aggregation/reasoning tasks are definitionally anti-compression |

Universal protocol: fixed decoder (Qwen3.5-0.8B) across all arms; arms =
no-context / truncation / text-summary or LLMLingua-2 at matched ratio /
BM25-RAG at matched token budget / bgkit (query-free) / bgkit
(query-conditioned, disclosed); ratio grid {4,8,16,32,64,128}× + aggressive
tail; report tokens-injected everywhere; Pass² for agentic runs.

Integration order (cheapest first): lm-eval-harness (BABILong + InfiniteBench
PassKey in one PR-tested harness) → RULER docker → BenchPress → LongMemEval →
CrossCodeEval → AppWorld/LOCA-bench (agentic, needs the P1 checkpoint).
