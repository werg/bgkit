# Q&A Document Collections for Language-Model and Retrieval Training

**Prepared:** 31 July 2026  
**Scope:** English-language datasets containing approximately 50–1,000 source documents per collection, together with questions and answers or closely related answer labels. PubMedQA is intentionally excluded.

## Executive summary

The strongest datasets for an end-to-end retrieval-augmented generation (RAG) training pipeline are:

1. **OR-ShARC** — the closest native match: 651 rule documents, 17,936 training samples, an open-retrieval task, conversational answers.
2. **MultiDoc2Dial** — 488 documents and thousands of multi-turn dialogues with gold document IDs and supporting span IDs.
3. **PolicyQA** — 115 privacy policies and 25,017 extractive QA examples; high training volume, but lower question diversity.
4. **ConditionalQA** — 652 long government documents with complex conditional answers and document-disjoint splits. Its stated NLP-research purpose matches the intended use of this report.
5. **ContractNLI** — 607 contracts with document-level decisions and evidence spans under CC BY 4.0; not literal QA, but straightforward to convert into instruction-style questions.

QuALITY, MultiRC/ERASER, ShARC, and MCTest are useful supplementary collections. They offer long-context, evidence-selection, conversational-rule, and simple multiple-choice supervision respectively, but most do not natively require retrieval from the full collection.

These collections are large enough for:

- supervised fine-tuning (SFT) or LoRA/adaptor training;
- dense or sparse retriever training;
- reranker training;
- evidence-selection training;
- multi-task RAG training.

They are not individually large enough to pretrain a general language model from scratch.

## At-a-glance comparison

| Dataset | Documents | Supervision volume | Native collection retrieval? | Gold evidence | Primary answer form | Practical access note |
|---|---:|---:|---|---|---|---|
| [OR-ShARC](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) | 651 rule texts | 17,936 train; 1,105 dev; 2,373 test | **Yes** | Gold rule document; dialogue grounding | Yes/no, irrelevant, or clarification question | CC BY-SA 3.0 stated by project |
| [MultiDoc2Dial](https://doc2dial.github.io/multidoc2dial/) | 488 web documents | 3,474 train dialogues; 661 validation dialogues; about 21,451 train turns | **Partly/native** | `doc_id` and span IDs on grounded turns | Conversational free text | Public JSON; verify dataset/content terms for intended use |
| [ConditionalQA](https://github.com/haitian-sun/ConditionalQA) | 652 long government pages | 2,338 train; 285 dev; 804 test questions | Derived by pooling documents | Source document plus answers and conditions | Extractive, yes/no, multiple-answer, unanswerable | Public repository; intended for NLP research |
| [PolicyQA](https://github.com/wasiahmad/PolicyQA) | 115 privacy policies | 25,017 QA examples derived from 714 human questions | Derived by pooling policies | Exact answer span and source policy | Extractive span | Code is MIT; audit terms of OPP-115/source policies |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | 607 NDAs | 17 hypotheses per document; roughly 10,000 document–hypothesis decisions | Derived by pooling contracts | Exact evidence-span indices | Entailment, contradiction, not mentioned | Dataset is CC BY 4.0 |
| [QuALITY](https://nyu-mll.github.io/quality/) | 265 long articles | 6,737 questions | Derived by pooling articles | Source article only | Four-way multiple choice | CC BY 4.0 stated by project |
| [MultiRC / ERASER](https://www.eraserbenchmark.com/) | 871 paragraphs/documents | About 6,000 multi-sentence questions | Derived by pooling documents | Supporting sentences/rationales | Multiple-choice, possibly multiple correct | Public research release; verify bundled terms |
| [ShARC](https://sharc-data.github.io/) | 948 rule snippets | 6,058 dialog-tree utterances and 6,637 scenarios; expanded instance counts depend on representation | No: rule text is supplied | Source rule and dialogue path | Yes/no, irrelevant, or clarification question | CC BY-SA 3.0 stated by project |
| [MCTest](https://mattr1.github.io/mctest/) | 660 short stories | 2,640 questions | Derived by pooling stories | Source story only | Four-way multiple choice | MSR-LA-style terms |

“Native collection retrieval” means the published task expects the system to find the relevant document from the corpus. “Derived” means the release supplies the source document with the question; a retrieval task can be constructed by hiding that document ID and pooling the corpus.

---

## 1. OR-ShARC

### Why it is a top candidate

OR-ShARC is the most direct match for training a retriever and a language model jointly. It adapts ShARC to an open-retrieval setting: the system searches a shared collection of rule texts before answering a user’s high-level question or asking a necessary clarification question.

### Data

- **Corpus:** 651 natural-language rule texts.
- **Training:** 17,936 samples.
- **Development:** 1,105 samples.
- **Test:** 2,373 samples.
- **Average dialogue length:** approximately 1.4 turns in the reported version.
- **Question rewriting:** ambiguous initial ShARC questions were rewritten where necessary so retrieval does not depend on seeing the gold rule first.

### Available supervision

- user scenario;
- initial question;
- dialogue history;
- relevant rule text/document;
- final decision or next clarification question.

This supports:

- query-to-document retriever training;
- document reranking;
- decision classification;
- follow-up-question generation;
- answer generation conditioned on retrieved text and history.

### Strengths

- Retrieval is part of the original task rather than an artificial conversion.
- The corpus size is exactly in the desired range.
- Almost 18,000 training samples provide useful SFT volume.
- Questions were rewritten to be usable without access to the gold document.
- It includes incomplete-information cases where asking a clarification is preferable to hallucinating.

### Limitations

- Rule texts are relatively short; this is not a long-document chunking benchmark.
- Outputs are specialized: yes/no/irrelevant or a follow-up question.
- The government-policy domain may produce strong domain and template regularities.
- CC BY-SA 3.0 includes attribution and ShareAlike conditions; downstream compliance should be reviewed.

### Sources

- [Official repository and data](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading)
- [Paper](https://arxiv.org/abs/2102.08633)
- [Original ShARC data](https://sharc-data.github.io/data.html)

---

## 2. MultiDoc2Dial

### Why it is a top candidate

MultiDoc2Dial contains realistic, goal-oriented conversations grounded in multiple public-information documents. Its annotations directly connect agent turns to source documents and supporting spans.

### Data

- **Corpus:** 488 documents.
- **Domains:** four public-service domains.
- **Total:** 4,796 conversations, averaging roughly 14 turns.
- **Training dialogues:** 3,474.
- **Validation dialogues:** 661.
- **Reported test dialogues:** 661; the public v1 download exposes only dummy test content, so plan around train and validation unless the challenge data is separately available.
- **Turn-level training rows:** approximately 21,451 in the common dataset representation.

### Available supervision

The document JSON contains:

- `doc_id`;
- raw and marked-up document content;
- structured spans;
- headings and parent titles.

Grounded dialogue turns contain references with:

- `doc_id`;
- `id_sp` for the supporting span;
- optional precondition/solution labels.

### Training uses

- conversational query-to-document retrieval;
- passage/span retrieval;
- answer generation;
- conversation-aware query rewriting;
- detecting irrelevant or unanswerable turns;
- training a model to carry grounding across document changes in one conversation.

### Strengths

- Direct document and span supervision.
- Natural free-text agent responses rather than only labels.
- Dialogue segments may be grounded in different documents.
- Documents and dialogues are already distributed as JSON.

### Limitations

- Converting every turn into an independent QA row loses conversational context.
- Some turns depend heavily on earlier dialogue history.
- The official public test file is not a normal labeled test split.
- Confirm dataset and underlying webpage terms before commercial model training.

### Sources

- [Project page](https://doc2dial.github.io/multidoc2dial/)
- [Data schema](https://doc2dial.github.io/multidoc2dial/data_readme.html)
- [Official repository](https://github.com/IBM/multidoc2dial)
- [Hugging Face dataset card](https://huggingface.co/datasets/IBM/multidoc2dial)
- [EMNLP paper](https://aclanthology.org/2021.emnlp-main.498/)

---

## 3. ConditionalQA

### Why it is useful

ConditionalQA targets answers that are correct only when particular conditions apply. The documents are long, structurally complex government pages, and the questions include multi-hop and compositional reasoning.

### Data

- **Full corpus:** 652 documents.
- **Train:** 436 documents and 2,338 questions.
- **Development:** 59 documents and 285 questions.
- **Test:** 136 documents and 804 questions.
- Documents can reach approximately 9,230 words.

The reported per-split document counts do not sum to 652; use the release’s IDs and split files as authoritative rather than reconstructing splits from headline totals.

### Answer types

- extractive answers;
- yes/no answers;
- multiple answers;
- unanswerable questions;
- answers associated with one or more applicability conditions.

### Training uses

- long-document answer generation;
- condition extraction and generation;
- multi-hop evidence selection;
- abstention/unanswerability;
- derived document retrieval by pooling the 652 documents.

### Strengths

- Questions were asked without workers seeing the complete answer, reducing simple answer-copying artifacts.
- Documents include explicit structure such as headings and sections.
- The split is document-oriented, which helps measure generalization to new documents.
- Complex answer structures are closer to real policy and procedural QA than simple extractive spans.

### Limitations

- Only 2,338 questions are in the official training set.
- Retrieval is not the native task; the source document is normally known.
- Answers are not verified by legal professionals and must not be treated as legal advice.

### Sources

- [Official repository](https://github.com/haitian-sun/ConditionalQA)
- [Project page](https://haitian-sun.github.io/conditionalqa/)
- [ACL paper](https://aclanthology.org/2022.acl-long.253/)

---

## 4. PolicyQA

### Why it is useful

PolicyQA offers the highest QA-example count among the compact document collections in this report. It is derived from the OPP-115 corpus of privacy policies and provides extractive answers.

### Data

- **Corpus:** 115 website privacy policies.
- **Examples:** 25,017 reading-comprehension examples.
- **Human-authored question inventory:** 714 questions.
- **Document split:** 75 training policies, 20 validation policies, and 20 test policies.
- **Average answer length reported in the paper:** approximately 13.5 words.

### Available supervision

- question;
- source passage/policy;
- short extractive answer;
- answer offsets or equivalent span annotations in the processed data.

### Training uses

- extractive answer generation;
- passage selection;
- domain-specific privacy-policy retrievers;
- derived document retrieval using policy IDs as positive documents;
- hard-negative training across policies discussing similar practices.

### Strengths

- Large number of supervised examples relative to only 115 documents.
- Short, grounded answers.
- Document-disjoint train/validation/test split.
- Strong domain consistency.

### Limitations

- Only 714 distinct human questions underlie 25,017 examples.
- Questions are deliberately generic so they can apply to many policy segments; random row splitting would cause severe leakage.

### Sources

- [Official repository](https://github.com/wasiahmad/PolicyQA)
- [Paper](https://aclanthology.org/2020.findings-emnlp.66/)
- [OPP-115 and privacy dataset portal](https://usableprivacy.org/data)

---

## 5. ContractNLI

### Why it is useful

ContractNLI is not presented as QA, but its fixed natural-language hypotheses are easily converted into questions. It supplies unusually clean document-level and evidence-span supervision.

### Data

- **Corpus:** 607 non-disclosure agreements.
- **Labels:** 17 recurring natural-language hypotheses per contract.
- **Split:** 70% train, 10% development, 20% test, stratified by document format.
- **Average contract:** approximately 2,254 tokens and 77.8 candidate spans.

### Native task

For every hypothesis and contract, predict:

- `Entailment`;
- `Contradiction`; or
- `NotMentioned`.

For entailed or contradicted decisions, identify all supporting evidence spans.

### Suggested QA conversion

Example:

```text
Hypothesis:
Some obligations of the agreement may survive termination.

Converted question:
Do any obligations under this agreement survive termination?

Converted answer:
Yes. [Optionally include the evidence text.]
```

Map:

- `Entailment` → “Yes” plus evidence;
- `Contradiction` → “No” plus evidence;
- `NotMentioned` → “The agreement does not specify this.”

Preserve the original three-way label separately; the natural-language conversion can lose nuances.

### Training uses

- question answering over contracts;
- document and evidence retrieval;
- explicit “not in document” training;
- multi-label evidence extraction;
- domain-specific legal language modeling.

### Strengths

- Full document text, span boundaries, labels and annotations are provided in JSON.
- Evidence may consist of multiple non-contiguous spans.
- Stable document IDs and original-document metadata are supplied.

### Limitations

- The same 17 hypotheses repeat across all contracts, limiting question diversity.
- It is specialized to NDAs.
- NLI-to-QA conversion introduces synthetic phrasing and should be documented.
- Legal interpretation remains difficult; model outputs must not be presented as legal advice.

### Sources

- [Official dataset page, schema and download](https://stanfordnlp.github.io/contract-nli/)
- [Repository](https://github.com/stanfordnlp/contract-nli)
- [Paper](https://aclanthology.org/2021.findings-emnlp.164/)

---

## 6. QuALITY

### Why it is useful

QuALITY provides difficult questions over documents averaging roughly 5,000 tokens. Annotators read the complete article before writing questions, so many examples depend on information distributed through the document.

### Data

- **Corpus:** 265 articles.
- **Questions:** 6,737 total.
- **QuALITY-HARD:** 3,360 questions identified as difficult.
- **Typical format:** one correct answer among four choices.
- **Sources:** fiction and nonfiction, including Project Gutenberg and Open American National Corpus material.

### Available supervision

- full source article;
- question;
- four answer choices;
- correct-choice label;
- difficulty-related validation metadata.

The release does not provide gold supporting passage spans comparable to MultiDoc2Dial or MultiRC.

### Training uses

- long-context multiple-choice SFT;
- answer selection;
- derived document retrieval by pooling all articles;
- long-context reranking;
- difficult-negative construction from incorrect choices.

### Strengths

- Human-written questions based on full-document reading.
- Long documents with questions that often cannot be solved from a short local window.
- CC BY 4.0 distribution.
- Useful counterbalance to policy/legal datasets.

### Limitations

- No fine-grained evidence annotations.
- Multiple-choice responses do not directly teach free-form grounded answers.
- If converted to open-ended QA, the correct option can be used as the answer but is not necessarily an ideal natural-language response.

### Sources

- [Project page](https://nyu-mll.github.io/quality/)
- [Repository](https://github.com/nyu-mll/quality)
- [NAACL paper](https://aclanthology.org/2022.naacl-main.391/)

---

## 7. MultiRC / ERASER MultiRC

### Why it is useful

MultiRC focuses on questions requiring information from multiple sentences. It is especially useful through the ERASER representation, which standardizes supporting rationales.

### Data

- **Corpus:** 871 paragraphs/documents.
- **Questions:** approximately 6,000 multi-sentence questions.
- Each question has several candidate answers.
- One or more answer options may be correct.
- The original annotation process asked workers to identify required sentences; questions required about 2.4 evidence sentences on average.

### Available supervision

- source document;
- question;
- answer options;
- independent correctness label per option;
- supporting sentence annotations/rationales.

### Training uses

- evidence-sentence retrieval;
- multi-label answer prediction;
- rationale-aware SFT;
- derived document retrieval;
- training models not to assume exactly one correct answer.

### Strengths

- Human supporting evidence.
- Requires multi-sentence aggregation.
- ERASER offers a normalized rationale format and evidence-focused metrics.
- Corpus size fits the requested range.

### Limitations

- Source units are generally paragraphs, not large operational documents.
- Multiple-answer multiple choice requires a nonstandard output format.
- The official website, repository code, ERASER packaging, and source documents may carry different terms; audit the exact files used.

### Sources

- [Official MultiRC page](https://cogcomp.seas.upenn.edu/multirc/)
- [Original paper](https://aclanthology.org/N18-1023/)
- [ERASER benchmark](https://www.eraserbenchmark.com/)
- [ERASER paper](https://aclanthology.org/2020.acl-main.408/)
- [ERASER MultiRC dataset card](https://huggingface.co/datasets/CogComp/eraser_multi_rc)

---

## 8. ShARC

### Why it is useful

ShARC trains a system to reason over natural-language rules and decide whether it can answer immediately or must ask a follow-up question. OR-ShARC should usually be preferred when retrieval training is required, but the original ShARC data remains useful for reader/generator training.

### Data

- **Corpus:** 948 distinct rule-text snippets.
- **Dialog-tree utterances:** 6,058.
- **Scenarios:** 6,637.
- Some processed representations expand paths or turns into a much larger number of model instances; always report which representation is used.

### Available supervision

- rule text;
- initial question;
- user scenario;
- dialogue history;
- next response:
  - `Yes`,
  - `No`,
  - `Irrelevant`, or
  - a clarification question.

### Training uses

- conversational decision making;
- follow-up-question generation;
- rule interpretation;
- incomplete-information detection;
- reader/generator pretraining before OR-ShARC retrieval training.

### Strengths

- Explicitly teaches clarification rather than unsupported guessing.
- Machine-readable and easy to convert into chat-style SFT.
- CC BY-SA 3.0 distribution.
- Same general domain as OR-ShARC, making staged training simple.

### Limitations

- The native task gives the relevant rule text, so retrieval is absent.
- Many questions are underspecified when detached from their supplied rule text; naively pooling documents creates a poor retrieval benchmark.
- Considerable overlap with OR-ShARC means the two datasets should not be treated as statistically independent.

### Sources

- [Project page](https://sharc-data.github.io/)
- [Data](https://sharc-data.github.io/data.html)
- [Paper](https://aclanthology.org/D18-1233/)
- [Hugging Face dataset card](https://huggingface.co/datasets/UCLNLP/sharc)

---

## 9. MCTest

### Why it is useful

MCTest is a small, simple, fully paired story-question dataset. It is useful for pipeline testing, low-cost SFT experiments, or as a general-domain supplement, but it is unlikely to stress a modern retrieval system.

### Data

- **Corpus:** 660 crowd-written short stories.
- **Questions:** four per story, or 2,640 total.
- **Answers:** four choices with one correct choice.
- Story language is intentionally accessible.

### Available supervision

- story ID and text;
- question;
- four answer candidates;
- gold answer label.

### Training uses

- multiple-choice answer selection;
- small-scale reader training;
- pipeline and schema validation;
- derived retrieval by indexing all stories and using the original story ID as the positive document.

### Strengths

- Simple and inexpensive to preprocess.
- Human-written stories and questions.
- Exact source document and correct answer are known.
- Corpus size fits the requested range.

### Limitations

- No supporting evidence spans.
- Only 2,640 questions.
- Short, child-accessible stories make retrieval and comprehension relatively easy.

### Sources

- [Dataset page and download](https://mattr1.github.io/mctest/)
- [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/)
- [EMNLP paper](https://aclanthology.org/D13-1020/)

