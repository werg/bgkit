---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
base_model:
- LiquidAI/LFM2.5-350M
- LiquidAI/LFM2.5-Encoder-350M
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
- context-compression
- long-context
- lfm2
---

# bgkit-350m: reading compressed context through the embedding layer

bgkit (background knowledge interaction transformer) gives a language model access to more text
than fits in its context budget. A bidirectional
encoder reads a text together with a prompt that states what the reader needs, compresses it by a
chosen ratio, and maps the result into the decoder's input embedding space. The decoder reads these
vectors in place of the text's tokens, in a slot among ordinary tokens; one input can hold many
slots.

This repository contains the model described in the accompanying paper (`paper.pdf`): a 350M
encoder and a 350M decoder, both fine-tuned from Liquid AI's LFM2.5 models, with the adapters merged
into the weights.

| Path | Contents | Loads with |
|---|---|---|
| `decoder/` | LFM2.5-350M (instruct) with the bgkit adapter merged; tokenizer and chat template | `AutoModelForCausalLM` |
| `encoder/` | LFM2.5-Encoder-350M with the bgkit adapter merged, in Liquid's layout | `AutoModelForMaskedLM` (`trust_remote_code=True`) |
| `compressor.safetensors`, `bgkit_config.json` | scoring head, prompt separator, survive embedding, projection block and MLP, interface constant | `modeling_bgkit.py` |
| `modeling_bgkit.py` | standalone inference: compress text, generate with slots | `torch`, `transformers>=5.5` |

## Usage

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("gabrielpickard/bgkit-350m")
sys.path.insert(0, path)
from modeling_bgkit import Bgkit

bg = Bgkit.from_pretrained(path, device="cuda")

# Question answering over a long context with 512 decoder positions for the context.
print(bg.answer(context=long_text, question="Who founded the company?", budget=512))

# Lower level: compress any text under a prompt, then place the vectors in a chat message.
slot = bg.compress(report_text, prompt="What were the main risks named?", ratio=1 / 16)
messages = [{"role": "user", "content": f"Report:\n{bg.sentinel}\n\nSummarise the main risks."}]
print(bg.generate(messages, [slot], max_new_tokens=128))
```

`bg.sentinel` (`<|reserved_6|>`) marks where a slot's vectors go; pass one tensor per sentinel, in
order. `compress` accepts `ratio` (fraction of tokens kept, between 1/1024 and 1/2) or `n_vectors`.
A single slot reads up to 8,192 tokens; `answer` cuts longer contexts into 2,048-token chunks, each
compressed under the question into its own slot. The encoding is specific to its prompt: compress
under the question you will ask.

## How it works

1. The encoder reads `[prompt][separator][text]` bidirectionally.
2. After layer 6 a scoring head ranks the text positions; the top `ceil(ratio * n)` are kept and
   marked, and the remaining layers run over the whole sequence, so each kept position can gather
   information from anywhere in the text.
3. The final states at the kept positions pass through an attention layer over the kept set and an
   MLP, and are rescaled to the norm of the decoder's token embeddings.
4. The decoder reads them where the slot token was. Its adapter was trained to read such slots.

## Results

From the paper, at a fixed number of decoder positions for the context (accuracy or LLM-judge
accuracy; *text* is the same decoder reading the text truncated to the budget):

| Benchmark | Context | Budget | bgkit | text | original decoder, full context |
|---|---|---|---|---|---|
| BABILong (qa1, qa2) | 15.4k tokens | 1,024 | 0.74 | 0.12 | 0.18 |
| HotpotQA | 1.4k | 256 | 0.64 | 0.23 | 0.45 |
| MuSiQue | 2.5k | 2,048 | 0.33 | 0.16 | 0.13 |
| LoCoMo | 18.8k | 1,024 | 0.28 | 0.09 | 0.19 |
| LongMemEval-S | 111k | 2,048 | 0.30 | 0.18 | — |
| RULER (needles) | 3.7k | 1,024 | 0.20 | 0.15 | 0.58 |

The released files were checked against the training code on 100 items each: HotpotQA at 256
positions (exact match 0.47 for both, 97 of 100 answers identical) and BABILong 16k at 1,024
positions (accuracy 0.74 for both, 99 of 100 identical).

## Limitations

- **Exact strings.** Arbitrary keys, numbers and identifiers are often lost at these ratios (RULER).
- **Task overlap.** Training included the training splits of HotpotQA, QuALITY and NarrativeQA, and
  bAbI tasks in filler text (BABILong's construction). MuSiQue, RULER, LongMemEval and LoCoMo were
  not trained on.
- **Small models.** Both models have 350M parameters; absolute accuracy on hard tasks is low.
- **Question-specific encodings.** An encoding made under one question serves that question; each new
  question needs a new encoder pass.
- **English.** Training and evaluation were in English.
- **Not included.** The paper's agent harness and the hierarchical repository encodings are not part
  of this release.

## Training

The decoder was trained to produce answers, agent turns and text continuations from inputs in which
the context appears only in compressed form, with distillation from a full-text reader, a contrast
between right and wrong encoder prompts, and distillation towards the original decoder on ordinary
chat to preserve its behaviour. Data: web text, code, books and dialogue; reading-comprehension and
multi-hop QA (SQuAD, TriviaQA, HotpotQA, QuALITY, NarrativeQA, bAbI); multi-session conversations
(MSC, Chronicles, SHARE) with generated questions; agent trajectories on software repositories and
web research generated by larger models. These datasets carry their own licences. Training ran on a
single NVIDIA DGX Spark.

## License

The weights are a derivative of LiquidAI/LFM2.5-350M and LiquidAI/LFM2.5-Encoder-350M and are
distributed under the [LFM Open License v1.0](LICENSE). Among its terms, commercial use is licensed
only to entities with annual revenue below US$10 million. See `NOTICE` for the modifications made.
This release is not affiliated with or endorsed by Liquid AI.

## Citation

```bibtex
@misc{pickard2026bgkit,
  title  = {bgkit: Reading Compressed Context through a Language Model's Embedding Layer},
  author = {Pickard, Gabriel},
  year   = {2026},
  note   = {https://huggingface.co/gabrielpickard/bgkit-350m}
}
```
