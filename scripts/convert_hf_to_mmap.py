#!/usr/bin/env python3
"""Convert Hugging Face QA datasets into the Phase 2 mmap schema."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa
from datasets import load_dataset

from bgkit.data.mmap_writer import build_csr_offsets, write_mmap_artifacts
from bgkit.data.repo_processing import looks_minified


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    split: str
    document_fields: tuple[str, ...]
    question_fields: tuple[str, ...]
    answer_fields: tuple[str, ...]
    tag_fields: tuple[str, ...] = ()
    config: str | None = None


_SPECS = {
    # PubMedQA (verified 2026-04-10):
    #   - "context" is a dict: {"contexts": list[str], "labels": list[str],
    #     "meshes": list[str], ...}. There is NO top-level "mesh_terms" field.
    #   - "question", "long_answer", "final_decision" are top-level strings.
    #   Tag extraction for meshes handled specially in convert_dataset().
    "pubmedqa": DatasetSpec(
        dataset_id="qiaojin/PubMedQA",
        split="pqa_labeled",
        document_fields=("context", "long_answer"),
        question_fields=("question",),
        answer_fields=("final_decision", "long_answer"),
        tag_fields=(),
    ),
    "pubmedqa_artificial": DatasetSpec(
        dataset_id="qiaojin/PubMedQA",
        split="pqa_artificial",
        document_fields=("context", "long_answer"),
        question_fields=("question",),
        answer_fields=("final_decision", "long_answer"),
        tag_fields=(),
    ),
    "newsqa": DatasetSpec(
        dataset_id="cnn_dailymail",
        config="3.0.0",
        split="train",
        document_fields=("article",),
        question_fields=("highlights",),
        answer_fields=("highlights",),
    ),
    # SearchQA (verified 2026-04-10):
    #   The canonical "search_qa" dataset uses a deprecated loading script
    #   that is no longer supported by the datasets library. Using
    #   "lucadiliello/searchqa" instead, which has Parquet files.
    #   Fields: context (str, concatenated snippets), question (str),
    #           answers (list[str]), key (str), labels (list[dict]).
    "searchqa": DatasetSpec(
        dataset_id="lucadiliello/searchqa",
        split="train",
        document_fields=("context",),
        question_fields=("question",),
        answer_fields=("answers",),
    ),
    "msmarco_passage": DatasetSpec(
        dataset_id="ms_marco",
        config="v2.1",
        split="train",
        document_fields=("passages",),
        question_fields=("query",),
        answer_fields=("answers",),
        tag_fields=("query_type",),
    ),
    # NarrativeQA (verified 2026-04-10):
    #   - "document" is a dict with: id, kind, url, file_size, word_count,
    #     start, end, summary (dict with "text" and "tokens").
    #     There is NO top-level "text" key in document; the readable text is
    #     in document["summary"]["text"].
    #   - "question" is a dict: {"text": str, "tokens": list[str]}.
    #   - "answers" is a list of dicts: [{"text": str, "tokens": list[str]}].
    #   Extraction handled specially in convert_dataset().
    "narrativeqa": DatasetSpec(
        dataset_id="deepmind/narrativeqa",
        split="train",
        document_fields=("document",),
        question_fields=("question",),
        answer_fields=("answers",),
    ),
    # KILT Wikipedia (verified 2026-04-10):
    #   "facebook/kilt_wikipedia" uses a deprecated loading script. Using
    #   "orionweller/kilt_wikipedia_split" (Parquet) instead.
    #   Fields: text, wikipedia_id, wikipedia_title, categories (str, comma-sep),
    #           anchors, history, section, sources.
    "kilt_wikipedia": DatasetSpec(
        dataset_id="orionweller/kilt_wikipedia_split",
        split="train",
        document_fields=("text",),
        question_fields=(),
        answer_fields=(),
        tag_fields=("categories",),
    ),
    "kilt_nq": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="nq",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_hotpotqa": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="hotpotqa",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_fever": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="fever",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_zsre": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="structured_zeroshot",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_trex": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="trex",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_wow": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="wow",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_eli5": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="eli5",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_aidayago2": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="aidayago2",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_wned": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="wned",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_cweb": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="cweb",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
    "kilt_triviaqa": DatasetSpec(
        dataset_id="facebook/kilt_tasks",
        config="triviaqa",
        split="train",
        document_fields=(),
        question_fields=("input",),
        answer_fields=("output",),
    ),
}


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_coerce_text(item) for item in value if item)
    if isinstance(value, dict):
        # NarrativeQA question/answer dicts have a "text" key
        if "text" in value:
            return _coerce_text(value["text"])
        # PubMedQA context dict has "contexts" (list of paragraph strings)
        if "contexts" in value:
            return _coerce_text(value["contexts"])
        return "\n".join(f"{k}: {_coerce_text(v)}" for k, v in value.items() if v)
    return str(value)


def _extract_first(record: dict, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = record.get(key)
        if value:
            return _coerce_text(value)
    return ""


def _extract_msmarco_document(record: dict) -> str:
    """Extract document text from MS MARCO passage format."""
    passages = record.get("passages")
    if not passages:
        return ""
    # MS MARCO passages is a dict with 'passage_text' list and 'is_selected' list
    if isinstance(passages, dict):
        texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])
        # Prefer selected passages, fall back to all
        if selected and any(selected):
            return "\n\n".join(
                text for text, sel in zip(texts, selected, strict=False) if sel and text
            )
        return "\n\n".join(text for text in texts if text)
    return _coerce_text(passages)


def _extract_msmarco_answer(record: dict) -> str:
    """Extract answer from MS MARCO format (answers is a list, some empty)."""
    answers = record.get("answers")
    if not answers:
        return ""
    if isinstance(answers, list):
        # Filter 'No Answer Present.' entries
        valid = [a for a in answers if a and a != "No Answer Present."]
        return valid[0] if valid else ""
    return _coerce_text(answers)


def _extract_narrativeqa_document(record: dict) -> str:
    """Extract document text from NarrativeQA format.

    NarrativeQA document structure (verified 2026-04-10):
      {"id": str, "kind": str, "url": str, "file_size": int,
       "word_count": int, "start": str, "end": str,
       "summary": {"text": str, "tokens": list[str]}}

    The actual readable content is in document["summary"]["text"].
    """
    document = record.get("document")
    if not document or not isinstance(document, dict):
        return ""
    summary = document.get("summary")
    if isinstance(summary, dict):
        return summary.get("text", "")
    if isinstance(summary, str):
        return summary
    return ""


def _extract_narrativeqa_question(record: dict) -> str:
    """Extract question text from NarrativeQA format.

    Question is a dict: {"text": str, "tokens": list[str]}.
    """
    question = record.get("question")
    if isinstance(question, dict):
        return question.get("text", "")
    if isinstance(question, str):
        return question
    return ""


def _extract_narrativeqa_answer(record: dict) -> str:
    """Extract answer from NarrativeQA format.

    Answers is a list of dicts: [{"text": str, "tokens": list[str]}, ...].
    """
    answers = record.get("answers")
    if not answers:
        return ""
    if isinstance(answers, list):
        for entry in answers:
            if isinstance(entry, dict):
                text = entry.get("text", "")
                if text:
                    return text
            elif isinstance(entry, str) and entry:
                return entry
    return _coerce_text(answers)


def _extract_pubmedqa_tags(record: dict) -> list[str]:
    """Extract mesh terms from PubMedQA context dict.

    PubMedQA context structure (verified 2026-04-10):
      {"contexts": list[str], "labels": list[str], "meshes": list[str], ...}
    Mesh terms are inside context["meshes"], NOT at top level.
    """
    context = record.get("context")
    if isinstance(context, dict):
        meshes = context.get("meshes", [])
        if isinstance(meshes, list):
            return [str(m) for m in meshes if m]
    return []


def _extract_kilt_answer(record: dict) -> str:
    """Extract answer text from KILT output format."""
    output = record.get("output")
    if not output:
        return ""
    if isinstance(output, list):
        for entry in output:
            if isinstance(entry, dict):
                answer = entry.get("answer")
                if answer:
                    return _coerce_text(answer)
            elif entry:
                return _coerce_text(entry)
    return _coerce_text(output)


def _extract_kilt_provenance(record: dict) -> list[str]:
    """Extract provenance Wikipedia IDs from KILT output."""
    output = record.get("output")
    if not output or not isinstance(output, list):
        return []
    prov_ids = []
    for entry in output:
        if isinstance(entry, dict):
            provenance = entry.get("provenance")
            if isinstance(provenance, list):
                for p in provenance:
                    if isinstance(p, dict) and p.get("wikipedia_id"):
                        prov_ids.append(str(p["wikipedia_id"]))
    return prov_ids


def convert_dataset(
    *,
    dataset_name: str,
    output_dir: Path,
    tokenizer_name: str,
    max_document_tokens: int,
    max_question_tokens: int,
    max_answer_tokens: int,
) -> dict:
    from transformers import AutoTokenizer

    spec = _SPECS[dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    load_kwargs = {}
    if spec.config:
        load_kwargs["name"] = spec.config
    dataset = load_dataset(spec.dataset_id, split=spec.split, **load_kwargs)

    document_chunks: list[np.ndarray] = []
    question_chunks: list[np.ndarray] = []
    answer_chunks: list[np.ndarray] = []
    document_lengths: list[int] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    skipped_minified = 0

    is_corpus_only = not spec.question_fields and not spec.answer_fields
    is_msmarco = dataset_name == "msmarco_passage"
    is_kilt_task = dataset_name.startswith("kilt_") and dataset_name != "kilt_wikipedia"
    is_narrativeqa = dataset_name == "narrativeqa"
    is_pubmedqa = dataset_name.startswith("pubmedqa")

    for idx, record in enumerate(dataset):
        # Dataset-specific extraction
        if is_msmarco:
            document_text = _extract_msmarco_document(record)
            question_text = _extract_first(record, spec.question_fields)
            answer_text = _extract_msmarco_answer(record)
        elif is_narrativeqa:
            document_text = _extract_narrativeqa_document(record)
            question_text = _extract_narrativeqa_question(record)
            answer_text = _extract_narrativeqa_answer(record)
        elif is_kilt_task:
            document_text = ""  # KILT tasks reference Wikipedia by provenance
            question_text = _extract_first(record, spec.question_fields)
            answer_text = _extract_kilt_answer(record)
        elif is_corpus_only:
            document_text = _extract_first(record, spec.document_fields)
            question_text = ""
            answer_text = ""
        else:
            document_text = _extract_first(record, spec.document_fields)
            question_text = _extract_first(record, spec.question_fields)
            answer_text = _extract_first(record, spec.answer_fields)

        if is_corpus_only:
            if not document_text:
                continue
        elif (not document_text and not is_kilt_task) or not question_text or not answer_text:
            continue

        # Quality filter in prose mode: catches the pathological cases
        # (CSV / HTML / base64 stuffed into a text field) without
        # flagging legitimate single-paragraph Wikipedia articles or
        # PubMedQA abstracts. KILT tasks have empty document_text by
        # design; skip the check there.
        if (
            document_text
            and not is_kilt_task
            and looks_minified(document_text, content_type="prose")
        ):
            skipped_minified += 1
            continue

        doc_ids = (
            tokenizer.encode(document_text, add_special_tokens=False)[:max_document_tokens]
            if document_text
            else []
        )
        question_ids = (
            tokenizer.encode(question_text, add_special_tokens=False)[:max_question_tokens]
            if question_text
            else []
        )
        answer_ids = (
            tokenizer.encode(answer_text, add_special_tokens=False)[:max_answer_tokens]
            if answer_text
            else []
        )

        if is_corpus_only:
            if not doc_ids:
                continue
            # Corpus-only: store empty question/answer arrays
            question_ids = question_ids or [0]
            answer_ids = answer_ids or [0]
        elif is_kilt_task:
            # KILT tasks have no inline documents — they reference Wikipedia
            # by provenance. Document tokens are intentionally empty; training
            # MUST use precomputed L0 (use_precomputed_l0: true).
            if not question_ids or not answer_ids:
                continue
            if not doc_ids:
                doc_ids = []
        else:
            if not doc_ids or not question_ids or not answer_ids:
                continue

        document_chunks.append(np.array(doc_ids, dtype=np.int32))
        question_chunks.append(np.array(question_ids, dtype=np.int32))
        answer_chunks.append(np.array(answer_ids, dtype=np.int32))
        document_lengths.append(len(doc_ids))
        question_lengths.append(len(question_ids))
        answer_lengths.append(len(answer_ids))

        tag_list = []
        # PubMedQA: mesh terms are nested inside context dict, not top-level
        if is_pubmedqa:
            tag_list = _extract_pubmedqa_tags(record)
        else:
            for field in spec.tag_fields:
                value = record.get(field)
                if isinstance(value, list):
                    tag_list.extend(str(item) for item in value)
                elif isinstance(value, str) and "," in value:
                    # Comma-separated tags (e.g. KILT Wikipedia categories)
                    tag_list.extend(t.strip() for t in value.split(",") if t.strip())
                elif value:
                    tag_list.append(str(value))

        # ``document_id`` is the stable per-article key used by the browse
        # tree and external taxonomy builders. Each HF dataset puts its
        # canonical id under a different field, so we probe them in order
        # of specificity:
        #   - document_id / wikipedia_id: KILT Wikipedia
        #   - pubid:                      PubMedQA (labeled + artificial)
        #   - query_id:                   MS MARCO
        #   - key:                        SearchQA
        #   - id:                         generic fallback
        #   - idx:                        last-resort enumeration (never
        #                                 user-meaningful; signals a bug)
        raw_doc_id = (
            record.get("document_id")
            or record.get("wikipedia_id")
            or record.get("pubid")
            or record.get("query_id")
            or record.get("key")
            or record.get("id")
            or idx
        )
        raw_id = record.get("id") or record.get("wikipedia_id") or record.get("pubid") or idx
        row_meta = {
            "id": str(raw_id),
            "document_id": str(raw_doc_id),
            "dataset_name": dataset_name,
            "tag_list_json": json.dumps(tag_list),
        }
        # Add KILT provenance
        if is_kilt_task:
            provenance = _extract_kilt_provenance(record)
            row_meta["provenance_json"] = json.dumps(provenance)
            if record.get("meta", {}).get("task"):
                row_meta["task_name"] = str(record["meta"]["task"])

        metadata_rows.append(row_meta)

    if not document_chunks:
        raise ValueError(f"No usable rows found while converting {dataset_name}")

    columns = {
        "id": pa.array([row["id"] for row in metadata_rows], type=pa.string()),
        "document_id": pa.array(
            [row["document_id"] for row in metadata_rows],
            type=pa.string(),
        ),
        "dataset_name": pa.array(
            [row["dataset_name"] for row in metadata_rows],
            type=pa.string(),
        ),
        "tag_list_json": pa.array(
            [row["tag_list_json"] for row in metadata_rows],
            type=pa.string(),
        ),
    }
    # Add optional columns present in some datasets
    if any("provenance_json" in row for row in metadata_rows):
        columns["provenance_json"] = pa.array(
            [row.get("provenance_json", "[]") for row in metadata_rows],
            type=pa.string(),
        )
    if any("task_name" in row for row in metadata_rows):
        columns["task_name"] = pa.array(
            [row.get("task_name", "") for row in metadata_rows],
            type=pa.string(),
        )
    metadata_table = pa.table(columns)

    return write_mmap_artifacts(
        output_dir=output_dir,
        tokens=np.concatenate(document_chunks),
        offsets=build_csr_offsets(np.array(document_lengths, dtype=np.int64)),
        manifest_extra={
            "dataset_name": dataset_name,
            "tokenizer": tokenizer_name,
            "total_question_tokens": int(sum(question_lengths)),
            "total_answer_tokens": int(sum(answer_lengths)),
            "skipped_minified": skipped_minified,
        },
        metadata_table=metadata_table,
        extra_arrays={
            "question_tokens.npy": np.concatenate(question_chunks),
            "question_offsets.npy": build_csr_offsets(np.array(question_lengths, dtype=np.int64)),
            "answer_tokens.npy": np.concatenate(answer_chunks),
            "answer_offsets.npy": build_csr_offsets(np.array(answer_lengths, dtype=np.int64)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", choices=sorted(_SPECS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--max-question-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    args = parser.parse_args()

    manifest = convert_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_document_tokens=args.max_document_tokens,
        max_question_tokens=args.max_question_tokens,
        max_answer_tokens=args.max_answer_tokens,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
