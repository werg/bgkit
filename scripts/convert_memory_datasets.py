#!/usr/bin/env python3
"""Convert multi-session memory datasets into the Phase 2 mmap schema.

Handles MSC, SHARE, Chronicles, PerLTQA, and LAPS from their HF/raw sources
with dataset-specific parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa

from bgkit.data.mmap_writer import build_csr_offsets, write_mmap_artifacts


def _tokenize_and_truncate(tokenizer, text: str, max_tokens: int) -> list[int]:
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)[:max_tokens]


# ---------------------------------------------------------------------------
# MSC: Multi-Session Chat (5 sessions per conversation, persona tracking)
# ---------------------------------------------------------------------------


def _parse_msc(tokenizer, max_doc: int, max_q: int, max_a: int):
    """Parse MSC from HuggingFace: nayohan/multi_session_chat.

    Dataset structure (verified 2026-04-10):
      Flat rows, one per session. Fields:
        - dialoug_id: int (conversation ID — note typo in dataset)
        - session_id: int (0-based session index within conversation)
        - dialogue: list[str] (utterances)
        - speaker: list[str] ("Speaker 1" / "Speaker 2")
        - persona1: list[str] (persona sentences for speaker 1)
        - persona2: list[str] (persona sentences for speaker 2)
        - dataset: str ("MSC")
    """
    from collections import defaultdict

    from datasets import load_dataset

    ds = load_dataset("nayohan/multi_session_chat", split="train")

    # Group rows by conversation (dialoug_id — sic, typo in HF dataset)
    conversations: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in ds:
        conv_id = row["dialoug_id"]
        sess_id = row["session_id"]
        conversations[conv_id][sess_id] = row

    records = []
    for conv_id, sessions_map in conversations.items():
        if len(sessions_map) < 2:
            continue

        # Build session texts in order
        session_texts = []
        for sess_id in sorted(sessions_map):
            row = sessions_map[sess_id]
            dialogue = row.get("dialogue", [])
            speakers = row.get("speaker", [])
            if dialogue and speakers:
                lines = []
                for utt, spk in zip(dialogue, speakers, strict=False):
                    lines.append(f"{spk}: {utt}")
                session_texts.append("\n".join(lines))
            elif dialogue:
                session_texts.append("\n".join(str(u) for u in dialogue))

        if len(session_texts) < 2:
            continue

        # Use earlier sessions as context, last session as Q/A source
        context = "\n\n[Session break]\n\n".join(session_texts[:-1])

        # Combine persona from last session's row
        last_row = sessions_map[max(sessions_map)]
        persona1 = last_row.get("persona1") or []
        persona2 = last_row.get("persona2") or []
        persona_text = " ".join(persona1 + persona2)

        question = "Based on previous conversations, what do you remember about the other person?"
        answer = persona_text if persona_text.strip() else session_texts[-1][:200]
        if not answer.strip():
            continue

        records.append({
            "context": context,
            "question": question,
            "answer": answer,
            "memory_type": "persona",
            "episode_id": str(conv_id),
        })

    return _build_arrays(records, tokenizer, max_doc, max_q, max_a, "msc")


# ---------------------------------------------------------------------------
# SHARE: 4 memory annotation types
# ---------------------------------------------------------------------------


def _parse_share(tokenizer, max_doc: int, max_q: int, max_a: int):
    """Parse SHARE from HuggingFace: eunwoneunwon/SHARE.

    Dataset structure (verified 2026-04-10):
      Cannot use ``load_dataset`` — the nested JSON breaks PyArrow. Must
      download the raw JSON via ``hf_hub_download`` and parse directly.

      Top-level: dict keyed by speaker-pair tuples (as strings).
      Each value: {"movie": str, "dialogue": [<session>, ...]}.
      Each session dict:
        - session: int
        - dialogues: list[{"speaker": str, "text": str, "label": list, "utterance": int}]
        - "<Name>'s persona": list[str]      (per-speaker persona entries)
        - "<Name>'s temporary event": list[str]
        - "Shared memory": list[str]
        - "Mutual event": list[str]
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("eunwoneunwon/SHARE", "data/train.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)

    records = []
    for pair_key, conv in data.items():
        dialogue_sessions = conv.get("dialogue", [])
        if not dialogue_sessions:
            continue

        # Build full dialogue text from all sessions
        session_texts = []
        for sess in dialogue_sessions:
            turns = sess.get("dialogues", [])
            lines = [
                f"{turn['speaker']}: {turn['text']}"
                for turn in turns
                if isinstance(turn, dict) and turn.get("text")
            ]
            if lines:
                session_texts.append("\n".join(lines))

        if not session_texts:
            continue
        full_dialog = "\n\n[Session break]\n\n".join(session_texts)

        # Extract memory annotations from each session
        for sess in dialogue_sessions:
            sess_keys = set(sess.keys())
            # Identify persona keys (format: "<Name>'s persona")
            for sk in sess_keys:
                if sk.endswith("'s persona"):
                    annotations = sess[sk]
                    if isinstance(annotations, list):
                        for ann in annotations:
                            if ann and isinstance(ann, str):
                                records.append({
                                    "context": full_dialog,
                                    "question": (
                                        "What persona entries can you recall "
                                        "from this conversation?"
                                    ),
                                    "answer": ann,
                                    "memory_type": "persona_entries",
                                    "episode_id": str(pair_key),
                                })
                elif sk.endswith("'s temporary event"):
                    annotations = sess[sk]
                    if isinstance(annotations, list):
                        for ann in annotations:
                            if ann and isinstance(ann, str):
                                records.append({
                                    "context": full_dialog,
                                    "question": (
                                        "What personal events can you recall "
                                        "from this conversation?"
                                    ),
                                    "answer": ann,
                                    "memory_type": "personal_events",
                                    "episode_id": str(pair_key),
                                })

            # Shared memory and Mutual event are fixed keys
            for mem_key, mem_type, question_fragment in [
                ("Shared memory", "shared_memories", "shared memories"),
                ("Mutual event", "mutual_events", "mutual events"),
            ]:
                annotations = sess.get(mem_key, [])
                if isinstance(annotations, list):
                    for ann in annotations:
                        if ann and isinstance(ann, str):
                            records.append({
                                "context": full_dialog,
                                "question": (
                                    f"What {question_fragment} can you recall "
                                    f"from this conversation?"
                                ),
                                "answer": ann,
                                "memory_type": mem_type,
                                "episode_id": str(pair_key),
                            })

    return _build_arrays(records, tokenizer, max_doc, max_q, max_a, "share")


# ---------------------------------------------------------------------------
# Chronicles: ConversationChronicles (5 sessions with summaries)
# ---------------------------------------------------------------------------


def _parse_chronicles(tokenizer, max_doc: int, max_q: int, max_a: int):
    """Parse Chronicles from HuggingFace: jihyoung/ConversationChronicles.

    Dataset structure (verified 2026-04-10):
      Fields:
        - dataID: str (e.g. "episode-96974")
        - relationship: str (e.g. "Classmates")
        - time_interval: list[str] (e.g. ["Start", "A few weeks after", ...])
        - summary: list[str] (one summary per session, all in a single list)
        - first_session_dialogue: list[str] (utterances)
        - first_session_speakers: list[str] (speaker labels)
        - second_session_dialogue / second_session_speakers: ...
        - ... through fifth_session_dialogue / fifth_session_speakers

      Note: summaries are in a single "summary" list field, NOT per-session
      fields like "first_session_summary".
    """
    from datasets import load_dataset

    ds = load_dataset("jihyoung/ConversationChronicles", split="train")
    records = []
    for row in ds:
        sessions = []
        for ordinal in ("first", "second", "third", "fourth", "fifth"):
            dialog_key = f"{ordinal}_session_dialogue"
            speakers_key = f"{ordinal}_session_speakers"
            dialog = row.get(dialog_key)
            speakers = row.get(speakers_key)
            if dialog:
                if isinstance(dialog, list) and speakers and isinstance(speakers, list):
                    lines = [
                        f"{spk}: {utt}"
                        for utt, spk in zip(dialog, speakers, strict=False)
                    ]
                    sessions.append("\n".join(lines))
                elif isinstance(dialog, list):
                    sessions.append("\n".join(str(turn) for turn in dialog))
                else:
                    sessions.append(str(dialog))

        if len(sessions) < 2:
            continue

        # Summaries are in a single list (one per session)
        summaries = row.get("summary", [])
        if isinstance(summaries, str):
            summaries = [summaries]

        # Context: concatenated sessions, question: recall, answer: summary
        context = "\n\n[Session break]\n\n".join(sessions[:-1])
        for i, summary in enumerate(summaries):
            if not isinstance(summary, str) or not summary.strip():
                continue
            records.append({
                "context": context,
                "question": f"Summarize session {i + 1} of the conversation.",
                "answer": summary,
                "memory_type": "knowledge_update",
                "episode_id": str(row.get("dataID", len(records))),
            })

        # Also create a cross-session recall question
        valid_summaries = [s for s in summaries if isinstance(s, str) and s.strip()]
        if valid_summaries and sessions:
            records.append({
                "context": context,
                "question": "What happened across all previous conversation sessions?",
                "answer": " ".join(valid_summaries),
                "memory_type": "temporal",
                "episode_id": str(row.get("dataID", len(records))),
            })

    return _build_arrays(records, tokenizer, max_doc, max_q, max_a, "chronicles")


# ---------------------------------------------------------------------------
# PerLTQA: Personal Long-Term QA (v2 dict format)
# ---------------------------------------------------------------------------


def _parse_perltqa(tokenizer, max_doc: int, max_q: int, max_a: int):
    """Parse PerLTQA from local clone or HF.

    NOTE (verified 2026-04-10): "Elvin-Yiming-Du/PerLTQA" does NOT exist on
    HuggingFace Hub (404).  The dataset must be obtained from the original
    GitHub repository: https://github.com/Elvin-Yiming-Du/PerLTQA
    Clone it locally and point DATA_DIR/perltqa/ at the en_v2 directory.

    Expects en_v2 format: memory file is dict keyed by character name,
    QA file has questions organized by memory type.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("Elvin-Yiming-Du/PerLTQA", split="train")
    except Exception:
        ds = None

    records = []
    if ds is not None:
        for row in ds:
            context = row.get("memory") or row.get("context") or ""
            if isinstance(context, dict):
                context = "\n".join(
                    f"{k}: {v}" if isinstance(v, str) else f"{k}: {json.dumps(v)}"
                    for k, v in context.items()
                )
            elif isinstance(context, list):
                context = "\n".join(str(item) for item in context)

            question = row.get("question") or ""
            answer = row.get("answer") or ""
            if not question or not answer:
                continue

            memory_type = row.get("memory_type") or row.get("type") or "profile"
            records.append({
                "context": str(context),
                "question": str(question),
                "answer": str(answer),
                "memory_type": str(memory_type),
                "episode_id": str(row.get("character", row.get("id", len(records)))),
            })
    else:
        raise RuntimeError(
            "PerLTQA not available from HuggingFace Hub (dataset does not exist). "
            "Clone from https://github.com/Elvin-Yiming-Du/PerLTQA and load "
            "the en_v2 data locally, or remove 'perltqa' from the conversion list."
        )

    return _build_arrays(records, tokenizer, max_doc, max_q, max_a, "perltqa")


# ---------------------------------------------------------------------------
# LAPS: Long-term Accumulated Preference and Situation
# ---------------------------------------------------------------------------


def _parse_laps(tokenizer, max_doc: int, max_q: int, max_a: int):
    """Parse LAPS dataset (recipe + movie domains with preference accumulation).

    NOTE (verified 2026-04-10): "informagi/laps" does NOT exist on HuggingFace
    Hub. The LAPS dataset must be obtained from the original source (the paper
    authors). If and when an HF version becomes available, the field names
    below should be re-verified against the actual schema.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("informagi/laps", split="train")
    except Exception:
        raise RuntimeError(
            "LAPS dataset not available from HuggingFace Hub "
            "(dataset 'informagi/laps' does not exist). "
            "Obtain the dataset from the original paper authors, "
            "or remove 'laps' from the conversion list."
        ) from None

    records = []
    if ds is not None:
        for row in ds:
            sessions = row.get("sessions") or row.get("dialogue_sessions") or []
            if isinstance(sessions, str):
                sessions = [sessions]

            # Build context from sessions
            session_texts = []
            for sess in sessions:
                if isinstance(sess, dict):
                    dialog = sess.get("dialogue") or sess.get("dialog") or ""
                    if isinstance(dialog, list):
                        dialog = "\n".join(str(turn) for turn in dialog)
                    session_texts.append(str(dialog))
                else:
                    session_texts.append(str(sess))

            if not session_texts:
                # Try flat dialogue format
                dialog = row.get("dialogue") or row.get("context") or ""
                if dialog:
                    session_texts = [str(dialog)]

            if not session_texts:
                continue

            context = "\n\n[Session break]\n\n".join(session_texts)

            # Preferences accumulate across sessions
            preferences = row.get("preferences") or {}
            if isinstance(preferences, dict):
                pref_text = "; ".join(
                    f"{cat}: {', '.join(vals) if isinstance(vals, list) else str(vals)}"
                    for cat, vals in preferences.items()
                    if vals
                )
            elif isinstance(preferences, str):
                pref_text = preferences
            else:
                pref_text = ""

            question = row.get("question") or "What are the user's accumulated preferences?"
            answer = row.get("answer") or pref_text
            if not answer.strip():
                continue

            records.append({
                "context": context,
                "question": str(question),
                "answer": str(answer),
                "memory_type": "preference",
                "episode_id": str(row.get("user_id", row.get("id", len(records)))),
            })

    return _build_arrays(records, tokenizer, max_doc, max_q, max_a, "laps")


# ---------------------------------------------------------------------------
# Common array builder
# ---------------------------------------------------------------------------


def _build_arrays(
    records: list[dict],
    tokenizer,
    max_doc: int,
    max_q: int,
    max_a: int,
    dataset_name: str,
) -> tuple[dict, list[dict]]:
    """Build mmap arrays from parsed records."""
    session_chunks: list[np.ndarray] = []
    question_chunks: list[np.ndarray] = []
    answer_chunks: list[np.ndarray] = []
    session_lengths: list[int] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    metadata_rows: list[dict] = []

    for record in records:
        session_ids = _tokenize_and_truncate(tokenizer, record["context"], max_doc)
        question_ids = _tokenize_and_truncate(tokenizer, record["question"], max_q)
        answer_ids = _tokenize_and_truncate(tokenizer, record["answer"], max_a)
        if not session_ids or not question_ids or not answer_ids:
            continue

        session_chunks.append(np.array(session_ids, dtype=np.int32))
        question_chunks.append(np.array(question_ids, dtype=np.int32))
        answer_chunks.append(np.array(answer_ids, dtype=np.int32))
        session_lengths.append(len(session_ids))
        question_lengths.append(len(question_ids))
        answer_lengths.append(len(answer_ids))
        metadata_rows.append({
            "id": str(record.get("episode_id", len(metadata_rows))),
            "document_id": str(record.get("episode_id", len(metadata_rows))),
            "dataset_name": dataset_name,
            "tag_list_json": json.dumps([record.get("memory_type", "")]),
            "memory_type": record.get("memory_type", ""),
        })

    if not session_chunks:
        raise ValueError(f"No valid records found for {dataset_name}")

    return {
        "session_chunks": session_chunks,
        "question_chunks": question_chunks,
        "answer_chunks": answer_chunks,
        "session_lengths": session_lengths,
        "question_lengths": question_lengths,
        "answer_lengths": answer_lengths,
        "metadata_rows": metadata_rows,
    }, metadata_rows


_DATASET_PARSERS = {
    "msc": _parse_msc,
    "share": _parse_share,
    "chronicles": _parse_chronicles,
    "perltqa": _parse_perltqa,
    "laps": _parse_laps,
}


def convert_memory_dataset(
    dataset_name: str,
    output_dir: Path,
    tokenizer_name: str,
    max_document_tokens: int = 8192,
    max_question_tokens: int = 512,
    max_answer_tokens: int = 512,
) -> dict:
    """Convert a memory dataset to mmap format."""
    from transformers import AutoTokenizer

    if dataset_name not in _DATASET_PARSERS:
        raise ValueError(
            f"Unknown memory dataset {dataset_name!r}. "
            f"Available: {sorted(_DATASET_PARSERS)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    parser = _DATASET_PARSERS[dataset_name]
    arrays, metadata_rows = parser(
        tokenizer, max_document_tokens, max_question_tokens, max_answer_tokens,
    )

    metadata_table = pa.table({
        "id": pa.array([row["id"] for row in metadata_rows], type=pa.string()),
        "document_id": pa.array([row["document_id"] for row in metadata_rows], type=pa.string()),
        "dataset_name": pa.array([row["dataset_name"] for row in metadata_rows], type=pa.string()),
        "tag_list_json": pa.array(
            [row["tag_list_json"] for row in metadata_rows], type=pa.string(),
        ),
        "memory_type": pa.array(
            [row["memory_type"] for row in metadata_rows], type=pa.string(),
        ),
    })

    return write_mmap_artifacts(
        output_dir=output_dir,
        tokens=np.concatenate(arrays["session_chunks"]),
        offsets=build_csr_offsets(np.array(arrays["session_lengths"], dtype=np.int64)),
        manifest_extra={
            "dataset_name": dataset_name,
            "tokenizer": tokenizer_name,
            "total_question_tokens": int(sum(arrays["question_lengths"])),
            "total_answer_tokens": int(sum(arrays["answer_lengths"])),
        },
        metadata_table=metadata_table,
        extra_arrays={
            "question_tokens.npy": np.concatenate(arrays["question_chunks"]),
            "question_offsets.npy": build_csr_offsets(
                np.array(arrays["question_lengths"], dtype=np.int64),
            ),
            "answer_tokens.npy": np.concatenate(arrays["answer_chunks"]),
            "answer_offsets.npy": build_csr_offsets(
                np.array(arrays["answer_lengths"], dtype=np.int64),
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_name",
        choices=sorted(_DATASET_PARSERS),
        help="Memory dataset to convert",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--max-question-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    args = parser.parse_args()

    manifest = convert_memory_dataset(
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
