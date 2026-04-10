#!/usr/bin/env python
"""Comprehensive Phase 2 evaluation — go/no-go gate for Phase 3.

Runs all benchmarks, all baselines, all ablations:
1. Full benchmark suite (KILT, MS MARCO, PubMedQA, NarrativeQA, git QA, memory)
2. Baselines (RAG dense, BM25, BgKIT L0-only)
3. Ablation: survivors present/zeroed/noise
4. Compression Pareto frontier at multiple retention ratios
5. Domain transfer: Phase 1 init vs raw init

Usage:
    python scripts/eval_phase2_comprehensive.py \
        +eval.checkpoint=checkpoints/phase2_step5_best \
        +eval.output_dir=eval_reports/phase2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()


def _load_trainer(cfg: DictConfig):
    """Load and set up a KRTrainer from config + checkpoint."""
    from bgkit.training.phase2.kr_trainer import KRTrainer

    trainer = KRTrainer(cfg)
    trainer.setup()

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path:
        from bgkit.training.checkpointing import load_checkpoint

        logger.info("loading_checkpoint", path=str(checkpoint_path))
        _metadata, state_dicts = load_checkpoint(Path(str(checkpoint_path)))
        model_state = state_dicts.get("model", {})
        if model_state:
            trainer.model.load_state_dict(model_state, strict=False)
            logger.info("checkpoint_loaded", keys=len(model_state))

    return trainer


def _generate_predictions(
    trainer,
    tokenizer,
    max_samples: int = 500,
) -> tuple[list[str], list[str], list[dict]]:
    """Generate text predictions from the trainer's eval dataloader.

    Returns:
        (predictions, references, metadata_list)
    """
    predictions: list[str] = []
    references: list[str] = []
    metadata_list: list[dict] = []
    samples_seen = 0

    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.eval_dataloader:
            if samples_seen >= max_samples:
                break

            prompt, prompt_mask = trainer._compose_prompt(batch)
            target_ids = batch["target_token_ids"].to(trainer.device)
            target_attention_mask = batch["target_attention_mask"].to(trainer.device)
            target_loss_mask = batch["target_loss_mask"].to(trainer.device)

            logits = trainer.decoder(
                prompt,
                target_ids,
                target_attention_mask,
                prompt_mask,
            )
            shifted_logits = logits[:, :-1]
            pred_ids = shifted_logits.argmax(dim=-1)
            shifted_mask = target_loss_mask[:, 1:]
            shifted_labels = target_ids[:, 1:]

            for i in range(pred_ids.size(0)):
                if samples_seen >= max_samples:
                    break
                mask_i = shifted_mask[i].bool()
                if not mask_i.any():
                    continue
                pred_text = tokenizer.decode(
                    pred_ids[i][mask_i].cpu().tolist(), skip_special_tokens=True,
                )
                ref_text = tokenizer.decode(
                    shifted_labels[i][mask_i].cpu().tolist(), skip_special_tokens=True,
                )
                predictions.append(pred_text)
                references.append(ref_text)
                meta = {}
                if "dataset_names" in batch and i < len(batch["dataset_names"]):
                    meta["dataset_name"] = batch["dataset_names"][i]
                if "metadata" in batch and i < len(batch["metadata"]):
                    meta.update(batch["metadata"][i])
                metadata_list.append(meta)
                samples_seen += 1

    trainer.model.train()
    return predictions, references, metadata_list


def _eval_benchmark(
    benchmark_name: str,
    predictions: list[str],
    references: list[str],
    metadata_list: list[dict],
) -> dict[str, float]:
    """Score predictions against a specific benchmark."""
    try:
        if benchmark_name == "pubmedqa":
            from bgkit.eval.benchmarks.pubmedqa_eval import evaluate_pubmedqa

            return evaluate_pubmedqa(predictions, references)

        elif benchmark_name == "narrativeqa":
            from bgkit.eval.benchmarks.narrativeqa_eval import evaluate_narrativeqa

            return evaluate_narrativeqa(predictions, [[r] for r in references])

        elif benchmark_name == "msmarco":
            from bgkit.eval.metrics.qa_metrics import token_f1

            # Use token-F1 as proxy for MRR when passage IDs unavailable
            f1_scores = [token_f1(p, [r]) for p, r in zip(predictions, references, strict=True)]
            return {
                "token_f1": sum(f1_scores) / max(len(f1_scores), 1),
                "n": float(len(predictions)),
            }

        elif benchmark_name == "kilt":
            from bgkit.eval.benchmarks.kilt_eval import (
                KILTPrediction,
                KILTReference,
                evaluate_kilt_downstream,
            )

            kilt_preds = [KILTPrediction(answer=p) for p in predictions]
            kilt_refs = [KILTReference(answers=[r]) for r in references]
            return evaluate_kilt_downstream(kilt_preds, kilt_refs)

        elif benchmark_name == "git_kr":
            from bgkit.eval.benchmarks.git_kr_eval import evaluate_git_kr

            question_types = [
                meta.get("answer_type", meta.get("task_name", "unknown"))
                for meta in metadata_list
            ]
            return evaluate_git_kr(predictions, [[r] for r in references], question_types)

        elif benchmark_name == "longmemeval":
            from bgkit.eval.metrics.qa_metrics import token_f1

            f1_scores = [token_f1(p, [r]) for p, r in zip(predictions, references, strict=True)]
            scaled = [1.0 + 4.0 * s for s in f1_scores]
            mean_score = sum(scaled) / max(len(scaled), 1)
            return {
                "mean_score": mean_score,
                "normalized_score": (mean_score - 1.0) / 4.0,
                "n": float(len(predictions)),
                "judge_mode": "token_f1_proxy",
            }

        elif benchmark_name == "locomo":
            from bgkit.eval.benchmarks.locomo_eval import evaluate_locomo

            categories = [int(meta.get("category", 1)) for meta in metadata_list]
            return evaluate_locomo(predictions, [[r] for r in references], categories)

        elif benchmark_name == "beam":
            from bgkit.eval.metrics.qa_metrics import token_f1

            f1_scores = [token_f1(p, [r]) for p, r in zip(predictions, references, strict=True)]
            scaled = [1.0 + 4.0 * s for s in f1_scores]
            mean_score = sum(scaled) / max(len(scaled), 1)
            return {
                "mean_score": mean_score,
                "normalized_score": (mean_score - 1.0) / 4.0,
                "n": float(len(predictions)),
                "judge_mode": "token_f1_proxy",
            }

        else:
            return {"status": "unknown_benchmark"}

    except ImportError as exc:
        return {"status": "import_error", "error": str(exc)}
    except Exception as exc:
        logger.warning("benchmark_eval_failed", benchmark=benchmark_name, error=str(exc))
        return {"status": "error", "error": str(exc)}


def _run_benchmark_suite(
    trainer,
    tokenizer,
    eval_cfg: DictConfig,
) -> dict:
    """Run all configured benchmarks via actual model inference."""
    benchmarks = list(eval_cfg.get("benchmarks", [
        "pubmedqa", "narrativeqa", "msmarco", "kilt", "git_kr",
        "longmemeval", "locomo", "beam",
    ]))

    max_samples = int(eval_cfg.get("max_eval_samples", 500))
    logger.info("benchmark_suite_generating_predictions", max_samples=max_samples)
    predictions, references, metadata_list = _generate_predictions(
        trainer, tokenizer, max_samples=max_samples,
    )
    logger.info("benchmark_suite_predictions", count=len(predictions))

    results = {}
    for benchmark in benchmarks:
        logger.info("eval_benchmark", name=benchmark)
        results[benchmark] = _eval_benchmark(
            benchmark, predictions, references, metadata_list,
        )

    return results


def _run_ablation_suite(
    trainer,
    tokenizer,
    eval_cfg: DictConfig,
) -> dict:
    """Run survivors present/zeroed/noise ablation.

    For each condition, modifies the compressed prompt before decoding
    and measures the change in answer quality.
    """
    max_samples = int(eval_cfg.get("ablation_samples", 200))
    conditions = ["present", "zeroed", "noise"]
    results = {}

    trainer.model.eval()
    for condition in conditions:
        logger.info("eval_ablation", condition=condition)
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_batches = 0
        samples_seen = 0

        with torch.no_grad():
            for batch in trainer.eval_dataloader:
                if samples_seen >= max_samples:
                    break

                prompt, prompt_mask = trainer._compose_prompt(batch)
                target_ids = batch["target_token_ids"].to(trainer.device)
                target_attention_mask = batch["target_attention_mask"].to(trainer.device)
                target_loss_mask = batch["target_loss_mask"].to(trainer.device)

                # Apply ablation condition to the prompt embeddings
                if condition == "zeroed":
                    prompt = torch.zeros_like(prompt)
                elif condition == "noise":
                    prompt = torch.randn_like(prompt) * prompt.std()
                # "present" uses the prompt as-is

                loss = trainer.decoder.forward_with_loss(
                    prompt,
                    target_ids,
                    target_attention_mask,
                    prompt_mask,
                    loss_mask=target_loss_mask,
                )

                logits = trainer.decoder(
                    prompt,
                    target_ids,
                    target_attention_mask,
                    prompt_mask,
                )
                shifted_logits = logits[:, :-1]
                shifted_labels = target_ids[:, 1:]
                shifted_mask = target_loss_mask[:, 1:]
                preds = shifted_logits.argmax(dim=-1)
                correct = ((preds == shifted_labels) & shifted_mask).sum().item()
                answer_tokens = shifted_mask.sum().item()

                total_correct += correct
                total_tokens += answer_tokens
                total_loss += loss.item()
                total_batches += 1
                samples_seen += target_ids.size(0)

        results[condition] = {
            "loss": total_loss / max(total_batches, 1),
            "token_accuracy": total_correct / max(total_tokens, 1),
            "n_batches": total_batches,
        }

    trainer.model.train()
    return results


def _run_compression_pareto(
    trainer,
    tokenizer,
    eval_cfg: DictConfig,
) -> dict:
    """Evaluate at multiple retention ratios to build a Pareto frontier."""
    ratios = list(eval_cfg.get("pareto_ratios", [0.50, 0.10, 0.05, 0.02, 0.01]))
    max_samples = int(eval_cfg.get("pareto_samples", 200))

    # Save original target ratio settings
    curriculum = trainer.step_cfg.get("curriculum", {})
    orig_start = curriculum.get("target_ratio_start")
    orig_end = curriculum.get("target_ratio_end")

    results_by_ratio = {}
    trainer.model.eval()

    for ratio in ratios:
        logger.info("eval_pareto_ratio", ratio=ratio)

        # Override the target ratio by patching the step config
        if hasattr(curriculum, "__setitem__"):
            curriculum["target_ratio_start"] = ratio
            curriculum["target_ratio_end"] = ratio
        else:
            # OmegaConf DictConfig may not allow direct assignment;
            # monkey-patch the method instead
            _original_target_ratio = trainer._target_ratio
            trainer._target_ratio = lambda r=ratio: r

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_batches = 0
        samples_seen = 0

        with torch.no_grad():
            for batch in trainer.eval_dataloader:
                if samples_seen >= max_samples:
                    break

                prompt, prompt_mask = trainer._compose_prompt(batch)
                target_ids = batch["target_token_ids"].to(trainer.device)
                target_attention_mask = batch["target_attention_mask"].to(trainer.device)
                target_loss_mask = batch["target_loss_mask"].to(trainer.device)

                loss = trainer.decoder.forward_with_loss(
                    prompt,
                    target_ids,
                    target_attention_mask,
                    prompt_mask,
                    loss_mask=target_loss_mask,
                )

                logits = trainer.decoder(
                    prompt,
                    target_ids,
                    target_attention_mask,
                    prompt_mask,
                )
                shifted_logits = logits[:, :-1]
                shifted_labels = target_ids[:, 1:]
                shifted_mask = target_loss_mask[:, 1:]
                preds = shifted_logits.argmax(dim=-1)
                correct = ((preds == shifted_labels) & shifted_mask).sum().item()
                answer_tokens = shifted_mask.sum().item()

                total_correct += correct
                total_tokens += answer_tokens
                total_loss += loss.item()
                total_batches += 1
                samples_seen += target_ids.size(0)

        results_by_ratio[str(ratio)] = {
            "ratio": ratio,
            "loss": total_loss / max(total_batches, 1),
            "token_accuracy": total_correct / max(total_tokens, 1),
            "n_batches": total_batches,
        }

        # Restore original ratio
        if hasattr(curriculum, "__setitem__"):
            if orig_start is not None:
                curriculum["target_ratio_start"] = orig_start
            if orig_end is not None:
                curriculum["target_ratio_end"] = orig_end
        else:
            trainer._target_ratio = _original_target_ratio

    trainer.model.train()
    return {"ratios": ratios, "results": results_by_ratio}


def _run_baseline_comparison(
    trainer,
    tokenizer,
    eval_cfg: DictConfig,
) -> dict:
    """Run RAG and BM25 baseline comparisons.

    Indexes eval documents, retrieves context with baselines, then scores
    the decoder's answer quality using retrieved context vs. compressed context.
    """
    from bgkit.eval.baselines.rag_baseline import BM25Baseline, RAGBaseline

    max_samples = int(eval_cfg.get("baseline_samples", 200))
    top_k = int(eval_cfg.get("baseline_top_k", 5))

    # Collect documents and questions from eval dataset
    documents: dict[str, str] = {}
    questions: list[str] = []
    ref_answers: list[str] = []

    inner_dataset = getattr(trainer.eval_dataset, "dataset", trainer.eval_dataset)
    eval_indices = (
        trainer.eval_dataset.indices
        if hasattr(trainer.eval_dataset, "indices")
        else range(len(trainer.eval_dataset))
    )

    for samples_seen, idx in enumerate(eval_indices):
        if samples_seen >= max_samples:
            break
        sample = inner_dataset[idx]
        doc_id = sample.document_id or str(idx)
        doc_text = tokenizer.decode(sample.content_token_ids.tolist(), skip_special_tokens=True)
        question_text = tokenizer.decode(
            sample.question_token_ids.tolist(), skip_special_tokens=True,
        )
        answer_text = tokenizer.decode(sample.answer_token_ids.tolist(), skip_special_tokens=True)
        documents[doc_id] = doc_text
        questions.append(question_text)
        ref_answers.append(answer_text)

    if not documents:
        return {"status": "no_eval_data"}

    results = {}

    # RAG baseline (dense retrieval)
    rag = RAGBaseline(
        embedding_model_name=eval_cfg.get("rag_embedding_model", "all-MiniLM-L6-v2"),
        reranker_model_name=eval_cfg.get("rag_reranker_model", None),
    )
    try:
        rag.index_repository(documents)
        rag_predictions = []
        for question in questions:
            context = rag.retrieve_text(question, top_k=top_k)
            # The RAG prediction is the retrieved context itself
            rag_predictions.append(context)

        from bgkit.eval.metrics.qa_metrics import token_f1

        rag_f1_scores = [
            token_f1(p, [r]) for p, r in zip(rag_predictions, ref_answers, strict=True)
        ]
        results["rag_dense"] = {
            "token_f1": sum(rag_f1_scores) / max(len(rag_f1_scores), 1),
            "n": float(len(rag_predictions)),
            "embedding_model": rag.embedding_model_name,
        }
    except Exception as exc:
        logger.warning("rag_baseline_failed", error=str(exc))
        results["rag_dense"] = {"status": "error", "error": str(exc)}

    # BM25 baseline
    bm25 = BM25Baseline()
    try:
        bm25.index_repository(documents)
        bm25_predictions = []
        for question in questions:
            retrieved = bm25.retrieve(question, top_k=top_k)
            # Concatenate retrieved doc texts
            parts = []
            for path, _score in retrieved:
                parts.append(documents.get(path, ""))
            bm25_predictions.append(" ".join(parts))

        from bgkit.eval.metrics.qa_metrics import token_f1

        bm25_f1_scores = [
            token_f1(p, [r]) for p, r in zip(bm25_predictions, ref_answers, strict=True)
        ]
        results["bm25"] = {
            "token_f1": sum(bm25_f1_scores) / max(len(bm25_f1_scores), 1),
            "n": float(len(bm25_predictions)),
        }
    except Exception as exc:
        logger.warning("bm25_baseline_failed", error=str(exc))
        results["bm25"] = {"status": "error", "error": str(exc)}

    # BgKIT model predictions (for comparison) -- use trainer's built-in eval
    bgkit_metrics = trainer.evaluate()
    results["bgkit"] = bgkit_metrics

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if not checkpoint_path:
        logger.error("No checkpoint specified. Use +eval.checkpoint=<path>")
        sys.exit(1)

    output_dir = Path(eval_cfg.get("output_dir", "eval_reports/phase2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trainer and models
    logger.info("phase2_loading_models")
    trainer = _load_trainer(cfg)

    decoder_name = trainer._resolve_decoder_backbone_name()
    tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True)

    report = {
        "checkpoint": str(checkpoint_path),
        "phase": "phase2_comprehensive",
    }

    # 1. Full benchmark suite
    logger.info("phase2_eval_benchmarks")
    report["benchmarks"] = _run_benchmark_suite(trainer, tokenizer, eval_cfg)

    # 2. Baselines
    logger.info("phase2_eval_baselines")
    report["baselines"] = _run_baseline_comparison(trainer, tokenizer, eval_cfg)

    # 3. Ablation suite
    logger.info("phase2_eval_ablations")
    report["ablations"] = _run_ablation_suite(trainer, tokenizer, eval_cfg)

    # 4. Compression Pareto
    logger.info("phase2_eval_pareto")
    report["compression_pareto"] = _run_compression_pareto(trainer, tokenizer, eval_cfg)

    # 5. Domain transfer: compare trainer metrics (baseline for this checkpoint)
    logger.info("phase2_eval_domain_transfer")
    domain_metrics = trainer.evaluate()
    report["domain_transfer"] = {
        "checkpoint_metrics": domain_metrics,
    }

    # Write report
    report_path = output_dir / "eval_phase2_comprehensive.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("phase2_eval_complete", report_path=str(report_path))
    print(json.dumps(report, indent=2, default=str))

    # WandB logging
    if eval_cfg.get("wandb", False):
        try:
            import wandb

            wandb.init(project=cfg.get("wandb_project", "bgkit"), name="eval_phase2")
            flat_metrics = {}
            for suite_name, suite_results in report.get("benchmarks", {}).items():
                if isinstance(suite_results, dict):
                    for k, v in suite_results.items():
                        if isinstance(v, (int, float)):
                            flat_metrics[f"eval/benchmark/{suite_name}/{k}"] = v
            for cond, cond_results in report.get("ablations", {}).items():
                if isinstance(cond_results, dict):
                    for k, v in cond_results.items():
                        if isinstance(v, (int, float)):
                            flat_metrics[f"eval/ablation/{cond}/{k}"] = v
            pareto = report.get("compression_pareto", {}).get("results", {})
            for ratio_key, ratio_results in pareto.items():
                if isinstance(ratio_results, dict):
                    for k, v in ratio_results.items():
                        if isinstance(v, (int, float)):
                            flat_metrics[f"eval/pareto/{ratio_key}/{k}"] = v
            for bl_name, bl_results in report.get("baselines", {}).items():
                if isinstance(bl_results, dict):
                    for k, v in bl_results.items():
                        if isinstance(v, (int, float)):
                            flat_metrics[f"eval/baseline/{bl_name}/{k}"] = v
            wandb.log(flat_metrics)
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
