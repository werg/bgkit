"""GPU liveness regression: content-bearing reps must beat zeroed reps.

Durable regression guard for the rep-COLLAPSE class (the git-repro encoder's
projected reps drifted off the decoder's readable manifold and stopped carrying
content). It reuses the EXACT machinery of ``scripts/diag_verbatim_decisive.py``
(splice via ``forward_with_single_splice`` in the REAL chat template built by
``SummarizationRoundRobinTrainer._build_chat_inputs`` — NOT a raw
``[reps|prompt|text]`` framing, which a prior canary got wrong and measured
garbage for Falcon), and asserts::

    rep_gain = mean(ce_zeroed - ce_reps) > 0.5   # reps are load-bearing

This needs a REAL encoder+decoder forward on GPU + a matched summarization
checkpoint + the summarization data pipeline, so it can only run inside the
training container. It is gpu+integration marked and SKIPS unless CUDA is present
AND ``BGKIT_LIVENESS_CKPT`` points at a checkpoint dir. On the host
``pytest tests/unit`` run it is always skipped — the ALWAYS-ON runtime norm guard
(``ReconstructionDecoder._maybe_guard_spliced_rep_norm``) is the primary,
zero-setup protection; this is the deeper periodic check.

Run in-container (trainer stopped, GPU free), e.g.::

    BGKIT_LIVENESS_CKPT=/workspace/checkpoints/phase1_summarization_round_robin_step51945_... \
      pytest tests/unit/training/test_rep_liveness_regression.py -v -m gpu
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

_LIVENESS_CKPT = os.environ.get("BGKIT_LIVENESS_CKPT")
# In-distribution operating point for the step-51945 checkpoint (L0=L1=0.316 ~=
# 0.10 end-to-end == the summarization curriculum END the ckpt was trained at).
_L0, _L1 = 0.316, 0.316
_MIN_REP_GAIN = float(os.environ.get("BGKIT_LIVENESS_MIN_GAIN", "0.5"))
_N_SAMPLES = int(os.environ.get("BGKIT_LIVENESS_N_SAMPLES", "6"))
_FAMILIES = [
    f.strip()
    for f in os.environ.get("BGKIT_LIVENESS_FAMILIES", "qwen35,falcon_h1").split(",")
    if f.strip()
]


def _load_diag_module():
    """Import ``scripts/diag_verbatim_decisive.py`` as a module to reuse its
    ``ce_with_slot`` / ``build_loss_mask`` helpers (the correct-template splice)."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "diag_verbatim_decisive.py"
    spec = importlib.util.spec_from_file_location("diag_verbatim_decisive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (container)")
@pytest.mark.skipif(
    not _LIVENESS_CKPT,
    reason="set BGKIT_LIVENESS_CKPT=<matched summarization checkpoint dir>",
)
def test_content_reps_beat_zeroed_in_chat_template():
    from hydra import compose, initialize_config_dir

    from bgkit.training.phase1.summarization_round_robin import (
        SummarizationRoundRobinTrainer,
    )
    from bgkit.utils.logging import setup_logging

    setup_logging()
    diag = _load_diag_module()

    configs_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=phase1_summarization_round_robin",
                f"step1_checkpoint={_LIVENESS_CKPT}",
                "training.max_total_source_tokens=3072",
            ],
        )

    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()

    # A handful of short single-sample eval batches (B=1 for clean per-sample CE).
    samples = []
    for flat_i in trainer._eval_flat_idx[: _N_SAMPLES * 4]:
        b = trainer._collate([int(flat_i)])
        n_src = sum(len(d) for d in b["source_docs"][0])
        if n_src > int(cfg.training.get("max_total_source_tokens", 3072)):
            continue
        samples.append(b)
        if len(samples) >= _N_SAMPLES:
            break
    assert samples, "no eval samples collected"

    for family in _FAMILIES:
        trainer.encoder.set_active_decoder_family(family)
        decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
        gains = []
        with torch.no_grad():
            for batch in samples:
                # CORRECT TEMPLATE: real chat prefix/suffix, not raw framing.
                prefix_ids, suffix_ids, suffix_masks, comp = trainer._build_chat_inputs(
                    family, batch,
                )
                trainer._target_ratio_start = trainer._target_ratio_end = _L0
                trainer.global_step = 0
                trainer._l1_introduction_step = 0
                trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = _L1
                enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, comp)
                survivors = enc_out.survivor_embeddings
                if int(survivors.shape[0]) == 0:
                    enc_out.release()
                    continue
                lm = diag.build_loss_mask(
                    prefix_ids, per_group, suffix_masks, trainer.device,
                )
                ce_reps = diag.ce_with_slot(
                    trainer, decoder, survivors, group_cu, prefix_ids, suffix_ids, lm,
                )
                ce_zero = diag.ce_with_slot(
                    trainer, decoder, torch.zeros_like(survivors), group_cu,
                    prefix_ids, suffix_ids, lm,
                )
                gains.append(ce_zero - ce_reps)
                enc_out.release()

        assert gains, f"{family}: no non-empty survivor samples"
        mean_gain = sum(gains) / len(gains)
        assert mean_gain > _MIN_REP_GAIN, (
            f"{family}: reps NOT load-bearing in the chat template — "
            f"mean rep_gain={mean_gain:.3f} <= {_MIN_REP_GAIN} "
            f"(ce_zeroed - ce_reps over {len(gains)} samples). This is the "
            f"rep-collapse regression the norm-band regularizer + guard defend."
        )
