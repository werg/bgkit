from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def test_falcon_phase1_configs_disable_decoder_lora_by_default():
    for name in ("phase1_falcon_l0.yaml", "phase1_falcon_l1.yaml"):
        cfg = OmegaConf.load(ROOT / "configs" / "training" / name)

        assert cfg.model.decoder.family == "falcon_h1"
        assert cfg.decoder_lora.enabled is False


def test_falcon_phase1_configs_use_aggressive_decoder_aware_batching():
    for name in ("phase1_falcon_l0.yaml", "phase1_falcon_l1.yaml"):
        cfg = OmegaConf.load(ROOT / "configs" / "training" / name)

        assert cfg.max_batch_tokens == 6144
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.sampler.cost_multiplier == 1.0
        assert cfg.sampler.eval_cost_multiplier == 1.0


def test_falcon_phase2_configs_disable_encoder_lora_by_default():
    for name in (
        "phase2_kb_stage_a_falcon.yaml",
        "phase2_kb_stage_b_falcon.yaml",
    ):
        cfg = OmegaConf.load(ROOT / "configs" / "training" / name)

        assert cfg.model.decoder.family == "falcon_h1"
        assert cfg.lora.enabled is False
        assert cfg.lora.train_l1_direct is True
