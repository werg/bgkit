from pathlib import Path

from hydra import compose, initialize_config_dir
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


def _compose_experiment(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[f"+experiment={name}"],
        )


def test_shared_decoder_training_contract_defaults_to_full_decoder_no_lora():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config")

    assert cfg.training.decoder_lora.enabled is False
    assert cfg.training.decoder_lora.r == 32
    assert cfg.training.decoder_lora.alpha == 64
    assert cfg.training.decoder_lora.peft_fused_backward is True
    assert cfg.training.decoder_lora.peft_fuse_gate_up is True
    assert cfg.training.freeze.decoder is False
    assert cfg.training.decoder_gradient_checkpointing is False
    assert cfg.training.decoder_layerwise_split.mode == "0"
    assert cfg.training.decoder_ce_strict is True


def test_qwen_training_configs_disable_decoder_lora_by_default():
    for name in (
        "phase1_step3.yaml",
        "phase1_step4.yaml",
        "phase1_step5.yaml",
        "phase1_step6.yaml",
        "phase3.yaml",
    ):
        cfg = OmegaConf.load(ROOT / "configs" / "training" / name)

        assert cfg.decoder_lora.enabled is False
        assert cfg.decoder_lora.r == 32
        assert cfg.decoder_lora.alpha == 64
        assert cfg.decoder_lora.peft_fused_backward is True
        assert cfg.decoder_lora.peft_fuse_gate_up is True


def _assert_qwen_step5_full_decoder_no_lora_contract(cfg):
    assert cfg.training.phase == "phase1_step5"
    assert cfg.training.decoder_lora.enabled is False
    assert cfg.training.freeze.decoder is False
    assert cfg.training.decoder_ce_impl == "cce"
    assert cfg.training.decoder_ce_strict is True
    assert cfg.training.gradient_checkpointing == "megatron"
    assert cfg.training.decoder_gradient_checkpointing is False
    assert cfg.training.decoder_layerwise_split.mode == "0"


def test_qwen_step5_default_experiment_trains_full_decoder_without_lora():
    _assert_qwen_step5_full_decoder_no_lora_contract(_compose_experiment("phase1_step5"))


def test_qwen_step5_frozen_decoder_alias_uses_no_lora_contract():
    cfg = _compose_experiment("phase1_step5_frozen_decoder")

    assert cfg.training.phase == "phase1_step5"
    assert cfg.training.decoder_lora.enabled is False
    assert cfg.training.freeze.decoder is True
    assert cfg.training.decoder_ce_impl == "cce"
    assert cfg.training.decoder_ce_strict is True
    assert cfg.training.gradient_checkpointing == "megatron"
    assert cfg.training.decoder_gradient_checkpointing is False
    assert cfg.training.decoder_layerwise_split.mode == "auto"


def test_qwen_step5_lora_baseline_stays_explicit():
    cfg = _compose_experiment("phase1_step5_lora_baseline")

    assert cfg.training.phase == "phase1_step5"
    assert cfg.training.decoder_lora.enabled is True
    assert cfg.training.decoder_lora.r == 32
    assert cfg.training.decoder_lora.alpha == 64
    assert cfg.training.decoder_lora.peft_fused_backward is True
    assert cfg.training.decoder_lora.peft_fuse_gate_up is True
    assert cfg.training.freeze.decoder is False
