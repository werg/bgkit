"""Config gating for gradient checkpointing.

``maybe_enable_gradient_checkpointing`` should keep checkpointing off by
default, honor ``compute.gradient_checkpointing: true`` to enable it, and
let ``training.gradient_checkpointing`` override ``compute`` when explicit.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from bgkit.training.gradient_utils import (
    configure_decoder_layerwise_split,
    gradient_checkpointing_requested,
    maybe_enable_decoder_gradient_checkpointing,
    maybe_enable_frozen_decoder_kernels,
    maybe_enable_gradient_checkpointing,
    validate_decoder_lora_freeze_contract,
)


def _cfg(**sections):
    return OmegaConf.create(sections)


@pytest.fixture(autouse=True)
def _clear_frozen_kernel_env(monkeypatch):
    monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV", raising=False)
    monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX", raising=False)
    monkeypatch.delenv(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM",
        raising=False,
    )
    monkeypatch.delenv(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX",
        raising=False,
    )
    monkeypatch.delenv(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX",
        raising=False,
    )
    monkeypatch.delenv("FLA_GDR_FUSE_QK_L2NORM_BWD", raising=False)
    monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CORE_BWD", raising=False)
    monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN", raising=False)
    monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN", raising=False)
    monkeypatch.delenv("BGKIT_FROZEN_ATTENTION_QKV_FUSION", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_SWIGLU_FUSION", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_BASE_FUSION", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_BASE_DX", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_DIRECT_DX_MAX_ROWS", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_RESIDUAL_FUSION", raising=False)
    monkeypatch.delenv("BGKIT_DECODER_MLP_QUACK_FUSION", raising=False)


class TestGradientCheckpointingRequested:
    def test_default_is_false(self):
        cfg = _cfg(compute={}, training={})
        assert gradient_checkpointing_requested(cfg) is False

    def test_compute_false_disables(self):
        cfg = _cfg(compute={"gradient_checkpointing": False}, training={})
        assert gradient_checkpointing_requested(cfg) is False

    def test_compute_true_enables(self):
        cfg = _cfg(compute={"gradient_checkpointing": True}, training={})
        assert gradient_checkpointing_requested(cfg) is True

    def test_training_overrides_compute_true(self):
        cfg = _cfg(
            compute={"gradient_checkpointing": False},
            training={"gradient_checkpointing": True},
        )
        assert gradient_checkpointing_requested(cfg) is True

    def test_training_overrides_compute_false(self):
        cfg = _cfg(
            compute={"gradient_checkpointing": True},
            training={"gradient_checkpointing": False},
        )
        assert gradient_checkpointing_requested(cfg) is False


class TestMaybeEnableGradientCheckpointing:
    def test_enables_when_requested(self):
        model = MagicMock()
        cfg = _cfg(compute={"gradient_checkpointing": True}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()

    def test_skips_when_disabled(self):
        model = MagicMock()
        cfg = _cfg(compute={"gradient_checkpointing": False}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()

    def test_default_skips(self):
        # No explicit key set anywhere: checkpointing defaults off.
        model = MagicMock()
        cfg = _cfg(compute={}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()


class TestMaybeEnableDecoderGradientCheckpointing:
    def test_skips_permanently_frozen_decoder_by_default(self):
        model = MagicMock()
        cfg = _cfg(
            compute={"gradient_checkpointing": "megatron"},
            training={"freeze": {"decoder": True}},
        )

        assert maybe_enable_decoder_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()

    def test_can_opt_back_in_for_frozen_decoder(self):
        model = MagicMock()
        cfg = _cfg(
            compute={"gradient_checkpointing": "megatron"},
            training={
                "freeze": {"decoder": True},
                "checkpoint_frozen_decoder": True,
            },
        )

        assert maybe_enable_decoder_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()

    def test_non_frozen_decoder_does_not_inherit_global_checkpointing(self):
        model = MagicMock()
        cfg = _cfg(
            compute={"gradient_checkpointing": "megatron"},
            training={"freeze": {"decoder": False}},
        )

        assert maybe_enable_decoder_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()

    def test_decoder_specific_false_overrides_global_checkpointing(self):
        model = MagicMock()
        cfg = _cfg(
            compute={"gradient_checkpointing": "megatron"},
            training={
                "freeze": {"decoder": False},
                "decoder_gradient_checkpointing": False,
            },
        )

        assert maybe_enable_decoder_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()

    def test_decoder_specific_true_uses_standard_checkpointing(self):
        model = MagicMock()
        cfg = _cfg(
            compute={"gradient_checkpointing": False},
            training={
                "freeze": {"decoder": False},
                "decoder_gradient_checkpointing": True,
            },
        )

        assert maybe_enable_decoder_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()


class TestMaybeEnableFrozenDecoderKernels:
    def test_skips_by_default(self):
        decoder = MagicMock()
        cfg = _cfg(training={"freeze": {"decoder": True}})

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {}
        decoder.enable_frozen_deltanet_channel_last_conv.assert_not_called()


class TestConfigureDecoderLayerwiseSplit:
    def test_skips_decoders_without_config_hook(self):
        decoder = object()
        cfg = _cfg(training={"decoder_layerwise_split": {"mode": "auto"}})

        configure_decoder_layerwise_split(decoder, cfg)

    def test_passes_structured_config_to_decoder_hook(self):
        decoder = MagicMock()
        cfg = _cfg(
            training={
                "decoder_layerwise_split": {
                    "mode": "auto",
                    "min_ratio": 2.5,
                    "min_prefix": 768,
                    "packed_deltanet": False,
                },
            },
        )

        configure_decoder_layerwise_split(decoder, cfg)

        decoder.set_qwen35_layerwise_split.assert_called_once_with(
            mode="auto",
            min_ratio=2.5,
            min_prefix=768,
            packed_deltanet=False,
        )

    def test_passes_scalar_config_to_decoder_hook(self):
        decoder = MagicMock()
        cfg = _cfg(training={"decoder_layerwise_split": "threshold"})

        configure_decoder_layerwise_split(decoder, cfg)

        decoder.set_qwen35_layerwise_split.assert_called_once_with(mode="threshold")

    def test_installs_channel_last_conv_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_channel_last_conv.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_channel_last_conv": True},
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_channel_last_conv": 18,
        }
        decoder.enable_frozen_deltanet_channel_last_conv.assert_called_once()

    def test_installs_stock_channel_last_fused_qkv_conv_l2norm_envs(self):
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_channel_last_conv.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_channel_last_conv": True,
                    "deltanet_stock_fused_qkv_conv_l2norm": True,
                    "deltanet_stock_fused_qkv_conv_l2norm_dx": True,
                    "deltanet_stock_fused_qkv_conv_split_dx": True,
                    "deltanet_fuse_qk_l2norm_bwd": True,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_stock_fused_qkv_conv_l2norm": 1,
            "deltanet_stock_fused_qkv_conv_l2norm_dx": 1,
            "deltanet_stock_fused_qkv_conv_split_dx": 1,
            "deltanet_fuse_qk_l2norm_bwd": 1,
            "deltanet_channel_last_conv": 18,
        }
        assert os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM"] == "1"
        assert os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX"] == "1"
        assert os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX"] == "1"
        assert os.environ["FLA_GDR_FUSE_QK_L2NORM_BWD"] == "1"

    def test_installs_deltanet_core_bwd_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_core_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_core_bwd": True},
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_core_bwd": 18,
        }
        decoder.enable_frozen_deltanet_core_bwd.assert_called_once()

    def test_installs_deltanet_residual_bwd_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_residual_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_residual_bwd": True},
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_residual_bwd": 18,
        }
        decoder.enable_frozen_deltanet_residual_bwd.assert_called_once()

    def test_installs_deltanet_residual_mlp_bwd_env_when_requested(self, monkeypatch):
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD", raising=False)
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_residual_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_residual_bwd": True,
                    "deltanet_residual_mlp_bwd": True,
                },
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_residual_mlp_bwd": 1,
                "deltanet_residual_bwd": 18,
            }
            assert os.environ["BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD"] == "1"
            decoder.enable_frozen_deltanet_residual_bwd.assert_called_once()
        finally:
            os.environ.pop("BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD", None)

    def test_installs_deltanet_input_rmsnorm_dx_with_residual(self, monkeypatch):
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", raising=False)
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_residual_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_residual_bwd": True,
                    "deltanet_input_rmsnorm_dx": True,
                },
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_input_rmsnorm_dx": 1,
                "deltanet_residual_bwd": 18,
            }
            assert os.environ["BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX"] == "1"
        finally:
            os.environ.pop("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", None)

    def test_installs_deltanet_input_rmsnorm_dx_with_core(self, monkeypatch):
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", raising=False)
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_core_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_input_rmsnorm_dx": True,
                },
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_core_bwd": 18,
                "deltanet_input_rmsnorm_dx": 1,
            }
            assert os.environ["BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX"] == "1"
        finally:
            os.environ.pop("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", None)

    def test_installs_deltanet_core_channel_last_env_when_requested(self, monkeypatch):
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV", raising=False)
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX", raising=False)
        monkeypatch.delenv(
            "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM",
            raising=False,
        )
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_core_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_channel_last_conv": True,
                    "deltanet_core_channel_last_conv_dx": True,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_core_bwd": 18,
            "deltanet_core_channel_last_conv": 1,
            "deltanet_core_channel_last_conv_dx": 1,
        }
        assert os.environ["BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV"] == "1"
        assert os.environ["BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX"] == "1"

    def test_installs_deltanet_core_fused_qkv_conv_l2norm_env_when_requested(
        self,
        monkeypatch,
    ):
        monkeypatch.delenv("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV", raising=False)
        monkeypatch.delenv(
            "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM",
            raising=False,
        )
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_core_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_channel_last_conv": True,
                    "deltanet_core_fused_qkv_conv_l2norm": True,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_core_bwd": 18,
            "deltanet_core_channel_last_conv": 1,
            "deltanet_core_fused_qkv_conv_l2norm": 1,
        }
        assert os.environ["BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV"] == "1"
        assert os.environ["BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM"] == "1"

    def test_installs_deltanet_core_seq_guards_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_core_bwd.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_min_seq_len": 1024,
                    "deltanet_core_max_seq_len": 4096,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "deltanet_core_bwd": 18,
        }
        assert os.environ["BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN"] == "1024"
        assert os.environ["BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN"] == "4096"

    def test_installs_deltanet_raw_gate_env_when_requested(self, monkeypatch):
        monkeypatch.delenv("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", raising=False)
        decoder = MagicMock()
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_raw_gate_in_kernel": True},
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_raw_gate_in_kernel": 1,
            }
            assert os.environ["BGKIT_DELTANET_RAW_GATE_IN_KERNEL"] == "1"
        finally:
            os.environ.pop("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", None)

    def test_installs_deltanet_raw_gate_with_channel_last_forward(self, monkeypatch):
        monkeypatch.delenv("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", raising=False)
        decoder = MagicMock()
        decoder.enable_frozen_deltanet_channel_last_conv.return_value = 18
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_channel_last_conv": True,
                    "deltanet_raw_gate_in_kernel": True,
                },
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_channel_last_conv": 18,
                "deltanet_raw_gate_in_kernel": 1,
            }
            assert os.environ["BGKIT_DELTANET_RAW_GATE_IN_KERNEL"] == "1"
        finally:
            os.environ.pop("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", None)

    def test_installs_deltanet_pair_qk_l2norm_env_when_requested(self, monkeypatch):
        monkeypatch.delenv("FLA_GDR_PAIR_QK_L2NORM_FWD", raising=False)
        decoder = MagicMock()
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_pair_qk_l2norm_fwd": True},
            },
        )

        try:
            assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
                "deltanet_pair_qk_l2norm_fwd": 1,
            }
            assert os.environ["FLA_GDR_PAIR_QK_L2NORM_FWD"] == "1"
        finally:
            os.environ.pop("FLA_GDR_PAIR_QK_L2NORM_FWD", None)

    def test_installs_attention_qkv_when_requested(self):
        decoder = MagicMock()
        decoder.enable_fused_attention_qkv.return_value = 6
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"attention_qkv": True},
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "attention_qkv": 6,
        }
        decoder.enable_fused_attention_qkv.assert_called_once_with()

    def test_installs_mlp_swiglu_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_mlp_swiglu_fusion.return_value = 24
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "mlp_swiglu": True,
                    "mlp_swiglu_triton_forward": True,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "mlp_swiglu": 24,
        }
        decoder.enable_frozen_mlp_swiglu_fusion.assert_called_once_with(
            use_triton_forward=True
        )

    def test_installs_mlp_base_dx_env_when_requested(self):
        decoder = MagicMock()
        decoder.enable_frozen_mlp_fusion.return_value = 24
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "mlp_base": True,
                    "mlp_base_dx": "adaptive",
                    "mlp_base_direct_dx_max_rows": 128,
                },
            },
        )

        assert maybe_enable_frozen_decoder_kernels(decoder, cfg) == {
            "mlp_base": 24,
        }
        assert os.environ["BGKIT_DECODER_MLP_BASE_DX"] == "adaptive"
        assert os.environ["BGKIT_DECODER_MLP_DIRECT_DX_MAX_ROWS"] == "128"

    def test_rejects_channel_last_conv_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"deltanet_channel_last_conv": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_deltanet_core_bwd_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"deltanet_core_bwd": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_deltanet_residual_bwd_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"deltanet_residual_bwd": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_channel_last_and_core_bwd_together(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_channel_last_conv": True,
                    "deltanet_core_bwd": True,
                },
            },
        )

        import pytest

        with pytest.raises(ValueError, match="mutually exclusive"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_deltanet_residual_with_other_deltanet_rewrites(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_residual_bwd": True,
                    "deltanet_core_bwd": True,
                },
            },
        )

        import pytest

        with pytest.raises(ValueError, match="mutually exclusive"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_stock_fused_qkv_conv_l2norm_without_channel_last(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_stock_fused_qkv_conv_l2norm": True,
                },
            },
        )

        with pytest.raises(ValueError, match="deltanet_channel_last_conv=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_stock_split_dx_without_stock_fused_qkv_conv_l2norm(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_channel_last_conv": True,
                    "deltanet_stock_fused_qkv_conv_split_dx": True,
                },
            },
        )

        with pytest.raises(
            ValueError,
            match="deltanet_stock_fused_qkv_conv_l2norm=true",
        ):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_raw_gate_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"deltanet_raw_gate_in_kernel": True},
            },
        )

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_raw_gate_with_deltanet_core_rewrite(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_raw_gate_in_kernel": True,
                    "deltanet_core_bwd": True,
                },
            },
        )

        with pytest.raises(ValueError, match="disable deltanet_core_bwd"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_raw_gate_with_deltanet_residual_rewrite(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_raw_gate_in_kernel": True,
                    "deltanet_residual_bwd": True,
                },
            },
        )

        with pytest.raises(ValueError, match="disable deltanet_residual_bwd"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_deltanet_residual_mlp_without_residual_rewrite(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_residual_mlp_bwd": True},
            },
        )

        with pytest.raises(ValueError, match="deltanet_residual_bwd=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_deltanet_input_rmsnorm_dx_without_core_or_residual(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_input_rmsnorm_dx": True},
            },
        )

        with pytest.raises(ValueError, match="deltanet_core_bwd or deltanet_residual_bwd"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_pair_qk_l2norm_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"deltanet_pair_qk_l2norm_fwd": True},
            },
        )

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_attention_qkv_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"attention_qkv": True},
            },
        )

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_core_channel_last_without_core_bwd(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {"deltanet_core_channel_last_conv": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match="requires frozen_decoder_kernels"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_core_channel_last_dx_without_core_channel_last(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_channel_last_conv_dx": True,
                },
            },
        )

        import pytest

        with pytest.raises(ValueError, match="deltanet_core_channel_last_conv=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_core_fused_qkv_conv_l2norm_without_core_channel_last(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_fused_qkv_conv_l2norm": True,
                },
            },
        )

        import pytest

        with pytest.raises(ValueError, match="deltanet_core_channel_last_conv=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_invalid_deltanet_core_seq_guards(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "deltanet_core_bwd": True,
                    "deltanet_core_min_seq_len": 4096,
                    "deltanet_core_max_seq_len": 1024,
                },
            },
        )

        with pytest.raises(ValueError, match="deltanet_core_min_seq_len"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_mlp_swiglu_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"mlp_swiglu": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_mlp_base_when_decoder_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "frozen_decoder_kernels": {"mlp_base": True},
            },
        )

        with pytest.raises(ValueError, match=r"requires training\.freeze\.decoder=true"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_invalid_mlp_base_dx(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "mlp_base": True,
                    "mlp_base_dx": "wide",
                },
            },
        )

        with pytest.raises(ValueError, match="mlp_base_dx"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_invalid_mlp_base_direct_dx_max_rows(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "mlp_base": True,
                    "mlp_base_direct_dx_max_rows": 0,
                },
            },
        )

        with pytest.raises(ValueError, match="mlp_base_direct_dx_max_rows"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)

    def test_rejects_multiple_mlp_kernel_modes(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "frozen_decoder_kernels": {
                    "mlp_swiglu": True,
                    "mlp_base": True,
                },
            },
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            maybe_enable_frozen_decoder_kernels(MagicMock(), cfg)


class TestDecoderLoraFreezeContract:
    def test_rejects_lora_on_permanently_frozen_decoder(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "decoder_lora": {"enabled": True},
            },
        )

        import pytest

        with pytest.raises(ValueError, match="no-LoRA frozen-decoder contract"):
            validate_decoder_lora_freeze_contract(cfg)

    def test_allows_frozen_decoder_without_lora(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": True},
                "decoder_lora": {"enabled": False},
            },
        )

        validate_decoder_lora_freeze_contract(cfg)

    def test_allows_lora_when_decoder_is_trainable(self):
        cfg = _cfg(
            training={
                "freeze": {"decoder": False},
                "decoder_lora": {"enabled": True},
            },
        )

        validate_decoder_lora_freeze_contract(cfg)


class TestMegatronGradientCheckpointing:
    """The ``"megatron"`` mode installs a selective per-op checkpoint policy."""

    @staticmethod
    def _build_layer_model():
        """Build a mock model with a mix of DeltaNet and FullAttn layers."""
        import torch.nn as nn

        class _Layer(nn.Module):
            def __init__(self, kind: str):
                super().__init__()
                # Mimic HF: each layer carries its own gradient_checkpointing flag.
                self.gradient_checkpointing = False
                self._gradient_checkpointing_func = None
                if kind == "deltanet":
                    self.linear_attn = nn.Module()
                else:
                    self.self_attn = nn.Module()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 3 DeltaNet + 1 FullAttention, repeated twice → 8 layers.
                kinds = (["deltanet"] * 3 + ["full"]) * 2
                self.layers = nn.ModuleList([_Layer(k) for k in kinds])

            def gradient_checkpointing_enable(self, **kwargs):
                # HF's enable propagates the flag to every layer.
                for layer in self.layers:
                    layer.gradient_checkpointing = True

        return _Model()

    def test_megatron_mode_swaps_checkpoint_func_on_all_layers(self):
        model = self._build_layer_model()
        cfg = _cfg(compute={}, training={"gradient_checkpointing": "megatron"})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True

        from bgkit.training.gradient_utils import _megatron_checkpoint_func

        assert all(layer.gradient_checkpointing is True for layer in model.layers)
        assert all(
            layer._gradient_checkpointing_func is _megatron_checkpoint_func
            for layer in model.layers
        )

    def test_megatron_returns_swapped_count(self):
        from bgkit.training.gradient_utils import _install_megatron_checkpoint_func

        model = self._build_layer_model()
        # Manually enable first, like maybe_enable_gradient_checkpointing does.
        model.gradient_checkpointing_enable()
        swapped = _install_megatron_checkpoint_func(model)
        assert swapped == 8

    def test_megatron_no_op_when_ckpt_not_enabled(self):
        """Don't quietly mask misconfiguration — only flip layers that
        actually had ckpt enabled."""
        from bgkit.training.gradient_utils import _install_megatron_checkpoint_func

        model = self._build_layer_model()
        # Skip enable.
        swapped = _install_megatron_checkpoint_func(model)
        assert swapped == 0

    def test_string_aliases_resolve(self):
        """Megatron/selective aliases all map to the Megatron-style policy."""
        from bgkit.training.gradient_utils import _coerce_gradient_checkpointing_value
        for alias in ("megatron", "selective", "selective_ops", "SELECTIVE"):
            assert _coerce_gradient_checkpointing_value(alias) == "megatron"
        assert _coerce_gradient_checkpointing_value("true") is True
        assert _coerce_gradient_checkpointing_value("false") is False
