from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from scripts import profile_training_step as profiler
from torch import nn


def _batch(token_count: int, cu_values: list[int]) -> dict[str, torch.Tensor]:
    return {
        "content_token_ids": torch.arange(token_count, dtype=torch.long),
        "content_cu_seqlens": torch.tensor(cu_values, dtype=torch.int32),
    }


def test_batch_bucket_signature_records_tensor_metadata_and_cu_lengths():
    sig = profiler._batch_bucket_signature(_batch(6, [0, 2, 6]))

    assert sig["tensors"] == [
        {
            "key": "content_cu_seqlens",
            "shape": [3],
            "dtype": "int32",
            "device": "cpu",
        },
        {
            "key": "content_token_ids",
            "shape": [6],
            "dtype": "int64",
            "device": "cpu",
        },
    ]
    assert sig["cu_lengths"]["content_cu_seqlens"] == {
        "count": 2,
        "total": 6,
        "min": 2,
        "max": 4,
        "l2_cost": 20,
        "lengths_hash": profiler._stable_json_hash([2, 4]),
    }


def test_static_bucket_summary_counts_micro_and_optimizer_step_buckets():
    repeated_step = [_batch(6, [0, 2, 6]), _batch(4, [0, 4])]
    different_step = [_batch(6, [0, 1, 6]), _batch(4, [0, 4])]
    signatures = profiler._batch_bucket_signatures(
        [repeated_step, repeated_step, different_step]
    )

    summary = profiler._summarize_static_buckets(signatures)

    assert summary["optimizer_steps"] == 3
    assert summary["microbatches"] == 6
    assert summary["unique_microbatch_buckets"] == 3
    assert summary["unique_optimizer_step_buckets"] == 2
    assert summary["optimizer_step_buckets"][0]["count"] == 2
    assert profiler._compact_static_bucket_summary(summary) == {
        "optimizer_steps": 3,
        "microbatches": 6,
        "unique_microbatch_buckets": 3,
        "unique_optimizer_step_buckets": 2,
        "omitted_microbatch_buckets": 0,
        "omitted_optimizer_step_buckets": 0,
    }


def test_fixed_step_batches_can_replay_first_bucket():
    first = [_batch(3, [0, 3])]
    second = [_batch(5, [0, 5])]
    prefetched = [first, second]

    assert profiler._fixed_step_batches(
        prefetched,
        1,
        repeat_first_fixed_step=True,
    ) is first
    assert profiler._fixed_step_batches(
        prefetched,
        1,
        repeat_first_fixed_step=False,
    ) is second


def test_trainable_contract_profile_requires_decoder_params_in_optimizer():
    encoder = nn.Linear(4, 4)
    decoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    optimizer = torch.optim.SGD(
        [
            {"params": list(encoder.parameters()), "lr": 1e-4, "base_lr": 1e-4},
            {"params": list(decoder.parameters()), "lr": 5e-5, "base_lr": 5e-5},
        ],
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "freeze": {"decoder": False},
                "decoder_lora": {"enabled": False},
            },
        },
    )
    trainer = SimpleNamespace(encoder=encoder, decoder=decoder, optimizer=optimizer, cfg=cfg)

    profile = profiler._trainable_contract_profile(trainer)

    assert profile["contract_ok"] is True
    assert profile["components"]["decoder"]["trainable_params"] == sum(
        p.numel() for p in decoder.parameters()
    )
    assert profile["optimizer"]["params_by_component"]["decoder"] == sum(
        p.numel() for p in decoder.parameters()
    )
    assert profile["optimizer"]["missing_decoder_optimizer_params"] == 0


def test_trainable_contract_profile_rejects_missing_decoder_optimizer_group():
    encoder = nn.Linear(4, 4)
    decoder = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(
        [{"params": list(encoder.parameters()), "lr": 1e-4, "base_lr": 1e-4}],
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "freeze": {"decoder": False},
                "decoder_lora": {"enabled": False},
            },
        },
    )
    trainer = SimpleNamespace(encoder=encoder, decoder=decoder, optimizer=optimizer, cfg=cfg)

    profile = profiler._trainable_contract_profile(trainer)

    assert profile["contract_ok"] is False
    assert profile["components"]["decoder"]["trainable_params"] > 0
    assert profile["optimizer"]["params_by_component"]["decoder"] == 0
    assert profile["optimizer"]["missing_decoder_optimizer_params"] == sum(
        p.numel() for p in decoder.parameters()
    )


def test_profile_resume_prefers_phase_checkpoint_over_last_file(tmp_path, monkeypatch):
    last = tmp_path / "phase1_falcon_l0_align_step1"
    phase_match = tmp_path / "phase1_step5_step2"
    last.mkdir()
    phase_match.mkdir()
    (tmp_path / ".last_checkpoint").write_text(str(last))

    monkeypatch.setattr(
        profiler,
        "resolve_latest_checkpoint",
        lambda checkpoint_dir, phase: phase_match,
    )
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": str(tmp_path),
            "resume_checkpoint": None,
            "training": {"phase": "phase1_step5"},
        },
    )

    assert profiler._resolve_resume_checkpoint(cfg) == phase_match


def test_profile_resume_uses_last_file_only_when_phase_has_no_checkpoint(
    tmp_path,
    monkeypatch,
):
    last = tmp_path / "phase1_falcon_l0_align_step1"
    last.mkdir()
    (tmp_path / ".last_checkpoint").write_text(str(last))

    monkeypatch.setattr(
        profiler,
        "resolve_latest_checkpoint",
        lambda checkpoint_dir, phase: None,
    )
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": str(tmp_path),
            "resume_checkpoint": None,
            "training": {"phase": "phase1_step5"},
        },
    )

    assert profiler._resolve_resume_checkpoint(cfg) == last
