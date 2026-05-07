import torch
from scripts import profile_training_step as profiler


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
