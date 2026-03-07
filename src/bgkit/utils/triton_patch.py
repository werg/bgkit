"""Monkey-patch Triton's autotuner for Blackwell sm_121 compatibility.

Some Triton kernels crash with "RuntimeError: Triton Error [CUDA]: invalid
argument" on sm_121 when certain autotuner configs (num_warps, etc.) are tried.
The upstream autotuner's _bench method only catches OutOfResources/PTXASError
and CompileTimeAssertionFailure, so CUDA RuntimeErrors propagate and kill
the entire autotuning run.

Fix: patch _bench to also catch Triton CUDA launch errors, returning
[inf, inf, inf] so the config is skipped but benchmarking continues normally
for the remaining configs. This preserves the standard autotuner's ability to
select the fastest working config.

Scoped to sm_121 only — no-op on other compute capabilities.

See: https://github.com/fla-org/flash-linear-attention/issues/607
See: https://github.com/fla-org/flash-linear-attention/issues/734
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_triton_autotuner():
    """Patch Triton autotuner _bench to tolerate CUDA launch errors on sm_121."""
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    if not (major == 12 and minor == 1):
        logger.debug("Not sm_121 (got sm_%d%d), skipping Triton autotuner patch", major, minor)
        return

    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return

    original_bench = Autotuner._bench

    def _safe_bench(self, *args, config, **meta):
        try:
            return original_bench(self, *args, config=config, **meta)
        except RuntimeError as e:
            err = str(e)
            if "Triton Error" in err and "invalid argument" in err:
                kernel_name = getattr(self, "base_fn", self.fn).__name__
                logger.warning(
                    "sm_121 autotuner: config %s failed for %s — %s",
                    config, kernel_name, err[:200],
                )
                return [float("inf"), float("inf"), float("inf")]
            raise

    Autotuner._bench = _safe_bench
    logger.info(
        "Triton autotuner patched for sm_121: "
        "CUDA launch errors treated as config failures"
    )
