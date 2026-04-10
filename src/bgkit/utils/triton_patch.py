"""Monkey-patch Triton's autotuner and kernel loader for Blackwell sm_121.

Two issues are addressed, both scoped to sm_121:

1. Autotuner _bench: some Triton kernels crash with "Triton Error [CUDA]:
   invalid argument" or "operation not permitted" when certain configs are
   tried. Upstream _bench only catches OutOfResources/PTXASError, so CUDA
   RuntimeErrors propagate and kill the autotuning run. Fix: also catch
   these errors and return [inf, inf, inf] so the config is skipped.

2. CompiledKernel._init_handles load_binary: CUDA error 800 "operation not
   permitted" from `driver.active.utils.load_binary(...)` when a new input
   shape forces a fresh kernel binary load. Root cause: cuModuleLoad family
   calls are disallowed inside a CUDA stream capture region. Long-running
   training accumulates new shape variants over time; eventually one tries
   to load while a stream is captured (autograd backward, inductor cudagraphs,
   etc.), triggering CUDA_ERROR_NOT_PERMITTED. The crash is intermittent
   because it requires the unlikely coincidence of (a) a never-before-seen
   kernel variant + (b) an active capture region.

   Fix: retry load_binary up to 3 times with synchronize() + empty_cache()
   between attempts. The synchronize forces the capture region to end (or
   the stream to drain), letting the next attempt load cleanly outside any
   capture. Combined with TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1 in the
   environment, this should eliminate the failure mode.

See: https://github.com/pytorch/pytorch/issues/87794 (CUDA error 800 + capture)
See: https://github.com/fla-org/flash-linear-attention/issues/609 (TMA on sm_121)
See: https://github.com/fla-org/flash-linear-attention/issues/638 (GatedDeltaNet bwd)
See: https://github.com/fla-org/flash-linear-attention/issues/734
"""

import logging
import time

logger = logging.getLogger(__name__)

_patched = False


def patch_triton_autotuner():
    """Patch Triton autotuner _bench and CompiledKernel._init_handles for sm_121."""
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
        logger.debug("Not sm_121 (got sm_%d%d), skipping Triton patches", major, minor)
        return

    # --- Patch 1: autotuner _bench ---
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
            if "Triton Error" in err and (
                "invalid argument" in err or "operation not permitted" in err
            ):
                kernel_name = getattr(self, "base_fn", self.fn).__name__
                logger.warning(
                    "sm_121 autotuner: config %s failed for %s — %s",
                    config, kernel_name, err[:200],
                )
                return [float("inf"), float("inf"), float("inf")]
            raise

    Autotuner._bench = _safe_bench

    # --- Patch 2: CompiledKernel._init_handles retry on load_binary errors ---
    try:
        from triton.compiler.compiler import CompiledKernel
    except ImportError:
        logger.warning("Could not import CompiledKernel — _init_handles patch skipped")
        logger.info("Triton autotuner patched for sm_121 (init_handles patch unavailable)")
        return

    original_init_handles = CompiledKernel._init_handles

    def _safe_init_handles(self):
        max_attempts = 3
        last_err = None
        for attempt in range(max_attempts):
            try:
                return original_init_handles(self)
            except RuntimeError as e:
                err = str(e)
                if "Triton Error" in err and "operation not permitted" in err:
                    last_err = e
                    kernel_name = getattr(self, "name", "<unknown>")
                    logger.warning(
                        "sm_121 init_handles: load_binary failed for %s "
                        "(attempt %d/%d) — %s",
                        kernel_name, attempt + 1, max_attempts, err[:200],
                    )
                    # Reset state so retry actually re-runs load_binary instead
                    # of returning the cached None module from the early-exit guard.
                    self.module = None
                    # Try to free CUDA resources before retry
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    time.sleep(0.5 * (attempt + 1))  # backoff
                    continue
                raise
        # All retries exhausted
        logger.error(
            "sm_121 init_handles: load_binary failed after %d attempts, giving up",
            max_attempts,
        )
        raise last_err  # type: ignore[misc]

    CompiledKernel._init_handles = _safe_init_handles

    logger.info(
        "Triton patched for sm_121: autotuner _bench tolerates CUDA launch errors, "
        "_init_handles retries on 'operation not permitted'"
    )
