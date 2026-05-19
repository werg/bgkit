"""Optional adapter for Luce's Qwen3.5 megakernel.

This backend adapts Luce's B=1 Qwen3.5 megakernel to BgKIT's spliced-embedding
decoder contract. The current fork exposes generation prefill plus an all-token
final-hidden prefill surface for training-forward parity; backward kernels are
still under active development.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from bgkit.models.decoder import GenerationOutput

PACKAGE_NAME = "luce_megakernel"
EXTENSION_NAME = "qwen35_megakernel_bf16_C"


@dataclass(frozen=True)
class LuceMegakernelStatus:
    """Import/build status for the optional Luce megakernel backend."""

    source_mounted: bool
    cache_present: bool
    package_importable: bool
    extension_importable: bool
    embedding_prefill_available: bool
    hidden_prefill_available: bool
    cuda_available: bool
    capability: tuple[int, int] | None
    backend: str | None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return (
            self.package_importable
            and self.extension_importable
            and self.embedding_prefill_available
            and self.hidden_prefill_available
            and self.cuda_available
        )


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir:
        roots.append(Path(checkpoint_dir) / ".luce-megakernel-native")
    roots.append(Path("/workspace/checkpoints/.luce-megakernel-native"))
    roots.append(Path("/workspace"))
    return roots


def ensure_luce_megakernel_path() -> None:
    """Add known container source/cache roots to ``sys.path`` when present."""

    for root in _candidate_roots():
        package_dir = root / PACKAGE_NAME
        if not package_dir.is_dir():
            continue
        # The Luce Python driver is imported as a package from ``root`` while
        # its compiled extension is emitted directly inside ``package_dir``.
        for path in (package_dir, root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))


def _cuda_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return int(major), int(minor)


def _select_backend(backend: str = "auto") -> str:
    normalized = backend.strip().lower()
    if normalized not in {"auto", "bf16", "nvfp4"}:
        raise ValueError("backend must be one of: auto, bf16, nvfp4")
    if normalized != "auto":
        return normalized
    capability = _cuda_capability()
    if capability is not None and capability[0] >= 12:
        return "nvfp4"
    return "bf16"


def import_luce_megakernel_module(backend: str = "auto") -> ModuleType:
    """Import the Luce Python driver module for the requested backend."""

    ensure_luce_megakernel_path()
    selected = _select_backend(backend)
    module_name = "model_nvfp4" if selected == "nvfp4" else "model"
    try:
        return importlib.import_module(f"{PACKAGE_NAME}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name not in {PACKAGE_NAME, f"{PACKAGE_NAME}.{module_name}"}:
            raise
        raise RuntimeError(
            "Luce megakernel package is not importable. Enable "
            "BGKIT_BOOTSTRAP_LUCE_MEGAKERNEL=1 in the Docker service so "
            "/workspace/luce_megakernel is copied and built into the checkpoint cache."
        ) from exc


def weights_from_hf_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    backend: str = "auto",
    verbose: bool = True,
    nvfp4_group_size: int | None = None,
) -> dict[str, Any]:
    """Pack an already-loaded Qwen3.5 HF/BgKIT state dict for Luce runtime."""

    module = import_luce_megakernel_module(backend)
    selected = _select_backend(backend)
    if selected != "nvfp4":
        raise RuntimeError("BgKIT Luce state-dict packing currently targets the NVFP4 backend")
    kwargs: dict[str, Any] = {
        "verbose": verbose,
        "backend": selected,
    }
    if nvfp4_group_size is not None:
        kwargs["nvfp4_group_size"] = int(nvfp4_group_size)
    try:
        return module.weights_from_state_dict(state_dict, **kwargs)
    except AttributeError as exc:
        raise RuntimeError(
            "The mounted Luce megakernel fork does not expose weights_from_state_dict; "
            "rebuild/update /home/werg/lucebox-hub on branch bgkit-sm121-adapter."
        ) from exc


def load_decoder(
    *,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    backend: str = "auto",
    verbose: bool = True,
    max_seq_len: int | None = None,
    **kwargs: Any,
) -> Any:
    """Construct Luce's stateful B=1 decoder.

    The returned object is the upstream ``Decoder`` instance. It supports token
    prompt generation through ``generate(prompt, max_tokens=...)`` and, on the
    NVFP4 path, ``prefill_tokens`` / ``step_many``. It is not an ``nn.Module``.
    """

    module = import_luce_megakernel_module(backend)
    decoder_kwargs = dict(model_name=model_name, verbose=verbose)
    if max_seq_len is not None:
        decoder_kwargs["max_seq_len"] = int(max_seq_len)
    decoder_kwargs.update(kwargs)
    return module.Decoder(**decoder_kwargs)


def load_decoder_from_hf_model(
    model: torch.nn.Module,
    *,
    tokenizer: Any = None,
    backend: str = "auto",
    verbose: bool = True,
    nvfp4_group_size: int | None = None,
) -> Any:
    """Construct a Luce decoder from an existing HF/BgKIT Qwen3.5 model."""

    module = import_luce_megakernel_module(backend)
    weights = weights_from_hf_state_dict(
        model.state_dict(),
        backend=backend,
        verbose=verbose,
        nvfp4_group_size=nvfp4_group_size,
    )
    kwargs: dict[str, Any] = {
        "weights": weights,
        "tokenizer": tokenizer,
        "backend": _select_backend(backend),
        "verbose": verbose,
    }
    if nvfp4_group_size is not None:
        kwargs["nvfp4_group_size"] = int(nvfp4_group_size)
    return module.Decoder(**kwargs)


@dataclass
class LuceSingleSpliceGenerator:
    """BgKIT-compatible B=1/B-loop generation adapter around Luce Decoder."""

    decoder: Any
    tokenizer: Any

    @property
    def device(self) -> torch.device:
        return self.decoder._embed_weight.device

    def _embed_prefix(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        ids = prefix_ids.to(device=self.device, dtype=torch.long)
        return self.decoder._embed_weight.index_select(0, ids).to(torch.bfloat16)

    def generate_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Generate from ``[prefix token embeddings | survivor embeddings]``.

        This matches ``ReconstructionDecoder.generate_with_single_splice`` for
        the greedy B-loop inference case. Nonzero-temperature sampling requires
        a future logits-returning Luce decode op and is intentionally rejected.
        """

        if temperature != 0.0:
            raise ValueError(
                "Luce megakernel splice generation currently supports greedy decoding only"
            )
        if not hasattr(self.decoder, "prefill_embeddings"):
            raise RuntimeError(
                "The mounted Luce megakernel fork does not expose prefill_embeddings; "
                "rebuild/update /home/werg/lucebox-hub on branch bgkit-sm121-adapter."
            )

        batch_size = int(survivor_cu_seqlens.shape[0]) - 1
        surv_cu = survivor_cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
        suffix_ids_dev = suffix_ids.to(device=self.device, dtype=torch.long)
        suf_len = int(suffix_ids_dev.shape[0])
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        prefix_emb = self._embed_prefix(prefix_ids)

        content_ids_list: list[torch.Tensor] = []
        full_ids_list: list[torch.Tensor] = []
        survivors = survivor_embeddings.to(device=self.device, dtype=torch.bfloat16)

        for b in range(batch_size):
            surv = survivors[surv_cu[b] : surv_cu[b + 1]]
            prefill_emb = torch.cat([prefix_emb, surv], dim=0).contiguous()
            first_id = int(self.decoder.prefill_embeddings(prefill_emb))
            generated: list[int] = [first_id]
            stopped = eos_id is not None and first_id == eos_id
            cur_id = first_id
            for _ in range(1, max_new_tokens):
                if stopped:
                    break
                cur_id = int(self.decoder.step(cur_id))
                generated.append(cur_id)
                if eos_id is not None and cur_id == eos_id:
                    stopped = True

            gen_ids = torch.tensor(generated, dtype=torch.long, device=self.device)
            full_ids_list.append(gen_ids)
            seq = gen_ids
            if eos_id is not None:
                while seq.shape[0] > 0 and int(seq[-1].item()) == eos_id:
                    seq = seq[:-1]
            if pad_id is not None:
                while seq.shape[0] > 0 and int(seq[-1].item()) == pad_id:
                    seq = seq[:-1]
            if suf_len > 0 and seq.shape[0] >= suf_len and seq[-suf_len:].equal(suffix_ids_dev):
                seq = seq[:-suf_len]
            content_ids_list.append(seq)

        content_text = [
            self.tokenizer.decode(ids.detach().cpu(), skip_special_tokens=True)
            for ids in content_ids_list
        ]
        return GenerationOutput(
            content_ids=content_ids_list,
            content_text=content_text,
            full_ids=full_ids_list,
        )


@dataclass
class LuceSpliceForwardOutput:
    """Forward output from Luce's spliced-embedding hidden prefill."""

    hidden_states: torch.Tensor
    token_ids: torch.Tensor
    loss_mask: torch.Tensor


@dataclass
class LuceSingleSpliceForward:
    """B=1 forward adapter exposing Luce final hidden states for CE work."""

    decoder: Any

    @property
    def device(self) -> torch.device:
        return self.decoder._embed_weight.device

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        ids_dev = ids.to(device=self.device, dtype=torch.long)
        return self.decoder._embed_weight.index_select(0, ids_dev).to(torch.bfloat16)

    def forward_with_single_splice_hidden(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> LuceSpliceForwardOutput:
        """Return final hidden states for ``[prefix | survivors | suffix]``.

        This mirrors the B=1 splice layout used by
        ``ReconstructionDecoder.forward_with_single_splice``. It deliberately
        stops before CE so we can compare and then replace BgKIT's loss/backward
        path one component at a time.
        """

        if not hasattr(self.decoder, "prefill_embeddings_hidden"):
            raise RuntimeError(
                "The mounted Luce megakernel fork does not expose "
                "prefill_embeddings_hidden; rebuild/update /home/werg/lucebox-hub "
                "on branch bgkit-sm121-adapter."
            )
        batch_size = int(survivor_cu_seqlens.shape[0]) - 1
        if batch_size != 1:
            raise ValueError("Luce hidden splice forward currently supports B=1")

        prefix_dev = prefix_ids.to(device=self.device, dtype=torch.long)
        suffix_dev = suffix_ids.to(device=self.device, dtype=torch.long)
        surv_cu = survivor_cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
        survivors = survivor_embeddings.to(device=self.device, dtype=torch.bfloat16)
        survivor = survivors[surv_cu[0] : surv_cu[1]]

        input_embeds = torch.cat(
            [self._embed_ids(prefix_dev), survivor, self._embed_ids(suffix_dev)],
            dim=0,
        ).contiguous()
        hidden = self.decoder.prefill_embeddings_hidden(input_embeds)
        token_ids = torch.cat(
            [
                prefix_dev,
                torch.zeros(int(survivor.shape[0]), dtype=torch.long, device=self.device),
                suffix_dev,
            ],
            dim=0,
        )
        if loss_mask is None:
            mask = torch.cat(
                [
                    torch.zeros(int(prefix_dev.shape[0]) + int(survivor.shape[0]),
                                dtype=torch.bool, device=self.device),
                    torch.ones(int(suffix_dev.shape[0]), dtype=torch.bool, device=self.device),
                ],
                dim=0,
            )
        else:
            mask = loss_mask.to(device=self.device, dtype=torch.bool)
        return LuceSpliceForwardOutput(
            hidden_states=hidden.unsqueeze(0),
            token_ids=token_ids.unsqueeze(0),
            loss_mask=mask.unsqueeze(0),
        )


def status(backend: str = "auto") -> LuceMegakernelStatus:
    """Return a non-throwing status snapshot for smoke tests and diagnostics."""

    ensure_luce_megakernel_path()
    source_mounted = Path("/workspace/luce_megakernel/setup.py").is_file()
    cache_present = any((root / PACKAGE_NAME / "setup.py").is_file() for root in _candidate_roots())
    capability = _cuda_capability()
    selected: str | None = None
    package_importable = False
    extension_importable = False
    embedding_prefill_available = False
    hidden_prefill_available = False
    error: str | None = None
    try:
        selected = _select_backend(backend)
        module = import_luce_megakernel_module(selected)
        package_importable = True
        embedding_prefill_available = hasattr(module.Decoder, "prefill_embeddings")
        hidden_prefill_available = hasattr(module.Decoder, "prefill_embeddings_hidden")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        importlib.import_module(EXTENSION_NAME)
        extension_importable = True
    except Exception as exc:
        if error is None:
            error = f"{type(exc).__name__}: {exc}"

    return LuceMegakernelStatus(
        source_mounted=source_mounted,
        cache_present=cache_present,
        package_importable=package_importable,
        extension_importable=extension_importable,
        embedding_prefill_available=embedding_prefill_available,
        hidden_prefill_available=hidden_prefill_available,
        cuda_available=torch.cuda.is_available(),
        capability=capability,
        backend=selected,
        error=error,
    )


def supports_spliced_embedding_prefill() -> bool:
    """Whether this backend can consume BgKIT survivor embeddings directly."""

    return True


def supports_spliced_hidden_prefill() -> bool:
    """Whether this backend can expose hidden states for spliced CE forward."""

    return True
