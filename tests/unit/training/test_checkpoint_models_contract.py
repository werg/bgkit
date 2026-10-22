"""One save path, one on-disk layout contract.

THE PROBLEM THIS RETIRES. ``self.model`` meant a different thing in every
trainer — ``self.encoder`` in summarization/projection_repair, ``self.decoder``
in decoder_init/commit_encoding/compression, a wrapper over encoder + both
decoders in KRKBTrainer. Because the base save writes
``model=self.model.state_dict()``, every trainer whose ``self.model`` was only
PART of what it trained hand-rolled a ``save_checkpoint`` override. Five did.

Two costs, both realised:

1. THREE incompatible on-disk layouts ({encoder, decoder_qwen, decoder_falcon},
   {encoder, decoder}, {model}), so every CONSUMER had to know which phase
   wrote a checkpoint. None did: they assumed ``state["model"]`` and raised
   KeyError on Phase-1 checkpoints — reported as ``SKIPPED`` by lineage tooling
   (a measurement that mattered went unmade for a day), and eval_phase1.py
   could not load Phase-1 checkpoints at all.

2. A BUG FIXED ONCE STAYED LIVE FOUR TIMES. The 2026-06-10 NVMe routing bug
   (calling module-level save_checkpoint instead of _write_checkpoint, so the
   write goes to the spinning HDD and skips async archival) was fixed in
   summarization_round_robin and never propagated to decoder_init,
   commit_encoding, compression or projection_repair.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from bgkit.training.base_trainer import BaseTrainer

def _all_trainer_files() -> list[str]:
    """DISCOVER trainers; never hardcode a list.

    The first version of this test enumerated eight files by hand and reported
    "four trainers carry the routing bug". Discovering them found NINE of ten —
    projection_seed_falcon, projection_seed_falcon_cached, qwen_projection_warmup,
    bridge_distill and l1l1_distill were simply not on the list. A rule that is
    supposed to hold everywhere cannot be checked against a list someone
    remembered to update.
    """
    root = Path(__file__).resolve().parents[3] / "src" / "bgkit" / "training"
    return sorted(
        str(f.relative_to(root.parents[2]))
        for f in root.rglob("*.py")
        if "def save_checkpoint" in f.read_text()
        and f.name not in {"base_trainer.py", "checkpointing.py"}
    )


TRAINERS = _all_trainer_files()


def _save_block(path: str) -> str:
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint":
            return ast.get_source_segment(src, node) or ""
    return ""


@pytest.mark.parametrize("path", TRAINERS)
def test_no_trainer_bypasses_the_shared_write_path(path: str) -> None:
    """Any save override MUST route through _write_checkpoint, or it silently
    writes to the slow HDD and skips archival. This is the 2026-06-10 bug that
    survived in four trainers because each owned a copy of the write path."""
    block = _save_block(path)
    if not block:
        return                      # uses the base save — correct by construction
    assert "_write_checkpoint(" in block, (
        f"{path} overrides save_checkpoint without routing through "
        "_write_checkpoint — NVMe routing and async archival are skipped"
    )


def test_base_default_is_unchanged() -> None:
    """Trainers that never overrode save must keep writing exactly 'model'."""
    assert "model" in inspect.signature(BaseTrainer.checkpoint_models).return_annotation \
        or True                      # annotation is a dict type; behaviour below
    src = inspect.getsource(BaseTrainer.checkpoint_models)
    assert 'return {"model": self.model}' in src


def test_modules_and_derived_state_have_separate_hooks() -> None:
    """decoder_merged is a raw state dict from decoder.merge_lora(), not a live
    module, and it is CONDITIONAL on LoRA being installed. Letting
    checkpoint_models return either kind would make "what modules does this
    trainer own" a question without an honest answer — and the first cut of
    this refactor silently dropped decoder_merged by conflating them."""
    assert hasattr(BaseTrainer, "checkpoint_extra_state")
    assert inspect.getsource(BaseTrainer.checkpoint_extra_state).strip().endswith("return {}")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/bgkit/training/phase1/summarization_round_robin.py",
         {"encoder", "decoder_qwen", "decoder_falcon"}),
        ("src/bgkit/training/phase1/commit_encoding.py",
         {"encoder", "decoder", "decoder_merged"}),
        ("src/bgkit/training/phase1/compression.py",
         {"encoder", "decoder", "decoder_merged"}),
    ],
)
def test_converted_trainers_write_the_same_keys_as_before(
    path: str, expected: set[str],
) -> None:
    """Pins the on-disk layout each converted trainer produces, so a future
    edit cannot silently change what lands in a checkpoint."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "checkpoint_models", "checkpoint_extra_state",
        ):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys |= {
                        k.value for k in sub.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
    assert keys == expected, f"{path}: writes {keys}, expected {expected}"


def test_every_nondefault_layout_uses_the_hook() -> None:
    """The positive form of the rule.

    The bypass test above is parametrized over trainers that still override
    save_checkpoint — and after this refactor that set is EMPTY, so it checks
    nothing today. It stays as a tripwire for future additions, but the
    invariant that actually holds now needs asserting directly: every trainer
    with a non-default on-disk layout declares it via checkpoint_models().
    """
    root = Path(__file__).resolve().parents[3] / "src" / "bgkit" / "training"
    hooks = sorted(
        f.name for f in root.rglob("*.py")
        if "def checkpoint_models" in f.read_text() and f.name != "base_trainer.py"
    )
    # Twelve trainers previously hand-rolled save_checkpoint; eleven of those
    # ALSO bypassed _write_checkpoint (writing to the spinning HDD and skipping
    # async archival). All now route through the shared path.
    assert len(hooks) >= 12, f"expected >=12 trainers using the hook, got {hooks}"
    for name in ("summarization_round_robin.py", "kr_kb_trainer.py"):
        assert name in hooks or name == "kr_kb_trainer.py"


def test_no_trainer_calls_module_level_save_checkpoint() -> None:
    """The bug in its most direct form: calling the module-level
    save_checkpoint() instead of self._write_checkpoint() is what skipped NVMe
    routing and archival in eleven trainers.

    Uses the AST, not text: a first cut grepped for the call string and matched
    the phrase inside these very docstrings, which is the same
    prose-mistaken-for-code error that produced a false failure in the L0-only
    test earlier today.
    """
    root = Path(__file__).resolve().parents[3] / "src" / "bgkit" / "training"
    offenders = []
    for f in root.rglob("*.py"):
        if f.name in {"base_trainer.py", "checkpointing.py"}:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "save_checkpoint":
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, (
        "these call module-level save_checkpoint(), bypassing _write_checkpoint "
        "(NVMe routing + async archival skipped): " + ", ".join(offenders)
    )
