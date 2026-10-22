"""Invariants every trainer must satisfy, checked by DISCOVERY not by list.

WHY DISCOVERY. A hand-written list of trainers found four carrying the
2026-06-10 NVMe routing bug. Walking the tree found ELEVEN — projection_seed_falcon,
projection_seed_falcon_cached, qwen_projection_warmup, bridge_distill,
l1l1_distill, joint_block_trainer and pruning_distill were simply not on the
list anyone remembered to update. A rule meant to hold everywhere cannot be
checked against an enumeration.

WHY THESE INVARIANTS. Each one was violated in production and cost real time:

  * module-level save_checkpoint()   11 trainers wrote to the spinning HDD and
                                     skipped async archival; the fix existed in
                                     one trainer for two months and could not
                                     propagate because each owned a copy.
  * undeclared rep splicing          summarization ran 52k steps splicing reps
                                     into a decoder with no rep-dependence
                                     metric; a downstream "regression" was then
                                     diagnosed for a day against a baseline
                                     nobody had measured.
  * duplicated helpers               three copies of rescale-to-embed-norm, so
                                     a representation contract could drift
                                     between call sites.

New trainers inherit these checks automatically. That is the point: the cost of
diverging is paid at CI time by the person diverging, not months later by
whoever is debugging.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[3] / "src" / "bgkit" / "training"
INFRA = {"base_trainer.py", "checkpointing.py", "checkpoint_manager.py",
         "checkpoint_registry.py", "checkpoint_archiver.py", "checkpoint_cli.py"}


def _trainer_files() -> list[Path]:
    return sorted(
        f for f in TRAINING_ROOT.rglob("*.py")
        if f.name not in INFRA and not f.name.startswith("__")
    )


def _ids(files: list[Path]) -> list[str]:
    return [str(f.relative_to(TRAINING_ROOT)) for f in files]


FILES = _trainer_files()
PARAMS = list(zip(FILES, _ids(FILES), strict=True))


@pytest.mark.parametrize(("path", "name"), PARAMS)
def test_no_module_level_save_checkpoint_call(path: Path, name: str) -> None:
    """Trainers must route writes through BaseTrainer._write_checkpoint.

    Calling the module-level save_checkpoint() directly skips NVMe fast-dir
    routing and async HDD archival. Eleven trainers did exactly this.
    """
    offenders = [
        node.lineno for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "save_checkpoint"
    ]
    assert not offenders, (
        f"{name} calls module-level save_checkpoint() at line(s) {offenders}; "
        "use self._write_checkpoint() so NVMe routing and archival apply"
    )


@pytest.mark.parametrize(("path", "name"), PARAMS)
def test_rep_splicing_trainers_declare_it(path: Path, name: str) -> None:
    """A trainer that hands reps to a decoder must declare SPLICES_REPS, so
    BaseTrainer can warn when its eval reports no rep-dependence metric."""
    src = path.read_text()
    splices = "forward_with_single_splice" in src or "EmbeddingSegment(" in src
    if not splices:
        return
    tree = ast.parse(src)
    declares = any(
        isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "SPLICES_REPS"
        or isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "SPLICES_REPS" for t in n.targets)
        for n in ast.walk(tree)
    )
    assert declares, (
        f"{name} splices reps into a decoder but does not declare "
        "SPLICES_REPS — its eval can then silently report no rep-dependence"
    )


def test_rescale_to_embed_norm_has_exactly_one_implementation() -> None:
    """A representation transform must not be reimplemented per caller, or the
    contract drifts between call sites. There were three copies."""
    root = TRAINING_ROOT.parents[0]
    impls = [
        f for f in root.rglob("*.py")
        if "def rescale_to_embed_norm" in f.read_text()
    ]
    assert len(impls) == 1, f"expected 1 implementation, found: {[f.name for f in impls]}"


def test_checkpoint_layout_is_declared_not_hand_written() -> None:
    """No trainer may override save_checkpoint; layout is declared via
    checkpoint_models() / checkpoint_extra_state() so metadata, NVMe routing
    and archival stay in one place."""
    offenders = []
    for f in _trainer_files():
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint":
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, (
        "override checkpoint_models()/checkpoint_extra_state() instead: "
        + ", ".join(offenders)
    )


def test_evaluate_template_exists_and_restores_mode_on_error() -> None:
    """The template must restore train mode even when a batch raises.

    Several hand-rolled overrides left models in eval mode on exception.
    Falcon-H1 takes a numerically different path in eval vs train (its .eval()
    Mixer path produced eval/loss 9.8 against train ~1), so a leaked mode
    silently changes later TRAINING, not just the failed eval.
    """
    import torch

    from bgkit.training.base_trainer import BaseTrainer

    class _Boom(BaseTrainer):
        def __init__(self) -> None:
            self.m = torch.nn.Linear(2, 2)
            self.m.train()
            self.eval_dataloader = [object()]

        def setup(self) -> None: ...
        def _forward_backward(self, batch): ...
        def evaluate(self) -> dict[str, float]:
            return {}
        def checkpoint_models(self):
            return {"m": self.m}
        def _eval_batch(self, batch):
            raise RuntimeError("batch exploded")

    t = _Boom()
    assert t.m.training is True
    with pytest.raises(RuntimeError, match="exploded"):
        t.evaluate_templated()
    assert t.m.training is True, "template leaked eval mode after an exception"


def test_evaluate_template_reduction_is_weighted() -> None:
    """Sums / weight, so unequal batches reduce correctly. Averaging per-batch
    means is a recurring source of quietly wrong eval numbers."""
    import torch

    from bgkit.training.base_trainer import BaseTrainer

    class _Weighted(BaseTrainer):
        def __init__(self) -> None:
            self.m = torch.nn.Linear(2, 2)
            self.eval_dataloader = [10.0, 90.0]     # sums
        def setup(self) -> None: ...
        def _forward_backward(self, batch): ...
        def evaluate(self) -> dict[str, float]:
            return {}
        def checkpoint_models(self):
            return {"m": self.m}
        def _eval_batch(self, batch):
            # batch is the loss SUM; weights differ (1 token vs 9 tokens)
            return {"loss": batch}, (1.0 if batch == 10.0 else 9.0)

    # token-weighted: (10 + 90) / (1 + 9) = 10.0   [mean-of-means would be 50.0]
    assert _Weighted().evaluate_templated()["loss"] == pytest.approx(10.0)
