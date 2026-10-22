"""Unit tests for ``BridgeDistillTrainer``: curriculum, mask construction,
freeze plan, and a tiny train_step over an SDPA-mocked stub encoder."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from omegaconf import OmegaConf
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.bridge_distill import (
    BridgeDistillTrainer,
    _build_l0_and_l1_forced_masks,
)

HIDDEN_DIM = 16


# --------------------------- mocks ---------------------------


class _StubBackbone(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        cu_seqlens=None,
        max_seqlen=None,
        position_ids=None,
        layer_hooks=None,
        **_kw,
    ):
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if layer_hooks and i in layer_hooks:
                h = layer_hooks[i](h)
        h = self.norm(h)
        return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=None)


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        head_dim = 4
        n_heads = hidden_dim // head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5


class _MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class _MockRotaryEmb(nn.Module):
    HEAD_DIM = 4

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        n = position_ids.shape[-1]
        cos = torch.ones(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        return cos, sin


def _make_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone_l0 = _StubBackbone(hidden_dim=hidden_dim)
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()

    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=True,
        with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5, "init_theta": -1.5},
        head_layer_index=1,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=False,
        with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5, "init_theta": -1.5},
        head_layer_index=1,
    )
    proj_layer = _MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = _MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)
    return BgKITEncoder(l0, l1, projection_block)


@pytest.fixture(autouse=True)
def _patch_packed_attention(monkeypatch):
    """Monkey-patch FA4 packed-attention to a CPU SDPA path so the projection
    block runs without a GPU. Mirrors test_decoder_init_projection's fixture.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    from bgkit.models import projection_block as pb

    def _sdpa_packed(self_attn, hidden, position_embeddings,
                     cu_seqlens, max_seqlen, position_ids=None,
                     is_causal=False):
        n = hidden.shape[0]
        if n == 0:
            return hidden
        head_dim = self_attn.head_dim
        # Qwen3.5 q_proj outputs 2*hidden (q + gate split). Match the existing fixture.
        n_heads = self_attn.q_proj.out_features // (head_dim * 2)
        n_kv_heads = self_attn.k_proj.out_features // head_dim
        q_gate = self_attn.q_proj(hidden)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self_attn.k_proj(hidden)
        v = self_attn.v_proj(hidden)
        q = q.reshape(n, n_heads, head_dim)
        k = k.reshape(n, n_kv_heads, head_dim)
        v = v.reshape(n, n_kv_heads, head_dim)
        q = self_attn.q_norm(q)
        k = self_attn.k_norm(k)

        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        cos, sin = position_embeddings
        q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
        q = q4.squeeze(0).transpose(0, 1).contiguous()
        k = k4.squeeze(0).transpose(0, 1).contiguous()
        v = v.contiguous()

        if n_kv_heads < n_heads:
            repeat = n_heads // n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        cu = cu_seqlens.tolist()
        outs: list[torch.Tensor] = []
        for b in range(len(cu) - 1):
            start, end = int(cu[b]), int(cu[b + 1])
            if end == start:
                continue
            qb = q[start:end].transpose(0, 1).unsqueeze(0)
            kb = k[start:end].transpose(0, 1).unsqueeze(0)
            vb = v[start:end].transpose(0, 1).unsqueeze(0)
            ob = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=None, is_causal=is_causal,
                scale=self_attn.scaling,
            )
            outs.append(ob.squeeze(0).transpose(0, 1))
        attn_out = torch.cat(outs, dim=0)
        attn_out = attn_out.reshape(n, n_heads * head_dim).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        return self_attn.o_proj(attn_out)

    monkeypatch.setattr(pb, "_packed_full_attention", _sdpa_packed)


def _make_trainer_skeleton() -> BridgeDistillTrainer:
    """Construct a BridgeDistillTrainer skeleton bypassing setup().

    Manually populates the attributes setup() would set, with mocked encoders
    and dataset-free state. Suitable for testing forward/backward, freeze
    plan, curriculum, and mask construction.
    """
    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step4p7",
            "max_steps": 10,
            "lr": 1e-3,
            "warmup_steps": 1,
            "eval_every": 0,
            "save_every": 0,
            "optimizer": "adamw",
        },
        "wandb": {"enabled": False},
        "seed": 42,
    })
    t = BridgeDistillTrainer(cfg)
    t.device = torch.device("cpu")
    t._teacher_ratio = 0.30  # 3/10 will survive
    t._min_per_sample = 0
    t._mse_weight = 1.0
    t._cos_weight = 1.0
    t._curriculum_steps = 100
    t._frac_extras_start = 1.0
    t._frac_extras_end = 0.31
    t._path_a_prob = 0.5
    t._unfreeze_l0_last_blocks = 1
    t._unfreeze_l1_first_blocks = 2
    t._unfreeze_bridge = True
    t._unfreeze_projection_block = True
    t._unfreeze_l0_norm = True
    t._unfreeze_l1_norm = True

    t._sampling_rng = torch.Generator(device=torch.device("cpu"))
    t._sampling_rng.manual_seed(0)

    t.encoder_teacher = _make_encoder()
    t.encoder_teacher.requires_grad_(False)
    t.encoder_teacher.eval()

    t.encoder_student = _make_encoder()
    # Mirror teacher weights exactly so Path A starts at parity (cos~1).
    t.encoder_student.load_state_dict(t.encoder_teacher.state_dict())
    t._freeze_for_bridge_distill(t.encoder_student)
    t.model = t.encoder_student
    t._decoder_state_dict = None
    t._accum_steps = 1
    return t


def _make_packed_batch(B: int, lengths: list[int], prompt_len: int = 2) -> dict:
    """Build a minimal commit-encoding-style batch keyed for the trainer's
    ``_content_inputs`` consumer. Single file per repo (cu_repo not used here)."""
    n = sum(lengths)
    content_ids = torch.randint(0, 60, (n,), dtype=torch.long)
    cu_file = torch.zeros(B + 1, dtype=torch.int32)
    cu_file[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    prompt_ids = torch.randint(0, 60, (B * prompt_len,), dtype=torch.long)
    prompt_cu = torch.tensor(
        [i * prompt_len for i in range(B + 1)], dtype=torch.int32,
    )
    return {
        "content_token_ids": content_ids,
        "cu_file_seqlens": cu_file,
        "prompt_token_ids": prompt_ids,
        "prompt_cu_seqlens": prompt_cu,
    }


# ----------------------------- tests -----------------------------


def test_curriculum_frac_extras_ramp():
    t = _make_trainer_skeleton()
    t._curriculum_steps = 100
    t._frac_extras_start = 1.0
    t._frac_extras_end = 0.5

    t.global_step = 0
    assert t._current_frac_extras() == pytest.approx(1.0)
    t.global_step = 50
    assert t._current_frac_extras() == pytest.approx(0.75)
    t.global_step = 100
    assert t._current_frac_extras() == pytest.approx(0.5)
    t.global_step = 200
    assert t._current_frac_extras() == pytest.approx(0.5)


def test_current_ratios_endpoint_balances_to_50_50():
    t = _make_trainer_skeleton()
    t._teacher_ratio = 0.20
    # Endpoint frac_extras=0.31 => r_l0 = 0.20 + 0.31*0.80 = 0.448 (~0.447 spec)
    r_l0, r_l1 = t._current_ratios(0.31)
    assert abs(r_l0 - 0.448) < 0.01
    # r_l1 = teacher_ratio / r_l0 — also ~0.447
    assert abs(r_l1 - 0.20 / r_l0) < 1e-6


def test_l0_l1_mask_construction_respects_per_sample_boundaries():
    teacher_mask = torch.tensor(
        [True, False, False, True, False, False,    # sample 0: keep [0, 3]
         True, False, False, False],                # sample 1: keep [6]
        dtype=torch.bool,
    )
    cu_file = torch.tensor([0, 6, 10], dtype=torch.int32)
    rng = torch.Generator(device="cpu").manual_seed(0)

    # frac=1.0 -> all doomed pass at L0; L1 mask reduces back to teacher mask.
    l0_mask, l1_mask = _build_l0_and_l1_forced_masks(teacher_mask, cu_file, 1.0, rng)
    assert l0_mask.all().item()
    # L1 mask is teacher_mask reindexed onto L0's reduced output (which equals
    # the full content under frac=1.0). So L1 mask == teacher_mask.
    assert l1_mask.equal(teacher_mask)

    # frac=0.0 -> no doomed pass at L0; L0 mask == teacher_mask.
    rng.manual_seed(0)
    l0_mask, l1_mask = _build_l0_and_l1_forced_masks(teacher_mask, cu_file, 0.0, rng)
    assert l0_mask.equal(teacher_mask)
    # L1 mask must be all-True over the reduced output (only teacher positions
    # made it through).
    assert l1_mask.shape == (int(teacher_mask.sum().item()),)
    assert l1_mask.all().item()


def test_l0_l1_mask_doomed_extras_stay_within_sample():
    teacher_mask = torch.zeros(20, dtype=torch.bool)
    teacher_mask[[0, 12]] = True
    cu_file = torch.tensor([0, 10, 20], dtype=torch.int32)
    rng = torch.Generator(device="cpu").manual_seed(7)

    l0_mask, _ = _build_l0_and_l1_forced_masks(teacher_mask, cu_file, 0.5, rng)
    # Must respect per-sample boundaries: sample 0 doomed = 9, sample 1 doomed = 9.
    # 0.5 * 9 = 4.5 -> ceil = 5 extras per sample; plus 1 teacher each -> 6 each.
    assert int(l0_mask[:10].sum().item()) == 6
    assert int(l0_mask[10:].sum().item()) == 6
    # Teacher positions must remain set.
    assert l0_mask[0].item() and l0_mask[12].item()


def test_freeze_plan_only_unfrozen_have_requires_grad():
    t = _make_trainer_skeleton()
    enc = t.encoder_student
    # Bridge / norms / projection / last L0 block / first 2 L1 blocks: trainable.
    assert any(p.requires_grad for p in enc.l0.auto_repro_head.parameters())
    assert any(p.requires_grad for p in enc.l0.norm.parameters())
    assert any(p.requires_grad for p in enc.l1.norm.parameters())
    assert any(p.requires_grad for p in enc.projection_block.parameters())
    assert any(p.requires_grad for p in list(enc.l0.backbone.layers)[-1].parameters())
    for layer in list(enc.l1.backbone.layers)[:2]:
        assert any(p.requires_grad for p in layer.parameters())

    # Heads / survive_embedding / prompt_separator / first L0 block: frozen.
    assert all(not p.requires_grad for p in enc.l0.head.parameters())
    assert all(not p.requires_grad for p in enc.l1.head.parameters())
    assert not enc.l0.survive_embedding.requires_grad
    assert not enc.l1.survive_embedding.requires_grad
    assert not enc.l0.prompt_separator_embedding.requires_grad
    assert all(
        not p.requires_grad
        for p in list(enc.l0.backbone.layers)[0].parameters()
    )
    # Last L1 block frozen.
    assert all(
        not p.requires_grad
        for p in list(enc.l1.backbone.layers)[-1].parameters()
    )


def test_path_a_loss_near_zero_when_student_equals_teacher():
    """Path A: student copies teacher weights -> proj outputs match -> loss ~ 0."""
    torch.manual_seed(0)
    t = _make_trainer_skeleton()
    batch = _make_packed_batch(B=2, lengths=[8, 6])
    ctx = t._content_inputs(t._move_batch(batch))
    mask_t, proj_t = t._teacher_forward(ctx)
    if int(mask_t.sum().item()) == 0:
        pytest.skip("teacher mask empty under random init; flaky seed")

    # Force Path A regardless of RNG.
    t._path_a_prob = 1.0
    t._sampling_rng.manual_seed(123)

    student_out = t._student_forward_path_a(ctx, mask_t)
    proj_s = student_out.survivor_embeddings
    assert proj_s.shape == proj_t.shape
    total, stats = t._distill_loss(proj_s, proj_t)
    # Cosine should be ~1, MSE ~0 (modulo floating-point).
    assert float(stats["diag/cos_sim"].item()) > 0.999
    assert float(stats["loss/mse"].item()) < 1e-6


def test_train_step_decreases_loss_and_unfrozen_get_grads():
    torch.manual_seed(1)
    t = _make_trainer_skeleton()
    # Diverge student from teacher so loss > 0 at step 0.
    with torch.no_grad():
        for p in t.encoder_student.l0.auto_repro_head.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in t.encoder_student.projection_block.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    # Force Path A only — Path B is high-variance under random init for the
    # tiny stub, masking convergence under a short-step budget.
    t._path_a_prob = 1.0

    opt = torch.optim.AdamW(t.trainable_parameters(), lr=1e-3)
    t.optimizer = opt

    # Use a single fixed batch so loss curve is deterministic across steps.
    batch = _make_packed_batch(B=2, lengths=[8, 6])
    losses: list[float] = []
    n_steps = 20
    for step in range(n_steps):
        t.global_step = step
        opt.zero_grad()
        metrics = t._forward_backward(batch)
        if metrics.get("skipped_empty_teacher", 0.0):
            pytest.skip("teacher mask empty under threshold init; flaky seed")
        losses.append(metrics["loss"])
        opt.step()

    # Compare loss windows to dampen single-step noise.
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early, (
        f"loss did not decrease: early-mean={early:.4f}, late-mean={late:.4f}; "
        f"trace={['%.4f' % v for v in losses]}"
    )

    # Path A doesn't engage the bridge (auto_repro_head). After Path A only,
    # projection_block + last L0 block params should have grads; the bridge
    # is exercised in the next sub-test (Path B).
    enc = t.encoder_student
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in enc.projection_block.parameters() if p.requires_grad
    )

    # Frozen components have NO grad regardless of path.
    for p in enc.l0.head.parameters():
        assert p.grad is None
    assert enc.l0.survive_embedding.grad is None
    first_l0 = list(enc.l0.backbone.layers)[0]
    for p in first_l0.parameters():
        assert p.grad is None


def test_path_b_routes_grad_through_bridge():
    """Path B (L0->bridge->L1) should populate grads on auto_repro_head and
    L1's first blocks."""
    torch.manual_seed(2)
    t = _make_trainer_skeleton()
    t._path_a_prob = 0.0  # always Path B
    t.global_step = 0  # frac_extras = 1.0
    opt = torch.optim.AdamW(t.trainable_parameters(), lr=1e-3)
    t.optimizer = opt

    batch = _make_packed_batch(B=2, lengths=[8, 6])
    metrics = t._forward_backward(batch)
    if metrics.get("skipped_empty_teacher", 0.0):
        pytest.skip("teacher mask empty")

    enc = t.encoder_student
    bridge_grads = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in enc.l0.auto_repro_head.parameters()
    )
    assert bridge_grads, "auto_repro_head should receive grad on Path B"

    l1_first = list(enc.l1.backbone.layers)[0]
    assert any(p.grad is not None for p in l1_first.parameters())


def test_path_b_runs_through_l1():
    torch.manual_seed(0)
    t = _make_trainer_skeleton()
    t._path_a_prob = 0.0  # always Path B
    t._sampling_rng.manual_seed(42)
    t.global_step = 0  # frac_extras=1.0 -> L0 keeps everyone

    batch = _make_packed_batch(B=2, lengths=[8, 6])
    ctx = t._content_inputs(t._move_batch(batch))
    mask_t, proj_t = t._teacher_forward(ctx)
    if int(mask_t.sum().item()) == 0:
        pytest.skip("teacher mask empty")

    student_out = t._student_forward_path_b(ctx, mask_t, frac_extras=1.0)
    # L1 must have run.
    assert student_out.l1 is not None
    # Final survivor count = teacher survivor count (forced by L1 mask).
    assert student_out.survivor_embeddings.shape[0] == int(mask_t.sum().item())
    assert student_out.survivor_embeddings.shape == proj_t.shape


def test_evaluate_reports_per_path_metrics(monkeypatch):
    torch.manual_seed(0)
    t = _make_trainer_skeleton()

    # Single-batch dataloader stub.
    batches = [_make_packed_batch(B=2, lengths=[8, 6])]
    t.eval_dataloader = batches
    t._memory_legacy_warned = False  # noop, just to be safe

    out = t.evaluate()
    assert "loss_path_A" in out
    assert "loss_path_B" in out
    assert "cosine_path_A" in out
    assert "cosine_path_B" in out
    assert "loss" in out
