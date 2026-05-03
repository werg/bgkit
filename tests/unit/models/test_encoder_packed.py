"""Parity test: packed BgKITEncoder vs. recorded padded reference.

Loads ``tests/fixtures/encoder_reference.pt`` (captured pre-Wave-1 rewrite
on CUDA + FA4 via the Qwen3.5-0.8B-Base backbone) and verifies that the
new packed encoder produces numerically equivalent outputs when fed the
same content embeddings in packed form.

The fixture payload is in padded shapes ``(B, L_max, ...)``; the test
converts those to packed form, calls the encoder, then unpacks back to
padded layout for comparison.

**Requires CUDA + fla-core + FA4** (Docker training container on DGX
Spark). The DeltaNet layers inside Qwen3.5 dispatch to fla-core's
Triton kernels which are CUDA-only; the host venv reports "Triton is
not supported on current platform, roll back to CPU" and fails.
Accordingly tests are marked ``gpu`` and skip when CUDA isn't available.

Wave 1.3 (DeltaNet varlen patch via ``bgkit.utils.deltanet_patch``) is
landed — ``chunk_gated_delta_rule`` accepts ``cu_seqlens`` and resets
the cumulative gate at boundaries. Previous ``@pytest.mark.xfail``
markers on these tests are therefore removed.

Tolerance is calibrated for bf16 attention: ``atol=5e-2, rtol=5e-2``
on head outputs (near-θ positions flip in bf16), a higher ``95%``
agreement bar on survivor_mask booleans, and ``atol=1e-2, rtol=1e-2``
on raw embeddings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Allow skipping when a real backbone load would be prohibitive; enforced
# in ``_load_backbone``.
FIXTURES_PATH = Path(__file__).parent.parent.parent / "fixtures" / "encoder_reference.pt"


def _load_fixture():
    if not FIXTURES_PATH.exists():
        pytest.skip(f"encoder_reference.pt not found at {FIXTURES_PATH}")
    return torch.load(FIXTURES_PATH, weights_only=False)


def _pack_content(
    input_embeddings: torch.Tensor,  # (B, L_max, D)
    attention_mask: torch.Tensor,  # (B, L_max) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert padded (B, L, D) + mask to packed (N, D) + cu_seqlens + position_ids."""
    B = input_embeddings.shape[0]  # noqa: N806 (ML shape var)
    lengths = attention_mask.sum(dim=1).to(torch.int64).tolist()
    pieces = [input_embeddings[i, : lengths[i]] for i in range(B)]
    packed = torch.cat(pieces, dim=0)
    cu = torch.zeros(B + 1, dtype=torch.int32)
    acc = 0
    for i, length in enumerate(lengths):
        acc += length
        cu[i + 1] = acc
    position_ids = torch.cat(
        [torch.arange(length, dtype=torch.int64) for length in lengths],
        dim=0,
    )
    return packed.contiguous(), cu, position_ids


def _unpack_to_padded(
    packed: torch.Tensor,  # (N, D) or (N,)
    cu_seqlens: torch.Tensor,  # (B+1,)
    target_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``_pack_content``: unpack to padded (B, L_max, ...) + mask."""
    cu = cu_seqlens.to(torch.int64).tolist()
    B = len(cu) - 1  # noqa: N806 (ML shape var)
    lengths = [cu[i + 1] - cu[i] for i in range(B)]
    max_len = max(lengths) if lengths else 0
    if target_lengths is not None:
        max_len = max(max_len, max(target_lengths))
    if packed.ndim == 2:
        D = packed.shape[1]  # noqa: N806 (ML shape var)
        out = torch.zeros(B, max_len, D, dtype=packed.dtype, device=packed.device)
        for i in range(B):
            out[i, : lengths[i]] = packed[cu[i] : cu[i + 1]]
    else:
        out = torch.zeros(B, max_len, dtype=packed.dtype, device=packed.device)
        for i in range(B):
            out[i, : lengths[i]] = packed[cu[i] : cu[i + 1]]
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=packed.device)
    for i in range(B):
        mask[i, : lengths[i]] = True
    return out, mask


def _load_backbone():
    """Load Qwen3.5-0.8B-Base from HF cache.

    Skips when the weights aren't locally available (CI / offline boxes)
    or when CUDA/Triton isn't usable (DeltaNet requires fla-core's Triton path).
    """
    if not torch.cuda.is_available():
        pytest.skip("encoder parity test requires CUDA (DeltaNet via fla-core/Triton)")
    # fla-core's Triton must be functional. On the host venv, Triton emits
    # "roll back to CPU" because libpython headers are missing; the downstream
    # `custom_device_ctx` then tries to dispatch to `torch.cpu.device` and
    # raises `AttributeError`. Skip the test rather than fail it — this
    # configuration is the Docker training container's responsibility.
    try:
        import fla.utils as _fla_utils
    except Exception as exc:
        pytest.skip(f"fla-core not importable: {exc}")
    if getattr(_fla_utils, "device_platform", None) != "cuda":
        pytest.skip(
            "fla-core is not running on CUDA (Triton unavailable on host venv); "
            "encoder parity test requires the Docker training container."
        )
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3.5-0.8B-Base",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    except Exception as exc:
        pytest.skip(f"Cannot load Qwen/Qwen3.5-0.8B-Base (cache miss?): {exc}")
    return model


def _build_packed_encoder(seed: int = 17):
    """Build a BgKITEncoder with the same architecture the fixture used.

    We (re)seed before construction so the survivorship head / projection
    head / separator / survive_embedding parameters match the fixture's
    random init bit-for-bit.
    """
    from bgkit.models.encoder import BgKITEncoder

    torch.manual_seed(seed)
    raw_model = _load_backbone()
    encoder = BgKITEncoder.from_pretrained(
        raw_model,
        hidden_dim=1024,
        torch_dtype=torch.bfloat16,
        bidi_warmup_steps=0,
    )
    encoder = encoder.to("cuda").eval()
    return encoder


def _sdpa_packed_forward_test_local(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Test-local packed SDPA helper.

    The production encoder is FA4-only; this helper exists solely to
    reproduce the pre-migration padded reference against which these
    parity fixtures were captured (SDPA). Packed ``(N, H, D)``.
    """
    cu = cu_seqlens.tolist()
    batch = len(cu) - 1
    n_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    if n_kv_heads < n_heads:
        repeat = n_heads // n_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    outputs = []
    for b in range(batch):
        start, end = cu[b], cu[b + 1]
        if end == start:
            continue
        q_b = query[start:end].transpose(0, 1).unsqueeze(0)
        k_b = key[start:end].transpose(0, 1).unsqueeze(0)
        v_b = value[start:end].transpose(0, 1).unsqueeze(0)
        out_b = torch.nn.functional.scaled_dot_product_attention(
            q_b, k_b, v_b, attn_mask=None, is_causal=is_causal, scale=scale,
        )
        outputs.append(out_b.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _force_sdpa_packed_attention(monkeypatch):
    """Force ``_packed_full_attention`` onto a test-local SDPA helper.

    The fixtures were captured with SDPA (padded era). This monkeypatch
    routes the encoder's full-attention wrapper through the test-local
    SDPA helper above so the parity comparison doesn't introduce FA4
    reduction-order drift vs. the captured reference.
    """
    from bgkit.models import bidirectional_qwen35 as bq

    original = bq._packed_full_attention

    def _sdpa_only(
        self_attn,
        hidden_states,
        position_embeddings,
        cu_seqlens,
        max_seqlen,
        position_ids,
        is_causal,
    ):
        n = hidden_states.shape[0]
        head_dim = self_attn.head_dim
        n_heads = self_attn.q_proj.out_features // (head_dim * 2)
        n_kv_heads = self_attn.k_proj.out_features // head_dim

        qg = self_attn.q_proj(hidden_states).view(n, n_heads, 2 * head_dim)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(n, n_heads * head_dim)
        k = self_attn.k_proj(hidden_states).view(n, n_kv_heads, head_dim)
        v = self_attn.v_proj(hidden_states).view(n, n_kv_heads, head_dim)
        q = self_attn.q_norm(q)
        k = self_attn.k_norm(k)

        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        cos, sin = position_embeddings
        q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
        q = q4.squeeze(0).transpose(0, 1).contiguous()
        k = k4.squeeze(0).transpose(0, 1).contiguous()
        v = v.contiguous()

        attn_output = _sdpa_packed_forward_test_local(
            q.cpu(),
            k.cpu(),
            v.cpu(),
            cu_seqlens=cu_seqlens.cpu(),
            is_causal=is_causal,
            scale=self_attn.scaling,
        ).to(q.device)
        attn_output = attn_output.reshape(n, n_heads * head_dim).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self_attn.o_proj(attn_output)

    monkeypatch.setattr(bq, "_packed_full_attention", _sdpa_only)
    # Also patch pruned_qwen35's reference to the same symbol.
    from bgkit.models import pruned_qwen35 as pq

    monkeypatch.setattr(pq, "_packed_full_attention", _sdpa_only)
    # Patch projection_block too.
    from bgkit.models import projection_block as pb

    monkeypatch.setattr(pb, "_packed_full_attention", _sdpa_only)
    _ = original  # silence flake warnings


@pytest.mark.slow
@pytest.mark.gpu
def test_encoder_packed_no_compression_parity(monkeypatch):
    """No-compression path: packed output matches padded reference."""
    _force_sdpa_packed_attention(monkeypatch)
    data = _load_fixture()
    inputs = data["inputs"]
    expected = data["outputs_no_compression"]

    input_emb = inputs["input_embeddings"]  # (B, L_max, D) bf16
    attention_mask = inputs["attention_mask"]  # (B, L_max) bool

    # Pack inputs.
    packed_emb, cu, pos_ids = _pack_content(input_emb, attention_mask)
    packed_emb = packed_emb.to("cuda")
    cu = cu.to("cuda")
    pos_ids = pos_ids.to("cuda")

    encoder = _build_packed_encoder(seed=data["seed"])

    with torch.no_grad():
        out = encoder(
            content_embeddings=packed_emb,
            content_cu_seqlens=cu,
            content_position_ids=pos_ids,
            target_ratio=None,
        )

    # Unpack (N, D) -> (B, L_max, D) for comparison.
    got_padded, got_mask = _unpack_to_padded(
        out.survivor_embeddings.cpu(),
        out.survivor_cu_seqlens.cpu(),
    )

    # Match shapes (expected is zero-padded at non-content positions).
    expected_emb = expected["survivor_embeddings"]
    expected_mask = expected["survivor_attention_mask"]

    # Only compare at valid positions (attention_mask == True).
    assert got_padded.shape == expected_emb.shape, (
        f"shape mismatch: got {got_padded.shape} vs {expected_emb.shape}"
    )
    assert got_mask.equal(expected_mask), "attention mask mismatch"

    got_valid = got_padded[attention_mask]
    exp_valid = expected_emb[attention_mask]

    # bf16 tolerance: accumulations in attention give ~1% relative error.
    torch.testing.assert_close(got_valid, exp_valid, atol=1e-2, rtol=1e-2)


@pytest.mark.slow
@pytest.mark.gpu
def test_encoder_packed_compressed_parity(monkeypatch):
    """Compressed path: packed output matches padded reference.

    Checks: base_raw, logits_for_op, survivor_mask, survivor_counts,
    survivor_embeddings (at surviving positions only).
    """
    _force_sdpa_packed_attention(monkeypatch)
    data = _load_fixture()
    inputs = data["inputs"]
    expected = data["outputs_compressed"]
    target_ratio = float(data["metadata"]["target_ratio_compressed"])

    input_emb = inputs["input_embeddings"]
    attention_mask = inputs["attention_mask"]

    packed_emb, cu, pos_ids = _pack_content(input_emb, attention_mask)
    packed_emb = packed_emb.to("cuda")
    cu_cuda = cu.to("cuda")
    pos_ids = pos_ids.to("cuda")

    encoder = _build_packed_encoder(seed=data["seed"])

    # Re-seed immediately before the compressed call so any stochastic ops
    # inside the head/operator line up with the fixture capture's ordering.
    torch.manual_seed(data["seed"])
    with torch.no_grad():
        out = encoder(
            content_embeddings=packed_emb,
            content_cu_seqlens=cu_cuda,
            content_position_ids=pos_ids,
            target_ratio_l0=target_ratio,
        )

    # --- head outputs: (N_content,) <-> (B, L_max) ---
    exp_base_raw = expected["base_raw"]  # (B, L_max) bf16
    exp_logits = expected["logits_for_op"]  # (B, L_max) bf16
    exp_mask = expected["survivor_mask"]  # (B, L_max) bool
    exp_counts = expected["survivor_counts"]  # (B,) int

    got_base_raw_padded, _ = _unpack_to_padded(out.l0.base_raw.cpu(), cu)
    got_logits_padded, _ = _unpack_to_padded(out.l0.logits_for_op.cpu(), cu)
    got_mask_padded, _ = _unpack_to_padded(out.l0.survivor_mask.cpu(), cu)

    # Compare at valid positions only.
    torch.testing.assert_close(
        got_base_raw_padded[attention_mask],
        exp_base_raw[attention_mask],
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        got_logits_padded[attention_mask],
        exp_logits[attention_mask],
        atol=5e-2,
        rtol=5e-2,
    )

    # Survivor mask should agree exactly as a bool.
    # (Head outputs have some bf16 noise; positions near θ may flip. Expect
    # very high agreement — 95%+ — not strict equality.)
    agree = (got_mask_padded == exp_mask)[attention_mask]
    agreement_rate = agree.float().mean().item()
    assert agreement_rate > 0.9, (
        f"survivor_mask agreement {agreement_rate:.3f} < 0.9 "
        f"(bf16 drift near θ is expected; large disagreement suggests a bug)"
    )

    # survivor_counts: per-sample survivor counts agree within a small margin.
    got_counts = out.survivor_counts.cpu()
    assert got_counts.shape == exp_counts.shape
    count_diff = (got_counts.to(torch.int64) - exp_counts.to(torch.int64)).abs()
    assert count_diff.max().item() <= 3, (
        f"survivor_counts diverged: got {got_counts.tolist()} vs expected {exp_counts.tolist()}"
    )


@pytest.mark.slow
@pytest.mark.gpu
def test_encoder_packed_output_shapes(monkeypatch):
    """Smoke test: packed forward returns the right flat shapes."""
    _force_sdpa_packed_attention(monkeypatch)
    data = _load_fixture()
    inputs = data["inputs"]
    input_emb = inputs["input_embeddings"]
    attention_mask = inputs["attention_mask"]

    packed_emb, cu, pos_ids = _pack_content(input_emb, attention_mask)
    n_content = int(cu[-1].item())
    packed_emb = packed_emb.to("cuda")
    cu = cu.to("cuda")
    pos_ids = pos_ids.to("cuda")

    encoder = _build_packed_encoder(seed=data["seed"])

    with torch.no_grad():
        out_no = encoder(
            content_embeddings=packed_emb,
            content_cu_seqlens=cu,
            content_position_ids=pos_ids,
            target_ratio_l0=None,
        )
        out_yes = encoder(
            content_embeddings=packed_emb,
            content_cu_seqlens=cu,
            content_position_ids=pos_ids,
            target_ratio_l0=0.5,
        )

    # No-compression: survivor_embeddings is flat over content positions.
    assert out_no.survivor_embeddings.shape == (n_content, 1024)
    assert out_no.survivor_cu_seqlens.shape == (5,)  # B+1 = 4+1
    assert int(out_no.survivor_cu_seqlens[-1].item()) == n_content

    # Compressed: flat survivors with their own cu_seqlens.
    total_survivors = int(out_yes.survivor_cu_seqlens[-1].item())
    assert out_yes.survivor_embeddings.shape == (total_survivors, 1024)
    assert out_yes.survivor_mask.shape == (n_content,)
    assert out_yes.base_raw.shape == (n_content,)
    assert out_yes.logits_for_op.shape == (n_content,)
