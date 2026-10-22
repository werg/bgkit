from __future__ import annotations

from bgkit2.agent.student import MAX_RATIO, MIN_RATIO, handle, slot_ratio


class StubStudent:
    checkpoint = "ck"

    def __init__(self):
        self.calls = []

    def generate_slots(self, messages, contexts, prompts, tools, n_reps, max_new_tokens=512):
        self.calls.append(("slots", messages, contexts, prompts, tools, n_reps, max_new_tokens))
        if len(contexts) != len(n_reps):
            raise ValueError("mismatch")
        return {
            "text": "ANSWER: x",
            "prefix_tokens": 10,
            "n_reps": sum(n_reps),
            "n_slots": len(contexts),
        }

    def generate_text(self, prompt, max_new_tokens=512):
        self.calls.append(("text", prompt, max_new_tokens))
        return {"text": "plain", "prefix_tokens": 3}


def test_slot_ratio_is_budget_over_length_clamped():
    assert slot_ratio(16, 64) == 0.25
    assert slot_ratio(1000, 64) == MAX_RATIO
    assert slot_ratio(1, 100_000) == MIN_RATIO
    assert slot_ratio(4, 0) == MAX_RATIO


def test_handle_routes_and_validates():
    st = StubStudent()
    assert handle(st, "/health", {})[0] == 200
    status, out = handle(st, "/generate", {"messages": [], "contexts": ["a"], "prompts": ["p"]})
    assert status == 400 and "n_reps" in out["error"]
    status, out = handle(
        st,
        "/generate",
        {
            "messages": [{"role": "user", "content": "q"}],
            "contexts": ["a", "b"],
            "prompts": ["p", "q"],
            "n_reps": [4, 8],
            "max_new_tokens": 32,
        },
    )
    assert status == 200 and out["n_reps"] == 12 and out["n_slots"] == 2
    assert st.calls[-1][5] == [4, 8] and st.calls[-1][6] == 32 and st.calls[-1][4] is None
    status, out = handle(
        st, "/generate", {"messages": [], "contexts": ["a"], "prompts": ["p"], "n_reps": [1, 2]}
    )
    assert status == 400 and "mismatch" in out["error"]
    status, out = handle(st, "/generate_full", {"prompt": "hello"})
    assert status == 200 and out["text"] == "plain"
    assert handle(st, "/nope", {})[0] == 404


def _cache_student(checkpoint: str = "ck"):
    """A StudentModel with a stub encoder whose reps are RANDOM per call: a cache hit is the
    only way two calls can return identical reps."""
    import contextlib
    from types import SimpleNamespace

    import torch

    from bgkit2.agent.student import StudentModel

    st = StudentModel.__new__(StudentModel)
    st.checkpoint, st.max_ctx_tokens, st.device = checkpoint, 64, torch.device("cpu")
    st.renderer = SimpleNamespace(
        encode_ctx=lambda t: torch.tensor([ord(c) % 50 for c in t] or [0]))
    calls = []

    def encoder(ids, mask, ratio=None, prompt_ids=None, prompt_mask=None):
        calls.append(int(ids.shape[0]))
        k = max(1, round(float(ratio.max()) * ids.shape[1]))
        rm = torch.zeros(ids.shape[0], k, dtype=torch.bool)
        for j in range(ids.shape[0]):
            rm[j, : max(1, round(float(ratio[j]) * int(mask[j].sum())))] = True
        return SimpleNamespace(reps=torch.randn(ids.shape[0], k, 4), rep_mask=rm)

    st.encoder = encoder
    st.core = SimpleNamespace(autocast=contextlib.nullcontext, finish_reps=lambda ids, out: out)
    return st, calls


def test_text_slot_cache_hit_returns_identical_reps():
    import torch

    st, calls = _cache_student()
    ctx = ["session one text " * 3, "session two text " * 2]
    r1, m1, n1 = st._encode_slots(ctx, ["INDEX", "INDEX"], [4, 3])
    assert calls == [2] and m1.sum(-1).tolist() == [4, 3]
    # the same slots again (next turn / next question): no encoder call, identical reps
    r2, m2, n2 = st._encode_slots(ctx, ["INDEX", "INDEX"], [4, 3])
    assert calls == [2] and n1 == n2
    assert torch.equal(r1[m1], r2[m2])
    # a new slot beside cached ones: only it is encoded; cached rows unchanged
    r3, m3, _ = st._encode_slots([ctx[0], "a read with a focus"], ["INDEX", "q?"], [4, 5])
    assert calls == [2, 1] and torch.equal(r3[0][m3[0]], r1[0][m1[0]])
    # any key component changes the key: prompt, rep count, checkpoint identity
    st._encode_slots([ctx[0]], ["other prompt"], [4])
    st._encode_slots([ctx[0]], ["INDEX"], [5])
    assert calls == [2, 1, 1, 1]
    other, _ = _cache_student("ck2")
    assert other.slot_key(ctx[0], "INDEX", 4) != st.slot_key(ctx[0], "INDEX", 4)
    s = st.slot_cache_stats()
    assert s["hits"] == 3 and s["misses"] == 5 and s["entries"] == 5
    assert handle(st, "/health", {})[1]["slot_cache"]["entries"] == 5


def test_text_slot_cache_is_bounded():
    st, calls = _cache_student()
    st.SLOT_CACHE_MAX_REPS = 6
    st._encode_slots(["aaaa aaaa", "bbbb bbbb"], ["p", "p"], [4, 4])
    assert st.slot_cache_stats()["reps"] == 8  # this call's slots are never evicted
    st._encode_slots(["cccc cccc"], ["p"], [4])
    s = st.slot_cache_stats()
    assert s["entries"] == 1 and s["reps"] == 4
    st._encode_slots(["cccc cccc"], ["p"], [4])
    assert calls == [2, 1]
