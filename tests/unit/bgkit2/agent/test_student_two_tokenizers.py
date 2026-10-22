"""The student / bench serving path on a two-tokenizer pairing (2026-09-26: the LFM2.5-8B-A1B
decoder of bgkit2_8b_a). Tiny random models, the REAL tokenizers: slot text, compression prompts
and spine leaves reach the encoder as encoder (LFM2.5-350M) ids; the rendering, the sentinel and
generation use the decoder's; every rep -- text slots and cached tree nodes alike -- is spliced
through the decoder-space adapter; trainable-only decoder checkpoints load strictly."""

from __future__ import annotations

import glob
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bgkit2.data.overview import overview_marker
from bgkit2.data.templates import SPLICE_SENTINEL

from ..conftest import tiny_config

M8 = "/mnt/external/bgkit-data/models/hf/LFM2.5-8B-A1B"
S350 = glob.glob(
    str(Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*"))

pytestmark = pytest.mark.skipif(not (Path(M8).is_dir() and S350), reason="tokenizers not on disk")

ENC_VOCAB, DEC_VOCAB = 65536, 128000
REP_DIM, DEC_DIM = 32, 48  # the rep space (the 350M's, tiny) != the decoder's embedding space


@pytest.fixture(scope="module")
def toks():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(M8), AutoTokenizer.from_pretrained(S350[0])


def _core(toks, trainable_only: bool = False):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model

    from bgkit2.models.decoder import SpliceDecoder
    from bgkit2.models.encoder import BgKIT2Encoder
    from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone
    from bgkit2.models.space_adapter import DecoderSpaceAdapter
    from bgkit2.training.config import Config
    from bgkit2.training.standalone import StandaloneModels

    t8, t3 = toks
    cfg = Config()
    cfg.run.dtype = "float32"
    cfg.model.decoder_id, cfg.model.ctx_tokenizer_id = M8, S350[0]
    cfg.model.decoder_save_trainable_only = trainable_only
    core = StandaloneModels(cfg, device="cpu")
    core.tok, core.ctx_tok = t8, t3  # what load_tokenizers reads from the two ids
    torch.manual_seed(0)
    core.encoder = BgKIT2Encoder(
        Lfm2BidirectionalBackbone(Lfm2Model(tiny_config(vocab_size=ENC_VOCAB))),
        head_layer=2, head_inner_dim=8,
    )
    lm = Lfm2ForCausalLM(tiny_config(vocab_size=DEC_VOCAB, hidden_size=DEC_DIM,
                                     pad_token_id=t8.pad_token_id, bos_token_id=t8.bos_token_id,
                                     eos_token_id=t8.eos_token_id))
    core.decoder = SpliceDecoder(lm)
    core.decoder.space_adapter = DecoderSpaceAdapter(REP_DIM, DEC_DIM)
    core.rep_space_table = torch.randn(300, REP_DIM)  # attach_space_adapter's table (tiny)
    core.finish_models()
    core.encoder.eval()
    core.decoder.eval()
    return core


def _student(toks):
    from bgkit2.agent.student import StudentModel

    return StudentModel.from_core(_core(toks), "ck")


def _spy_encoder(student):
    seen = []
    orig = student.encoder.forward

    def spy(ids, mask, **kw):
        seen.append((ids.clone(), mask.clone(), kw.get("prompt_ids"), kw.get("prompt_mask")))
        return orig(ids, mask, **kw)

    student.encoder.forward = spy
    return seen


def _spy_build_batch(student):
    seen = []
    orig = student.decoder.build_batch

    def spy(prefix_ids, reps, rep_mask, suffix_ids, **kw):
        sb = orig(prefix_ids, reps, rep_mask, suffix_ids, **kw)
        seen.append(SimpleNamespace(prefix=prefix_ids, suffix=suffix_ids, reps=reps,
                                    extra=kw.get("extra"), mid=kw.get("mid_ids"),
                                    embeds=sb.inputs_embeds))
        return sb

    student.decoder.build_batch = spy
    return seen


def test_student_renderer_splits_the_two_tokenizers(toks):
    t8, t3 = toks
    st = _student(toks)
    assert st.tok is t8 and st.renderer.split_tokenizers and st.renderer.ctx_tok is t3
    assert st.renderer.sentinel_id == t8.convert_tokens_to_ids("<|img_row_1_col_1|>")
    assert st.renderer.eos_id == t8.convert_tokens_to_ids("<|im_end|>") == t8.eos_token_id


def test_text_slots_encode_encoder_ids_and_splice_through_the_adapter(toks):
    t8, t3 = toks
    st = _student(toks)
    enc_seen, dec_seen = _spy_encoder(st), _spy_build_batch(st)
    ctx = ["def area(r):\n    return 3.14159 * r * r\n", "The capital of France is Paris."]
    prompts = ["read geometry.py: the area formula", ""]
    msgs = [{"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": f"First:\n{SPLICE_SENTINEL}\nSecond:\n{SPLICE_SENTINEL}\n"
                                        "What does area return?"}]
    out = st.generate_slots(msgs, ctx, prompts, None, [4, 3], max_new_tokens=3)
    assert out["n_slots"] == 2 and isinstance(out["text"], str)
    ids, mask, p_ids, p_mask = enc_seen[0]
    for j, c in enumerate(ctx):  # the encoder read ENCODER ids of the slot text ...
        assert ids[j][mask[j].bool()].tolist() == t3(c, add_special_tokens=False)["input_ids"]
    # ... and of the compression prompt (the default one for an empty prompt)
    assert p_ids[0][p_mask[0].bool()].tolist() == t3(
        prompts[0], add_special_tokens=False)["input_ids"]
    assert p_ids[1][p_mask[1].bool()].tolist() == t3(
        "Compress the tool result.\n", add_special_tokens=False)["input_ids"]
    b = dec_seen[-1]
    assert b.reps.shape[-1] == REP_DIM and b.embeds.shape[-1] == DEC_DIM  # adapted at splice
    dec_ids = [*b.prefix[0].tolist(), *b.mid[0][0].tolist(), *b.suffix[0].tolist()]
    # the decoder read its own rendering: no sentinel left, and the same text as the 8B template
    assert st.renderer.sentinel_id not in dec_ids and max(dec_ids) < len(t8)
    full = t8.apply_chat_template(
        [{**m, "content": m["content"].replace(SPLICE_SENTINEL, "<|img_row_1_col_1|>")}
         for m in msgs], tokenize=True, add_generation_prompt=True)
    full = full if isinstance(full, list) else full["input_ids"]
    assert [x for x in full if x != st.renderer.sentinel_id] == dec_ids


def test_cached_tree_nodes_are_rep_space_and_go_through_the_adapter(toks):
    st = _student(toks)
    dec_seen = _spy_build_batch(st)
    node = torch.randn(5, REP_DIM)

    class Cache:
        repos = ("r",)

        def get(self, repo, n=""):
            assert repo == "r"
            return (node,)

    st.tree_root_cache = Cache()
    msgs = [{"role": "user", "content": f"Repository map:\n{SPLICE_SENTINEL}\nWhere is main?"}]
    out = st.generate_slots(msgs, [overview_marker("r", 5)], [""], None, [5], max_new_tokens=2)
    assert out["n_reps"] == 5
    b = dec_seen[-1]
    assert torch.equal(b.reps[0, :5], node) and b.embeds.shape[-1] == DEC_DIM
    ad = st.decoder.space_adapter(node[None], torch.ones(1, 5, dtype=torch.bool))
    start = int(b.prefix[0].shape[0])
    assert torch.allclose(b.embeds[0, start : start + 5], ad[0], atol=1e-5)


def test_full_text_arm_reads_decoder_ids_without_the_adapter(toks):
    t8, _ = toks
    st = _student(toks)
    dec_seen = _spy_build_batch(st)
    prompt = t8.apply_chat_template([{"role": "user", "content": "Say hi."}], tokenize=False,
                                    add_generation_prompt=True)
    out = st.generate_text(prompt, max_new_tokens=2)
    b = dec_seen[-1]
    assert b.reps is None
    assert b.prefix[0].tolist() == t8(prompt, add_special_tokens=False)["input_ids"]
    assert out["prefix_tokens"] == len(b.prefix[0])


def test_spine_leaves_are_encoder_ids(toks, monkeypatch):
    _t8, t3 = toks
    st = _student(toks)
    seen = {}

    def spine_output(ids, p_ids, **kw):
        seen["ids"], seen["p"] = ids, p_ids
        raise RuntimeError("stop")

    monkeypatch.setattr(st.core, "spine_output", spine_output)
    payload = {"compaction": 1, "leaf": "assistant: done",
               "child": {"compaction": 1, "leaf": "user: fix the bug", "child": None}}
    with pytest.raises(RuntimeError, match="stop"):
        st._encode_spine_slot(payload, "compaction: the bug")
    from bgkit2.agent.context import compaction_leaves

    leaves = compaction_leaves(payload)
    assert [x.tolist() for x in seen["ids"][0]] == [
        t3(x, add_special_tokens=False)["input_ids"] for x in leaves]
    assert seen["p"][0].tolist() == t3("compaction: the bug", add_special_tokens=False)["input_ids"]


def test_trainable_only_decoder_state_loads_strictly(toks):
    from bgkit2.models.decoder import TrainableDecoderState

    core = _core(toks, trainable_only=True)
    sd = {k: v.clone() for k, v in TrainableDecoderState(core.decoder).state_dict().items()}
    assert sd and all(k.startswith("space_adapter.") for k in sd)  # no LoRA in this tiny core
    with torch.no_grad():
        core.decoder.space_adapter.linear.weight.add_(1.0)
    core.load_decoder_state(sd, source="ck")
    assert torch.equal(core.decoder.space_adapter.linear.weight, sd["space_adapter.linear.weight"])
    with pytest.raises(RuntimeError, match="missing"):
        core.load_decoder_state({k: v for k, v in sd.items() if "linear" not in k})
    # a full-decoder config refuses the trainable-only state (the base would be missing)
    core.cfg2.model.decoder_save_trainable_only = False
    with pytest.raises(RuntimeError):
        core.load_decoder_state(sd)


def test_load_tokenizers_reads_both_ids_and_keeps_host_ones(toks):
    from bgkit2.training.config import Config
    from bgkit2.training.standalone import StandaloneModels

    cfg = Config()
    cfg.model.decoder_id, cfg.model.ctx_tokenizer_id = M8, S350[0]
    core = StandaloneModels(cfg, device="cpu")
    core.load_tokenizers()
    assert len(core.tok) == len(toks[0]) and len(core.ctx_tok) == len(toks[1])
    r = core.chat_renderer()
    assert r.split_tokenizers and r.enc_pad_id < ENC_VOCAB <= r.pad_id
    cfg1 = Config()
    cfg1.model.decoder_id = S350[0]
    one = StandaloneModels(cfg1, device="cpu")
    one.tok = toks[1]
    one.load_tokenizers()  # the host's tokenizer is kept; one tokenizer -> ctx_tok None
    assert one.tok is toks[1] and one.ctx_tok is None and not one.chat_renderer().split_tokenizers
