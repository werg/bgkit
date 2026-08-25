#!/usr/bin/env python
"""GENERATION probe — confound-free. Feed the reps, greedily decode, LOOK.

Every CE-gap metric so far is confounded (AR teacher-forcing prior floors
ce_zeroed; the raw-embedding text-oracle is off the decoder's operating norm).
This probe sidesteps all of it: encode a piece of content with the KNOWN-GOOD
51945 encoder, splice the COMPRESSED reps, and GENERATE greedily with the real
verbatim-reconstruction prompt. Then print target-vs-generated.

If in-distribution prose/code reconstructs verbatim AND git diffs/files do too
-> the decoder+reps reconstruct diffs fine; the git-repro wall is the tree
pipeline / step-286 training (a bug). If the decoder paraphrases / drifts even
in-distribution -> reps carry gist, not verbatim tokens (compression limit).

Run (container, trainer STOPPED, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    -e DIAG_CONTENT_JSON=/workspace/bgkit/scripts/git_repro_content.json \
    train-phase2-kb-git-repro-fullbackprop \
    python /workspace/bgkit/scripts/diag_git_repro_gen_probe.py \
    +experiment=phase1_summarization_round_robin \
    step1_checkpoint=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459 \
    training.max_total_source_tokens=4096
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import structlog
import torch
from omegaconf import DictConfig

from bgkit.data.chat_template import TOOL_CONFIGS, tokenize_with_sentinel
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()

RECON_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the "
        "bgkit_read_file tool for reading file contents."
    ),
    "user_prompt": "Read the file `{file_path}`",
    "compression_prompt": "Return the file contents verbatim",
    "response_prefix": "Here are the contents of `{file_path}`:",
}
RECON_CONFIG = TOOL_CONFIGS["file_read_repro"]

CONTENT_JSON = os.environ.get(
    "DIAG_CONTENT_JSON", "/workspace/bgkit/scripts/git_repro_content.json",
)
SRC_CAP = int(os.environ.get("DIAG_SRC_CAP", "1200"))
MAX_NEW = int(os.environ.get("DIAG_MAX_NEW", "300"))
# (l0, l1): near-lossless and the git-repro leaf retention.
RATIOS = ((0.999, None), (0.63, 0.63))


def build_loss_mask(prefix_ids, per_group, suffix_masks, device):
    full = []
    for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
        full.append(torch.cat([
            torch.zeros(int(pre.shape[0]), dtype=torch.bool, device=device),
            torch.zeros(int(per_group[i]), dtype=torch.bool, device=device),
            sm.to(device),
        ]))
    return torch.cat(full)


def ce_with_slot(decoder, survivors, group_cu, prefix_ids, suffix_ids, loss_mask):
    out = decoder.forward_with_single_splice(
        survivor_embeddings=survivors, survivor_cu_seqlens=group_cu,
        prefix_ids=prefix_ids, suffix_ids=suffix_ids, loss_mask=loss_mask,
    )
    loss = out.loss if hasattr(out, "loss") else out
    return float(loss.item())


def build_recon_inputs(trainer, family, text, label, language):
    dev = trainer.device
    dec_tok = trainer.tokenizer_qwen if family == "qwen35" else trainer.tokenizer_falcon
    target_t = torch.as_tensor(
        dec_tok.encode(text, add_special_tokens=False)[:SRC_CAP], dtype=torch.long,
    )
    out = tokenize_with_sentinel(
        dec_tok, RECON_VARIANT, RECON_CONFIG, file_path=label, language=language,
        content_token_ids=target_t, encoder_tokenizer=trainer.encoder_tokenizer,
    )
    tok = out["token_ids"]
    lm = out["loss_mask"]
    ss = int(out["bgkit_splice_start"].item())
    sl = int(out["bgkit_splice_len"].item())
    prefix = tok[:ss].to(dev)
    suffix = tok[ss + sl:].to(dev)
    suffix_mask = lm[ss + sl:].to(torch.bool).to(dev)
    comp = out["compression_prompt_ids"].to(torch.long).to(dev)
    return prefix, suffix, suffix_mask, comp


@torch.no_grad()
def probe(trainer, family, label, ctype, text, language):
    dev = trainer.device
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dec_tok = trainer.tokenizer_qwen if family == "qwen35" else trainer.tokenizer_falcon
    prefix, suffix, suffix_mask, comp = build_recon_inputs(
        trainer, family, text, label, language)
    embed_dec = decoder._get_inner_model_and_head()[0].get_input_embeddings()
    enc_ids = np.asarray(
        trainer.encoder_tokenizer.encode(text, add_special_tokens=False)[:SRC_CAP],
        dtype=np.int64,
    )
    batch = {"source_docs": [[enc_ids]], "dataset_names": ["multi_news"],
             "group_ids": [label]}
    tgt_txt = dec_tok.decode(
        dec_tok.encode(text, add_special_tokens=False)[:SRC_CAP],
    )
    # text-oracle: full uncompressed source embeddings in the slot (ceiling).
    src_dec = torch.as_tensor(
        dec_tok.encode(text, add_special_tokens=False)[:SRC_CAP], dtype=torch.long,
        device=dev)
    oracle_emb = embed_dec(src_dec).to(dtype=torch.bfloat16)
    gc_o = torch.tensor([0, int(src_dec.shape[0])], dtype=torch.int32, device=dev)
    lm_o = build_loss_mask([prefix], [int(src_dec.shape[0])], [suffix_mask], dev)
    ce_oracle = ce_with_slot(decoder, oracle_emb, gc_o, [prefix], [suffix], lm_o)
    print("\n" + "#" * 100)
    print(f"# {family} | {ctype} | {label} | src_enc_tokens={int(enc_ids.shape[0])} "
          f"| ce_oracle={ce_oracle:.3f}")
    print("#" * 100)
    print("---- TARGET (first 500 chars) ----")
    print(tgt_txt[:500])
    for r0, r1 in RATIOS:
        trainer._target_ratio_start = trainer._target_ratio_end = r0
        trainer.global_step = 0
        if r1 is None:
            trainer._l1_introduction_step = 10**9
        else:
            trainer._l1_introduction_step = 0
            trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = r1
        enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, [comp])
        survivors = enc_out.survivor_embeddings
        K = int(survivors.shape[0])
        lm = build_loss_mask([prefix], per_group, [suffix_mask], dev)
        ce_reps = ce_with_slot(decoder, survivors, group_cu, [prefix], [suffix], lm)
        ce_zero = ce_with_slot(
            decoder, torch.zeros_like(survivors), group_cu, [prefix], [suffix], lm)
        rep_gain = ce_zero - ce_reps
        rep_ratio = rep_gain / max(ce_zero - ce_oracle, 1e-6)
        nr = float(survivors.detach().float().norm(dim=-1).mean()) if K > 0 else 0.0
        print(f"\n---- METRICS @ l0={r0},l1={r1}: ce_reps={ce_reps:.3f} "
              f"ce_zero={ce_zero:.3f} rep_gain={rep_gain:.3f} "
              f"rep_ratio={rep_ratio:.3f} surv_norm={nr:.2f} K={K} ----")
        try:
            gen = decoder.generate_with_single_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=group_cu,
                prefix_ids=prefix,
                suffix_ids=suffix,
                tokenizer=dec_tok,
                max_new_tokens=MAX_NEW,
                temperature=0.0,
            )
            ct = getattr(gen, "content_text", None)
            gtxt = (ct[0] if isinstance(ct, list) and ct else str(ct))
        except Exception as exc:  # noqa: BLE001
            gtxt = f"<GEN ERROR: {type(exc).__name__}: {exc}>"
        print(f"\n---- GENERATED @ l0={r0},l1={r1}  (K={K} reps) ----")
        print(str(gtxt)[:500])
        enc_out.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase1_summarization_round_robin", cfg.training.phase
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()

    # Optional: override the (base) weights with a flat phase2_kb checkpoint
    # (keys `encoder.*`, `decoders.{qwen35,falcon_h1}.*`) to probe the TRAINED
    # git-repro encoder via the same clean single-shot path.
    raw_ckpt = os.environ.get("DIAG_RAW_CKPT", "").strip()
    if raw_ckpt:
        raw = torch.load(os.path.join(raw_ckpt, "model.pt"),
                         map_location="cpu", weights_only=False)
        enc_sd = {k[len("encoder."):]: v for k, v in raw.items()
                  if k.startswith("encoder.")}
        q_sd = {k[len("decoders.qwen35."):]: v for k, v in raw.items()
                if k.startswith("decoders.qwen35.")}
        f_sd = {k[len("decoders.falcon_h1."):]: v for k, v in raw.items()
                if k.startswith("decoders.falcon_h1.")}
        me, ue = trainer.encoder.load_state_dict(enc_sd, strict=False)
        mq, uq = trainer.decoder_qwen.load_state_dict(q_sd, strict=False)
        mf, uf = trainer.decoder_falcon.load_state_dict(f_sd, strict=False)
        logger.info("raw_ckpt_injected", path=raw_ckpt,
                    enc_missing=len(me), enc_unexpected=len(ue),
                    qwen_missing=len(mq), qwen_unexpected=len(uq),
                    falcon_missing=len(mf), falcon_unexpected=len(uf))
        print(f"[RAW CKPT] enc missing={len(me)} unexpected={len(ue)} | "
              f"qwen missing={len(mq)} unexpected={len(uq)} | "
              f"falcon missing={len(mf)} unexpected={len(uf)}")

    # Optional: load a SEPARATE-FILE checkpoint (encoder.pt + decoder_merged.pt),
    # e.g. a phase1_step6 code-reconstruction ckpt. Remaps the single-block
    # `projection_block.*` layout to per-family `projection_blocks.qwen35.*`.
    sep_ckpt = os.environ.get("DIAG_SEP_CKPT", "").strip()
    if sep_ckpt:
        enc_raw = torch.load(os.path.join(sep_ckpt, "encoder.pt"),
                             map_location="cpu", weights_only=False)

        def _remap(k):
            k = k.replace("projection_block.transformer_layer.",
                          "projection_blocks.qwen35.transformer_layers.0.")
            k = k.replace("projection_block.", "projection_blocks.qwen35.")
            return k
        enc_sd = {_remap(k): v for k, v in enc_raw.items()}
        # Drop shape-mismatched keys (e.g. threshold anchor grids 7 vs 6) —
        # strict=False ignores missing/unexpected but NOT size mismatch.
        cur_enc = trainer.encoder.state_dict()
        dropped = [k for k, v in enc_sd.items()
                   if k in cur_enc and tuple(cur_enc[k].shape) != tuple(v.shape)]
        for k in dropped:
            enc_sd.pop(k)
        if dropped:
            print(f"[SEP CKPT] dropped {len(dropped)} shape-mismatched enc keys: "
                  f"{dropped[:6]}")
        dec_file = ("decoder_merged.pt"
                    if os.path.exists(os.path.join(sep_ckpt, "decoder_merged.pt"))
                    else "decoder.pt")
        dec_raw = torch.load(os.path.join(sep_ckpt, dec_file),
                             map_location="cpu", weights_only=False)
        me, ue = trainer.encoder.load_state_dict(enc_sd, strict=False)
        mq, uq = trainer.decoder_qwen.load_state_dict(dec_raw, strict=False)
        # projection-load sanity: any qwen35 projection key left unloaded?
        proj_missing = [k for k in me if "projection_blocks.qwen35" in k]
        logger.info("sep_ckpt_injected", path=sep_ckpt, dec_file=dec_file,
                    enc_missing=len(me), enc_unexpected=len(ue),
                    proj_qwen_missing=len(proj_missing),
                    qwen_missing=len(mq), qwen_unexpected=len(uq))
        print(f"[SEP CKPT] {sep_ckpt} dec={dec_file}\n"
              f"  enc missing={len(me)} (proj_qwen_missing={len(proj_missing)}) "
              f"unexpected={len(ue)} | qwen missing={len(mq)} unexpected={len(uq)}")
        if proj_missing:
            print("  !! projection_blocks.qwen35 keys NOT loaded:", proj_missing[:6])
        # show a few non-projection missing enc keys (expected: falcon proj,
        # l1l1_bridge, section_separator — harmless for single-shot qwen).
        other = [k for k in me if "projection_blocks.qwen35" not in k][:8]
        print("  enc missing (non-qwen-proj, expected harmless):", other)

    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()

    content = json.load(open(CONTENT_JSON))
    items = []
    # a couple of short diffs + one short gold file
    for e in content.get("diffs", [])[:2]:
        items.append((f"diff_{e['n_tok_qwen']}", "git_diff", e["text"], "diff"))
    gf = content.get("gold_files", [])
    if gf:
        items.append((f"goldfile_{gf[0]['n_tok_qwen']}", "git_gold_file", gf[0]["text"], "text"))
    # one in-distribution summ source (code/prose) for reference
    max_src = int(cfg.training.get("max_total_source_tokens", 4096))
    for flat_i in trainer._eval_flat_idx[:30]:
        b = trainer._collate([int(flat_i)])
        if sum(len(d) for d in b["source_docs"][0]) > max_src:
            continue
        txt = trainer.encoder_tokenizer.decode(
            np.concatenate([np.asarray(d) for d in b["source_docs"][0]]).tolist(),
        )
        items.append((f"summsrc_{b['dataset_names'][0]}", "summ_source", txt, "text"))
        break

    families = [
        f.strip() for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    for family in families:
        trainer.encoder.set_active_decoder_family(family)
        for label, ctype, text, lang in items:
            probe(trainer, family, label, ctype, text, lang)

    Path("/workspace/checkpoints/diag_git_repro_gen_probe.done").write_text("done")
    logger.info("gen_probe_complete")


if __name__ == "__main__":
    main()
