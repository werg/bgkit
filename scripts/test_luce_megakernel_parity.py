#!/usr/bin/env python3
"""Parity checks for the BgKIT-adapted Luce Qwen3.5 megakernel.

Checks:
1. Luce token prefill and embedding prefill are equivalent when the embeddings
   are ``embed_tokens[token_ids]``.
2. Luce survivor-splice generation matches ``ReconstructionDecoder`` greedy
   generation on a short B=1 splice prompt, using the same loaded decoder
   weights.
3. Luce survivor-splice hidden prefill matches ``ReconstructionDecoder`` final
   hidden states closely enough to anchor the training CE/backward path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.inference.luce_megakernel import (
    LuceSingleSpliceForward,
    LuceSingleSpliceGenerator,
    load_decoder_from_hf_model,
)
from bgkit.inference.luce_megakernel import (
    status as luce_status,
)
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.utils.attention_backend import resolve_decoder_attention_implementation


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: dict[str, Any]


def _ids(tokenizer, text: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor(ids, dtype=dtype, device=device)


def _load_models(model_name: str, *, verbose: bool) -> tuple[Any, Any, ReconstructionDecoder, Any]:
    attn_impl = resolve_decoder_attention_implementation("auto", decoder_family="qwen35")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_decoder = ReconstructionDecoder(model, hidden_dim=1024, decoder_family="qwen35")
    luce_decoder = load_decoder_from_hf_model(
        model,
        tokenizer=tokenizer,
        backend="nvfp4",
        verbose=verbose,
    )
    return model, tokenizer, hf_decoder, luce_decoder


def _token_prefill_parity(
    luce_decoder,
    tokenizer,
    *,
    prompt: str,
    decode_steps: int,
) -> CheckResult:
    device = luce_decoder._embed_weight.device
    token_ids = _ids(tokenizer, prompt, device=device, dtype=torch.int32)
    if int(token_ids.numel()) < 2:
        raise ValueError("token parity prompt must encode to at least two tokens")

    first_token = int(luce_decoder.prefill_tokens(token_ids))
    seq_tokens = [first_token]
    cur = first_token
    for _ in range(max(0, decode_steps - 1)):
        cur = int(luce_decoder.step(cur))
        seq_tokens.append(cur)

    embeds = luce_decoder._embed_weight.index_select(0, token_ids.long()).to(torch.bfloat16)
    first_embed = int(luce_decoder.prefill_embeddings(embeds.contiguous()))
    seq_embeds = [first_embed]
    cur = first_embed
    for _ in range(max(0, decode_steps - 1)):
        cur = int(luce_decoder.step(cur))
        seq_embeds.append(cur)

    return CheckResult(
        name="token_prefill_vs_embedding_prefill",
        ok=seq_tokens == seq_embeds,
        details={
            "prompt": prompt,
            "prompt_len": int(token_ids.numel()),
            "decode_steps": decode_steps,
            "token_path": seq_tokens,
            "embedding_path": seq_embeds,
            "token_text": tokenizer.decode(seq_tokens, skip_special_tokens=True),
            "embedding_text": tokenizer.decode(seq_embeds, skip_special_tokens=True),
        },
    )


def _splice_generation_parity(
    model,
    tokenizer,
    hf_decoder: ReconstructionDecoder,
    luce_decoder,
    *,
    prefix: str,
    survivor_text: str,
    suffix: str,
    max_new_tokens: int,
) -> CheckResult:
    device = next(model.parameters()).device
    prefix_ids = _ids(tokenizer, prefix, device=device, dtype=torch.long)
    survivor_ids = _ids(tokenizer, survivor_text, device=device, dtype=torch.long)
    suffix_ids = _ids(tokenizer, suffix, device=device, dtype=torch.long)
    embed = model.get_input_embeddings()
    survivor_embeddings = embed(survivor_ids).to(torch.bfloat16)
    survivor_cu = torch.tensor(
        [0, int(survivor_embeddings.shape[0])],
        dtype=torch.int32,
        device=device,
    )

    with torch.inference_mode():
        hf_out = hf_decoder.generate_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        luce_out = LuceSingleSpliceGenerator(
            decoder=luce_decoder,
            tokenizer=tokenizer,
        ).generate_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

    hf_ids = hf_out.full_ids[0].detach().cpu().tolist()
    luce_ids = luce_out.full_ids[0].detach().cpu().tolist()
    common_prefix = 0
    for a, b in zip(hf_ids, luce_ids, strict=False):
        if a != b:
            break
        common_prefix += 1

    return CheckResult(
        name="splice_generation_vs_reconstruction_decoder",
        ok=hf_ids == luce_ids,
        details={
            "prefix": prefix,
            "survivor_text": survivor_text,
            "suffix": suffix,
            "prefix_len": int(prefix_ids.numel()),
            "survivor_len": int(survivor_ids.numel()),
            "max_new_tokens": max_new_tokens,
            "hf_ids": hf_ids,
            "luce_ids": luce_ids,
            "common_prefix_tokens": common_prefix,
            "hf_text": hf_out.content_text[0],
            "luce_text": luce_out.content_text[0],
        },
    )


def _splice_hidden_parity(
    model,
    tokenizer,
    hf_decoder: ReconstructionDecoder,
    luce_decoder,
    *,
    prefix: str,
    survivor_text: str,
    suffix: str,
    max_abs_tol: float,
    mean_abs_tol: float,
    loss_abs_tol: float,
) -> CheckResult:
    device = next(model.parameters()).device
    prefix_ids = _ids(tokenizer, prefix, device=device, dtype=torch.long)
    survivor_ids = _ids(tokenizer, survivor_text, device=device, dtype=torch.long)
    suffix_ids = _ids(tokenizer, suffix, device=device, dtype=torch.long)
    embed = model.get_input_embeddings()
    survivor_embeddings = embed(survivor_ids).to(torch.bfloat16)
    survivor_cu = torch.tensor(
        [0, int(survivor_embeddings.shape[0])],
        dtype=torch.int32,
        device=device,
    )

    with torch.inference_mode():
        hf_out = hf_decoder.forward_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=[prefix_ids],
            suffix_ids=[suffix_ids],
            return_hidden_states=True,
        )
        luce_out = LuceSingleSpliceForward(
            decoder=luce_decoder,
        ).forward_with_single_splice_hidden(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
        )

    hf_hidden = hf_out.hidden_states.detach().float()
    luce_hidden = luce_out.hidden_states.detach().float()
    diff = (hf_hidden - luce_hidden).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms = float(torch.sqrt((diff * diff).mean()).item())
    with torch.inference_mode():
        luce_loss = hf_decoder._compute_lm_ce(
            lm_head=hf_out.lm_head,
            hidden_states=luce_out.hidden_states,
            token_ids_full=hf_out.token_ids,
            attention_mask=hf_out.attention_mask,
            loss_mask_full=hf_out.loss_mask,
            chunk_size=None,
        )
    hf_loss = float(hf_out.loss.detach().float().item())
    luce_loss_value = float(luce_loss.detach().float().item())
    loss_abs = abs(hf_loss - luce_loss_value)
    ok = (
        hf_hidden.shape == luce_hidden.shape
        and max_abs <= max_abs_tol
        and mean_abs <= mean_abs_tol
        and loss_abs <= loss_abs_tol
    )

    return CheckResult(
        name="splice_hidden_vs_reconstruction_decoder",
        ok=ok,
        details={
            "prefix": prefix,
            "survivor_text": survivor_text,
            "suffix": suffix,
            "shape": tuple(hf_hidden.shape),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rms_abs": rms,
            "hf_loss": hf_loss,
            "luce_hidden_loss": luce_loss_value,
            "loss_abs": loss_abs,
            "max_abs_tol": max_abs_tol,
            "mean_abs_tol": mean_abs_tol,
            "loss_abs_tol": loss_abs_tol,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--mode",
        choices=("token", "splice", "hidden", "both", "all"),
        default="both",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefix", default="Complete this phrase: The capital of")
    parser.add_argument("--survivor-text", default=" France")
    parser.add_argument("--suffix", default=" is")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--hidden-max-abs-tol", type=float, default=0.5)
    parser.add_argument("--hidden-mean-abs-tol", type=float, default=0.06)
    parser.add_argument("--hidden-loss-abs-tol", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    st = luce_status("nvfp4")
    print(json.dumps({"luce_status": asdict(st)}, indent=2, sort_keys=True))
    if not st.usable:
        raise SystemExit(2)

    model, tokenizer, hf_decoder, luce_decoder = _load_models(args.model, verbose=not args.quiet)
    results: list[CheckResult] = []
    if args.mode in {"token", "both", "all"}:
        results.append(
            _token_prefill_parity(
                luce_decoder,
                tokenizer,
                prompt=args.prompt,
                decode_steps=args.decode_steps,
            )
        )
    if args.mode in {"splice", "both", "all"}:
        results.append(
            _splice_generation_parity(
                model,
                tokenizer,
                hf_decoder,
                luce_decoder,
                prefix=args.prefix,
                survivor_text=args.survivor_text,
                suffix=args.suffix,
                max_new_tokens=args.max_new_tokens,
            )
        )
    if args.mode in {"hidden", "all"}:
        results.append(
            _splice_hidden_parity(
                model,
                tokenizer,
                hf_decoder,
                luce_decoder,
                prefix=args.prefix,
                survivor_text=args.survivor_text,
                suffix=args.suffix,
                max_abs_tol=args.hidden_max_abs_tol,
                mean_abs_tol=args.hidden_mean_abs_tol,
                loss_abs_tol=args.hidden_loss_abs_tol,
            )
        )

    payload = {"results": [asdict(r) for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(r.ok for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
