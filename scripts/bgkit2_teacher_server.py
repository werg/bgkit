#!/usr/bin/env python
"""OpenAI-compatible chat server around a HF causal LM (the teacher / digest
model for trajectory generation). Runs inside a GPU container (the vllm-node
image has transformers 5.15 + fastapi); the model's OWN chat template renders
messages + tools; the reply carries parsed content / tool_calls /
reasoning_content (see bgkit2.agent.teacher).

Co-tenancy guard: the trainer shares the unified memory. Requests are refused
(HTTP 503, Retry-After) while host ``available`` memory is below
``--min-avail-gb``; generation is serialised (one request at a time) so the
peak is one prefill + KV cache of ``--max-model-len`` tokens.

Usage (in the container):
  python scripts/bgkit2_teacher_server.py --model IFM/K2-Horizon-7B --port 8095 \
      --min-avail-gb 14 --max-model-len 65536
"""

from __future__ import annotations

import argparse
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from bgkit2.agent.teacher import TeacherEngine, parse_k2_output


def host_available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 0.0


def build(args) -> FastAPI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True, device_map=args.device
    )
    model.eval()
    parse = parse_k2_output if "k2" in args.model.lower() else (lambda t: parse_k2_output(t))
    engine = TeacherEngine(model, tok, parse=parse, device=args.device)
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "model": args.model, "host_available_gb": host_available_gb()}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        avail = host_available_gb()
        if avail < args.min_avail_gb:
            return JSONResponse(
                {"error": f"host memory low: {avail:.1f} GB available"},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        n_prompt = 0
        with lock:
            t0 = time.perf_counter()
            try:
                out = engine.chat(
                    body["messages"],
                    body.get("tools"),
                    max_tokens=min(int(body.get("max_tokens", 2048)), args.max_new_tokens),
                    temperature=float(body.get("temperature", 0.0)),
                    top_p=float(body.get("top_p", 1.0)),
                    chat_template_kwargs=body.get("chat_template_kwargs")
                    or args.default_template_kwargs,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            n_prompt = out["usage"]["prompt_tokens"]
            out["usage"]["wall_s"] = time.perf_counter() - t0
        if n_prompt > args.max_model_len:
            out["warning"] = f"prompt of {n_prompt} tokens exceeds --max-model-len"
        out.update(id="chatcmpl-local", object="chat.completion", model=args.model)
        return out

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--min-avail-gb", type=float, default=14.0)
    ap.add_argument(
        "--template-kwargs",
        default='{"tool_call_format": "json", "reasoning_effort": "low"}',
        help="JSON dict passed to apply_chat_template when the request has none",
    )
    args = ap.parse_args()
    import json

    args.default_template_kwargs = json.loads(args.template_kwargs)
    app = build(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
