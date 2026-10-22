#!/usr/bin/env python
"""Serve a bgkit2 checkpoint to the agent harness (GPU container).

Endpoints (JSON):
  GET  /health
  POST /generate       {messages, contexts, prompts, tools, n_reps, max_new_tokens}
                       -> {text, prefix_tokens, n_reps, n_slots}   (compressed arm)
  POST /generate_full  {prompt, max_new_tokens} -> {text, prefix_tokens} (full-text arm)

One request at a time on the GPU (a lock); the harness runs episodes
concurrently and queues here. Launch via the compose service
``serve-bgkit2-student`` (port 8097):
  CK=eval_keep/<name> docker compose -f docker/docker-compose.yaml up -d serve-bgkit2-student
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.agent.student import StudentModel, handle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True, help="hydra experiment name of the checkpoint")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--tree-root-cache", default="",
                    help="per-repo root reps (bgkit2_cache_repo_trees.py): fills overview slots")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--thinking", action="store_true",
                    help="always-thinking serving: the decoder's own generation prompt, reasoning "
                         "returned as a <think> block (StudentModel.thinking)")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentModel(cfg, Path(args.checkpoint), device)
    student.thinking = bool(args.thinking)
    if args.tree_root_cache:
        n = student.load_tree_root_cache(args.tree_root_cache)
        print(f"tree root cache: {n} repos", flush=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(*handle(student, self.path, {}))

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n).decode() or "{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "bad json"})
                return
            with lock:
                status, payload = handle(student, self.path, body)
            self._send(status, payload)

        def log_message(self, fmt, *a):  # quiet
            return

    print(f"student server: {args.checkpoint} on {args.host}:{args.port} ({device})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
