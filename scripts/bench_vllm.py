#!/usr/bin/env python3
"""Quick throughput benchmark for vLLM servers.

Usage:
    python scripts/bench_vllm.py --url http://localhost:8090 --concurrency 1 10 25 50
"""

import argparse
import asyncio
import time

import httpx

# Sample prompts of varying length to simulate real workload
PROMPTS = [
    (
        "Describe what this Python file does:\n"
        "```python\nimport os\nimport sys\nfrom pathlib import Path\n\n"
        "def walk_tree(root):\n    for p in Path(root).rglob('*'):\n"
        "        if p.is_file() and p.suffix == '.py':\n"
        "            yield p\n```"
    ),
    (
        "Describe what this configuration file does:\n"
        "```yaml\nname: ci\non: [push, pull_request]\njobs:\n"
        "  test:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - run: pip install -e .[dev]\n      - run: pytest\n```"
    ),
    (
        "Describe this JavaScript module:\n"
        "```javascript\nimport React, { useState, useEffect }"
        " from 'react';\n\nexport function DataTable({ endpoint,"
        " columns }) {\n  const [data, setData] = useState([]);\n"
        "  const [loading, setLoading] = useState(true);\n\n"
        "  useEffect(() => {\n    fetch(endpoint)\n"
        "      .then(r => r.json())\n"
        "      .then(d => { setData(d); setLoading(false); });\n"
        "  }, [endpoint]);\n\n"
        "  if (loading) return <div>Loading...</div>;\n"
        "  return (\n    <table>\n"
        "      <thead><tr>{columns.map(c => <th key={c}>"
        "{c}</th>)}</tr></thead>\n"
        "      <tbody>{data.map((row, i) => <tr key={i}>"
        "{columns.map(c => <td key={c}>{row[c]}</td>)}"
        "</tr>)}</tbody>\n    </table>\n  );\n}\n```"
    ),
    (
        "Describe this Rust function:\n"
        "```rust\nuse std::collections::HashMap;\n\n"
        "pub fn word_frequency(text: &str) -> HashMap<String, usize>"
        " {\n    let mut freq = HashMap::new();\n"
        "    for word in text.split_whitespace() {\n"
        "        let w = word.to_lowercase()"
        ".trim_matches(|c: char| !c.is_alphanumeric())"
        ".to_string();\n        if !w.is_empty() {\n"
        "            *freq.entry(w).or_insert(0) += 1;\n"
        "        }\n    }\n    freq\n}\n```"
    ),
    (
        "Describe this Dockerfile:\n"
        "```dockerfile\nFROM python:3.12-slim\nWORKDIR /app\n"
        "COPY pyproject.toml .\n"
        'RUN pip install --no-cache-dir .\nCOPY src/ src/\n'
        "EXPOSE 8000\n"
        'CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]\n```'
    ),
]

SYSTEM = (
    "Write a single dense paragraph describing this file. "
    "Include: what it does, key exports, and dependencies. "
    "No headers or bullet points."
)


async def bench_one(
    client: httpx.AsyncClient, model: str, prompt: str, sem: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0,
    }
    async with sem:
        t0 = time.monotonic()
        resp = await client.post("/v1/chat/completions", json=payload)
        elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code, "elapsed": elapsed}

    data = resp.json()
    usage = data.get("usage", {})
    return {
        "ok": True,
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "content": (data["choices"][0]["message"].get("content") or "")[:80],
    }


async def run_bench(url: str, model: str, concurrency: int, num_requests: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=120.0) as client:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
        t0 = time.monotonic()
        results = await asyncio.gather(*[bench_one(client, model, p, sem) for p in prompts])
        wall_time = time.monotonic() - t0

    ok = [r for r in results if r["ok"]]
    failed = len(results) - len(ok)
    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_completion = sum(r["completion_tokens"] for r in ok)
    total_tokens = total_prompt + total_completion

    avg_latency = sum(r["elapsed"] for r in ok) / len(ok) if ok else 0
    p50 = sorted(r["elapsed"] for r in ok)[len(ok) // 2] if ok else 0
    p99 = sorted(r["elapsed"] for r in ok)[int(len(ok) * 0.99)] if ok else 0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time_s": round(wall_time, 2),
        "ok": len(ok),
        "failed": failed,
        "total_tokens": total_tokens,
        "total_completion": total_completion,
        "tok_per_s": round(total_tokens / wall_time, 1),
        "completion_tok_per_s": round(total_completion / wall_time, 1),
        "avg_latency_s": round(avg_latency, 3),
        "p50_latency_s": round(p50, 3),
        "p99_latency_s": round(p99, 3),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if omitted)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 25, 50])
    parser.add_argument("--requests", type=int, default=50, help="Requests per concurrency level")
    args = parser.parse_args()

    # Auto-detect model
    model = args.model
    if not model:
        async with httpx.AsyncClient(base_url=args.url, timeout=10) as c:
            r = await c.get("/v1/models")
            model = r.json()["data"][0]["id"]
    print(f"Model: {model}")
    print(f"URL: {args.url}")
    print()

    # Warmup
    print("Warming up...")
    await run_bench(args.url, model, 1, 2)

    print(f"{'concurrency':>11} {'ok':>4} {'fail':>4} {'wall_s':>7} {'tok/s':>7} "
          f"{'comp_tok/s':>10} {'avg_lat':>8} {'p50_lat':>8} {'p99_lat':>8}")
    print("-" * 80)

    for c in args.concurrency:
        r = await run_bench(args.url, model, c, args.requests)
        print(f"{r['concurrency']:>11} {r['ok']:>4} {r['failed']:>4} {r['wall_time_s']:>7.1f} "
              f"{r['tok_per_s']:>7.1f} {r['completion_tok_per_s']:>10.1f} "
              f"{r['avg_latency_s']:>8.3f} {r['p50_latency_s']:>8.3f} {r['p99_latency_s']:>8.3f}")


if __name__ == "__main__":
    asyncio.run(main())
