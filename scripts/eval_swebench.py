#!/usr/bin/env python3
"""SWE-bench evaluation harness.

Three commands:
  generate  — Run student model on SWE-bench tasks, produce predictions.jsonl
  evaluate  — Run swebench.harness.run_evaluation on predictions
  ablation  — Knowledge source ablation (all / fs+git / fs-only / no-bgkit)

The generation step runs an interactive agent loop per instance:
  1. Checkout repo at base_commit via pygit2
  2. Load BgKIT context from pre-computed cache
  3. Present issue + context to student model
  4. Student generates tool-call sequences (read_file / edit_file / done)
  5. Parse and execute each tool call against the working tree
  6. Feed results back, repeat until "done" or turn limit
  7. Extract cumulative diff as model_patch

Usage:
    python scripts/eval_swebench.py generate \
        --checkpoint checkpoints/phase3_best \
        --repos-dir data/swe_repos \
        --output predictions.jsonl \
        --subset lite

    python scripts/eval_swebench.py evaluate \
        --predictions predictions.jsonl

    python scripts/eval_swebench.py ablation \
        --checkpoint checkpoints/phase3_best \
        --repos-dir data/swe_repos \
        --output-dir ablation_results
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Tool-call protocol: the student model emits structured tool calls that we
# parse and execute against a working-tree checkout.
# ---------------------------------------------------------------------------

# Regex pattern for tool-call parsing in generated text.
# The model is trained on Qwen3.5's native tool-call format, emitting
# JSON inside <tool_call> tags:
#   <tool_call>
#   {"name": "read_file", "arguments": {"path": "src/foo.py"}}
#   </tool_call>

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from generated text.

    Parses Qwen3.5's native JSON tool-call format:
        <tool_call>
        {"name": "read_file", "arguments": {"path": "src/foo.py"}}
        </tool_call>
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"name": name, "arguments": arguments})
    return calls


def _execute_tool(tool: dict, work_dir: Path) -> str:
    """Execute a tool call against the working tree.

    Returns the tool result string to feed back to the model.
    """
    name = tool["name"]
    args = tool["arguments"]

    if name == "read_file":
        fpath = work_dir / args.get("path", "")
        if not fpath.exists():
            return f"Error: file not found: {args.get('path', '')}"
        if not fpath.is_file():
            return f"Error: not a file: {args.get('path', '')}"
        try:
            content = fpath.read_text(errors="replace")
            # Truncate very long files
            if len(content) > 50_000:
                content = content[:50_000] + "\n... (truncated)"
            return content
        except Exception as exc:
            return f"Error reading file: {exc}"

    elif name == "edit_file":
        fpath = work_dir / args.get("path", "")
        old = args.get("old", "")
        new = args.get("new", "")
        if not fpath.exists():
            # Creating a new file
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(new)
            return f"Created {args.get('path', '')}"
        try:
            content = fpath.read_text()
            if old and old in content:
                content = content.replace(old, new, 1)
                fpath.write_text(content)
                return f"Edited {args.get('path', '')}"
            elif not old:
                # Append or overwrite
                fpath.write_text(new)
                return f"Wrote {args.get('path', '')}"
            else:
                return f"Error: old string not found in {args.get('path', '')}"
        except Exception as exc:
            return f"Error editing file: {exc}"

    elif name == "list_files":
        target = work_dir / args.get("path", "")
        if not target.exists():
            return f"Error: path not found: {args.get('path', '')}"
        if target.is_file():
            return str(args.get("path", ""))
        files = sorted(str(p.relative_to(work_dir)) for p in target.rglob("*") if p.is_file())
        return "\n".join(files[:500])

    elif name == "run_command":
        cmd = args.get("command", "")
        if not cmd:
            return "Error: empty command"
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(work_dir),
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if len(output) > 10_000:
                output = output[:10_000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as exc:
            return f"Error running command: {exc}"

    elif name == "done":
        return "__DONE__"

    else:
        return f"Error: unknown tool '{name}'"


def _extract_diff(work_dir: Path) -> str:
    """Extract the cumulative diff from a working tree."""
    try:
        result = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=str(work_dir), capture_output=True, text=True, timeout=10,
        )
        diff = result.stdout

        # Also include untracked files as new-file diffs
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(work_dir), capture_output=True, text=True, timeout=10,
        )
        for fpath in untracked.stdout.strip().splitlines():
            if not fpath:
                continue
            try:
                content = (work_dir / fpath).read_text(errors="replace")
                diff += f"\ndiff --git a/{fpath} b/{fpath}\n"
                diff += "new file mode 100644\n"
                diff += "--- /dev/null\n"
                diff += f"+++ b/{fpath}\n"
                lines = content.splitlines(keepends=True)
                diff += f"@@ -0,0 +1,{len(lines)} @@\n"
                for line in lines:
                    diff += f"+{line}"
                if lines and not lines[-1].endswith("\n"):
                    diff += "\n\\ No newline at end of file\n"
            except Exception:
                pass
        return diff
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Interactive generation loop
# ---------------------------------------------------------------------------


def _checkout_repo(repo_path: Path, base_commit: str, work_dir: Path) -> bool:
    """Clone/checkout repo at base_commit into work_dir."""
    try:
        import pygit2

        # Clone from local repo
        repo = pygit2.clone_repository(str(repo_path), str(work_dir))
        commit = repo.revparse_single(base_commit)
        repo.checkout_tree(commit.peel(pygit2.Tree))
        repo.set_head(commit.id)
        return True
    except Exception:
        # Fallback: git clone + checkout
        try:
            subprocess.run(
                ["git", "clone", str(repo_path), str(work_dir)],
                capture_output=True, timeout=60,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(work_dir), capture_output=True, timeout=10,
            )
            return True
        except Exception:
            return False


def _load_bgkit_context(
    instance_id: str,
    bgkit_cache_dir: str | None,
    device,
):
    """Load pre-computed BgKIT context for an instance."""
    if not bgkit_cache_dir:
        return None, None

    cache_path = Path(bgkit_cache_dir)
    if not cache_path.exists():
        return None, None

    try:
        from bgkit.data.datasets.precomputed_l0_cache import PrecomputedL0Cache

        import torch

        cache = PrecomputedL0Cache(str(cache_path))
        survivors = cache.get_survivors(instance_id)
        survivors = survivors.unsqueeze(0).to(device)  # (1, K, D)
        mask = torch.ones(1, survivors.size(1), dtype=torch.bool, device=device)
        return survivors, mask
    except (KeyError, FileNotFoundError):
        return None, None


def _build_prompt(issue_text: str, conversation: list[dict]) -> str:
    """Build the model prompt from issue + conversation history."""
    parts = [
        "You are a coding assistant. Fix the issue described below by reading "
        "and editing files in the repository.\n\n"
        "Available tools (use JSON inside <tool_call> tags):\n"
        "- read_file: "
        '<tool_call>{"name": "read_file", '
        '"arguments": {"path": "..."}}</tool_call>\n'
        "- edit_file: "
        '<tool_call>{"name": "edit_file", '
        '"arguments": {"path": "...", "old": "...", '
        '"new": "..."}}</tool_call>\n'
        "- list_files: "
        '<tool_call>{"name": "list_files", '
        '"arguments": {"path": "..."}}</tool_call>\n'
        "- run_command: "
        '<tool_call>{"name": "run_command", '
        '"arguments": {"command": "..."}}</tool_call>\n'
        "- done: "
        '<tool_call>{"name": "done", '
        '"arguments": {}}</tool_call>\n\n'
        f"Issue:\n{issue_text}\n",
    ]

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"\n[{role}]\n{content}")

    return "\n".join(parts)


def generate_prediction(
    instance: dict,
    model,
    tokenizer,
    device,
    repos_dir: Path,
    bgkit_cache_dir: str | None = None,
    max_turns: int = 30,
    max_new_tokens: int = 2048,
) -> dict:
    """Run the interactive agent loop for a single SWE-bench instance."""
    import torch

    instance_id = instance["instance_id"]
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    issue_text = instance.get("problem_statement", instance.get("text", ""))

    # Find repo path
    repo_path = repos_dir / repo.replace("/", "_")
    if not repo_path.exists():
        repo_path = repos_dir / repo
    if not repo_path.exists():
        return {
            "instance_id": instance_id,
            "model_name_or_path": "bgkit",
            "model_patch": "",
            "error": f"repo not found: {repo}",
        }

    # Create temp working directory and checkout
    with tempfile.TemporaryDirectory(prefix=f"swe_{instance_id}_") as tmpdir:
        work_dir = Path(tmpdir)
        if not _checkout_repo(repo_path, base_commit, work_dir):
            return {
                "instance_id": instance_id,
                "model_name_or_path": "bgkit",
                "model_patch": "",
                "error": f"checkout failed: {base_commit}",
            }

        # Load BgKIT context
        bgkit_survivors, bgkit_mask = _load_bgkit_context(
            instance_id, bgkit_cache_dir, device,
        )

        # Interactive loop
        conversation: list[dict] = []
        stats = {"turns": 0, "reads": 0, "edits": 0}

        for turn in range(max_turns):
            prompt_text = _build_prompt(issue_text, conversation)
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

            # Truncate if too long
            max_ctx = getattr(model.config, "max_position_embeddings", 32768) - max_new_tokens
            if input_ids.size(1) > max_ctx:
                input_ids = input_ids[:, -max_ctx:]

            # Generate
            with torch.no_grad():
                if bgkit_survivors is not None:
                    # Prefix injection: [bgkit_context | input_tokens]
                    input_embeds = model.get_input_embeddings()(input_ids)
                    combined_embeds = torch.cat([bgkit_survivors, input_embeds], dim=1)
                    combined_mask = torch.cat([
                        bgkit_mask,
                        torch.ones_like(input_ids, dtype=torch.bool),
                    ], dim=1)
                    output_ids = model.generate(
                        inputs_embeds=combined_embeds,
                        attention_mask=combined_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    # Strip input positions
                    new_ids = output_ids[:, combined_embeds.size(1):]
                else:
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    new_ids = output_ids[:, input_ids.size(1):]

            response_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
            conversation.append({"role": "assistant", "content": response_text})

            # Parse and execute tool calls
            tool_calls = _parse_tool_calls(response_text)
            if not tool_calls:
                # Model didn't emit a tool call — treat as done
                break

            stats["turns"] += 1
            tool_results = []
            done = False
            for tool in tool_calls:
                result = _execute_tool(tool, work_dir)
                if result == "__DONE__":
                    done = True
                    break
                tool_results.append(f"[{tool['name']}] {result}")
                if tool["name"] == "read_file":
                    stats["reads"] += 1
                elif tool["name"] == "edit_file":
                    stats["edits"] += 1

            if done:
                break

            # Feed results back
            result_text = "\n\n".join(tool_results)
            conversation.append({"role": "tool", "content": result_text})

        # Extract diff
        model_patch = _extract_diff(work_dir)

    return {
        "instance_id": instance_id,
        "model_name_or_path": "bgkit",
        "model_patch": model_patch,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_generate(args) -> None:
    """Generate predictions for SWE-bench tasks."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_name = {
        "lite": "SWE-bench/SWE-bench_Lite",
        "verified": "SWE-bench/SWE-bench_Verified",
        "full": "SWE-bench/SWE-bench",
    }.get(args.subset, args.subset)

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="test")
    if args.max_instances:
        ds = ds.select(range(min(args.max_instances, len(ds))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if (checkpoint_path / "metadata.json").exists():
        # BgKIT checkpoint format — load decoder backbone
        from bgkit.training.checkpointing import load_checkpoint

        _meta, state_dicts = load_checkpoint(checkpoint_path)
        model_state = state_dicts.get("model", {})
        decoder_state = {
            k.replace("decoder.backbone.", "", 1): v
            for k, v in model_state.items()
            if k.startswith("decoder.backbone.")
        }
        model_name = args.model_name or "Qwen/Qwen3.5-0.8B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        if decoder_state:
            model.load_state_dict(decoder_state, strict=False)
    else:
        # HF model path
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path), trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name or "Qwen/Qwen3.5-0.8B", trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    repos_dir = Path(args.repos_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-predicted instances
    done_ids = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line).get("instance_id"))
        print(f"Resuming: {len(done_ids)} instances already done")

    print(f"Generating predictions for {len(ds)} instances...")
    with output_path.open("a") as fout:
        for i, instance in enumerate(ds):
            iid = instance["instance_id"]
            if iid in done_ids:
                continue

            prediction = generate_prediction(
                instance=instance,
                model=model,
                tokenizer=tokenizer,
                device=device,
                repos_dir=repos_dir,
                bgkit_cache_dir=args.bgkit_cache,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
            )
            fout.write(json.dumps(prediction, default=str) + "\n")
            fout.flush()

            has_patch = bool(prediction.get("model_patch", "").strip())
            stats = prediction.get("stats", {})
            print(
                f"  [{i + 1}/{len(ds)}] {iid}: "
                f"{'patch' if has_patch else 'empty'}, "
                f"{stats.get('turns', 0)} turns, "
                f"{stats.get('edits', 0)} edits"
            )

    print(f"Predictions saved to {output_path}")


def cmd_evaluate(args) -> None:
    """Evaluate predictions using swebench harness."""
    try:
        from swebench.harness.run_evaluation import main as swebench_main
    except ImportError:
        print(
            "swebench not installed. Install with: pip install swebench",
            file=sys.stderr,
        )
        _analyze_predictions(str(args.predictions))
        return

    print(f"Running SWE-bench evaluation on {args.predictions}...")
    eval_args = [
        "--dataset_name", args.dataset_name,
        "--predictions_path", str(args.predictions),
        "--run_id", args.run_id,
        "--max_workers", str(args.max_workers),
    ]
    if args.namespace:
        eval_args.extend(["--namespace", args.namespace])

    sys.argv = ["swebench_eval"] + eval_args
    swebench_main()


def _analyze_predictions(predictions_path: str) -> None:
    """Basic prediction analysis without swebench harness."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    total = len(predictions)
    with_patch = sum(1 for p in predictions if p.get("model_patch", "").strip())
    empty = total - with_patch

    print("Prediction analysis:")
    print(f"  Total: {total}")
    print(f"  With patch: {with_patch}")
    print(f"  Empty: {empty}")

    if with_patch > 0:
        avg_patch_lines = sum(
            len(p["model_patch"].splitlines())
            for p in predictions if p.get("model_patch", "").strip()
        ) / with_patch
        print(f"  Avg patch lines: {avg_patch_lines:.1f}")

    # Stats summary
    all_stats = [p.get("stats", {}) for p in predictions if p.get("stats")]
    if all_stats:
        avg_turns = sum(s.get("turns", 0) for s in all_stats) / len(all_stats)
        avg_reads = sum(s.get("reads", 0) for s in all_stats) / len(all_stats)
        avg_edits = sum(s.get("edits", 0) for s in all_stats) / len(all_stats)
        print(f"  Avg turns: {avg_turns:.1f}")
        print(f"  Avg reads: {avg_reads:.1f}")
        print(f"  Avg edits: {avg_edits:.1f}")

    errors = [p for p in predictions if p.get("error")]
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e['instance_id']}: {e['error']}")


def cmd_ablation(args) -> None:
    """Knowledge source ablation study."""
    conditions = {
        "all": {"bgkit_cache": args.bgkit_cache, "desc": "All BgKIT context"},
        "no_bgkit": {"bgkit_cache": None, "desc": "No BgKIT context"},
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for condition, cfg in conditions.items():
        print(f"\n=== Ablation: {condition} — {cfg['desc']} ===")
        pred_path = output_dir / f"predictions_{condition}.jsonl"

        # Build args for generate
        gen_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            repos_dir=args.repos_dir,
            output=str(pred_path),
            subset=args.subset,
            max_instances=args.max_instances,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            model_name=args.model_name,
            bgkit_cache=cfg["bgkit_cache"],
        )
        cmd_generate(gen_args)

    # Compare results
    report = {}
    for condition in conditions:
        pred_path = output_dir / f"predictions_{condition}.jsonl"
        if pred_path.exists():
            preds = []
            with pred_path.open() as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))
            with_patch = sum(1 for p in preds if p.get("model_patch", "").strip())
            report[condition] = {
                "total": len(preds),
                "with_patch": with_patch,
                "patch_rate": with_patch / max(len(preds), 1),
            }

    report_path = output_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nAblation report: {report_path}")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate predictions")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--repos-dir", required=True)
    gen.add_argument("--output", default="predictions.jsonl")
    gen.add_argument("--subset", default="lite", choices=["lite", "verified", "full"])
    gen.add_argument("--max-instances", type=int, default=None)
    gen.add_argument("--max-turns", type=int, default=30)
    gen.add_argument("--max-new-tokens", type=int, default=2048)
    gen.add_argument("--model-name", default=None)
    gen.add_argument("--bgkit-cache", default=None)

    ev = sub.add_parser("evaluate", help="Evaluate predictions")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Verified")
    ev.add_argument("--run-id", default="bgkit-eval")
    ev.add_argument("--max-workers", type=int, default=4)
    ev.add_argument("--namespace", default="")

    ab = sub.add_parser("ablation", help="Knowledge source ablation")
    ab.add_argument("--checkpoint", required=True)
    ab.add_argument("--repos-dir", required=True)
    ab.add_argument("--output-dir", default="ablation_results")
    ab.add_argument("--subset", default="lite", choices=["lite", "verified", "full"])
    ab.add_argument("--max-instances", type=int, default=None)
    ab.add_argument("--max-turns", type=int, default=30)
    ab.add_argument("--max-new-tokens", type=int, default=2048)
    ab.add_argument("--model-name", default=None)
    ab.add_argument("--bgkit-cache", default=None)

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "ablation":
        cmd_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
