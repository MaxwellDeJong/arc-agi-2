"""Orchestrator: dispatches Gemini CLI agents to E2B sandboxes.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching agents to E2B sandboxes with Gemini CLI pre-installed
- Writing logs (raw_stream, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)

Each agent runs in its own E2B Firecracker microVM with network restricted
to generativelanguage.googleapis.com via SNI domain filtering.

Usage:
  uv run python orchestrator.py --tasks 0934a4d8 --num-agents 1
  uv run python orchestrator.py --tasks all --num-agents 3
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Force line-buffered stdout so background runs show live progress
sys.stdout.reconfigure(line_buffering=True)

# --- Paths ---
ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"
AGENT_RUNNER_PATH = ROOT / "agent_runner.py"


# ── E2B Sandbox Helpers ───────────────────────────────────────────────────

# E2B resource configuration
E2B_CPU_COUNT = 2               # vCPUs per sandbox
E2B_COST_PER_VCPU_HOUR = 0.05  # E2B pricing: $0.05/hour per vCPU

# Status event formatters for pretty-printing agent status events
_EVENT_FORMATTERS: dict[str, Callable[[dict], str]] = {
    "started": lambda e: f"started (model={e.get('model', '?')})",
    "iteration": lambda e: f"iteration {e.get('iteration', '?')}/{e.get('max_iterations', '?')}",
    "transform_validation": lambda e: f"transform {'PASS' if e.get('all_pass') else 'FAIL'} (iter {e.get('iteration', '?')})",
    "submitted": lambda e: f"submit #{e.get('attempt', '?')}",
    "done": lambda e: f"done — {e.get('attempts', 0)} attempts, {e.get('elapsed', '?')}s",
    "results_written": lambda e: "results written",
    "error": lambda e: f"ERROR: {e.get('msg', '')}",
}


async def run_agent_in_e2b(
    task_id: str,
    agent_id: str,
    raw_task: dict,
    test_index: int,
    model: str,
    max_iterations: int,
    soft_training_feedback: bool,
) -> dict:
    """Run a Gemini CLI agent inside an E2B Firecracker sandbox.

    Network isolation uses E2B's SNI-based domain filtering: all outbound traffic
    is denied by default, with an allowlist for Google API endpoints only.
    """
    from e2b import ALL_TRAFFIC, AsyncSandbox

    envs: dict[str, str] = {}
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        envs["GEMINI_API_KEY"] = gemini_key

    config = {
        "task_id": task_id,
        "agent_id": agent_id,
        "raw_task": raw_task,
        "test_index": test_index,
        "model": model,
        "max_iterations": max_iterations,
        "soft_training_feedback": soft_training_feedback,
    }

    network = {
        "deny_out": [ALL_TRAFFIC],
        "allow_out": ["generativelanguage.googleapis.com"],
    }

    # Retry sandbox creation with exponential backoff (handles concurrency limits)
    sandbox = None
    sandbox_start = time.time()
    for attempt in range(5):
        try:
            sandbox = await AsyncSandbox.create(
                template="arc-gemini-solver",
                envs=envs,
                network=network,
                timeout=43500,  # sandbox lifetime: 12h + 5min buffer
            )
            sandbox_start = time.time()
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
            print(f"  [e2b] {agent_id}: sandbox create failed (attempt {attempt + 1}/5), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)

    try:
        # Write config to /root/ — deleted by agent_runner before starting Gemini
        await sandbox.files.write("/root/config.json", json.dumps(config))
        # Upload agent_runner.py into sandbox
        await sandbox.files.write("/app/agent_runner.py", AGENT_RUNNER_PATH.read_text())

        # stdout/stderr callbacks for live status
        def on_stdout(output) -> None:
            line = output.line if hasattr(output, "line") else str(output)
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    evt_type = event.get("event", "?")
                    aid = event.get("agent_id", "?")
                    formatter = _EVENT_FORMATTERS.get(evt_type)
                    if formatter:
                        detail = formatter(event)
                        print(f"  [status] {aid}: {detail}")
            except (json.JSONDecodeError, TypeError):
                pass

        def on_stderr(output) -> None:
            line = output.line if hasattr(output, "line") else str(output)
            if line.strip():
                print(f"  [e2b-stderr] {agent_id}: {line[:200]}")

        await sandbox.commands.run(
            "python3 /app/agent_runner.py",
            user="root",
            timeout=43200 + 120,  # 12h + 2min buffer
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )

        # Read results
        results_content = await sandbox.files.read("/workspace/results.json")
        result = json.loads(results_content)

        # Calculate E2B cost
        sandbox_duration = time.time() - sandbox_start
        e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR

        # Add E2B cost tracking to results
        result["e2b_cost"] = e2b_cost
        result["e2b_duration"] = sandbox_duration
        result["total_cost"] = result.get("cost", 0) + e2b_cost

        print(f"  [e2b-cost] {agent_id}: Gemini=${result.get('cost', 0):.4f}, E2B=${e2b_cost:.4f}, Total=${result['total_cost']:.4f}, Duration={sandbox_duration:.1f}s")

        return result

    except Exception as e:
        sandbox_duration = time.time() - sandbox_start
        e2b_cost = (sandbox_duration / 3600) * E2B_CPU_COUNT * E2B_COST_PER_VCPU_HOUR

        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": [],
            "elapsed": 0,
            "cost": 0,
            "e2b_cost": e2b_cost,
            "e2b_duration": sandbox_duration,
            "total_cost": e2b_cost,
            "turns": 0,
            "error": f"E2B sandbox error: {e}",
            "raw_lines": [],
            "stderr": "",
        }
    finally:
        await sandbox.kill()


# ── Task Loading ────────────────────────────────────────────────────────────

_ALL_TASKS: dict[str, dict] | None = None


def _load_all_tasks() -> dict[str, dict]:
    """Load challenges into {task_id: {train, test}} (cached)."""
    global _ALL_TASKS
    if _ALL_TASKS is None:
        if not CHALLENGES_FILE.exists():
            raise FileNotFoundError(f"Challenges file not found: {CHALLENGES_FILE}")
        challenges = json.loads(CHALLENGES_FILE.read_text())
        _ALL_TASKS = challenges
    return _ALL_TASKS


def load_task_ids(tasks_arg: str) -> list[str]:
    """Parse --tasks argument into list of task IDs."""
    if tasks_arg == "all":
        return sorted(_load_all_tasks().keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def load_task_json(task_id: str) -> dict:
    """Load a single task from challenges."""
    all_tasks = _load_all_tasks()
    if task_id not in all_tasks:
        raise KeyError(f"Task {task_id} not found")
    return all_tasks[task_id]


# ── Transcript Parsing ──────────────────────────────────────────────────────

_TOOL_NAME_MAP: dict[str, str] = {
    "run_shell_command": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "write_new_file": "Write",
    "edit_file": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list_directory": "Glob",
}


def _map_tool_params(gemini_name: str, params: dict) -> dict:
    """Map Gemini CLI tool parameters to viewer-compatible format."""
    if gemini_name == "run_shell_command":
        return {"command": params.get("command", ""), "description": params.get("description", "")}
    if gemini_name == "read_file":
        return {"file_path": params.get("file_path", "")}
    if gemini_name in ("write_file", "write_new_file"):
        return {"file_path": params.get("file_path", ""), "content": params.get("content", "")}
    if gemini_name == "edit_file":
        return {
            "file_path": params.get("file_path", ""),
            "old_string": params.get("old_string", ""),
            "new_string": params.get("new_string", ""),
            "replace_all": params.get("replace_all", False),
        }
    if gemini_name == "glob":
        return {"pattern": params.get("pattern", "")}
    if gemini_name == "grep":
        return {"pattern": params.get("pattern", ""), "path": params.get("path", "")}
    if gemini_name == "list_directory":
        return {"pattern": params.get("dir_path", "") + "/*"}
    return params


def parse_gemini_stream_json(raw_lines: list[str], task_id: str) -> list[dict]:
    """Transform Gemini stream-json JSONL into viewer-compatible transcript entries."""
    entries: list[dict] = []
    turn_counter = 0
    current_blocks: list[dict] = []
    pending_text = ""

    def flush_text():
        nonlocal pending_text
        if pending_text.strip():
            current_blocks.append({"type": "text", "text": pending_text.strip()})
        pending_text = ""

    def flush_assistant():
        nonlocal current_blocks, turn_counter
        if current_blocks:
            turn_counter += 1
            entries.append({
                "type": "assistant",
                "turn": turn_counter,
                "content": current_blocks,
            })
            current_blocks = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt_type = obj.get("type", "")

        if evt_type == "message":
            role = obj.get("role", "")
            content = obj.get("content", "")
            is_delta = obj.get("delta", False)

            if role == "assistant":
                if is_delta:
                    pending_text += content
                else:
                    flush_text()
                    if content.strip():
                        current_blocks.append({"type": "text", "text": content.strip()})

        elif evt_type == "tool_use":
            flush_text()

            gemini_name = obj.get("tool_name", "")
            tool_id = obj.get("tool_id", "")
            params = obj.get("parameters", {})

            viewer_name = _TOOL_NAME_MAP.get(gemini_name, gemini_name)
            viewer_params = _map_tool_params(gemini_name, params)

            if gemini_name == "run_shell_command" and "submit.py" in params.get("command", ""):
                cmd = params.get("command", "")
                grid = _extract_grid_from_submit_cmd(cmd)
                if grid is not None:
                    current_blocks.append({
                        "type": "tool_use",
                        "name": "submit",
                        "id": tool_id,
                        "input": {"output": grid, "test_index": 0},
                    })
                else:
                    current_blocks.append({
                        "type": "tool_use",
                        "name": viewer_name,
                        "id": tool_id,
                        "input": viewer_params,
                    })
            else:
                current_blocks.append({
                    "type": "tool_use",
                    "name": viewer_name,
                    "id": tool_id,
                    "input": viewer_params,
                })

        elif evt_type == "tool_result":
            flush_text()
            flush_assistant()

            tool_id = obj.get("tool_id", "")
            status = obj.get("status", "")
            output = obj.get("output", "")
            is_error = status == "error"

            if isinstance(output, str) and len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"

            entries.append({
                "type": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                    **({"is_error": True} if is_error else {}),
                }],
            })

        elif evt_type == "result":
            flush_text()
            flush_assistant()

            stats = obj.get("stats", {})
            entries.append({
                "type": "result",
                "cost": 0,
                "num_turns": turn_counter,
                "usage": {
                    "input_tokens": stats.get("input_tokens", 0),
                    "output_tokens": stats.get("output_tokens", 0),
                    "total_tokens": stats.get("total_tokens", 0),
                    "cached_tokens": stats.get("cached", 0),
                },
            })

    flush_text()
    flush_assistant()

    return entries


def _extract_grid_from_submit_cmd(cmd: str) -> list[list[int]] | None:
    """Try to extract a 2D grid from a submit.py command string."""
    match = re.search(r"submit\.py\s+['\"]?(\[.+\])['\"]?\s*$", cmd)
    if match:
        try:
            grid = json.loads(match.group(1))
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                return grid
        except json.JSONDecodeError:
            pass
    return None


# ── Log Writing (local) ────────────────────────────────────────────────────

def write_agent_logs(
    result: dict,
    task_id: str,
    log_dir: Path,
) -> None:
    """Write log files from result's raw_lines."""
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_lines: list[str] = result.get("raw_lines", [])

    # raw_stream.jsonl
    raw_stream_path = log_dir / "raw_stream.jsonl"
    with open(raw_stream_path, "w") as f:
        for line in raw_lines:
            f.write(line + "\n")

    # transcript.jsonl (viewer-compatible)
    transcript_entries = parse_gemini_stream_json(raw_lines, task_id)
    transcript_path = log_dir / "transcript.jsonl"
    with open(transcript_path, "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")

    # readable.md
    readable_path = log_dir / "readable.md"
    with open(readable_path, "w") as rf:
        agent_id = result.get("agent_id", "unknown")
        test_index = result.get("test_index", 0)
        rf.write(f"# Agent Log: {agent_id} (task: {task_id}, test: {test_index})\n\n")

        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rf.write(f"[raw] {line}\n")
                continue

            evt_type = obj.get("type", "")

            if evt_type == "message" and obj.get("role") == "assistant":
                content = obj.get("content", "")
                if obj.get("delta"):
                    rf.write(content)
                else:
                    rf.write(f"\n**Assistant:**\n{content}\n\n")

            elif evt_type == "tool_use":
                tool_name = obj.get("tool_name", "")
                params = obj.get("parameters", {})
                if tool_name == "run_shell_command":
                    rf.write(f"\n\n**Tool: {tool_name}**\n```\n$ {params.get('command', '')}\n```\n\n")
                else:
                    input_str = json.dumps(params, indent=2)[:500]
                    rf.write(f"\n\n**Tool: {tool_name}**\n```\n{input_str}\n```\n\n")

            elif evt_type == "tool_result":
                output = obj.get("output", "")[:2000]
                status = obj.get("status", "")
                rf.write(f"**Tool Result ({status}):**\n```\n{output}\n```\n\n")

            elif evt_type == "result":
                stats = obj.get("stats", {})
                rf.write(
                    f"---\n**Result:** "
                    f"tokens={stats.get('total_tokens', '?')}, "
                    f"duration={stats.get('duration_ms', '?')}ms, "
                    f"tool_calls={stats.get('tool_calls', '?')}\n"
                )

    # attempts.jsonl
    attempts_path = log_dir / "attempts.jsonl"
    with open(attempts_path, "w") as f:
        for attempt in result.get("attempts", []):
            f.write(json.dumps(attempt) + "\n")

    # stderr.log (if any)
    stderr = result.get("stderr", "")
    if stderr:
        (log_dir / "stderr.log").write_text(stderr)

    # error.log (if any)
    if "error" in result:
        (log_dir / "error.log").write_text(result["error"])


# ── Retry with exponential backoff ────────────────────────────────────────

MAX_RETRIES = 10
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 600.0  # 10 min cap

logger = logging.getLogger(__name__)


async def _retry_e2b_call(coro_fn, *, agent_id: str) -> dict:
    """Call an async function with exponential backoff + jitter on transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(kw in err_str for kw in (
                "deadline exceeded", "unavailable", "connection",
                "timeout", "reset by peer", "broken pipe",
                "eof", "transport", "503", "502",
                "429", "rate limit", "resource_exhausted",
                "overloaded", "too many requests",
                "stopped or disabled",
            ))
            if not is_transient or attempt == MAX_RETRIES:
                raise
            backoff = min(INITIAL_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
            jitter = random.uniform(0, backoff * 0.5)
            wait = backoff + jitter
            logger.warning(
                f"[{agent_id}] Attempt {attempt}/{MAX_RETRIES} failed: {e} — "
                f"retrying in {wait:.1f}s"
            )
            print(
                f"  retry {agent_id} attempt {attempt}/{MAX_RETRIES} failed "
                f"({type(e).__name__}), retrying in {wait:.0f}s..."
            )
            await asyncio.sleep(wait)

    # Unreachable, but satisfies type checker
    raise RuntimeError(f"[{agent_id}] All {MAX_RETRIES} retries exhausted")


# ── Incremental Result Writing ──────────────────────────────────────────────

def _write_agent_result(run_dir: Path, task_id: str, agent_id: str, agent_data: dict) -> None:
    """Atomically write/update a single agent's result into the task file.

    This provides a safety net for early termination: if the process is killed
    mid-run, tasks where some-but-not-all agents completed will still have
    those agents' grids on disk for submission.py to use.

    No lock needed: asyncio is single-threaded, no await between read and write.
    """
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    task_file = task_results_dir / f"{task_id}.json"
    tmp_file = task_results_dir / f"{task_id}.json.tmp"

    # Read existing data or start fresh
    if task_file.exists():
        try:
            data = json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError):
            data = {"agents": {}}
    else:
        data = {"agents": {}}

    # Add/update this agent's entry
    data.setdefault("agents", {})[agent_id] = agent_data

    # Atomic write: tmp then rename
    tmp_file.write_text(json.dumps(data, indent=2))
    os.rename(str(tmp_file), str(task_file))


# ── Per-Task Orchestration ──────────────────────────────────────────────────

async def process_task(
    task_id: str,
    args: argparse.Namespace,
    run_dir: Path,
    backend_queue: asyncio.Queue[str] | None,
) -> dict:
    """Orchestrate N agents per test input via E2B, save results independently."""
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])

    agent_metas: list[tuple[str, int, Path]] = []  # (agent_id, test_index, log_dir)

    async def _dispatch(agent_id: str, kwargs: dict, test_index: int, log_dir: Path) -> dict:
        """Acquire a backend slot, run the agent, release the slot.

        Writes logs and incremental results immediately when the agent finishes,
        before waiting for other agents (safety net for early termination).
        """
        if backend_queue is None:
            result = await _retry_e2b_call(
                lambda kw=kwargs: run_agent_in_e2b(**kw),
                agent_id=agent_id,
            )
        else:
            token = await backend_queue.get()
            try:
                result = await _retry_e2b_call(
                    lambda kw=kwargs: run_agent_in_e2b(**kw),
                    agent_id=agent_id,
                )
            finally:
                backend_queue.put_nowait(token)

        # Write logs immediately so partial results survive early termination
        if not isinstance(result, Exception):
            write_agent_logs(result, task_id, log_dir)

            # Write incremental result for early-termination safety
            attempts = result.get("attempts", [])
            agent_data = {
                "test_index": test_index,
                "attempts": [a["grid"] for a in attempts],
                "cost": result.get("cost", 0),
                "e2b_cost": result.get("e2b_cost", 0),
                "e2b_duration": result.get("e2b_duration", 0),
                "total_cost": result.get("total_cost", 0),
                "turns": result.get("turns", 0),
                "usage": result.get("usage", {}),
            }
            _write_agent_result(run_dir, task_id, agent_id, agent_data)

        return result

    agent_coros: list = []

    for ti in range(num_tests):
        for ei in range(args.num_agents):
            agent_id = f"{task_id}_ens{ei}_t{ti}"
            agent_log_dir = run_dir / "logs" / task_id / f"t{ti}" / f"agent{ei}"
            agent_metas.append((agent_id, ti, agent_log_dir))

            _kwargs = dict(
                task_id=task_id,
                agent_id=agent_id,
                raw_task=raw_task,
                test_index=ti,
                model=args.model,
                max_iterations=args.max_iterations,
                soft_training_feedback=args.soft_training_feedback,
            )
            agent_coros.append(_dispatch(agent_id, _kwargs, ti, agent_log_dir))

    agent_results = await asyncio.gather(*agent_coros, return_exceptions=True)

    # Collect per-agent results (logs already written in _dispatch)
    per_agent: dict[str, dict] = {}
    submitted_tests: set[int] = set()

    for (agent_id, ti, log_dir), result in zip(agent_metas, agent_results):
        if isinstance(result, Exception):
            per_agent[agent_id] = {
                "test_index": ti, "attempts": [], "error": str(result),
            }
            # Write error log
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "error.log").write_text(str(result))
            continue

        attempts = result.get("attempts", [])
        has_grid = any(a.get("grid") is not None for a in attempts)
        if has_grid:
            submitted_tests.add(ti)
        per_agent[agent_id] = {
            "test_index": ti,
            "attempts": [a["grid"] for a in attempts],
            "cost": result.get("cost", 0),
            "e2b_cost": result.get("e2b_cost", 0),
            "e2b_duration": result.get("e2b_duration", 0),
            "total_cost": result.get("total_cost", 0),
            "turns": result.get("turns", 0),
            "usage": result.get("usage", {}),
        }

    submitted = len(submitted_tests)
    total = num_tests

    valid_results = [r for r in agent_results if isinstance(r, dict)]
    total_cost = sum(r.get("cost", 0) for r in valid_results)  # Gemini API cost
    total_e2b_cost = sum(r.get("e2b_cost", 0) for r in valid_results)  # E2B infrastructure cost
    elapsed = max((r.get("elapsed", 0) for r in valid_results), default=0)

    # Aggregate token usage across all agents
    total_usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    for r in valid_results:
        usage = r.get("usage", {})
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["cached_tokens"] += usage.get("cached_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

    score_data = {
        "submitted": submitted,
        "total": total,
        "elapsed": round(elapsed, 1),
        "gemini_api_cost": round(total_cost, 4),
        "e2b_cost": round(total_e2b_cost, 4),
        "total_cost": round(total_cost + total_e2b_cost, 4),
        "usage": total_usage,
    }

    task_result = {
        "score": score_data,
        "agents": per_agent,
    }
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(json.dumps(task_result, indent=2))

    return {
        "task_id": task_id,
        "score": score_data,
    }


# ── Main Orchestration ─────────────────────────────────────────────────────

async def run_all(args: argparse.Namespace):
    task_ids = load_task_ids(args.tasks)
    print(f"Loaded {len(task_ids)} tasks")

    # Run directory: resume existing or create new
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = RESULTS / args.resume
        if not run_dir.exists():
            raise RuntimeError(f"Resume directory not found: {run_dir}")
        print(f"Resuming run: {run_dir}")
    else:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{args.name}_{run_stamp}" if args.name else run_stamp
        run_dir = RESULTS / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

    # Symlink results/latest -> this run
    RESULTS.mkdir(parents=True, exist_ok=True)
    latest = RESULTS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    print(f"Run directory: {run_dir}")

    # Load already-completed tasks for resume
    completed_tasks: dict[str, dict] = {}
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    for f in task_results_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            completed_tasks[f.stem] = data
        except Exception:
            pass
    if completed_tasks:
        print(f"Found {len(completed_tasks)} already-completed tasks, skipping them")

    remaining_ids = [tid for tid in task_ids if tid not in completed_tasks]
    print(f"Running {len(remaining_ids)} tasks ({len(task_ids) - len(remaining_ids)} skipped)")

    all_scores: dict[str, dict] = {}
    total_submitted = 0
    total_tests = 0
    total_cost = 0.0

    # Seed totals from completed tasks
    for tid, data in completed_tasks.items():
        score = data.get("score", {})
        all_scores[tid] = score
        total_submitted += score.get("submitted", 0)
        total_tests += score.get("total", 0)
        total_cost += score.get("total_cost", score.get("cost", 0))

    completed = len(completed_tasks)

    # Shared slot queue: each token represents one E2B sandbox slot
    backend_queue: asyncio.Queue[str] | None = None
    if args.concurrency > 0:
        backend_queue = asyncio.Queue()
        for _ in range(args.concurrency):
            backend_queue.put_nowait("gemini")

    async def _process_and_report(task_id: str):
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(task_id, args, run_dir, backend_queue)
        except Exception as e:
            completed += 1
            print(f"[{completed}/{len(task_ids)}] ERROR {task_id}: {e}")
            return

        score = result["score"]
        total_submitted += score["submitted"]
        total_tests += score["total"]
        total_cost += score.get("total_cost", 0)
        all_scores[task_id] = score

        completed += 1
        s = score["submitted"]
        t = score["total"]
        print(
            f"[{completed}/{len(task_ids)}] "
            f"{'ok' if s == t else 'XX'} {task_id}  "
            f"{s}/{t} submitted  "
            f"({score.get('elapsed', 0):.0f}s)"
        )

    # Shuffle so tasks aren't processed in alphabetical order
    random.shuffle(remaining_ids)

    # Dispatch all tasks concurrently; backend_queue limits actual sandbox count
    await asyncio.gather(
        *[_process_and_report(tid) for tid in remaining_ids],
        return_exceptions=True,
    )

    # Write summary — raw data only, scoring is done post-hoc via majority voting
    summary = {
        "model": args.model,
        "num_agents": args.num_agents,
        "max_iterations": args.max_iterations,
        "soft_training_feedback": args.soft_training_feedback,
        "num_tasks": len(task_ids),
        "total_tests": total_tests,
        "total_cost": round(total_cost, 2),
        "tasks": all_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Done! {len(task_ids)} tasks, {total_tests} test inputs")
    print(f"Score with majority voting + pass@2 post-hoc")
    print(f"Summary: {run_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Gemini CLI Solver (E2B)")
    parser.add_argument("--tasks", default="all",
                        help="'all' (default) | comma-separated IDs")
    parser.add_argument("--num-agents", type=int, default=1,
                        help="Agents per test input (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max transform loop iterations per agent (default: 10)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Gemini model name (default: gemini-3.1-pro-preview)")
    parser.add_argument("--name", default=None,
                        help="Name prefix for results directory (e.g. 'GEMINI_3_FLASH_1x')")
    parser.add_argument("--resume", default=None,
                        help="Resume a previous run directory")
    parser.add_argument("--soft-training-feedback", action="store_true", default=False,
                        help="Use softer training failure message ('Try again' instead of 'Try a fundamentally different approach')")
    parser.add_argument("--concurrency", type=int, default=40,
                        help="Max simultaneous E2B sandboxes (agent-level). Default: 40. Set to 0 for unlimited.")
    args = parser.parse_args()

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
