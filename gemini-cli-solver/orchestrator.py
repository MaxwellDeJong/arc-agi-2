"""Orchestrator: dispatches local LLM agents to Docker containers.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching local LLM agents to Docker containers with the solver image
- Writing logs (raw_stream, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)

Each agent runs in its own Docker container with the solver image.
The orchestrator builds the image from gemini-cli-solver/Dockerfile at startup.

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
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import aiodocker

# Force line-buffered stdout so background runs show live progress
sys.stdout.reconfigure(line_buffering=True)

# --- Paths ---
ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"
AGENT_RUNNER_PATH = ROOT / "agent_runner.py"
LOCAL_CLI_PATH = ROOT / "local_agent_cli.py"

# --- Cached file contents (read once at startup, not per container) ---
_AGENT_RUNNER_CONTENT: str | None = None
_LOCAL_CLI_CONTENT: str | None = None

# --- Shared Docker client (one connection, reused across all containers) ---
_docker_client: aiodocker.Docker | None = None


def _get_docker_client() -> aiodocker.Docker:
    global _docker_client
    if _docker_client is None:
        _docker_client = aiodocker.Docker()
    return _docker_client


def _get_agent_runner_content() -> str:
    global _AGENT_RUNNER_CONTENT
    if _AGENT_RUNNER_CONTENT is None:
        _AGENT_RUNNER_CONTENT = AGENT_RUNNER_PATH.read_text()
    return _AGENT_RUNNER_CONTENT


def _get_local_cli_content() -> str:
    global _LOCAL_CLI_CONTENT
    if _LOCAL_CLI_CONTENT is None:
        _LOCAL_CLI_CONTENT = LOCAL_CLI_PATH.read_text()
    return _LOCAL_CLI_CONTENT


# ── Docker Sandbox Helpers ────────────────────────────────────────────────

# Docker containers have no per-usage infrastructure cost.

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


async def _build_image(image: str) -> None:
    """Build the Docker image from the Dockerfile in gemini-cli-solver/."""
    print(f"[docker] Building image {image} ...")
    proc = await asyncio.create_subprocess_exec(
        "docker", "build", "-t", image, str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        output = stdout.decode() if stdout else ""
        raise RuntimeError(f"docker build failed:\n{output}")
    print(f"[docker] Image {image} built successfully")


async def run_agent_in_docker(
    task_id: str,
    agent_id: str,
    raw_task: dict,
    test_index: int,
    model: str,
    max_iterations: int,
    soft_training_feedback: bool,
    image: str = "arc-solver:latest",
    container_memory_gb: float = 2.0,
    container_cpus: float = 1.0,
) -> dict:
    """Run a Gemini CLI agent inside a Docker container.

    Mounts three host temp directories as volumes:
      /root      — config.json (read by agent_runner.py)
      /app       — agent_runner.py
      /workspace — results.json written here by agent_runner.py
    """
    container_start = time.time()
    container = None
    tmp_root = tmp_app = tmp_workspace = None
    docker_client = _get_docker_client()

    try:
        # Step 1 — Create temp directories and write files
        tmp_root = tempfile.mkdtemp()
        tmp_app = tempfile.mkdtemp()
        tmp_workspace = tempfile.mkdtemp()

        config = {
            "task_id": task_id,
            "agent_id": agent_id,
            "raw_task": raw_task,
            "test_index": test_index,
            "model": model,
            "max_iterations": max_iterations,
            "soft_training_feedback": soft_training_feedback,
        }
        (Path(tmp_root) / "config.json").write_text(json.dumps(config))
        (Path(tmp_app) / "agent_runner.py").write_text(_get_agent_runner_content())
        (Path(tmp_app) / "local_agent_cli.py").write_text(_get_local_cli_content())

        # Step 2 — Build environment variables
        envs: dict[str, str] = {}
        vllm_url = os.environ.get("VLLM_BASE_URL", "http://host.docker.internal:8000/v1")
        # Containers can't reach 'localhost' — remap to the Docker host gateway
        # so users can keep the natural http://localhost:8000/v1 in their .env.
        vllm_url = vllm_url.replace("://localhost:", "://host.docker.internal:") \
                            .replace("://127.0.0.1:", "://host.docker.internal:")
        envs["VLLM_BASE_URL"] = vllm_url
        envs["VLLM_API_KEY"] = os.environ.get("VLLM_API_KEY", "not-needed")

        # Step 3 — Run the container
        memory_bytes = int(container_memory_gb * 1024 * 1024 * 1024)
        nano_cpus = int(container_cpus * 10 ** 9)
        container = await docker_client.containers.run(
            config={
                "Image": image,
                "Cmd": ["python3", "/app/agent_runner.py"],
                "Env": [f"{k}={v}" for k, v in envs.items()],
                "HostConfig": {
                    "Binds": [
                        f"{tmp_root}:/root",
                        f"{tmp_app}:/app",
                        f"{tmp_workspace}:/workspace",
                    ],
                    "Memory": memory_bytes,
                    "NanoCpus": nano_cpus,
                    "AutoRemove": False,
                    "ExtraHosts": ["host.docker.internal:host-gateway"],
                },
            }
        )

        stderr_lines: list[str] = []
        async for log_line in container.log(stdout=True, stderr=True, follow=True):
            line = log_line.rstrip("\n")
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    evt_type = event.get("event", "")
                    aid = event.get("agent_id", "?")
                    formatter = _EVENT_FORMATTERS.get(evt_type)
                    if formatter:
                        detail = formatter(event)
                        print(f"  [status] {aid}: {detail}")
            except (json.JSONDecodeError, TypeError):
                if line.strip():
                    print(f"  [docker-stderr] {agent_id}: {line[:200]}")
                    stderr_lines.append(line)

        exit_info = await container.wait()
        exit_code = exit_info.get("StatusCode", -1)

        container_duration = time.time() - container_start

        # Step 4 — Read results from workspace volume
        results_path = Path(tmp_workspace) / "results.json"
        try:
            result = json.loads(results_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as read_err:
            return {
                "task_id": task_id,
                "agent_id": agent_id,
                "test_index": test_index,
                "attempts": [],
                "elapsed": 0,
                "cost": 0,
                "container_duration": container_duration,
                "turns": 0,
                "error": f"Docker container error: could not read results.json (exit_code={exit_code}): {read_err}",
                "raw_lines": [],
                "stderr": "\n".join(stderr_lines),
            }

        # Step 5 — Return result
        return {
            **result,
            "container_duration": container_duration,
        }

    except Exception as e:
        container_duration = time.time() - container_start
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": [],
            "elapsed": 0,
            "cost": 0,
            "container_duration": container_duration,
            "turns": 0,
            "error": f"Docker container error: {e}",
            "raw_lines": [],
            "stderr": "",
        }
    finally:
        # Step 6 — Remove container and clean up temp directories
        if container is not None:
            try:
                await container.delete(force=True)
            except Exception:
                pass
        for tmp_dir in (tmp_root, tmp_app, tmp_workspace):
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


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


async def _retry_backend_call(coro_fn, *, agent_id: str) -> dict:
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
    """Orchestrate N agents per test input via Docker containers, save results independently."""
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])

    agent_metas: list[tuple[str, int, Path]] = []  # (agent_id, test_index, log_dir)

    async def _dispatch(agent_id: str, kwargs: dict, test_index: int, log_dir: Path) -> dict:
        """Acquire a backend slot, run the agent, release the slot.

        Writes logs and incremental results immediately when the agent finishes,
        before waiting for other agents (safety net for early termination).
        """
        if backend_queue is None:
            result = await _retry_backend_call(
                lambda kw=kwargs: run_agent_in_docker(**kw),
                agent_id=agent_id,
            )
        else:
            token = await backend_queue.get()
            try:
                result = await _retry_backend_call(
                    lambda kw=kwargs: run_agent_in_docker(**kw),
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
                "container_duration": result.get("container_duration", 0),
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
                image=args.image,
                container_memory_gb=args.container_memory_gb,
                container_cpus=args.container_cpus,
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
            "container_duration": result.get("container_duration", 0),
            "turns": result.get("turns", 0),
            "usage": result.get("usage", {}),
        }

    submitted = len(submitted_tests)
    total = num_tests

    valid_results = [r for r in agent_results if isinstance(r, dict)]
    total_cost = sum(r.get("cost", 0) for r in valid_results)
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
        "api_cost": round(total_cost, 4),
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
        total_cost += score.get("api_cost", score.get("cost", 0))

    completed = len(completed_tasks)

    # Pre-load agent scripts into memory so every container creation skips disk I/O
    _get_agent_runner_content()
    _get_local_cli_content()

    # Build the Docker image before dispatching any agents
    await _build_image(args.image)

    # Shared slot queue: each token represents one Docker container slot
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
        total_cost += score.get("api_cost", 0)
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

    # Close the shared Docker client
    global _docker_client
    if _docker_client is not None:
        await _docker_client.close()
        _docker_client = None

    print(f"\n{'='*60}")
    print(f"Done! {len(task_ids)} tasks, {total_tests} test inputs")
    print(f"Score with majority voting + pass@2 post-hoc")
    print(f"Summary: {run_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Local LLM Solver (Docker)")
    parser.add_argument("--tasks", default="all",
                        help="'all' (default) | comma-separated IDs")
    parser.add_argument("--num-agents", type=int, default=1,
                        help="Agents per test input (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Max transform loop iterations per agent (default: 10)")
    parser.add_argument("--model", default="google/gemma-3-27b-it",
                        help="Model name to request from the local inference server (default: google/gemma-3-27b-it)")
    parser.add_argument("--name", default=None,
                        help="Name prefix for results directory (e.g. 'GEMINI_3_FLASH_1x')")
    parser.add_argument("--resume", default=None,
                        help="Resume a previous run directory")
    parser.add_argument("--soft-training-feedback", action="store_true", default=False,
                        help="Use softer training failure message ('Try again' instead of 'Try a fundamentally different approach')")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max simultaneous Docker containers. Default: 16. Set to 0 for unlimited. "
                             "Higher values keep the GPU saturated by ensuring inference requests are "
                             "always queued while other containers execute code.")
    parser.add_argument("--image", default="arc-solver:latest",
                        help="Docker image to use for agent containers (default: arc-solver:latest)")
    parser.add_argument("--container-memory-gb", type=float, default=2.0,
                        help="Memory limit per container in GiB (default: 2.0). "
                             "Lower values allow more containers to run concurrently.")
    parser.add_argument("--container-cpus", type=float, default=1.0,
                        help="CPU limit per container (default: 1.0). "
                             "Lower values allow more containers to run concurrently.")
    args = parser.parse_args()

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
