#!/usr/bin/env python3
"""Local agent CLI — drop-in replacement for @google/gemini-cli.

Speaks the same stream-json output format and CLI flags as the Gemini CLI
subset used by agent_runner.py, but calls a vLLM server via the
OpenAI-compatible API instead of Google's API.

Usage:
    python3 local_agent_cli.py -y -m <model> -o stream-json -p "<prompt>"
    python3 local_agent_cli.py -y -m <model> -o stream-json --resume latest
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────

# Maximum characters of tool output added to the model's context window.
# Local models have limited context (32K–128K tokens); keeping tool output
# short prevents silent truncation by the inference engine.
# TODO: Implement a proper context management strategy (e.g. summarising or
# dropping old tool results) when approaching the model's context limit.
TOOL_OUTPUT_TRUNCATION_LIMIT = 8_000

# ── Tool Definitions ───────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or relative path to the file."},
                    "content": {"type": "string", "description": "The full content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_new_file",
            "description": "Write content to a new file, creating it or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or relative path to the file."},
                    "content": {"type": "string", "description": "The full content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact string to replace."},
                    "new_string": {"type": "string", "description": "Replacement string."},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                    "description": {"type": "string", "description": "Brief description of what the command does."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "File or directory to search in. Defaults to cwd."},
                },
                "required": ["pattern"],
            },
        },
    },
]


# ── Output helpers ─────────────────────────────────────────────────────────

def emit(obj: dict) -> None:
    """Write a JSON event line to stdout."""
    print(json.dumps(obj), flush=True)


# ── Tool execution ─────────────────────────────────────────────────────────

def execute_tool(tool_call, cwd: str) -> tuple[str, bool]:
    """Execute a single tool call; return (result_text, is_error)."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return f"Invalid tool arguments JSON: {e}", True

    try:
        if name in ("write_file", "write_new_file"):
            file_path = args["file_path"]
            content = args["content"]
            p = Path(file_path) if Path(file_path).is_absolute() else Path(cwd) / file_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"File written successfully: {file_path}", False

        elif name == "read_file":
            file_path = args["file_path"]
            p = Path(file_path) if Path(file_path).is_absolute() else Path(cwd) / file_path
            try:
                return p.read_text(), False
            except FileNotFoundError:
                return f"File not found: {file_path}", True

        elif name == "edit_file":
            file_path = args["file_path"]
            old_string = args["old_string"]
            new_string = args["new_string"]
            replace_all = args.get("replace_all", False)
            p = Path(file_path) if Path(file_path).is_absolute() else Path(cwd) / file_path
            try:
                original = p.read_text()
            except FileNotFoundError:
                return f"File not found: {file_path}", True
            if old_string not in original:
                return f"String not found in {file_path}: {old_string!r}", True
            if replace_all:
                updated = original.replace(old_string, new_string)
            else:
                updated = original.replace(old_string, new_string, 1)
            p.write_text(updated)
            return f"File edited successfully: {file_path}", False

        elif name == "run_shell_command":
            command = args["command"]
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = result.stdout + result.stderr
            if len(output) > TOOL_OUTPUT_TRUNCATION_LIMIT:
                output = output[:TOOL_OUTPUT_TRUNCATION_LIMIT] + "\n... (truncated)"
            return output, False

        elif name == "glob":
            pattern = args["pattern"]
            matches = sorted(Path(cwd).glob(pattern))
            relative = [str(m.relative_to(cwd)) for m in matches]
            return "\n".join(relative) if relative else "(no matches)", False

        elif name == "grep":
            pattern = args["pattern"]
            search_path = args.get("path") or cwd
            if not Path(search_path).is_absolute():
                search_path = str(Path(cwd) / search_path)
            result = subprocess.run(
                ["grep", "-r", "-n", pattern, search_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd,
            )
            output = result.stdout + result.stderr
            if len(output) > TOOL_OUTPUT_TRUNCATION_LIMIT:
                output = output[:TOOL_OUTPUT_TRUNCATION_LIMIT] + "\n... (truncated)"
            return output if output.strip() else "(no matches)", False

        else:
            return f"Unknown tool: {name}", True

    except subprocess.TimeoutExpired:
        return "Command timed out after 300s", True
    except Exception as e:
        return f"Tool execution error: {e}\n{traceback.format_exc()}", True


# ── Session persistence ────────────────────────────────────────────────────

def load_session(session_dir: Path) -> dict:
    """Load the latest session from disk; raises FileNotFoundError if absent."""
    session_file = session_dir / "latest.json"
    if not session_file.exists():
        raise FileNotFoundError(f"No session file found at {session_file}")
    return json.loads(session_file.read_text())


def save_session(session_dir: Path, session: dict) -> None:
    """Write session to disk, overwriting any existing latest.json."""
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "latest.json").write_text(json.dumps(session, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Local vLLM agent CLI")
    parser.add_argument("-y", action="store_true", help="Auto-confirm (accepted and ignored)")
    parser.add_argument("-m", "--model", required=True, help="Model name to request from the vLLM server")
    parser.add_argument("-o", "--output", default="stream-json", help="Output format (only stream-json is implemented)")
    parser.add_argument("-p", "--prompt", default=None, help="Initial user prompt")
    parser.add_argument("--resume", default=None, help="Resume a prior session (only 'latest' is supported)")
    args = parser.parse_args()

    if args.resume is None and args.prompt is None:
        print("Error: either --prompt or --resume is required", file=sys.stderr)
        return 1

    if args.resume is not None and args.resume != "latest":
        print(f"Error: only 'latest' is supported for --resume, got {args.resume!r}", file=sys.stderr)
        return 1

    cwd = os.getcwd()

    settings_path = Path(cwd) / ".local_agent" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except Exception:
            pass

    base_url = os.environ.get("VLLM_BASE_URL") or settings.get("base_url", "http://host.docker.internal:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY") or settings.get("api_key", "not-needed")
    max_turns: int = settings.get("max_turns", 200)
    max_time_minutes: int = settings.get("max_time_minutes", 360)

    model = args.model
    client = OpenAI(base_url=base_url, api_key=api_key)
    session_dir = Path(cwd) / ".local_agent" / "sessions"

    token_stats: dict[str, int] = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    tool_call_count = 0
    start_time = time.time()

    zero_stats = {"input": 0, "cached": 0, "output_tokens": 0, "total_tokens": 0, "duration_ms": 0, "tool_calls": 0}

    # llama.cpp reports prompt_tokens as the FULL context size on every call
    # (entire conversation history re-tokenized each turn). Accumulating these
    # raw values would count early-turn tokens O(N) times. Instead we track the
    # delta: how many new tokens were added to the context since the last call.
    # We persist last_prompt_tokens across --resume sessions so the resumed
    # session's first call doesn't re-count the previous session's context.
    prev_prompt_tokens: int = 0

    if args.resume == "latest":
        try:
            session = load_session(session_dir)
        except FileNotFoundError as e:
            emit({"type": "result", "stats": zero_stats})
            print(f"Error: {e}", file=sys.stderr)
            return 1

        messages: list[dict] = session["messages"]
        token_stats = session.get("token_stats", token_stats)
        prev_prompt_tokens = session.get("last_prompt_tokens", 0)

        feedback_text = sys.stdin.read()
        if feedback_text.strip():
            messages.append({"role": "user", "content": feedback_text})
    else:
        gemini_md_path = Path(cwd) / "GEMINI.md"
        if gemini_md_path.exists():
            system_content = gemini_md_path.read_text()
        else:
            system_content = "You are a helpful AI assistant."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": args.prompt},
        ]

    turns = 0
    exit_code = 0

    try:
        while turns < max_turns and (time.time() - start_time) < max_time_minutes * 60:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=False,
                )
            except Exception as e:
                err_str = str(e)
                print(f"vLLM API error: {err_str}", file=sys.stderr)
                exit_code = 1
                # Context overflow: the session is permanently unresumable because
                # every subsequent --resume call would reload the same overflowing
                # history.  Delete the session file so agent_runner stops retrying.
                if "exceed_context_size_error" in err_str or "context_length_exceeded" in err_str:
                    session_file = session_dir / "latest.json"
                    if session_file.exists():
                        session_file.unlink()
                        print("Context overflow: deleted session to prevent resume loop.", file=sys.stderr)
                break

            assistant_message = response.choices[0].message

            msg_dict: dict = {"role": "assistant", "content": assistant_message.content}
            if assistant_message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_message.tool_calls
                ]
            messages.append(msg_dict)

            if assistant_message.content:
                emit({"type": "message", "role": "assistant", "content": assistant_message.content, "delta": False})

            if response.usage:
                cur_prompt = response.usage.prompt_tokens or 0
                # Only count tokens added since the last API call; this
                # neutralises llama.cpp's full-context re-reporting each turn.
                incremental_input = max(cur_prompt - prev_prompt_tokens, 0)
                token_stats["input_tokens"] += incremental_input
                token_stats["output_tokens"] += response.usage.completion_tokens or 0
                prev_prompt_tokens = cur_prompt

            if not assistant_message.tool_calls:
                break

            for tool_call in assistant_message.tool_calls:
                try:
                    parameters = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    parameters = {}

                emit({
                    "type": "tool_use",
                    "tool_name": tool_call.function.name,
                    "tool_id": tool_call.id,
                    "parameters": parameters,
                })

                result_text, is_error = execute_tool(tool_call, cwd)

                display_text = result_text
                if len(display_text) > TOOL_OUTPUT_TRUNCATION_LIMIT:
                    display_text = display_text[:TOOL_OUTPUT_TRUNCATION_LIMIT] + "\n... (truncated)"

                emit({
                    "type": "tool_result",
                    "tool_id": tool_call.id,
                    "status": "error" if is_error else "ok",
                    "output": display_text,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

                tool_call_count += 1
                turns += 1

    except Exception as e:
        print(f"Unhandled error in tool-use loop: {e}\n{traceback.format_exc()}", file=sys.stderr)
        exit_code = 1

    duration_ms = int((time.time() - start_time) * 1000)
    total_tokens = token_stats["input_tokens"] + token_stats["output_tokens"]

    emit({
        "type": "result",
        "stats": {
            "input": token_stats["input_tokens"],
            "cached": token_stats["cached_tokens"],
            "output_tokens": token_stats["output_tokens"],
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "tool_calls": tool_call_count,
        },
    })

    save_session(session_dir, {
        "model": model,
        "messages": messages,
        "token_stats": token_stats,
        "last_prompt_tokens": prev_prompt_tokens,
    })

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
