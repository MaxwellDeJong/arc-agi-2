#!/usr/bin/env python3
"""Build a Kaggle submission from Gemini CLI solver results.

For each task and each test index:
  - Pool ALL grids from all agents into a single list
  - Rank by frequency count across the pool
  - attempt_1 = most common grid (rank 1)
  - attempt_2 = second most common grid (rank 2), or [[0]] fallback

Writes submission.json and cost_breakdown.json to project root,
and scores against ground truth.

Usage:
  python3 submission.py          # reads from gemini-cli-solver/results/latest
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


Grid = list[list[int]]


def canonicalize_grid(grid: Grid) -> str:
    """Stable JSON string for a grid (for hashing / counting)."""
    return json.dumps(grid, separators=(",", ":"))


def top_k_vote(grids: list[Grid], k: int = 2) -> list[Grid]:
    """Return the top-k most common grids from a pool, ordered by frequency.

    If the pool is empty, returns an empty list.  Ties are broken by
    whichever grid was seen first (Counter.most_common is stable).
    """
    if not grids:
        return []
    counts: Counter[str] = Counter()
    lookup: dict[str, Grid] = {}
    for g in grids:
        key = canonicalize_grid(g)
        counts[key] += 1
        lookup[key] = g
    return [lookup[key] for key, _ in counts.most_common(k)]


# ── Solver result loading ─────────────────────────────────────────────────


def load_solver_grids(results_dir: Path) -> dict[str, dict[int, list[Grid]]]:
    """Load solver results from a task_results/ directory.

    Schema: {agents: {agent_id: {test_index, attempts, ...}}}.

    Returns: {task_id: {test_index: [grid, grid, ...]}}
    Each agent produces one grid; we pool all grids from all agents
    for each test index. Majority voting selects the top 2 for submission.
    """
    task_results_dir = results_dir / "task_results"
    if not task_results_dir.is_dir():
        print(f"Warning: task_results dir not found: {task_results_dir}", file=sys.stderr)
        return {}

    out: dict[str, dict[int, list[Grid]]] = {}

    for f in sorted(task_results_dir.glob("*.json")):
        task_id = f.stem
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping corrupt {f}: {e}", file=sys.stderr)
            continue
        agents: dict[str, dict] = data.get("agents", {})
        per_test: dict[int, list[Grid]] = {}

        for agent_info in agents.values():
            ti: int = agent_info.get("test_index", 0)
            attempts: list[Grid] = agent_info.get("attempts", [])
            if ti not in per_test:
                per_test[ti] = []
            per_test[ti].extend(attempts)

        out[task_id] = per_test

    return out


# ── Summary loading ───────────────────────────────────────────────────────


def load_summary(results_dir: Path) -> dict | None:
    """Load summary.json from a solver results directory."""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        return None
    return json.loads(summary_file.read_text())


# ── Ground truth loading ──────────────────────────────────────────────────


def load_ground_truth(data_dir: Path) -> dict[str, list[Grid]]:
    """Load ground truth from arc-agi_evaluation_solutions.json.

    Returns: {task_id: [gt_output_for_test_0, gt_output_for_test_1, ...]}
    """
    solutions_file = data_dir / "arc-agi_evaluation_solutions.json"
    if not solutions_file.exists():
        print(f"Warning: ground truth not found: {solutions_file}", file=sys.stderr)
        return {}
    return json.loads(solutions_file.read_text())


# ── Scoring ───────────────────────────────────────────────────────────────


def score_submission(
    submission: dict[str, list[dict[str, Grid | None]]],
    ground_truth: dict[str, list[Grid]],
) -> tuple[float, int, int]:
    """Score a Kaggle submission against ground truth.

    Returns: (arc_mean_score, num_tasks_scored, total_correct_tasks)
    """
    per_task_scores: dict[str, float] = {}

    for task_id, gt_outputs in ground_truth.items():
        if task_id not in submission:
            per_task_scores[task_id] = 0.0
            continue

        preds = submission[task_id]
        correct = 0
        for ti, gt in enumerate(gt_outputs):
            if ti >= len(preds):
                continue
            pack = preds[ti] or {}
            a1 = pack.get("attempt_1")
            a2 = pack.get("attempt_2")
            if (a1 is not None and a1 == gt) or (a2 is not None and a2 == gt):
                correct += 1
        per_task_scores[task_id] = correct / max(len(gt_outputs), 1)

    if not per_task_scores:
        return 0.0, 0, 0

    arc_mean = sum(per_task_scores.values()) / len(per_task_scores)
    perfect_tasks = sum(1 for s in per_task_scores.values() if s == 1.0)
    return arc_mean, len(per_task_scores), perfect_tasks


# ── Cost breakdown ────────────────────────────────────────────────────────


def _aggregate_costs_from_task_results(results_dir: Path) -> dict[str, dict]:
    """Fallback: aggregate per-agent costs from task_results/*.json files.

    Used when summary.json is missing (e.g. process killed before writing it).
    Returns: {task_id: {"api_cost": float, "e2b_cost": float, "total_cost": float, "usage": dict}}
    """
    task_results_dir = results_dir / "task_results"
    if not task_results_dir.is_dir():
        return {}

    out: dict[str, dict] = {}
    for f in sorted(task_results_dir.glob("*.json")):
        task_id = f.stem
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        agents = data.get("agents", {})
        api_cost = 0.0
        e2b_cost = 0.0
        total_cost = 0.0
        usage: dict[str, int] = {}
        for agent_info in agents.values():
            api_cost += agent_info.get("cost", 0)
            e2b_cost += agent_info.get("e2b_cost", 0)
            total_cost += agent_info.get("total_cost", 0)
            for k, v in agent_info.get("usage", {}).items():
                usage[k] = usage.get(k, 0) + (v if isinstance(v, int) else 0)
        out[task_id] = {
            "api_cost": api_cost,
            "e2b_cost": e2b_cost,
            "total_cost": total_cost,
            "usage": usage,
        }
    return out


def extract_cost_breakdown(results_dir: Path, num_tasks: int) -> dict:
    """Extract cost breakdown from the Gemini CLI solver.

    Falls back to aggregating from task_results/*.json when summary.json
    is missing (e.g. process killed before writing it).
    """
    summary = load_summary(results_dir)

    breakdown = {
        "_documentation": {
            "purpose": "Cost breakdown",
            "generated_by": "submission.py",
            "e2b_pricing": {
                "vcpus": 2,
                "cost_per_vcpu_hour_usd": 0.05,
                "total_hourly_rate_usd": 0.10,
                "formula": "(duration_seconds / 3600) * vcpus * cost_per_vcpu_hour"
            },
            "note": "API costs calculated from Gemini CLI token reports using published pricing"
        },
        "gemini_cli": {},
        "metadata": {
            "e2b_vcpus": 2,
            "e2b_cost_per_vcpu_hour": 0.05,
            "num_tasks": num_tasks
        }
    }

    if summary:
        tasks = summary.get("tasks", {})
        api_cost = sum(task.get("gemini_api_cost", 0.0) for task in tasks.values())
        e2b_cost = sum(task.get("e2b_cost", 0.0) for task in tasks.values())
        total = api_cost + e2b_cost

        total_usage = {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
        }
        for task in tasks.values():
            usage = task.get("usage", {})
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["cached_tokens"] += usage.get("cached_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)

        breakdown["gemini_cli"] = {
            "model": summary.get("model", "unknown"),
            "gemini_api_cost": round(api_cost, 4),
            "e2b_cost": round(e2b_cost, 4),
            "total_cost": round(total, 4),
            "usage": total_usage,
            "per_task": {
                task_id: {
                    "gemini_api_cost": round(task.get("gemini_api_cost", 0.0), 4),
                    "e2b_cost": round(task.get("e2b_cost", 0.0), 4),
                    "total_cost": round(task.get("total_cost", 0.0), 4),
                    "elapsed_seconds": round(task.get("elapsed", 0.0), 2),
                    "usage": task.get("usage", {})
                }
                for task_id, task in tasks.items()
            }
        }
    else:
        # Fallback: aggregate from task_results/*.json
        task_costs = _aggregate_costs_from_task_results(results_dir)
        if task_costs:
            api_cost = sum(t["api_cost"] for t in task_costs.values())
            e2b_cost = sum(t["e2b_cost"] for t in task_costs.values())
            total = api_cost + e2b_cost
            total_usage: dict[str, int] = {}
            for t in task_costs.values():
                for k, v in t["usage"].items():
                    total_usage[k] = total_usage.get(k, 0) + v
            breakdown["gemini_cli"] = {
                "model": "unknown (from task_results)",
                "gemini_api_cost": round(api_cost, 4),
                "e2b_cost": round(e2b_cost, 4),
                "total_cost": round(total, 4),
                "usage": total_usage,
                "per_task": {
                    task_id: {
                        "gemini_api_cost": round(t["api_cost"], 4),
                        "e2b_cost": round(t["e2b_cost"], 4),
                        "total_cost": round(t["total_cost"], 4),
                        "usage": t["usage"],
                    }
                    for task_id, t in task_costs.items()
                }
            }
        else:
            breakdown["gemini_cli"] = {
                "model": "N/A",
                "gemini_api_cost": 0.0,
                "e2b_cost": 0.0,
                "total_cost": 0.0,
                "usage": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0},
                "per_task": {}
            }

    return breakdown


def print_cost_report(cost_breakdown: dict) -> None:
    """Print human-readable cost breakdown to console."""
    print(f"\n{'='*60}")
    print("COST BREAKDOWN")
    print(f"{'='*60}")

    g = cost_breakdown["gemini_cli"]
    g_usage = g.get("usage", {})
    print(f"\nGemini CLI Solver:")
    print(f"  Model:         {g['model']}")
    print(f"  Gemini API:    ${g['gemini_api_cost']:.4f}")
    print(f"  E2B Infra:     ${g['e2b_cost']:.4f}")
    print(f"  TOTAL:         ${g['total_cost']:.4f}")
    if g_usage:
        print(f"  Tokens:        input={g_usage.get('input_tokens', 0):,}, "
              f"cached={g_usage.get('cached_tokens', 0):,}, "
              f"output={g_usage.get('output_tokens', 0):,}")
    print(f"{'='*60}")


def write_cost_breakdown_file(cost_breakdown: dict, output_dir: Path) -> None:
    """Write cost breakdown to cost_breakdown.json."""
    cost_file = output_dir / "cost_breakdown.json"
    cost_file.write_text(json.dumps(cost_breakdown, indent=2))
    print(f"Wrote cost breakdown to: {cost_file}")


# ── Transcript security check ────────────────────────────────────────────


_SUSPICIOUS_PATTERNS = [
    # API key env var names
    r"GEMINI_API_KEY",
    r"GOOGLE_API_KEY",
    # Env inspection
    r"printenv",
    r"os\.environ",
    r"/proc/self/environ",
    r"/proc/\d+/environ",
]

_COMPILED_PATTERNS = [re.compile(p) for p in _SUSPICIOUS_PATTERNS]


def check_transcripts(results_dir: Path) -> list[str]:
    """Scan transcript logs for API key access or env inspection attempts.

    Returns a list of warning strings (empty if clean).
    """
    warnings: list[str] = []

    logs_dir = results_dir / "logs"
    if not logs_dir.is_dir():
        return warnings

    for transcript_file in sorted(logs_dir.rglob("transcript.jsonl")):
        rel = transcript_file.relative_to(results_dir)
        for line_num, line in enumerate(transcript_file.read_text().splitlines(), 1):
            for pattern in _COMPILED_PATTERNS:
                match = pattern.search(line)
                if match:
                    warnings.append(
                        f"[gemini-cli] {rel}:{line_num}: "
                        f"found '{match.group()}'"
                    )

    return warnings


# ── Main ──────────────────────────────────────────────────────────────────


def build_submission(
    solver_grids: dict[str, dict[int, list[Grid]]],
    ground_truth: dict[str, list[Grid]],
) -> dict[str, list[dict[str, Grid | None]]]:
    """Build Kaggle submission from solver output.

    For each task/test, grids are ranked by frequency:
      attempt_1 = most common grid (rank 1)
      attempt_2 = second most common grid (rank 2), or [[0]] fallback
    """
    all_task_ids = sorted(set(ground_truth.keys()) | set(solver_grids.keys()))
    submission: dict[str, list[dict[str, Grid | None]]] = {}

    for task_id in all_task_ids:
        num_tests = len(ground_truth.get(task_id, []))
        if num_tests == 0:
            # Infer from solver outputs
            tests = solver_grids.get(task_id, {})
            all_indices = set(tests.keys())
            num_tests = max(all_indices, default=-1) + 1

        preds: list[dict[str, Grid | None]] = []
        tests = solver_grids.get(task_id, {})

        for ti in range(num_tests):
            pool = tests.get(ti, [])
            top = top_k_vote(pool, k=2)

            preds.append({
                "attempt_1": top[0] if len(top) >= 1 else [[0]],
                "attempt_2": top[1] if len(top) >= 2 else [[0]],
            })

        submission[task_id] = preds

    return submission


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    results_dir = script_dir / "gemini-cli-solver" / "results" / "latest"
    output_path = script_dir / "submission.json"

    # Resolve symlinks (e.g. results/latest -> actual run dir)
    if results_dir.is_symlink():
        results_dir = results_dir.resolve()

    print(f"Results dir: {results_dir}")

    # Load results
    solver_grids = load_solver_grids(results_dir)
    print(f"Loaded {len(solver_grids)} tasks from Gemini CLI solver")

    # Load ground truth
    ground_truth = load_ground_truth(data_dir)

    # Build submission
    submission = build_submission(solver_grids, ground_truth)
    print(f"Built submission with {len(submission)} tasks")

    # Score
    if ground_truth:
        arc_mean, num_scored, perfect = score_submission(submission, ground_truth)
        print(f"\n{'='*60}")
        print(f"ARC-mean score: {arc_mean * 100:.2f}% ({arc_mean * num_scored:.2f}/{num_scored})")
        print(f"Perfect tasks:  {perfect}/{num_scored}")
        print(f"{'='*60}")

    # Cost breakdown
    print(f"\n{'='*60}")
    print("Extracting cost breakdown...")
    cost_breakdown = extract_cost_breakdown(results_dir, num_tasks=len(ground_truth))
    print_cost_report(cost_breakdown)
    write_cost_breakdown_file(cost_breakdown, script_dir)

    # Security: check transcripts for API key access attempts
    print(f"\n{'='*60}")
    print("Checking transcripts for suspicious patterns...")
    security_warnings = check_transcripts(results_dir)
    if security_warnings:
        print(f"\n  WARNING: {len(security_warnings)} suspicious pattern(s) found:")
        for w in security_warnings:
            print(f"    {w}")
    else:
        print("  Clean — no suspicious patterns found.")
    print(f"{'='*60}")

    # Write submission
    output_path.write_text(json.dumps(submission))
    print(f"\nWrote submission to: {output_path}")


if __name__ == "__main__":
    main()
