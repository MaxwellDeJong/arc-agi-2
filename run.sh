#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Load environment variables ──────────────────────────────────────────
set -a && source "$SCRIPT_DIR/.env" && set +a

# ── 12-hour time circuit breaker ─────────────────────────────────────────
WALL_CLOCK_LIMIT=${WALL_CLOCK_LIMIT:-43200}   # 12 hours (override for testing)
DEADLINE_BUFFER=${DEADLINE_BUFFER:-300}        # 5 min for submission.py + safety
START_EPOCH=$(date +%s)

# ── Parse arguments ──────────────────────────────────────────────────────
SMOKE_TASK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE_TASK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: ./run.sh [--smoke <task_id>]" >&2
            exit 1
            ;;
    esac
done

# ── Configure based on mode ──────────────────────────────────────────────
if [[ -n "$SMOKE_TASK" ]]; then
    echo "=== ARC-AGI Smoke Test: $SMOKE_TASK ==="
    GEMINI_CLI_TASKS="$SMOKE_TASK"
    GEMINI_CLI_AGENTS=6
    GEMINI_CLI_MAX_ITERATIONS=10
    GEMINI_CLI_CONCURRENCY=1
    RUN_NAME="SMOKE_${SMOKE_TASK}"
else
    echo "=== Final ARC-AGI Run ==="
    GEMINI_CLI_TASKS="all"
    GEMINI_CLI_AGENTS=12
    GEMINI_CLI_MAX_ITERATIONS=10
    GEMINI_CLI_CONCURRENCY=132
    RUN_NAME="FINAL_RUN"
fi

echo "Script dir: $SCRIPT_DIR"
echo ""

# ── Start solver ─────────────────────────────────────────────────────────
echo "--- Starting Gemini CLI solver ---"
echo "[gemini-cli] $GEMINI_CLI_AGENTS agents, max-iterations $GEMINI_CLI_MAX_ITERATIONS"
echo ""

cd "$SCRIPT_DIR/gemini-cli-solver"
uv run python orchestrator.py \
    --tasks "$GEMINI_CLI_TASKS" \
    --num-agents "$GEMINI_CLI_AGENTS" \
    --max-iterations "$GEMINI_CLI_MAX_ITERATIONS" \
    --concurrency "${GEMINI_CLI_CONCURRENCY:-4}" \
    --name "$RUN_NAME" &
PID_SOLVER=$!

echo "Solver PID: $PID_SOLVER"
echo ""

# ── SIGTERM safety net ───────────────────────────────────────────────────
# If run.sh itself is killed (e.g. by Kaggle harness), kill solver and
# run submission.py so we still produce a submission.json with partial results.
_cleanup() {
    echo ""
    echo "--- SIGTERM/SIGINT received — cleaning up ---"
    kill "$PID_SOLVER" 2>/dev/null
    sleep 5
    kill -9 "$PID_SOLVER" 2>/dev/null
    wait "$PID_SOLVER" 2>/dev/null
    # Kill watchdog if running
    [[ -n "${PID_WATCHDOG:-}" ]] && kill "$PID_WATCHDOG" 2>/dev/null
    echo "--- Producing submission.json from partial results ---"
    cd "$SCRIPT_DIR"
    python3 submission.py || true
    echo "=== Done (via signal handler) ==="
    exit 1
}
trap _cleanup SIGTERM SIGINT

# ── Deadline watchdog ────────────────────────────────────────────────────
# Sleep until DEADLINE_BUFFER before the wall-clock limit, then kill solver.
SLEEP_SECS=$(( WALL_CLOCK_LIMIT - DEADLINE_BUFFER ))
if (( SLEEP_SECS > 0 )); then
    (
        sleep "$SLEEP_SECS"
        ELAPSED=$(( $(date +%s) - START_EPOCH ))
        echo ""
        echo "--- WATCHDOG: ${ELAPSED}s elapsed, killing solver (${DEADLINE_BUFFER}s before ${WALL_CLOCK_LIMIT}s limit) ---"
        kill "$PID_SOLVER" 2>/dev/null
        sleep 15
        kill -9 "$PID_SOLVER" 2>/dev/null
    ) &
    PID_WATCHDOG=$!
    echo "Watchdog PID: $PID_WATCHDOG (will fire in ${SLEEP_SECS}s)"
else
    echo "Warning: WALL_CLOCK_LIMIT ($WALL_CLOCK_LIMIT) <= DEADLINE_BUFFER ($DEADLINE_BUFFER), no watchdog started"
fi
echo ""

# Wait for solver — don't exit on failure so submission.py can run with partial results
set +e
wait "$PID_SOLVER"
SOLVER_EXIT=$?
set -e

# Kill the watchdog — solver has exited (naturally or via watchdog kill)
[[ -n "${PID_WATCHDOG:-}" ]] && kill "$PID_WATCHDOG" 2>/dev/null && wait "$PID_WATCHDOG" 2>/dev/null || true

echo ""
echo "--- Solver result ---"
if [[ $SOLVER_EXIT -eq 0 ]]; then
    echo "[gemini-cli] SUCCESS"
else
    echo "[gemini-cli] FAILED (exit code $SOLVER_EXIT)"
fi
echo ""

# ── Build submission ─────────────────────────────────────────────────────
echo "--- Building submission ---"
cd "$SCRIPT_DIR"
python3 submission.py

echo ""
if [[ $SOLVER_EXIT -ne 0 ]]; then
    echo "=== Done (with solver errors) ==="
    exit 1
else
    echo "=== Done! ==="
fi
