#!/bin/bash
# Replacement-study eval populate, chunked one model at a time.
#
# Why chunked: theriad.eval.run's populate saves the grid only after the whole
# plan completes (run.py _cmd_populate). One invocation per model means the grid
# is written 15 times instead of once, so an interruption costs at most the
# model in flight rather than the entire run.
#
# Resume is automatic: plan_grid() plans around cells already in the grid, so
# re-running this script skips whatever already landed.
#
# Frontier tier (grok-4.5, claude-opus-5, gpt-4o) is deliberately excluded.
#
#   Usage:  ./scripts/run_replacement_study.sh
#   Log:    eval_run.log in the repo root
#
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
GRID="$REPO/eval_grid.json"
LOG="$REPO/eval_run.log"

# --- environment ------------------------------------------------------------
if [ -d "$REPO/.venv" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
else
  echo "WARNING: no .venv found; using whatever python3 is on PATH." >&2
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$REPO/.env" ]; then
  OPENROUTER_API_KEY="$(grep -m1 'OPENROUTER_API_KEY' "$REPO/.env" \
    | sed 's/.*OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*//' \
    | tr -d '"'"'"'' | tr -d '\r')"
  export OPENROUTER_API_KEY
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set and not found in .env" >&2
  exit 1
fi

python3 -c "import httpx" 2>/dev/null || {
  echo "ERROR: httpx not importable. Activate the venv or 'pip install -e .'" >&2
  exit 1
}

# The model list is derived from theriad.models.registry rather than hardcoded.
#
# Why: a hand-maintained list drifts from the ladder the router actually selects
# from, and the drift is silent. The Aug 8-11 runs measured gpt-5.4-mini,
# kimi-k2.7-code and gemini-2.5-flash (none of them in the registry, so the
# router can never pick them) while never measuring muse-spark-1.2, grok-4.5,
# muse-spark-1.1 or laguna-s-2.1. Unmeasured models fall back to the uniform
# Beta prior (p=0.5) and, if unpriced, cost 0 — see theriad/eval/routers.py —
# so their absence flatters them instead of failing loudly.
#
# Models already in eval_grid.json still appear here; plan_grid skips recorded
# cells, so a re-listed model costs one no-op chunk rather than another call.
MODEL_PLAN="$(python3 - <<'PY'
from theriad.models import tiers as T
from theriad.models.registry import ModelRegistry

# Eval coverage flag per registry tier. --cheap measures every item
# (item_fraction 1.0), --premium half, --frontier a fifth; see
# theriad/eval/run.py. Everything not premium/frontier gets full coverage.
FLAG_FOR_TIER = {T.TIER_FRONTIER: "frontier", T.TIER_PREMIUM: "premium"}
# Lower rank = more coverage. A model in several tiers (grok-4.5 is premium and
# frontier; deepseek-v4-flash is cheap_bulk and premium) is measured once at the
# most generous coverage it qualifies for, so no tier is left short.
RANK = {"cheap": 0, "premium": 1, "frontier": 2}

registry = ModelRegistry()
best: dict[str, str] = {}
for tier in registry.all_tiers():
    flag = FLAG_FOR_TIER.get(tier, "cheap")
    for cand in registry.candidates_for_tier(tier):
        current = best.get(cand.model)
        if current is None or RANK[flag] < RANK[current]:
            best[cand.model] = flag

# Frontier first: those chunks are the fewest, the dearest, and the ones whose
# absence silently breaks the escalation boundary, so an interrupted run should
# lose them last.
for model, flag in sorted(best.items(), key=lambda kv: (-RANK[kv[1]], kv[0])):
    print(f"{flag} {model}")
PY
)" || { echo "ERROR: could not enumerate models from the registry." >&2; exit 1; }

if [ -z "$MODEL_PLAN" ]; then
  echo "ERROR: registry enumeration returned no models." >&2
  exit 1
fi

# Seconds to idle between chunks. The first run's failures clustered
# immediately after long chunks, which is the signature of accumulated
# rate-limit pressure; pausing lets the budget refill before the next model.
COOLDOWN_S="${COOLDOWN_S:-60}"

FAILED=()

# Per-chunk spend ceiling. Frontier and premium cells cost multiples of a cheap
# cell, so a flat $1.00 would abort those chunks part-way rather than measure
# them. Override from the environment to tighten or loosen.
BUDGET_CHEAP="${BUDGET_CHEAP:-1.00}"
BUDGET_PREMIUM="${BUDGET_PREMIUM:-2.00}"
BUDGET_FRONTIER="${BUDGET_FRONTIER:-3.00}"

run_chunk() {  # $1 = tier flag (frontier|premium|cheap), $2 = model slug
  local budget
  case "$1" in
    frontier) budget="$BUDGET_FRONTIER" ;;
    premium)  budget="$BUDGET_PREMIUM" ;;
    *)        budget="$BUDGET_CHEAP" ;;
  esac
  echo "=== [$(date -u +%H:%M:%S)] START $2 ($1, budget \$$budget) ==="
  python3 -m theriad.eval.run \
    --corpus "$REPO/eval_corpus.jsonl" --grid "$GRID" \
    --"$1" "$2" --samples 1 \
    populate --allow-paid-calls --yes --budget "$budget"
  local code=$?
  echo "=== [$(date -u +%H:%M:%S)] EXIT $code for $2 ==="
  # populate now exits non-zero if any call errored, so a chunk that failed
  # silently in the first run cannot report success here.
  if [ "$code" -ne 0 ]; then
    FAILED+=("$2")
    echo "!!! $2 had errors — see above; its cells were NOT recorded and will"
    echo "!!! be retried the next time this script runs."
  fi
  sleep "$COOLDOWN_S"
}

{
  echo "########## RUN STARTED $(date -u) ##########"
  # Herestring, not a pipe: the loop must run in this shell so FAILED survives.
  while read -r flag model; do
    [ -n "$model" ] || continue
    run_chunk "$flag" "$model"
  done <<< "$MODEL_PLAN"
  echo "########## ALL CHUNKS DONE $(date -u) ##########"
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo "########## ${#FAILED[@]} MODEL(S) HAD ERRORS ##########"
    printf '  %s\n' "${FAILED[@]}"
    echo "Re-run this script to retry them."
  else
    echo "########## ALL MODELS CLEAN ##########"
  fi
} 2>&1 | tee -a "$LOG"

echo
echo "Grid: $GRID"
echo "Next: python3 -m theriad.eval.run --corpus eval_corpus.jsonl --grid eval_grid.json \\"
echo "        report --values 0.001,0.01,0.10,1.00 --json report.json"
