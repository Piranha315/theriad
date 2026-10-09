#!/bin/bash
# Bring meta/muse-spark-1.2 — the frontier tier primary — to full 200-item
# coverage.
#
# Why this one and not grok-4.6: grok prices the accuracy *ceiling*, which is
# useful but not a tier the router should reach by default. muse-spark-1.2 is
# the actual frontier primary and at 38/200 cells it is the only tier that
# cannot be evaluated at all — every escalation decision into frontier scores
# on the Beta(1,1) prior (p=0.5), which is the flattering-by-absence failure
# that has already produced two wrong conclusions in this eval.
#
# --cheap here means item_fraction=1.0 in the planner, NOT "a cheap model".
# Deliberate: escalation targets must be scored on the same items as the base
# tier or the comparison is between different question sets.
#
# Resume is automatic — plan_grid skips recorded cells, so re-running after an
# interruption costs only what is still missing.
set -uo pipefail

cd ~/Cost-Optimization || exit 1
source .venv/bin/activate 2>/dev/null

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  OPENROUTER_API_KEY="$(grep -m1 'OPENROUTER_API_KEY' .env \
    | sed 's/.*OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*//' \
    | tr -d '"'"'"'' | tr -d '\r')"
  export OPENROUTER_API_KEY
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set and not found in .env" >&2
  exit 1
fi

MODEL="meta/muse-spark-1.2"
BUDGET="${BUDGET:-4.00}"

echo "=== [$(date -u +%H:%M:%S)] FILL $MODEL (budget \$$BUDGET) ==="
python3 -m theriad.eval.run \
  --corpus eval_corpus.jsonl --grid eval_grid.json \
  --cheap "$MODEL" --samples 1 \
  populate --allow-paid-calls --yes --budget "$BUDGET"
echo "=== [$(date -u +%H:%M:%S)] EXIT $? for $MODEL ==="
echo "########## FRONTIER FILL DONE $(date -u) ##########"
