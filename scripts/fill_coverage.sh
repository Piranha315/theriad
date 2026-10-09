#!/bin/bash
# One-off: bring escalation targets to full 200-item coverage so the fallback /
# cross-tier sweeps can be measured on the same items as the base tier. The
# asymmetric coverage in grid.py is right for cost control but makes escalation
# targets unscoreable — most of their cells fall back to the Beta(1,1) prior.
# --cheap here means item_fraction=1.0 in the planner, not "a cheap model".
cd ~/Cost-Optimization || exit 1
source .venv/bin/activate 2>/dev/null
export OPENROUTER_API_KEY="$(grep -m1 'OPENROUTER_API_KEY' .env | sed 's/.*OPENROUTER_API_KEY[[:space:]]*=[[:space:]]*//' | tr -d '"'"'"'' | tr -d '\r')"
for spec in "z-ai/glm-5.2 1.00" "x-ai/grok-4.6 3.00" "meta/muse-spark-1.2 3.00"; do
  set -- $spec
  echo "=== [$(date -u +%H:%M:%S)] FILL $1 (budget \$$2) ==="
  python3 -m theriad.eval.run --corpus eval_corpus.jsonl --grid eval_grid.json \
    --cheap "$1" --samples 1 populate --allow-paid-calls --yes --budget "$2"
  echo "=== [$(date -u +%H:%M:%S)] EXIT $? for $1 ==="
  sleep 30
done
echo "########## FILL DONE $(date -u) ##########"
