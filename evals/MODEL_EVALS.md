# Theriad Model Evaluation Guide

> **Purpose**: This document explains how Theriad uses public benchmarks and
> LMArena data to initialize model tiers, why public benchmarks alone are
> insufficient, and how Theriad-specific evals should be structured in future.

---

## 1. Using LMArena as a Quality Prior

Theriad uses LMArena leaderboard rankings as a **starting quality estimate**
(`quality_prior` field on `ModelCandidate`) when populating the model registry.

### What LMArena tells us

- Relative ranking on open-ended human preference (ELO score)
- Code/webdev specialization through the Coding Arena sub-leaderboard
- Broad multilingual and instruction-following quality

### What LMArena does **not** tell us

- Cost per token (prices change frequently)
- Latency and reliability of specific providers
- Quality on Theriad's specific task distribution (extraction, bulk, summarize)
- Whether a particular OpenRouter slug maps 1:1 to the arena model

### Policy

> LMArena rankings initialize `quality_prior`. Theriad-specific cost/success
> logs from real usage override public priors over time (ADR D-017).

Never treat LMArena as the final truth. A model that ranks highly on open-ended
conversation may underperform on structured extraction or JSON output tasks.

---

## 2. Recommended Benchmark Families by Task Type

These are reference benchmarks for initializing tier quality estimates.
They are NOT run automatically by Theriad — they must be executed externally
in a controlled eval harness (see Section 4).

| Task Type | Recommended Benchmarks / Priors |
|-----------|----------------------------------|
| General reasoning | MMLU-Pro, GPQA, LiveBench |
| Math | AIME 2024, MATH-500 |
| Coding (general) | LiveCodeBench, HumanEval+, MBPP |
| Coding (webdev) | LMArena Code/WebDev Arena, WebDev Arena ELO |
| Long context | RULER, NIAH (Needle-In-A-Haystack), LongBench |
| Structured output | JSON schema validity, exact-match extraction tests |
| Tool use | Berkeley Function Calling Leaderboard (BFCL) |
| Translation | FLORES-200 |

---

## 3. Why Public Benchmarks Do Not Replace Theriad-Specific Evaluation

Public benchmarks measure models on standardised test sets. Theriad routes
real user prompts, which differ in:

- **Distribution**: users send extraction, rewrite, and QA tasks, not MMLU questions
- **Cost sensitivity**: a 0.1% quality drop is acceptable if cost falls 80%
- **Output format**: many prompts require JSON, structured data, or concise answers
- **Latency**: a model that scores 90% but p99 latency is 30 s may be worse than
  one scoring 85% with p99 of 3 s

### Theriad's evaluation goal

> Find the **cheapest model that clears a task-specific quality threshold**,
> not the best model overall.

This means a model that is #10 on MMLU-Pro but cheap and fast may be the
right choice for `cheap_bulk`, while the #1 model is reserved for `premium`.

---

## 4. Quality Eval Structure (Future Implementation)

When implementing a Theriad-specific eval harness, follow these principles:

### 4.1 Never call providers by default

```bash
# ✓ Dry-run estimate (safe, free)
theriad simulate prompts.jsonl --baseline openai/gpt-4o

# ✗ Live eval (costs money, requires credentials)
theriad eval prompts.jsonl --live --allow-paid-calls
```

Any command that calls a live provider must:
1. Require `--allow-paid-calls` flag
2. Print an estimate of expected cost before proceeding
3. Require confirmation (`--yes`) or prompt the user interactively

### 4.2 Per-task quality thresholds

Each tier should have a minimum acceptable quality score for each task type.
Example:

| Tier | Task | Min Score | Metric |
|------|------|-----------|--------|
| cheap_bulk | JSON extraction | ≥ 0.95 exact match | exact match |
| cheap_code | HumanEval | ≥ 0.70 pass@1 | pass@1 |
| strong_reasoning | MMLU-Pro | ≥ 0.75 accuracy | accuracy |

### 4.3 Eval dataset format

Eval datasets should be JSONL files (same as `theriad simulate`):

```json
{"id": "eval_001", "prompt": "...", "expected": "...", "task_type": "extraction", "quality_threshold": 0.95}
```

### 4.4 Result logging

Eval results should be stored in `~/.theriad/evals/` as JSONL with:
- timestamp, model, tier, task_type, score, latency_ms, cost_usd

---

## 5. Costs of Running Quality Evals

Quality evals that call live providers **cost money**. Budget estimates:

| Eval set size | Baseline model | Estimated cost |
|---------------|----------------|----------------|
| 100 prompts | openai/gpt-4o | ~$0.50–2.00 |
| 1000 prompts | openai/gpt-4o | ~$5–20 |
| 100 prompts | qwen/qwen3-8b | ~$0.005–0.05 |

Always run `theriad simulate` first to estimate costs before running live evals.

### Current replacement study

The implemented harness is invoked as the `theriad` Python module (the former
`fusion.eval.run` path no longer exists). Build a fixed, vintage-tagged corpus:

```bash
python3 -m theriad.eval.run \
  --corpus eval_corpus.jsonl \
  build-corpus --sources mmlu,gsm8k,mmlu_pro --per-source 60
```

The named model set keeps the benchmark aligned with both the models currently
routed and the priceable replacement candidates. Planning is offline and
does not require provider credentials:

```bash
python3 -m theriad.eval.run \
  --corpus eval_corpus.jsonl --grid eval_grid.json \
  --model-set replacement-study --samples 5 plan
```

Inspect the printed per-model cost table before spending. Population is the
only paid step and retains both the explicit paid-call acknowledgement and the
hard budget guard.

**The budget guard refuses upfront, it does not stop partway.** `run.py` compares
the *whole* plan estimate against `--budget` and aborts if it exceeds it, so
`--budget` must be set above the figure `plan` printed or nothing runs at all.
Cost scales linearly with `--samples` over the 200-item corpus and the 18-model
set:

| `--samples` | calls  | est. USD |
|-------------|--------|----------|
| 1           |  2,614 |  $5.20   |
| 2           |  5,228 | $10.40   |
| 3           |  7,842 | $15.60   |
| 5           | 13,070 | $26.00   |

The grid resumes, so starting at `--samples 1` and adding samples later is
cheaper per decision than committing to 5 up front:

```bash
python3 -m theriad.eval.run \
  --corpus eval_corpus.jsonl --grid eval_grid.json \
  --model-set replacement-study --samples 1 \
  populate --allow-paid-calls --yes --budget 6.00
```

Two caveats on the current model set. `poolside/laguna-xs-2.1:free` has no
pricing entry and is scored as free rather than known-zero.

And OpenRouter returns `Insufficient credits` for key-level credit limits as
well as for an exhausted balance, so the message alone does not tell you which
you hit. The Aug 1 benchmark run took 60 of these in a 15-second burst and then
resumed successful paid calls three minutes later on a funded account — a
pattern that points at a limit trip, not a drained balance. Before a large
populate, check the key's credit and rate limits, not just the account total.

Reports are offline and can be regenerated for different values of a correct
answer without repeating provider calls:

```bash
python3 -m theriad.eval.run \
  --corpus eval_corpus.jsonl --grid eval_grid.json \
  report --values 0.001,0.01,0.10,1.00 --json report.json
```

---

## 6. Versioning Model Tier Quality Priors

When updating `quality_prior` values in `theriad/models/registry.py`:

1. Record the source and date in the `source` field (e.g., `"lmarena_code_2025q2"`)
2. Add a note in `DECISIONS.md` explaining what changed and why
3. Run `theriad simulate` on your standard eval set to project cost impact
4. If tier assignments change (e.g., moving a model from `cheap_code` to `code_strong`),
   run a full regression test suite

---

*Last updated: 2026-08-07 — replacement-study cost table measured against the
200-item corpus; prior `--budget 8.00` example corrected (it would have been
refused at a $26.00 estimate). Tier config reflects OpenRouter pricing and
LMArena rankings as of 2025-Q2.*
