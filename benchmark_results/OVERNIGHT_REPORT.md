# Theriad Overnight Maintenance Report
**Generated:** 2026-06-29  
**Session type:** Overnight-safe stabilization pass (no destructive actions)

---

## What Changed

### New files created
| File | Purpose |
|------|---------|
| `theriad/benchmark/preflight.py` | Disk/RAM/Ollama/API-key checks before any benchmark run |
| `theriad/benchmark/allowlist.py` | Overnight-safe local model filter; blocks large models by default |
| `theriad/benchmark/resume.py` | Load completed keys from existing JSONL; skip on re-run |
| `theriad/benchmark/presets.py` | Named presets: `smoke`, `overnight_safe`, `full` |

### Modified files
| File | Change |
|------|--------|
| `theriad/benchmark/runner.py` | Added resume support, per-row fsync, output caps, allowlist wiring |
| `theriad/benchmark/run.py` | Added `--resume`, `--preset`, `--allow-large-local-models`, `--skip-preflight` flags; preflight wired into CLI |
| `theriad/benchmark/types.py` | Added `output_preview`, `output_sha256`, `_make_output_preview()` |
| `theriad/benchmark/summary.py` | Added `median_latency_ms` to `ModelStats` and `_median()` helper |
| `theriad/benchmark/providers/ollama_lane.py` | Allowlist filtering in `models()`; output preview capture |
| `theriad/benchmark/providers/openrouter_lane.py` | Output preview capture |
| `theriad/accounting/pricing.py` | Added `PricingLookup` NamedTuple + `lookup()` method; added `xiaomi/mimo-v2.5` and `xiaomi/mimo-v2.5-20260422` at $0.105/$0.28 per 1M |
| `theriad/benchmark/formatting.py` | `format_usd()` — `None→"?"`, `0.0→"$0"`, tiny costs adaptive precision |

### Pricing fix (from previous session)
- `xiaomi/mimo-v2.5-20260422` now resolves to known pricing ($0.105 input / $0.28 output per 1M).  
- Tiny costs like `$0.0000483` now display correctly instead of `"?"`.

---

## Tests Run

```
454 passed, 1 warning
```

**New test files:**
- `tests/unit/benchmark/test_overnight.py` — 52 tests covering:
  - Preflight disk thresholds (< 10 GB blocked, 10–20 GB warn + blocks local, ≥ 20 GB OK)
  - Resume skips completed rows, loads error rows as retryable
  - Checkpoint: one row written per call, partial results readable after crash
  - Allowlist blocks large models by default; `--allow-large-local-models` overrides heuristic
  - Explicit block list (`qwen3.6:27b`, `deepseek-r1:latest`) not overrideable
  - Output caps applied per-category before lane call
  - Presets have correct budgets and caps
  - `output_preview` / `output_sha256` roundtrip through JSONL
  - Partial summary on incomplete JSONL
  - Median latency computed correctly
  - Cost formatting: MiMo shows real cost, unknown shows `"?"`

**Pre-existing tests all still pass** (402 tests across all modules).

---

## Preflight Check Results (this machine, at run time)

| Check | Value | Status |
|-------|-------|--------|
| Disk | 10.8 GB free | ⚠ WARN |
| Memory | 3.5 GB available | ✓ OK |
| OpenRouter key | present | ✓ OK |
| Ollama | unreachable | ⚠ WARN (not running) |
| Budget | $2.00 | ✓ OK |
| **Overall** | | **READY TO RUN** (WARN, not BLOCKED) |

**Disk interpretation:** 10.8 GB is in the 10–20 GB warn zone.  
- Local (Ollama) benchmark is **disabled** by preflight (`local_allowed = False`).  
- OpenRouter benchmark is **allowed**.

---

## Overnight Benchmark Attempt

**Command:** `python -m theriad.benchmark.run --preset overnight_safe --resume --budget 2.00 --yes`

**Outcome:** Benchmark ran in sandbox environment which has a SOCKS proxy configured.  
The proxy requires the `socksio` package (not installed), blocking outbound HTTPS.  
30 calls were attempted and all failed with:  
`Using SOCKS proxy, but the 'socksio' package is not installed.`

**Recovery:** The 30 error rows were removed from `latest.jsonl`. The file was restored to the original 60 successful rows.

**No money was spent. No data was lost.**

---

## Existing Benchmark Results (from previous run on user's machine)

| Provider | Model | N | Avg Latency | Median Latency | Avg tok/s | Total Cost | Quality |
|----------|-------|---|-------------|----------------|-----------|------------|---------|
| openrouter | `xiaomi/mimo-v2.5-20260422` | 29 | 4,814 ms | 3,127 ms | 20.0 | $0.001007 | 69% |
| openrouter | `qwen/qwen3.6-plus-04-02` | 1 | 35,110 ms | 35,110 ms | 53.7 | $0.000194 | 100% |
| ollama | `llama3.1:8b` | 30 | 3,844 ms | 3,521 ms | 16.1 | $0.000000 | 83% |

**Total rows:** 60 / 60 complete (both openrouter and ollama lanes finished)  
**Total known spend:** $0.001201  
**Budget used:** < 0.1% of $2.00 ceiling  

### Notes on quality scores
- MiMo (69%) reflects free-tier rate limiting and some thin outputs; not a quality signal.
- llama3.1:8b (83%) is solid for a local 8B model.
- The single qwen3.6-plus row (100%) is statistically meaningless.

---

## Rows Completed vs Remaining

| Provider | Model | Completed | Remaining |
|----------|-------|-----------|-----------|
| openrouter | openrouter/auto | 30 | 0 |
| ollama | llama3.1:8b | 30 | 0 |
| **Total** | | **60** | **0** |

The previous benchmark run is **fully complete** for the configured providers/models.

---

## Crash Risk Assessment

| Risk | Before | After |
|------|--------|-------|
| Lose all results on crash | HIGH | LOW — fsync after every row |
| Duplicate work on restart | HIGH | LOW — `--resume` skips completed keys |
| Disk exhaustion from large models | HIGH | LOW — allowlist + preflight |
| Budget overrun | MEDIUM | LOW — $2 hard ceiling + preflight check |
| Large local models running overnight | HIGH | LOW — blocked by default allowlist |

---

## No Destructive Actions Were Taken

- ✅ No files deleted
- ✅ No Docker prune or cache cleanup
- ✅ No `rm -rf` outside project output directories  
- ✅ No subscriptions purchased
- ✅ No fine-tuning attempted
- ✅ No large models run (Ollama not running; sandbox blocked network)
- ✅ Budget spent: $0.00 (sandbox blocked all calls)

---

## Recommended Next Steps (for tomorrow)

### 1. Run the summary on existing results
```bash
python -m theriad.benchmark.summary benchmark_results/latest.jsonl
```

### 2. When you want more benchmark coverage, run a fresh overnight-safe pass on your actual machine:
```bash
python -m theriad.benchmark.run \
  --preset overnight_safe \
  --resume \
  --budget 2.00
```
This will:
- Run preflight (check disk, RAM, Ollama, API key)
- Automatically skip the 60 already-complete rows
- Only run what's missing (0 rows currently — consider adding new models)

### 3. To add more models to benchmark, edit the provider lane files:
- `theriad/benchmark/providers/openrouter_lane.py` — add model slugs to `_OPENROUTER_MODELS`
- `theriad/benchmark/providers/ollama_lane.py` — models auto-discovered from running Ollama

### 4. To benchmark additional local models (if Ollama starts):
```bash
# Start Ollama, then:
python -m theriad.benchmark.run \
  --providers ollama \
  --preset overnight_safe \
  --resume \
  --budget 0.00
```
Ollama models will be filtered by the allowlist automatically.  
To include large models: add `--allow-large-local-models` (qwen3.6:27b and deepseek-r1 remain blocked regardless).

### 5. Disk space
Free up disk to reach ≥ 20 GB to re-enable local model benchmarking.  
Currently: ~10.8 GB free (warn zone).

---

## Command Reference

```bash
# Full benchmark (all providers, resume-safe, $10 budget)
python -m theriad.benchmark.run --preset full --resume --yes

# Overnight safe (small models only, $2 budget, resume)
python -m theriad.benchmark.run --preset overnight_safe --resume --budget 2.00

# Quick smoke test (6 prompts, fast)
python -m theriad.benchmark.run --preset smoke --yes --skip-preflight

# View results
python -m theriad.benchmark.summary benchmark_results/latest.jsonl

# Partial summary (works even on incomplete files)
python -m theriad.benchmark.summary benchmark_results/
```
