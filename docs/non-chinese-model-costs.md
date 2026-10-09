# Cost of routing without China-origin models

> Status: advisory replacement study, not active routing configuration. The
> source of truth for current traffic is `theriad/models/registry.py`; the
> active registry intentionally includes China-origin candidates, including the
> unrestricted frontier primary.

**Question:** if the router is restricted to models from non-China vendors
(security/supply-chain concern), how much more does inference cost, and what are
the cheap substitutes?

**Short answer:** it is **cheaper** — about **-12% blended, -4% on a code-heavy
mix** — if `premium` moves to `openai/gpt-5.6-luna`. The security constraint is
not a cost to absorb; paying for it is the wrong framing. Two other tiers get
cheaper independently, and the free tiers are unaffected because they already
contain no China-origin models.

Prices are USD per 1M tokens from `theriad/accounting/pricing.py` and
`theriad/accounting/openrouter_prices.json`. Token profiles per tier use the
heuristics in `theriad/workflow/cost_simulator.py`. Premium and frontier
candidates are seeded at **list** price, not the promotional OpenRouter rate:
seeds feed the eval budget guard, and over-estimating spend is the safe error.

---

## 1. Where China-origin models sit today

From `theriad/models/registry.py`:

| Tier | Candidate | Vendor | Origin |
|---|---|---|---|
| `cheap_bulk` | `xiaomi/mimo-v2.5` (fallback) | Xiaomi | CN |
| `cheap_bulk` | `deepseek/deepseek-v4-flash-0731` (fallback) | DeepSeek | CN |
| `cheap_code` | `qwen/qwen3.6-plus` (fallback) | Alibaba | CN |
| `cheap_code` | `xiaomi/mimo-v2.5` (fallback) | Xiaomi | CN |
| `code_strong` | `deepseek/deepseek-v4-flash-0731` (primary) | DeepSeek | CN |
| `code_strong` | `qwen/qwen3.6-plus` (fallback) | Alibaba | CN |
| `strong_reasoning` | `deepseek/deepseek-v4-flash-0731` (fallback) | DeepSeek | CN |
| `premium` | `deepseek/deepseek-v4-flash-0731` (primary) | DeepSeek | CN |
| `premium` | `z-ai/glm-5.2` (fallback) | Zhipu | CN |
| `frontier` | `z-ai/glm-5.3-flash` (primary) | Z.AI | CN |

Everything else already qualifies:

- `free_or_floor`, `free_general`, `free_verify`, `free_summarize`, `strong_reasoning`
  primary — `nvidia/nemotron-3-*:free` (US)
- `free_code` — `poolside/laguna-xs-2.1:free` (US), `cohere/north-mini-code:free` (Canada)
- `premium` fallback — `openai/gpt-5.6-luna` (US)

So **every free tier is already China-free**, and the `strong_reasoning` primary
is too. Only the paid path is exposed, and only when free capacity is exhausted.

## 2. Substitutes, per tier

Cost is per 1,000 requests at that tier's token profile. `x` is the multiple of
the current China-origin candidate.

### `cheap_bulk` — 1,200 in / 400 out

| Model | Origin | $/1k req | vs current |
|---|---|---|---|
| `xiaomi/mimo-v2.5` *(current)* | CN | 0.280 | — |
| **`openai/gpt-oss-120b`** (Groq direct) | US | 0.420 | 1.50x |
| `google/gemini-2.5-flash` | US | 0.420 | 1.50x |
| `openai/gpt-oss-20b` (Groq direct) | US | 0.210 | **0.75x** |

Recommend `gpt-oss-120b` primary, `gemini-2.5-flash` fallback. `gpt-oss-20b` is
*cheaper than the status quo* and worth benchmarking for the trivial end of bulk
(extraction, rewrite) before defaulting to the 120b.

### `cheap_code` — 2,500 in / 2,500 out

| Model | Origin | $/1k req | vs current |
|---|---|---|---|
| `qwen/qwen3.6-plus` *(active fallback)* | CN | 5.688 | — |
| **`openai/gpt-oss-120b`** | US | 1.875 | **0.33x** |
| `gpt-5.4-mini` | US | 13.125 | 2.31x |
| `claude-haiku-4-5` | US | 15.000 | 2.64x |

This tier gets **~3x cheaper**. `qwen3.6-plus` carries a $1.95/1M output price
that is poor value on an output-heavy code profile; `gpt-oss-120b` at $0.60/1M
output beats it outright. This is a cost win available regardless of the
security question.

### `code_strong` — 2,500 in / 2,500 out

| Model | Origin | $/1k req | vs current |
|---|---|---|---|
| `deepseek/deepseek-v4-flash-0731` *(new active primary)* | CN | 1.050 | — |
| **`gpt-5.4-mini`** | US | 13.125 | 12.50x |
| `claude-haiku-4-5` | US | 15.000 | 14.29x |
| `claude-sonnet-5` | US | 45.000 | 42.86x |

DeepSeek is ~81% cheaper on the offline seed rates than Kimi was, with a 1M
context instead of 262K. The quality claim is mixed rather than universal:
OpenRouter's current code-category arena signal is 47.3% for DeepSeek versus
52.9% for Kimi, while the catalog's Artificial Analysis coding index is 69.1
versus 60.8. DeepSeek is also text-only, so image requests must still pass
through the vision fallback gate. This is a cost/context decision that needs a
Theriad-specific code evaluation.

### `strong_reasoning` — 1,600 in / 900 out

| Model | Origin | $/1k req | vs current |
|---|---|---|---|
| `deepseek/deepseek-v4-flash-0731` *(active fallback)* | CN | 0.476 | — |
| **`google/gemini-2.5-flash`** | US | 0.780 | 1.64x |
| `openai/gpt-oss-120b` | US | 0.780 | 1.64x |

Low impact in practice: the tier's primary is already the free NVIDIA Nemotron
Super, so this only prices overflow traffic.

### `premium` — 1,600 in / 900 out

| Model | Origin | $/1k req | vs current |
|---|---|---|---|
| `deepseek/deepseek-v4-flash-0731` *(active primary)* | CN | 0.48 | — |
| **`openai/gpt-5.6-luna`** @ OpenRouter promo (0.10/0.60) | US | **0.70** | **0.21x** |
| **`openai/gpt-5.6-luna`** @ list (0.20/1.20) | US | **1.40** | **0.41x** |
| `x-ai/grok-4.5` | US | 8.60 | 2.53x |
| `anthropic/claude-opus-5` | US | 30.50 | 8.96x |

**`openai/gpt-5.6-luna` is the answer here, and it is cheaper than what it
replaces.** At OpenAI's list price of $0.20/$1.20 per 1M it costs **0.41x**
GLM-5.2 on this tier's profile; at OpenRouter's current promotional rate
($0.10/$0.60) it is **0.21x**. Either way the premium tier gets cheaper, not
more expensive — which inverts the original premise of this document.

It is also not a quality downgrade. On the Artificial Analysis Intelligence
Index, GPT-5.6 Luna (max) scores **51 — tied with GLM-5.2 (max)**.

The catch is **latency**, and it is substantial: Luna (max) takes ~121s
end-to-end against GLM-5.2's ~13s, roughly 9x. The lower effort settings trade
index points for speed — Luna (xhigh) is 49 at ~40s, Luna (high) is 46 at
~10s. If premium traffic is user-facing, `high` or `xhigh` is likely the right
operating point and the tier's quality ceiling drops accordingly. Measure this
before committing: reasoning effort is a per-request parameter, so the choice
does not have to be permanent.

`x-ai/grok-4.5` remains a credible alternative at 2.53x, and scores 54 — three
index points above both Luna (max) and GLM-5.2, at ~17s. Note the disagreement
between metrics here: on Artificial Analysis *cost per task* Grok looks close to
GLM ($0.35 vs $0.30) because it is less verbose across their suite, but on
*per-token* pricing against this tier's profile it is 2.53x. Which figure is
right depends on how verbose Grok is on your workload, and that is precisely
what the outcome grid measures.

Cached input on Grok is $0.50/1M, a 75% discount for workloads with a stable
prefix.

## 3. Blended impact

The blended figures below are the pre-DeepSeek code-route snapshot retained for
audit; rerun the simulator before quoting them as current post-Kimi-removal
totals.

Assuming the free tiers absorb ~60% of requests (they cost $0 on both sides) and
using the recommended substitute per tier:

| Traffic mix | Current | `premium`→Luna (list) | `premium`→Grok 4.5 |
|---|---|---|---|
| Balanced (bulk .15 / code .10 / code_strong .08 / reasoning .05 / premium .02) | $1.55 /1k | **$1.37 — -12%** | $1.51 — -2% |
| Code-heavy (.10 / .15 / .20 / .05 / .05) | $3.19 /1k | **$3.06 — -4%** | $3.42 — +7% |
| Bulk-heavy (.30 / .05 / .03 / .05 / .01) | $0.74 /1k | **$0.67 — -10%** | $0.74 — -1% |

At OpenRouter's promotional Luna rate the balanced figure improves further to
-13%. Every mix is cheaper than the status quo, so the exercise is no longer
"what does security cost" but "why were we paying more for GLM-5.2".

The `premium` substitute still dominates sensitivity — it is the only tier where
the candidates differ by more than a factor of two — but the direction has
flipped. Even Grok, at 2.53x on the tier, is roughly break-even blended because
`cheap_code` saves more than `premium` costs.

## 4. Recommendation

1. **Swap `cheap_code` to `openai/gpt-oss-120b` now.** It is cheaper *and*
   removes a China-origin primary. No trade-off to weigh.
2. **Swap `cheap_bulk` and `strong_reasoning` fallbacks** to `gpt-oss-120b` /
   `gemini-2.5-flash`. ~1.5x on tiers that are already fractions of a cent.
3. **Keep `code_strong` on `deepseek/deepseek-v4-flash-0731`** for the lower
   cost and larger context, while treating the arena/AA disagreement as a
   reason to run the Theriad code evaluation rather than a universal quality
   claim.
4. **Move `premium` to `openai/gpt-5.6-luna`** — pending a benchmark run and a
   decision on reasoning effort. It ties GLM-5.2 on the Intelligence Index and
   costs 0.41x at list, so this is a cost improvement rather than a concession.
   The open question is latency: Luna (max) is ~9x slower than GLM-5.2, so
   confirm the effort setting against your latency budget before switching.
   `x-ai/grok-4.5` is the alternative if latency matters more than price — three
   index points higher, ~17s, 2.53x.

Note that "non-China" here is vendor origin, not hosting. Nothing above changes
where inference physically runs — all paid candidates still route through
OpenRouter or a direct US endpoint. If the concern is data residency rather than
model provenance, this analysis is the wrong lever.

## 5. Is there a tier above `premium`?

Yes, and it is worth adding — but the two obvious candidates split.

Costs below are **Artificial Analysis cost-per-task**, not per-token. That is the
right metric for tier ordering because it prices verbosity: a model that scores
the same for half the output tokens is genuinely cheaper. It is *not* a
substitute for per-token pricing in `theriad/accounting` — AA's task mix is not
Theriad's.

### The intelligence/cost Pareto frontier

The table below is the prior Artificial Analysis study snapshot; it predates
Muse Spark 1.2. The current routing decision and its live OpenRouter evidence
are recorded below.

Nothing cheaper scores as high:

| Model | Origin | Intelligence | $/task | e2e response |
|---|---|---|---|---|
| GPT-5.6 Luna (max) | US | 51 | 0.29 | 121.0s |
| **Grok 4.5 (high)** | US | **54** | **0.35** | **17.0s** |
| **Claude Opus 5 (medium)** | US | **56** | **0.62** | **15.4s** |
| GPT-5.6 Sol (high) | US | 56 | 0.62 | 20.1s |
| Kimi K3 | CN | 57 | 0.72 | 80.7s |
| GPT-5.6 Sol (xhigh) | US | 58 | 0.94 | 41.0s |
| Claude Opus 5 (high) | US | 59 | 1.06 | 25.1s |
| Claude Opus 5 (xhigh) | US | 60 | 1.56 | 51.5s |
| Claude Opus 5 (max) | US | 61 | 2.03 | 68.9s |

### Grok 4.5 (high) — yes

It is the cheapest way to buy anything meaningfully above 51, at +$0.09/task
over Luna (max) for +3 index points, and it is dramatically faster end-to-end
(17.0s vs 121.0s). It also strictly dominates the slower high-effort Luna
option in this study, which scores 51 for $0.29/task and takes **121s** to
respond.

### Kimi K3 — no

Not because it is Chinese. Because it does not earn a slot:

- **It is not the quality ceiling.** At 57 it sits below Opus 5 high (59),
  Opus 5 xhigh (60), Sol max (59), and Fable 5 (60). A top tier exists to be the
  best available option, and K3 is not it.
- **It is not the value pick either.** Claude Opus 5 (medium) scores 56 for
  $0.62 — one index point lower for 14% less money and, decisively,
  **15.4s vs 80.7s** end-to-end. At 33 tokens/s K3 is the slowest model on the
  board.

K3 is on the price frontier only in the narrow sense that nothing else scores
exactly 57 for less. Adjust for latency and it is dominated. Adding it would
reintroduce CN exposure in exchange for nothing.

### Current routing decision (2026-08-26)

- **`code_strong` primary → `deepseek/deepseek-v4-flash-0731`**. Kimi K2.7 is
  removed from active registry/default routing; its pricing entries remain only
  for historical logs and explicit legacy requests.
- **`premium` primary → `deepseek/deepseek-v4-flash-0731`** (China-origin), with
  Luna as the first non-China fallback and the existing candidates retained.
  The registry keeps this deliberate price/quality exception documented in its
  candidate note; this page's non-China recommendation remains a constraint,
  not the unconditional default.
- **`frontier` primary → `z-ai/glm-5.3-flash`** (Z.AI, CN) for the default
  unrestricted route, with Muse Spark 1.2 retained as the first U.S.-origin
  fallback and Grok 4.6 as the independent fallback. Artificial Analysis
  reports an Intelligence Index of 57.46 for GLM 5.3 Flash versus 56.76 for
  Muse Spark 1.2. OpenRouter currently lists GLM at the temporary discounted
  rate of $0.075/$0.25 per 1M input/output tokens; its catalog reports a
  1M-token context, multimodal input, and mandatory reasoning.
- **`model_origin_policy=us_only` remains unchanged in behavior:** the origin
  filter excludes GLM and resolves the frontier tier to Muse Spark 1.2, then
  Grok 4.6. The offline registry/accounting seed for GLM is the conservative
  non-promotional $0.15/$0.50 list rate; the live catalog overlay picks up the
  discounted rate while it lasts.
- Eval runs still pin Muse Spark 1.2 to the **Meta-only OpenRouter provider with
  fallbacks disabled** (`META_ONLY` in `theriad/eval/executors.py`). GLM 5.3
  Flash is included in the frontier replacement-study set, but no paid
  Theriad-specific evaluation was run as part of this routing change. Claude
  Opus remains priceable and present as the incumbent comparison control.

These are public benchmark priors, not Theriad benchmark results. The
replacement-study result still gates a claim that quality is validated.

## 6. Caveats

- Quality priors for the substitutes are **not measured on Theriad's corpus** —
  this includes the DeepSeek code route, GLM 5.3 Flash, and Muse Spark 1.2, as well as
  `gpt-oss-120b` in `cheap_code` and `code_strong`. They
  should go through `theriad.benchmark` before a quality claim is promoted (per
  the policy in `evals/MODEL_EVALS.md`).
- No production traffic logs exist in this repo, so the mixes in §3 are stated
  assumptions, not observations. Per-tier multiples in §2 are mix-independent
  and are the durable numbers here.
- Prices are the offline seeds. Live OpenRouter prices override them at runtime
  (`theriad/accounting/live_prices.py`); re-run this comparison against a fresh
  catalog before acting on the margins.
- Premium and frontier prices are now seeded in `theriad/accounting/pricing.py`
  (`openai/gpt-5.6-luna`, `x-ai/grok-4.6`, `z-ai/glm-5.3-flash`,
  `meta/muse-spark-1.2`, and the Opus comparison control), so the
  cost simulator and eval budget guard can price them. Quality figures still come
  from Artificial Analysis rather than a Theriad benchmark run — per
  `evals/MODEL_EVALS.md`, that run gates promotion.
- Luna is seeded at OpenAI list price. OpenRouter's promotional rate is currently
  half that, which would make the tier cheaper still, but a seed that understates
  cost is the one failure the budget guard cannot catch — so the promo is
  documented, not baked in. The live catalog overlay picks up the real rate at
  runtime.
- §5 is built entirely on Artificial Analysis cost-per-task and latency figures.
  Per-token prices for Grok 4.5, Kimi K3, and the Opus 5 reasoning-effort
  variants are **not** in this repo's tables, so §5 costs cannot be compared
  directly against the $/1k-req figures in §2. Add them to
  `theriad/accounting/pricing.py` before acting.
- Reasoning-effort variants (`high`/`xhigh`/`max`) are distinct operating points,
  not distinct models. Whether the router can select one depends on the provider
  adapter exposing a reasoning-effort parameter — unverified for all three
  vendors named in §5.

---

## Postscript (2026-08-02): the premium tier now leads with a China-origin model

`deepseek/deepseek-v4-flash-0731` is the `premium` primary. That directly
contradicts everything above, and the contradiction is deliberate rather than an
oversight, so it is recorded here instead of leaving the document and the
registry disagreeing.

**What changed.** The 0731 release scores **50** on the AA Intelligence Index — a
10-point jump over the April V4 Flash and one point behind GPT-5.6 Luna — at the
same price as the April model. On this tier's profile that is **$0.48 per 1k
requests against Luna's $1.40**, roughly a third, or $0.31 at OpenRouter's
current rate. It sits on AA's Pareto frontier for intelligence versus cost per
task. Nothing non-China comes close on that trade.

**What this document is still for.** The analysis above stands on its own terms:
if the constraint is applied, it is now *cheaper* to apply it than not, because
Luna beats GLM-5.2 on price at equal index. What changed is not the arithmetic
but the priority — a price/quality decision was taken to override the origin
constraint at the head of the chain.

**What that costs, concretely.** China-origin exposure on the paid path goes
*up*, not down: `premium` traffic now lands on DeepSeek by default, where before
it landed on GLM-5.2. If the original security concern is live, this is the tier
to revisit first, and the fix is a one-line role swap — Luna is the very next
candidate and needs no other change.

**What remains true regardless.** Every free tier is still China-free, and those
absorb most traffic. `cheap_code` is still the largest unclaimed saving in the
document (~3x on `openai/gpt-oss-120b`) and is unaffected by this decision.
