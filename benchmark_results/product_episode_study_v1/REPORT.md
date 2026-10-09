# Router episode study

Result: **inconclusive**. Production routing remains unchanged.

Committed study spend (including unresolved reservations): **$2.339057 / $100**.
Known incurred attempt charges: **$0.05334086**; unresolved attempt charges: **12**.

| Stage | Episodes | Successes | Known episode spend | Unknown charges | Operational failures |
|---|---:|---:|---:|---:|---:|
| pilot | 10 | 8 | $0.002531 | 0 | 2 |
| development | 89 | 53 | $0.042298 | 12 | 33 |
| test | 20 | 12 | $0.005851 | 0 | 4 |
| smoke | 10 | 6 | $0.002661 | 0 | 2 |

## Locked test observations

Baseline success: **60%**; challenger success: **60%**.
Mean episode cost: **$0.00041046** baseline and **$0.00017466** challenger.

| Family | Baseline success | Challenger success | Baseline mean cost | Challenger mean cost | Baseline p50 / p95 seconds | Challenger p50 / p95 seconds |
|---|---:|---:|---:|---:|---:|---:|
| chat | 50% | 50% | $0.00005682 | $0.00004311 | 3.4 / 4.4 | 6.2 / 6.8 |
| coding | 100% | 100% | $0.00000000 | $0.00013984 | 19.8 / 30.7 | 5.0 / 7.5 |
| tools | 100% | 50% | $0.00020367 | $0.00024085 | 37.3 / 40.6 | 42.4 / 58.3 |
| search | 50% | 100% | $0.00120931 | $0.00009867 | 28.9 / 37.6 | 15.7 / 18.5 |
| agents | 0% | 0% | $0.00058250 | $0.00035084 | 104.0 / 142.7 | 73.5 / 111.8 |

Diagnostic one-sided 95% bounds: quality difference lower bound **-20.0 percentage points**; cost difference upper bound **$-0.00015461 per episode**.
These small-sample bootstrap bounds cannot establish improvement.

The pilot power estimate called for **1,355 independent groups** for 80% power at a five-point lift; the frozen test had **10**.

All stage spending is reserved before dispatch. Unknown charges retain the full reservation and cannot support a cost claim. Interrupted trajectories stop collection.

## Interpretation

- The locked test has 10 independent groups; consult the power plan before interpreting a five-point target.
- This implementation run did not freeze a complete source snapshot before its final test.
- Product fixture controls omit account HTTP gates, personal memory and live circuit state; no real-traffic cost claim.
- Terminal project agents and frozen search evidence do not establish live browser or live web-search performance.
- Deterministic structured grading covers frozen criteria, not broad open-ended conversational quality.
- No production promotion occurs; a favorable point estimate is not an improvement claim.

## Reproduce

`.venv/bin/python scripts/run_router_episode_study.py report --directory /Users/puranvs/Cost-Optimization/benchmark_results/product_episode_study_v1`

Detailed paired intervals, family latency, policy identities, frozen fixtures and durable attempts are in adjacent JSON/JSONL files.
