# Public episode corpus — acquired 2026-09-10

This is a usable-format **candidate pool**, not a claim of 1,163 statistically certified independent observations or unseen model-pretraining data. No model outcomes were collected and no paid calls were made.

| New source | Tasks | Train | Calibration | Threshold selection | Held-out test | Pilot | Smoke |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 500 | 219 | 84 | 73 | 98 | 16 | 10 |
| HumanEval | 163 | 70 | 23 | 32 | 30 | 5 | 3 |
| HotpotQA | 500 | 238 | 80 | 64 | 92 | 16 | 10 |

`new-episodes.jsonl` contains 1,163 new task rows; `combined-episodes.jsonl` adds the unchanged 50-task seed for **1,213 rows total**. The latter preserves coverage of all six splits and five existing workload families. It contains 225 test candidates in total, including the original five.

The three added sources clear the basic training/calibration row-count shortage. Tools and agents still have only their original seed fixtures: their test strata have one group each and cannot support inference. Do not claim a powered five-family study.

## Source and independence controls

- Original HumanEval and GSM8K data are pinned to Git commits; HotpotQA's public Hugging Face mirror is pinned to its dataset revision after the original CMU endpoint timed out. Every retained source file has a SHA-256 checksum. The source READMEs and license/attribution files are preserved.
- Original problem wording is preserved; explicit output-format instructions are added. No paraphrased, renamed, or randomized synthetic expansion is used.
- The available local history screened is the 200-row legacy benchmark plus the current 50-task seed. Legacy answer-format wrappers are removed for comparison. Exact/near-duplicate wording is excluded. This does not attest to every older corpus or provider's training set.
- GSM8K arithmetic traces are grouped after replacing constants, with at most one selected problem per trace template; all 48 templates represented by the legacy GSM8K rows are excluded (1,608 source rows). This conservative filter changes the problem distribution and is not production weighting.
- HumanEval reference ASTs are compared after normalizing function/variable names and constants. References are parsed only, never executed on the host. One structurally/lexically related candidate is excluded.
- Every selected HotpotQA task has a disjoint set of **all original context titles**, even titles omitted from the adapted fixture. The runner receives all supporting paragraphs and up to two bounded distractors, at most four documents. This is a frozen-evidence adaptation, not the original ten-document or live-web benchmark. Strict short-answer grading differs from the official normalization/F1 metric.
- Groups are assigned reproducibly to train/calibration/threshold/test/pilot/smoke before model calls. These are router-study splits, not the upstream dataset's original train/test labels. Do not fit or tune on the held-out rows or their reference answers.

Mechanical checks reduce known dependencies; they cannot establish semantic independence or rule out public-benchmark memorization. These remain screened candidate groups pending substantive corpus review. Treat the output as research coverage, not a promotion decision.

## What was verified

- All 1,163 new rows and all 1,213 combined rows validate against the current Episode schema and split/group invariants.
- All 18 retained source files pass checksum verification.
- All 1,000 text-answer fixtures accept the stored answer and reject a deliberately wrong answer using the actual grader.
- All 163 code references and their test harnesses parse; required entry points, check functions, and assertions are present. Reference correctness has **not** been executed; the real sandbox remains unavailable.
- All 500 frozen-search fixtures return and fetch their documents through the actual FrozenEvidenceService without network access.
- Eight importer regression tests pass. Unicode line separators are escaped so the existing runner's `splitlines()` parsing preserves one JSON record per line.

See `validation.json`, `acquisition-report.json`, `rejections.jsonl`, and `ATTRIBUTION.md` for the detailed evidence. `reference-answers.jsonl` is grader/reference material and must never be appended to a model's task input.

## Additional native agent data

The package retains **2,449 raw tau2 task definitions**: airline 50, retail 114, telecom 2,285, with a pinned source revision and MIT license. `adapter-needed.json` explicitly gives these **zero usable Episode rows** and **no independent-group count**. They need native domain tools/databases, a user simulator, official state/action graders, and lineage grouping. Many telecom tasks are compositional variants. They cannot simply be relabeled as independent project-workspace episodes.

## Reproduction and next use

The offline builder is `scripts/build_public_episode_corpus.py`. It verifies the acquisition-cache manifest and refuses to overwrite a package. The full-row validation script is `scripts/validate_public_episode_corpus.py`. Both avoid provider calls and reference-code execution.

For a new reviewed study, supply the absolute `combined-episodes.jsonl` path to the existing study runner's `prepare --corpus` option and use a fresh study directory. Preparation must refresh model eligibility/prices; no budget here authorizes collection. Review the candidate groups and add runnable tools/agents tasks before a final five-family test. Restore the real coding sandbox before any coding collection. Model-provider contamination remains a limitation even after these gates.
