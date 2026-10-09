# GIST-small prompt-difficulty training

This workflow fine-tunes
[`avsolatorio/GIST-small-Embedding-v0`](https://huggingface.co/avsolatorio/GIST-small-Embedding-v0)
to rank prompts by outcome-derived difficulty. It does **not** treat vocabulary
sophistication as difficulty, and it does not produce a calibrated probability
that any particular model will succeed.

## Dataset decision

| Dataset | Use | Rows / splits | License | Why |
|---|---|---:|---|---|
| [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset) | Default primary | 109,101 train; 10,000 validation | Apache-2.0 | Broad real prompts with a 1–5 estimate of weak-model performance relative to GPT-4. |
| [`asdwb/route_balance_model_estimator`](https://huggingface.co/datasets/asdwb/route_balance_model_estimator) | Default outcome supplement | 14,919/3,634 Qwen and 8,086/1,979 Gemma train/test | CC-BY-4.0 | The same prompt is scored across several model sizes; this is direct router supervision. |
| [`agentlans/prompt-difficulty`](https://huggingface.co/datasets/agentlans/prompt-difficulty) | Research reference only | 99,758 train | CC-BY-4.0 | Broad coverage, but labels are eight models' opinions rather than observed task success. The card reports that harmful prompts are spuriously rated difficult. |
| [`JiaqiXue/R2-Bench`](https://huggingface.co/datasets/JiaqiXue/R2-Bench) | Future token-budget extension | 30,968 queries; roughly 4.95M scored generations | MIT | Excellent direct supervision, but the current auto-converted table omits model/budget provenance and is about 2.8 GB of Parquet. Do not silently aggregate it until the adapter preserves those fields. |
| [`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench) | Excluded pending license clarification | 30k+ prompts across 11 models | Not declared | Strong schema fit, but no dataset license is declared on the Hub card. |
| [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | Diagnostic only | 7,787 | CC-BY-SA-4.0 | Easy/Challenge is useful as a transfer slice, but training on it would encourage source-format recognition. |
| [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Diagnostic only | 500 | Not declared on this card | Has levels 1–5, but it is tiny and intended as an evaluation set. |

The two default repositories and the base model are pinned by immutable Hub
revision in `theriad/routing/gist_training.py`.

## Target construction

The sources have incompatible raw scales, so they are not concatenated naively:

- RouteLLM: `raw_difficulty = (5 - mixtral_score) / 4`.
- RouteBalance: `raw_difficulty = 1 - mean(cross-model DeepEval score)`;
  truncated generations are ignored and safety prompts are excluded by default.
- Each source is mapped through the empirical CDF of its **training** labels.
  Validation and test use those frozen training boundaries.
- Duplicate prompts are grouped globally. Any prompt appearing in a test split
  is excluded from training in every source.

The learned score is therefore a relative difficulty rank in `[0, 1]`. It must
be calibrated against an untouched Theriad model/outcome grid before production
routing. It must not be interpreted as `P(model succeeds)`.

## Installation

```bash
python -m pip install -e '.[training]'
```

No `datasets` or `pyarrow` installation is required. Pinned JSONL files are
streamed over HTTPS and never cached as full local copies. A full preparation
streams roughly 436 MB from Hugging Face.

## Commands

Validate the live schemas and preprocessing with a bounded sample, without
loading an embedding model:

```bash
python scripts/train_gist_difficulty.py \
  --prepare-only \
  --max-routellm-train 20 \
  --max-routellm-eval 20 \
  --max-routebalance-train 20 \
  --max-routebalance-eval 20 \
  --output-dir /tmp/gist-difficulty-prepare
```

Run the default fine-tuning job:

```bash
python scripts/train_gist_difficulty.py \
  --output-dir artifacts/gist-difficulty
```

The default trains only the final two BERT blocks plus the scalar head. This is
deliberately conservative: GIST's own model card warns that fine-tuning coverage
can improve some domains while degrading others. Compare against the frozen
encoder baseline:

```bash
python scripts/train_gist_difficulty.py \
  --train-last-n-layers 0 \
  --output-dir artifacts/gist-difficulty-head-only
```

On Apple Silicon, automatic device selection uses MPS and forces eager BERT
attention because PyTorch's MPS SDPA backend does not support training dropout.
Use `--device cpu` only as a slower fallback if another MPS operation fails.

The output contains:

- `encoder/`: the fine-tuned Sentence Transformer;
- `difficulty_head.safetensors`: the scalar ranking head;
- `prepared/{train,validation,test}.jsonl`: auditable normalized examples;
- `prepared/manifest.json`: input counts and pinned source URLs;
- `training_manifest.json`: parameters, licenses, metrics, and deployment gate.

## Required promotion checks

Do not replace the current router from these training metrics alone. Promotion
requires all of the following:

1. The fine-tuned encoder beats the frozen-encoder baseline on RouteBalance's
   untouched official test prompts.
2. It also beats the current surface-feature predictor on an untouched Theriad
   corpus, split by source or vintage rather than random rows.
3. A calibration layer maps its rank score to model-specific success
   probabilities and is evaluated with calibration error as well as rank.
4. Routing is shadowed with a global kill switch and observed cost ceiling
   before it changes model selection.
