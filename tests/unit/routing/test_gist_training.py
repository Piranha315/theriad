from __future__ import annotations

import json

import pytest

from scripts.train_gist_difficulty import model_kwargs_for_device
from theriad.routing.gist_training import (
    DifficultyExample,
    assign_routebalance_split,
    calibrate_target_quantiles,
    deduplicate_examples,
    iter_jsonl_lines,
    iter_utf8_lf_lines,
    normalise_prompt,
    route_balance_example,
    route_llm_example,
)


def test_normalise_prompt_decodes_conversation_json():
    assert normalise_prompt('["first turn", "second turn"]') == "first turn second turn"


def test_route_llm_reverses_weak_model_score():
    easy = route_llm_example({"prompt": "easy", "mixtral_score": 5}, "train")
    hard = route_llm_example({"prompt": "hard", "mixtral_score": 1}, "train")
    assert easy is not None and easy.target == 0.0
    assert hard is not None and hard.target == 1.0


def test_route_llm_rejects_invalid_score():
    assert route_llm_example({"prompt": "x", "mixtral_score": 0}, "train") is None
    assert route_llm_example({"prompt": "x", "mixtral_score": 6}, "train") is None


def _routebalance_row(*scores, harmful=False, truncated_index=None):
    models = {}
    for index, score in enumerate(scores):
        models[f"model-{index}"] = {
            "is_truncated": index == truncated_index,
            "llm_judge_scores": {"deepeval-llama3.1-8b-it_reference": score},
        }
    return {
        "request_id": "row-1",
        "prompt": "Explain the result",
        "dataset": "test-source",
        "is_harmful": harmful,
        "models": models,
    }


def test_routebalance_uses_mean_success_and_ignores_truncation():
    example = route_balance_example(
        _routebalance_row(0.2, 0.8, 1.0, truncated_index=2),
        config="qwen",
        split="train",
    )
    assert example is not None
    assert example.target == pytest.approx(0.5)


def test_routebalance_excludes_harmful_by_default():
    row = _routebalance_row(0.5, harmful=True)
    assert route_balance_example(row, config="qwen", split="train") is None
    assert route_balance_example(
        row, config="qwen", split="train", include_harmful=True
    ) is not None


def test_routebalance_split_preserves_official_test():
    example = route_balance_example(
        _routebalance_row(0.5), config="qwen", split="test"
    )
    assert example is not None
    assert assign_routebalance_split(example, original_split="test").split == "test"


def test_quantile_calibration_uses_training_boundaries_for_test():
    rows = [
        DifficultyExample("a", 0.0, "source", "train", "a", "p"),
        DifficultyExample("b", 0.5, "source", "train", "b", "p"),
        DifficultyExample("c", 1.0, "source", "train", "c", "p"),
        DifficultyExample("d", 0.75, "source", "test", "d", "p"),
    ]
    calibrated, boundaries = calibrate_target_quantiles(rows)
    test = next(row for row in calibrated if row.split == "test")
    assert boundaries["source"] == [0.0, 0.5, 1.0]
    assert test.target == pytest.approx(2 / 3)


def test_quantile_calibration_gives_ties_the_same_midrank():
    rows = [
        DifficultyExample("a", 0.0, "source", "train", "a", "p"),
        DifficultyExample("b", 0.0, "source", "train", "b", "p"),
        DifficultyExample("c", 1.0, "source", "train", "c", "p"),
    ]
    calibrated, _ = calibrate_target_quantiles(rows)
    values = [row.target for row in calibrated if row.target < 0.5]
    assert values == pytest.approx([1 / 3, 1 / 3])


def test_deduplication_keeps_duplicate_prompt_out_of_training():
    rows = [
        DifficultyExample("same prompt", 0.2, "a", "train", "1", "a"),
        DifficultyExample("same prompt", 0.8, "b", "test", "2", "b"),
    ]
    result = deduplicate_examples(rows)
    assert len(result) == 1
    assert result[0].split == "test"
    assert result[0].target == pytest.approx(0.8)


def test_jsonl_parser_reports_bad_line_number():
    with pytest.raises(ValueError, match="line 2"):
        list(iter_jsonl_lines(['{"ok": true}', "not-json"]))


def test_jsonl_parser_rejects_non_object():
    with pytest.raises(ValueError, match="not an object"):
        list(iter_jsonl_lines([json.dumps([1, 2, 3])]))


def test_lf_framing_preserves_unicode_line_separator_inside_json_string():
    chunks = [
        b'{"prompt": "first',
        "\u2028second".encode(),
        b'"}\n{"prompt": "third"}\n',
    ]
    rows = list(iter_jsonl_lines(iter_utf8_lf_lines(chunks)))
    assert rows == [{"prompt": "first\u2028second"}, {"prompt": "third"}]


def test_mps_forces_eager_attention_for_training_dropout():
    assert model_kwargs_for_device("mps") == {"attn_implementation": "eager"}
    assert model_kwargs_for_device("cpu") == {}
    assert model_kwargs_for_device("cuda") == {}
