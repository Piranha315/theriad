"""Regression tests for grading correctness and non-destructive offline regrades."""
from __future__ import annotations

import json

import pytest

from scripts.regrade_router_benchmark import build_regrade_audit, main
from theriad.benchmark.types import GRADER_VERSION, BenchmarkPrompt, _extract_mc_letter


def prompt(answer="4", answer_type="exact_match", **kwargs):
    return BenchmarkPrompt("item", "qa", "Question", answer=answer, answer_type=answer_type, **kwargs)


@pytest.mark.parametrize("output", [
    "A careful analysis is required; I cannot determine the answer.",
    "The answer is ambiguous.", "ANSWER: Apple", "ANSWER: A or B", "ANSWER: A. or B",
    "B or D, I think B", "I think it is B maybe",
    "We can rule out (A) and (B).", "A. first choice\nB. second choice",
    "ANSWER: A\nANSWER: I don't know",
])
def test_prose_ambiguous_choices_and_option_lists_are_not_answers(output):
    assert _extract_mc_letter(output) is None


@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
@pytest.mark.parametrize("template", ["ANSWER: {letter}", "The answer is **{letter}**.", r"\boxed{{{letter}}}"])
def test_explicit_choices_preserve_all_ten_letters(letter, template):
    assert _extract_mc_letter(template.format(letter=letter)) == letter


@pytest.mark.parametrize("output", [
    r"Thus, the correct answer is \(\boxed{C}\).",
    "The answer is:\n\\[\n\\boxed{C}\n\\]",
    "The final answer is $\\boxed{\\text{1,400.00}}$\n\nC",
])
def test_math_delimiters_and_final_choice_after_numeric_work_are_preserved(output):
    assert _extract_mc_letter(output) == "C"


@pytest.mark.parametrize("output", [
    "ANSWER: 4/5", "ANSWER: 4 + 5 = 9", "4.5", "40", "4 or 5", "4-5",
    "4 is wrong", "4 apples or oranges", "4 times two", "4,", "4e1",
])
def test_numeric_answers_do_not_ignore_calculations_or_alternatives(output):
    assert prompt().quality_check(output) is False


@pytest.mark.parametrize(("answer", "output"), [
    ("4", "4.0"), ("4", "ANSWER: $4."), ("4", r"ANSWER: $\boxed{4}$"),
    ("48", "48g"), ("10", "ANSWER: 10 liters"), ("14", "14 years ago"),
    ("1000", "ANSWER: 1,000"), ("1000000", "1,000,000"),
    ("-0.5", "-.5"), ("0.8", "ANSWER: 4/5"), ("4", "4e0"),
])
def test_numeric_equivalence_boxing_and_legitimate_units(answer, output):
    assert prompt(answer).quality_check(output) is True


def test_numeric_comparison_does_not_round_distinct_large_integers_together():
    assert prompt("9007199254740992").quality_check("9007199254740993") is False


def test_unit_in_answer_key_requires_full_match():
    assert prompt("4 oranges").quality_check("4 apples") is False


@pytest.mark.parametrize("kwargs", [
    {"answer": None, "answer_type": None},
    {"answer": None, "answer_type": "multiple_choice"},
    {"answer": "", "answer_type": "exact_match"},
    {"answer": "Z", "answer_type": "multiple_choice"},
    {"answer": "4", "answer_type": "unknown"},
    {"answer": None, "answer_type": None, "expected_properties": [""]},
])
def test_missing_or_invalid_rubric_is_explicitly_ungradable(kwargs):
    item = prompt(**kwargs)
    grade = item.grade("")
    assert grade.status == "ungradable"
    assert grade.grader_version == GRADER_VERSION
    assert grade.passed is None
    assert item.quality_check("") is None


def test_empty_answer_to_valid_question_is_incorrect_not_ungradable():
    assert prompt().grade("").status == "incorrect"
    assert prompt().quality_check("") is False


def write_sources(tmp_path, rows, *, trials=1, truncated=0):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"id": "item", "category": "qa", "prompt": "Question", "answer": "4", "answer_type": "exact_match"}) + "\n")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"cells": [{"model": "model", "item_id": "item", "trials": trials, "successes": trials, "truncated": truncated, "cost_usd": 0.02 * (trials + truncated)}]}))
    transcripts = tmp_path / "responses.jsonl"
    transcripts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return corpus, grid, transcripts


def row(**kwargs):
    return {"model": "model", "item_id": "item", "output": "ANSWER: 4/5", "graded_correct": True, "truncated": False, "cost_usd": 0.02, **kwargs}


def test_regrade_preserves_original_grade_and_cost_and_refuses_overwrites(tmp_path):
    sources = write_sources(tmp_path, [row()])
    original_bytes = [path.read_bytes() for path in sources]
    output = tmp_path / "audit.json"
    argv = ["--corpus", str(sources[0]), "--grid", str(sources[1]), "--transcripts", str(sources[2]), "--output", str(output)]
    assert main(argv) == 0
    report = json.loads(output.read_text())
    assert report["summary"]["changed_scored_grades"] == 1
    assert report["summary"]["coverage_complete"]
    assert report["cells"][0]["original"]["successes"] == 1
    assert report["cells"][0]["successes"] == 0
    assert report["cells"][0]["recorded_cost_usd"] == 0.02
    assert report["attempts"][0]["original_graded_correct"] is True
    assert report["attempts"][0]["grade"]["status"] == "incorrect"
    assert report["replacement_grid"] is False
    assert [path.read_bytes() for path in sources] == original_bytes
    with pytest.raises(SystemExit):
        main(argv)
    for source in sources:
        with pytest.raises(SystemExit):
            main([*argv[:-1], str(source)])
    assert [path.read_bytes() for path in sources] == original_bytes


def test_missing_transcript_is_not_imputed_from_old_success(tmp_path):
    sources = write_sources(tmp_path, [row(output="4")], trials=2)
    report = build_regrade_audit(*sources)
    assert not report["summary"]["coverage_complete"]
    cell = report["cells"][0]
    assert cell["original"]["successes"] == 2
    assert cell["graded_trials"] == 1
    assert cell["successes"] == 1


def test_truncation_remains_ungradable_and_spend_is_retained(tmp_path):
    sources = write_sources(tmp_path, [row(truncated=True)], trials=0, truncated=1)
    report = build_regrade_audit(*sources)
    assert report["summary"]["status_counts"] == {"ungradable": 1}
    assert report["cells"][0]["graded_trials"] == 0
    assert report["cells"][0]["recorded_cost_usd"] == 0.02


@pytest.mark.parametrize("kwargs", [{"output": None}, {"item_id": "missing"}, {"cost_usd": None}])
def test_missing_output_corpus_entry_or_price_prevents_complete_regrade(tmp_path, kwargs):
    sources = write_sources(tmp_path, [row(**kwargs)])
    report = build_regrade_audit(*sources)
    assert not report["summary"]["coverage_complete"]


def test_duplicate_or_extra_attempts_do_not_look_like_complete_coverage(tmp_path):
    sources = write_sources(tmp_path, [row(), row()])
    report = build_regrade_audit(*sources)
    assert not report["summary"]["coverage_complete"]
