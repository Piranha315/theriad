"""
Tests for HLE (Humanity's Last Exam) integration.

Covers:
- MC answer extraction from varied model response formats
- Exact-match normalisation and comparison
- Legacy expected_properties fallback is unchanged
- hle_converter: _is_image_question, _row_to_prompt, fetch_sample (mocked)
- prompts.py round-trips answer / answer_type through JSONL
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from theriad.benchmark.types import (
    BenchmarkPrompt,
    _extract_mc_letter,
    _normalize_exact,
)


# ── _extract_mc_letter ────────────────────────────────────────────────────────

class TestExtractMcLetter:

    def _ex(self, text: str) -> str | None:
        return _extract_mc_letter(text)

    def test_bare_letter_at_start(self):
        assert self._ex("A") == "A"
        assert self._ex("B\n") == "B"

    def test_letter_with_period(self):
        assert self._ex("C. Because the sky is blue.") == "C"

    def test_letter_with_paren(self):
        assert self._ex("(D) is the answer.") == "D"

    def test_answer_is_phrasing(self):
        assert self._ex("The answer is C.") == "C"
        assert self._ex("answer is: B") == "B"
        assert self._ex("The answer is **E**") == "E"

    def test_bold_markdown(self):
        assert self._ex("**A**") == "A"
        assert self._ex("**A**. This is correct.") == "A"

    def test_lowercase_letter(self):
        assert self._ex("a") == "A"
        assert self._ex("the answer is c.") == "C"

    def test_multiline_letter_on_own_line(self):
        text = "Let me think...\nB\nTherefore..."
        assert self._ex(text) == "B"

    def test_letter_colon(self):
        assert self._ex("A: Correct because...") == "A"

    def test_returns_none_when_no_letter(self):
        assert self._ex("I don't know.") is None
        assert self._ex("The result is 42.") is None
        assert self._ex("") is None

    def test_ambiguous_prose_has_no_answer(self):
        assert self._ex("B or D, I think B") is None

    def test_answer_is_takes_priority_over_fallback(self):
        # "The answer is D" should win even if A appears earlier in the text
        assert self._ex("A is a possibility, but the answer is D.") == "D"


# ── _normalize_exact ──────────────────────────────────────────────────────────

class TestNormalizeExact:

    def test_lowercases(self):
        assert _normalize_exact("HELLO") == "hello"

    def test_collapses_whitespace(self):
        assert _normalize_exact("  foo   bar  ") == "foo bar"

    def test_strips_newlines(self):
        assert _normalize_exact("foo\nbar") == "foo bar"


# ── BenchmarkPrompt.quality_check ─────────────────────────────────────────────

class TestQualityCheck:

    def _prompt(self, **kw) -> BenchmarkPrompt:
        defaults = dict(id="t1", category="test", prompt="q", max_output_tokens=100)
        defaults.update(kw)
        return BenchmarkPrompt(**defaults)

    # MC path
    def test_mc_correct_letter(self):
        p = self._prompt(answer="A", answer_type="multiple_choice")
        assert p.quality_check("The answer is A.") is True

    def test_mc_wrong_letter(self):
        p = self._prompt(answer="A", answer_type="multiple_choice")
        assert p.quality_check("The answer is B.") is False

    def test_mc_case_insensitive_answer_field(self):
        p = self._prompt(answer="c", answer_type="multiple_choice")
        assert p.quality_check("C.") is True

    def test_mc_no_letter_in_output_is_false(self):
        p = self._prompt(answer="A", answer_type="multiple_choice")
        assert p.quality_check("I'm not sure.") is False

    # Exact-match path
    def test_exact_match_equal(self):
        p = self._prompt(answer="42", answer_type="exact_match")
        assert p.quality_check("42") is True

    def test_exact_match_case_insensitive(self):
        p = self._prompt(answer="Paris", answer_type="exact_match")
        assert p.quality_check("paris") is True

    def test_exact_match_whitespace_normalised(self):
        p = self._prompt(answer="hello world", answer_type="exact_match")
        assert p.quality_check("  hello   world  ") is True

    def test_exact_match_wrong_answer(self):
        p = self._prompt(answer="Paris", answer_type="exact_match")
        assert p.quality_check("London") is False

    # Legacy expected_properties path
    def test_legacy_substring_match(self):
        p = self._prompt(expected_properties=["foo", "bar"])
        assert p.quality_check("contains foo and bar") is True
        assert p.quality_check("contains foo only") is False

    def test_no_check_returns_unknown(self):
        p = self._prompt()
        assert p.quality_check("anything") is None

    def test_answer_field_takes_precedence_over_expected_properties(self):
        # When answer + answer_type are set, expected_properties is not consulted.
        p = self._prompt(
            answer="A",
            answer_type="multiple_choice",
            expected_properties=["MISSING_KEYWORD"],
        )
        assert p.quality_check("A") is True


# ── hle_converter helpers ─────────────────────────────────────────────────────

class TestHleConverter:

    def test_is_image_question_null(self):
        from theriad.benchmark.hle_converter import _is_image_question
        assert _is_image_question({"image": None}) is False
        assert _is_image_question({}) is False

    def test_is_image_question_empty_string(self):
        from theriad.benchmark.hle_converter import _is_image_question
        assert _is_image_question({"image": ""}) is False

    def test_is_image_question_dict_with_src(self):
        from theriad.benchmark.hle_converter import _is_image_question
        assert _is_image_question({"image": {"src": "data:image/png;base64,abc"}}) is True

    def test_is_image_question_dict_no_src(self):
        from theriad.benchmark.hle_converter import _is_image_question
        assert _is_image_question({"image": {"src": "", "bytes": None}}) is False

    def test_row_to_prompt_mcq(self):
        from theriad.benchmark.hle_converter import _row_to_prompt
        row = {
            "question": "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6",
            "answer": "B",
            "answer_type": "multiple_choice",
            "category": "Mathematics",
        }
        result = _row_to_prompt(row, 0)
        assert result["id"] == "hle_0000"
        assert result["answer"] == "B"
        assert result["answer_type"] == "multiple_choice"
        assert result["category"] == "mathematics"
        assert "max_output_tokens" not in result
        assert "multiple-choice" in result["prompt"]

    def test_row_to_prompt_exact(self):
        from theriad.benchmark.hle_converter import _row_to_prompt
        row = {
            "question": "Name the capital of France.",
            "answer": "Paris",
            "answer_type": "exact_match",
            "category": "Geography",
        }
        result = _row_to_prompt(row, 5)
        assert result["id"] == "hle_0005"
        assert result["answer"] == "Paris"
        assert result["answer_type"] == "exact_match"
        assert result["category"] == "geography"
        assert "max_output_tokens" not in result

    def test_row_to_prompt_category_slugified(self):
        from theriad.benchmark.hle_converter import _row_to_prompt
        row = {"question": "Q", "answer": "A", "answer_type": "multiple_choice",
               "category": "Computer Science"}
        result = _row_to_prompt(row, 0)
        assert result["category"] == "computer_science"

    def test_row_to_prompt_normalises_mc_type_variants(self):
        from theriad.benchmark.hle_converter import _row_to_prompt
        for variant in ("mc", "multiplechoice", "multiple_choice", "Multiple_Choice"):
            row = {"question": "Q", "answer": "A", "answer_type": variant, "category": "X"}
            assert _row_to_prompt(row, 0)["answer_type"] == "multiple_choice"

    def test_fetch_sample_mocked(self):
        """fetch_sample uses _fetch_page and _fetch_total_rows; verify filtering works."""
        from theriad.benchmark.hle_converter import fetch_sample

        fake_rows = [
            {
                "question": f"Question {i}?",
                "answer": "A",
                "answer_type": "multiple_choice",
                "category": "Mathematics",
                "image": None,
            }
            for i in range(20)
        ]
        # Add one image question that should be filtered
        fake_rows.append({
            "question": "Look at this image.",
            "answer": "B",
            "answer_type": "multiple_choice",
            "category": "Vision",
            "image": {"src": "data:image/png;base64,abc"},
        })

        with (
            patch("theriad.benchmark.hle_converter._fetch_total_rows", return_value=21),
            patch("theriad.benchmark.hle_converter._fetch_page", return_value=fake_rows),
        ):
            prompts = fetch_sample(sample=5, text_only=True)

        assert len(prompts) == 5
        # No image questions should be included
        assert all(p["category"] != "vision" for p in prompts)

    def test_fetch_sample_includes_image_when_all_types(self):
        from theriad.benchmark.hle_converter import fetch_sample

        fake_rows = [
            {
                "question": f"Question {i}?",
                "answer": "A",
                "answer_type": "multiple_choice",
                "category": "Vision",
                "image": {"src": "data:image/png;base64,abc"},
            }
            for i in range(10)
        ]

        with (
            patch("theriad.benchmark.hle_converter._fetch_total_rows", return_value=10),
            patch("theriad.benchmark.hle_converter._fetch_page", return_value=fake_rows),
        ):
            prompts = fetch_sample(sample=3, text_only=False)

        assert len(prompts) == 3

    def test_fetch_sample_category_filter(self):
        from theriad.benchmark.hle_converter import fetch_sample

        fake_rows = [
            {"question": f"Q{i}", "answer": "A", "answer_type": "multiple_choice",
             "category": "Mathematics" if i % 2 == 0 else "Physics", "image": None}
            for i in range(10)
        ]

        with (
            patch("theriad.benchmark.hle_converter._fetch_total_rows", return_value=10),
            patch("theriad.benchmark.hle_converter._fetch_page", return_value=fake_rows),
        ):
            prompts = fetch_sample(sample=10, category_filter="mathematics")

        assert all(p["category"] == "mathematics" for p in prompts)


# ── prompts.py round-trip ──────────────────────────────────────────────────────

class TestPromptsRoundtrip:

    def test_load_prompts_reads_answer_fields(self, tmp_path: Path):
        from theriad.benchmark.prompts import load_prompts

        jsonl = tmp_path / "hle.jsonl"
        rows = [
            {"id": "hle_0000", "category": "mathematics", "prompt": "Q1?",
             "max_output_tokens": 16, "expected_properties": [],
             "answer": "B", "answer_type": "multiple_choice"},
            {"id": "hle_0001", "category": "geography", "prompt": "Q2?",
             "max_output_tokens": 128, "expected_properties": [],
             "answer": "Paris", "answer_type": "exact_match"},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        prompts = load_prompts(jsonl)
        assert len(prompts) == 2

        p0 = prompts[0]
        assert p0.answer == "B"
        assert p0.answer_type == "multiple_choice"
        assert p0.quality_check("The answer is B.") is True
        assert p0.quality_check("The answer is A.") is False

        p1 = prompts[1]
        assert p1.answer == "Paris"
        assert p1.answer_type == "exact_match"
        assert p1.quality_check("paris") is True
        assert p1.quality_check("London") is False

    def test_load_prompts_without_answer_fields_is_backward_compatible(self, tmp_path: Path):
        from theriad.benchmark.prompts import load_prompts

        jsonl = tmp_path / "classic.jsonl"
        row = {"id": "c1", "category": "code", "prompt": "Write hello world.",
               "max_output_tokens": 200, "expected_properties": ["hello"]}
        jsonl.write_text(json.dumps(row) + "\n")

        prompts = load_prompts(jsonl)
        assert len(prompts) == 1
        p = prompts[0]
        assert p.answer is None
        assert p.answer_type is None
        assert p.quality_check("hello world") is True
        assert p.quality_check("goodbye world") is False
