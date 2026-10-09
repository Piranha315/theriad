"""Fixture structure and trusted grader checks; never dispatch model calls."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import PurePosixPath

import pytest

from theriad.routing.episode_fixtures import build_seed_corpus, corpus_manifest
from theriad.routing.episode_runtime import _text_grade
from theriad.routing.episode_study import Episode


def test_fifty_distinct_templates_have_fixed_balanced_splits():
    rows = build_seed_corpus()
    assert len(rows) == len({row.template_id for row in rows}) == len({row.group_id for row in rows}) == 50
    assert len({row.prompt for row in rows}) == 50
    assert Counter(row.family for row in rows) == dict.fromkeys(("chat", "coding", "tools", "search", "agents"), 10)
    for family in {row.family for row in rows}:
        assert Counter(row.split for row in rows if row.family == family) == {"pilot": 1, "train": 4, "validation": 2, "test": 2, "smoke": 1}
    assert [row.content_hash for row in rows] == [row.content_hash for row in build_seed_corpus()]


def test_repeats_never_inflate_independent_groups_or_cross_splits():
    rows = build_seed_corpus(repeats=3)
    groups = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    assert len(rows) == 150
    assert len(groups) == 50
    for repetitions in groups.values():
        assert len(repetitions) == 3
        assert len({row.split for row in repetitions}) == 1
        assert all(row.fixture == repetitions[0].fixture and row.prompt == repetitions[0].prompt for row in repetitions)
        assert len({id(row.fixture) for row in repetitions}) == 3
    assert corpus_manifest(rows)["independent_groups"] == 50


def test_filtering_development_data_does_not_reassign_test_groups():
    all_rows = build_seed_corpus()
    development = build_seed_corpus(splits=["pilot", "train", "validation"])
    assert len(development) == 35
    assert {row.episode_id: row.content_hash for row in development} == {
        row.episode_id: row.content_hash for row in all_rows if row.split in {"pilot", "train", "validation"}
    }


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"repeats": True}, {"repeats": 1.5}, {"splits": []}, {"splits": ["unknown"]}])
def test_invalid_corpus_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_seed_corpus(**kwargs)


def test_wire_roundtrip_and_provenance_are_explicit():
    from dataclasses import asdict

    rows = build_seed_corpus()
    for row in rows:
        assert Episode.from_dict(json.loads(json.dumps(asdict(row)))).content_hash == row.content_hash
        assert row.fixture_version and row.grader_version
        assert row.source in {"newly_authored", "public_reference_with_authored_question"}
        if row.family == "search":
            assert row.source == "public_reference_with_authored_question"
            assert row.fixture["snapshot_kind"] == "authored_summary_of_public_reference"
            assert row.fixture["network_required"] is False
            assert all(doc["url"].startswith("https://") and doc["text"] for doc in row.fixture["documents"])
            assert set(row.grader["answer"]["citations"]) <= {doc["url"] for doc in row.fixture["documents"]}
    manifest = corpus_manifest(rows)
    assert manifest["powered_for_joint_acceptance"] is False
    assert manifest["live_route_evidence"] is False
    assert manifest["imported_public_benchmark_items"] == 0
    assert manifest["browser_environment_coverage"] == "not_covered_project_agent_fixtures_only"


def test_chat_includes_real_history_long_inputs_and_two_different_image_tasks():
    chat = [row for row in build_seed_corpus() if row.family == "chat"]
    histories = [row for row in chat if "multiturn" in row.tags]
    assert len(histories) == 2
    assert all([message["role"] for message in row.fixture["messages"]] == ["user", "assistant", "user"] for row in histories)
    long = [row for row in chat if "long_context" in row.tags]
    assert len(long) == 2
    assert all(len(row.prompt) > 5000 for row in long)
    images = [row for row in chat if "vision" in row.tags]
    assert len(images) == 2
    assert {row.fixture["image"]["fixture_origin"] for row in images} == {"authored_receipt", "authored_schedule"}
    assert all("expected" not in row.fixture["image"] and "vision" in row.capabilities for row in images)
    absence = next(row for row in long if "abstention" in row.tags)
    assert "contains no case-999" not in absence.prompt


@pytest.mark.parametrize("row", [row for row in build_seed_corpus() if row.family in {"chat", "search"}], ids=lambda row: row.template_id)
def test_correct_structured_answers_pass_and_wrong_or_empty_answers_fail(row):
    answer = row.grader["answer"]
    correct = json.dumps(answer) if row.grader["kind"] == "json" else str(answer)
    assert _text_grade(row, correct)[0] is True
    assert _text_grade(row, "")[0] is False
    assert _text_grade(row, "{}" if correct != "{}" else "[]")[0] is False
    if row.family == "search":
        wrong = dict(answer)
        wrong["citations"] = ["https://invalid.example/unsupported"]
        assert _text_grade(row, json.dumps(wrong))[0] is False
        fact = next(key for key in answer if key != "citations")
        wrong = dict(answer)
        wrong[fact] = not answer[fact] if isinstance(answer[fact], bool) else "wrong fact"
        assert _text_grade(row, json.dumps(wrong))[0] is False


def test_workspace_expectations_are_bounded_and_retain_untouched_inputs():
    for row in build_seed_corpus():
        if row.family not in {"tools", "agents"}:
            continue
        assert row.grader["kind"] == "workspace"
        fixture = row.fixture
        assert fixture["files"] and (fixture.get("expected_files") or fixture.get("expected_json"))
        for mapping in (fixture["files"], fixture.get("expected_files", {}), fixture.get("expected_json", {})):
            assert all(not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts for path in mapping)
        for path, text in fixture["files"].items():
            if path not in fixture.get("expected_json", {}):
                assert fixture["expected_files"][path] == text


def _references():
    """Handwritten trusted implementations validate the handwritten hidden tests."""
    def stable_unique(values):
        return list(dict.fromkeys(values))

    def merge_intervals(intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        return merged

    def encode_runs(text):
        from itertools import groupby

        return [(letter, len(list(group))) for letter, group in groupby(text)]

    def sum_csv(text):
        values = [value.strip() for value in text.split(",") if value.strip()]
        if any(re.fullmatch(r"[+-]?\d+", value) is None for value in values):
            raise ValueError
        return sum(map(int, values))

    def get_path(data, keys, default=None):
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                return default
            data = data[key]
        return data

    def rolling_average(values, k):
        if type(k) is not int or k <= 0:
            raise ValueError
        return [sum(values[i:i + k]) / k for i in range(len(values) - k + 1)]

    def split_escaped(text):
        fields, current, escaped = [], "", False
        for letter in text:
            if escaped:
                current += letter
                escaped = False
            elif letter == "\\":
                escaped = True
            elif letter == ",":
                fields.append(current)
                current = ""
            else:
                current += letter
        return fields + [current + ("\\" if escaped else "")]

    def topological_layers(dependencies):
        pending = {node: set(deps) for node, deps in dependencies.items()}
        for node in set().union(*pending.values()):
            pending.setdefault(node, set())
        layers = []
        while pending:
            layer = sorted(node for node, deps in pending.items() if not deps)
            if not layer:
                raise ValueError
            layers.append(layer)
            for node in layer:
                del pending[node]
            for deps in pending.values():
                deps.difference_update(layer)
        return layers

    def total_money(amounts):
        return format(sum(map(Decimal, amounts), Decimal(0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")

    def lower_bound(values, target):
        left, right = 0, len(values)
        while left < right:
            mid = (left + right) // 2
            if values[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    return locals()


@pytest.mark.parametrize("row", [row for row in build_seed_corpus() if row.family == "coding"], ids=lambda row: row.template_id)
def test_hidden_code_tests_accept_trusted_reference_and_reject_broken_function(row):
    entry = row.fixture["entry_point"]
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry)
    assert entry in row.prompt
    trusted_test = compile(row.fixture["tests"], f"<fixture-tests:{entry}>", "exec")
    exec(trusted_test, _references())
    with pytest.raises(AssertionError):
        exec(trusted_test, {entry: lambda *_args, **_kwargs: None})


def test_four_way_seed_never_manufactures_independent_training_variants():
    from theriad.routing.episode_fixtures import build_four_way_seed_corpus

    rows = build_four_way_seed_corpus()
    assert len(rows) == len({row.group_id for row in rows}) == 50
    assert not any("authored_variant" in row.template_id for row in rows)
    for family in {row.family for row in rows}:
        assert Counter(row.split for row in rows if row.family == family) == {
            "pilot": 1, "train": 3, "calibration": 2, "threshold_selection": 2, "test": 1, "smoke": 1}
    assert all("underpowered" in row.tags for row in rows)
    assert corpus_manifest(rows)["powered_for_joint_acceptance"] is False
    filtered = build_four_way_seed_corpus(splits=["train", "calibration", "threshold_selection"])
    assert len(filtered) == 35
    assert {row.episode_id: row.content_hash for row in filtered} == {
        row.episode_id: row.content_hash for row in rows if row.split in {"train", "calibration", "threshold_selection"}}


def test_known_generated_variants_cannot_claim_separate_independent_groups():
    from dataclasses import replace
    from theriad.routing.episode_study import validate_episodes

    seed = build_seed_corpus()[0]
    first = replace(seed, episode_id="chat.authored_variant_01.trial-1", group_id="chat.authored_variant_01",
                    template_id="chat.authored_variant_01", prompt="Return only the sum of 2 and 7.")
    second = replace(first, episode_id="chat.authored_variant_02.trial-1", group_id="chat.authored_variant_02",
                     template_id="chat.authored_variant_02", prompt="Return only the sum of 3 and 8.")
    with pytest.raises(ValueError, match="(related|lineage)"):
        validate_episodes([first, second], require_all_families=False)
    assert validate_episodes([first, replace(second, group_id=first.group_id)], require_all_families=False)["independent_groups"] == 1


def test_declared_parent_template_relationship_cannot_cross_independent_groups():
    from dataclasses import replace
    from theriad.routing.episode_study import validate_episodes

    first = build_seed_corpus()[0]
    second = replace(first, episode_id="child", group_id="child", template_id="child",
                     prompt="A differently parameterized version of the task.",
                     fixture={"parent_template_id": first.template_id})
    with pytest.raises(ValueError, match="(related|lineage)"):
        validate_episodes([first, second], require_all_families=False)
