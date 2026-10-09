from __future__ import annotations

from dataclasses import replace

from theriad.routing.study_adapters import (
    BFCLAdapter,
    StructuredCompletionRequest,
    StructuredCompletionResponse,
    StructuredToolCall,
    check_bfcl_calls,
    collection_eligibility,
    run_bfcl_smoke_fixtures,
    run_tau_environment_smoke,
)

from .test_study import _prompt


class _NoToolsProvider:
    supports_tool_calls = False


class _ToolProvider:
    supports_tool_calls = True

    async def complete_with_tools(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResponse:
        assert len(request.tool_schemas) == 1
        return StructuredCompletionResponse(
            model=request.model,
            content="",
            tool_calls=(
                StructuredToolCall("first", {"x": 1}, 0),
                StructuredToolCall("second", {"x": 2}, 1),
            ),
        )


class _FixtureOfficialChecker:
    available = True

    def check(self, **_kwargs):
        return {
            "official_checker_id": "fixture-official-bfcl",
            "official_checker_revision": "fixture-v1",
            "official_checker_result": {"valid": True, "error": []},
            "authoritative": True,
        }


class _DottedToolProvider:
    supports_tool_calls = True

    async def complete_with_tools(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResponse:
        sent_name = request.tool_schemas[0]["name"]
        assert "." not in sent_name
        assert len(sent_name) <= 64
        assert request.tool_schemas[0]["parameters"]["type"] == "object"
        assert (
            request.tool_schemas[0]["parameters"]["properties"]["confidence"]["type"]
            == "number"
        )
        return StructuredCompletionResponse(
            model=request.model,
            content="",
            tool_calls=(StructuredToolCall(sent_name, {"sku": "123"}, 0),),
        )


class _CanonicalNameChecker(_FixtureOfficialChecker):
    def check(self, **kwargs):
        assert kwargs["tool_schemas"][0]["name"] == "walmart.check_price"
        assert kwargs["tool_schemas"][0]["parameters"]["type"] == "dict"
        assert (
            kwargs["tool_schemas"][0]["parameters"]["properties"]["confidence"][
                "type"
            ]
            == "float"
        )
        assert kwargs["candidate_tool_calls"][0].name == "walmart.check_price"
        return super().check(**kwargs)


def _bfcl_row():
    return replace(
        _prompt(1, "train"),
        source="bfcl-v4",
        task_format="tool_use",
        lane="general",
        label_type="tool_state",
        grader={
            "kind": "bfcl_official",
            "executor_id": "bfcl-checker",
            "executor_version": "bfcl-v1",
            "evidence_policy": "sha256_trusted_result",
            "tool_schemas": [{"name": "first"}],
            "official_checker_id": "bfcl-checker",
            "official_checker_version": "bfcl-v1",
            "official_checker_available": True,
            "task_mode": "multiple",
            "ground_truth": [
                {"name": "first", "arguments": {"x": 1}},
                {"name": "second", "arguments": {"x": 2}},
            ],
        },
    )


def test_bfcl_preserves_multiple_calls_and_tool_schema():
    result = BFCLAdapter(official_checker=_FixtureOfficialChecker()).collect(
        _ToolProvider(), _bfcl_row(), model="fixture"
    )
    assert result.status == "graded"
    assert result.grade is True
    assert [call.name for call in result.tool_calls] == ["first", "second"]
    assert result.evidence["ground_truth_preserved"] is True


def test_bfcl_aliases_provider_invalid_names_and_restores_canonical_name():
    row = replace(
        _bfcl_row(),
        grader={
            **_bfcl_row().grader,
            "task_mode": "simple",
            "tool_schemas": [
                {
                    "name": "walmart.check_price",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "sku": {"type": "string"},
                            "confidence": {"type": "float"},
                        },
                    },
                }
            ],
            "ground_truth": [
                {"name": "walmart.check_price", "arguments": {"sku": "123"}}
            ],
        },
    )

    result = BFCLAdapter(official_checker=_CanonicalNameChecker()).collect(
        _DottedToolProvider(), row, model="fixture"
    )

    assert result.status == "graded"
    assert result.grade is True
    assert result.tool_calls[0].name == "walmart.check_price"
    aliases = result.evidence["provider_tool_name_aliases"]
    assert set(aliases.values()) == {"walmart.check_price"}
    assert all("." not in alias and len(alias) <= 64 for alias in aliases)


def test_bfcl_endpoint_without_tool_calls_is_ineligible_not_failed():
    result = BFCLAdapter().collect(_NoToolsProvider(), _bfcl_row(), model="fixture")
    assert result.status == "ineligible_tool_capability"
    assert result.eligible is False
    assert result.grade is None


def test_bfcl_parallel_and_irrelevance_are_not_collapsed():
    parallel = [
        StructuredToolCall("lookup", {"id": 1}, 0, parallel_group="g"),
        StructuredToolCall("lookup", {"id": 2}, 1, parallel_group="g"),
    ]
    passed, evidence = check_bfcl_calls(
        task_mode="parallel", expected_tool_calls=parallel, candidate_tool_calls=parallel
    )
    assert passed

    passed, _ = check_bfcl_calls(
        task_mode="simple",
        expected_tool_calls=["calc(n=20, k=5, p=1/6)"],
        candidate_tool_calls=[StructuredToolCall("calc", {"n": 20, "k": 5, "p": 1 / 6}, 0)],
        tool_schemas=[
            {
                "name": "calc",
                "parameters": {"properties": {"n": {}, "k": {}, "p": {}}},
            }
        ],
    )
    assert passed
    assert len(evidence["candidate_tool_calls"]) == 2
    passed, _ = check_bfcl_calls(
        task_mode="irrelevance", expected_tool_calls=[], candidate_tool_calls=[]
    )
    assert passed


def test_fixture_smokes_are_provider_free_and_report_real_executor_gap():
    bfcl = run_bfcl_smoke_fixtures()
    tau = run_tau_environment_smoke()
    assert bfcl["provider_calls"] == 0
    assert bfcl["official_checker_executable"] is False
    assert tau["provider_calls"] == 0
    assert tau["official_environment_executable"] is False


def test_bfcl_and_tau_rows_are_not_eligible_without_executor_evidence():
    row = _bfcl_row()
    blocked = replace(
        row,
        grader={**row.grader, "official_checker_available": False},
    )
    assert collection_eligibility(blocked)[0] is False
