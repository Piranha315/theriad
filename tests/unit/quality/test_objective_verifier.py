from theriad.quality.objective_verifier import verify_objective_response


def test_unverifiable_general_answer_returns_none():
    assert verify_objective_response("Tell me a joke", "A joke") is None
    assert verify_objective_response("Fix this source code", "Done") is None


def test_json_request_accepts_valid_json_and_rejects_prose():
    good = verify_objective_response("Return JSON only", '```json\n{"ok": true}\n```')
    bad = verify_objective_response("Return JSON only", "The answer is yes")
    assert good and good.passed
    assert bad and not bad.passed


def test_minimal_json_schema_checks_type_and_required_keys():
    result = verify_objective_response(
        "Return the record",
        '{"name": "A"}',
        expected_json_schema={"type": "object", "required": ["name", "id"]},
    )
    assert result and not result.passed
    assert "id" in result.details


def test_citation_check_requires_a_link_not_self_reported_confidence():
    bad = verify_objective_response("Explain this and cite sources", "I am confident [1].")
    good = verify_objective_response(
        "Explain this and cite sources", "See [the source](https://example.com)."
    )
    assert bad and not bad.passed
    assert good and good.passed


def test_external_tool_signal_can_drive_failure():
    result = verify_objective_response(
        "Update the database",
        "Done",
        external_signals=[("database_state", False, "row absent")],
    )
    assert result and not result.passed
    assert result.source == "database_state"
