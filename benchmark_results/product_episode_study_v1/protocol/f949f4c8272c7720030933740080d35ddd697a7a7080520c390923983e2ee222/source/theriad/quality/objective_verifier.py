"""Deterministic acceptance checks for objectively verifiable responses."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveVerification:
    applicable: bool
    passed: bool
    source: str
    score: float
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_JSON_REQUEST = re.compile(
    r"\b(?:return|respond|reply|output|format|provide|emit)\b.{0,35}\bjson\b|\bjson only\b",
    re.I | re.S,
)
_CITATION_REQUEST = re.compile(
    r"\b(?:cite|citations?|references?|provide (?:your )?sources|list (?:the )?sources|with sources)\b",
    re.I,
)
_CITATION_OUTPUT = re.compile(r"https?://|\[[^\]]+\]\(https?://", re.I)


def _strip_json_fence(output: str) -> str:
    stripped = output.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.I | re.S)
    return match.group(1) if match else stripped


def _check_schema(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if expected_type in type_map and not isinstance(value, type_map[expected_type]):
        return False, f"expected JSON type {expected_type}"
    if isinstance(value, dict):
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            return False, f"missing required keys: {missing}"
    return True, "schema passed"


def verify_objective_response(
    prompt: str,
    output: str,
    *,
    expected_json_schema: dict[str, Any] | None = None,
    external_signals: Iterable[tuple[str, bool, str]] = (),
) -> ObjectiveVerification | None:
    """Return None when no objective acceptance check applies.

    External signals are produced by trusted executors (tests, database state,
    browser/tool state), never by parsing the model's claimed confidence.
    """

    checks: list[tuple[str, bool, str]] = list(external_signals)
    if expected_json_schema is not None or _JSON_REQUEST.search(prompt):
        try:
            value = json.loads(_strip_json_fence(output))
            passed, details = (
                _check_schema(value, expected_json_schema)
                if expected_json_schema is not None
                else (True, "valid JSON")
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            passed, details = False, f"invalid JSON: {type(exc).__name__}"
        checks.append(("json_schema" if expected_json_schema else "json", passed, details))
    if _CITATION_REQUEST.search(prompt):
        passed = bool(_CITATION_OUTPUT.search(output))
        checks.append(("citations", passed, "citation link present" if passed else "no citation link"))
    if not checks:
        return None
    failures = [f"{source}: {details}" for source, passed, details in checks if not passed]
    passed = not failures
    return ObjectiveVerification(
        applicable=True,
        passed=passed,
        source="+".join(source for source, _passed, _details in checks),
        score=sum(1 for _source, ok, _details in checks if ok) / len(checks),
        details="; ".join(failures) if failures else "all objective checks passed",
    )
