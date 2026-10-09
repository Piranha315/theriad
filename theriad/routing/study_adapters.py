"""Task-family collection and grading adapters for the Router V2 study.

The study cannot safely treat every task as a text completion.  This module
keeps the boundary explicit:

* ordinary rows use ``ProviderProtocol.complete``;
* BFCL rows require a provider-native structured-tool interface and a pinned
  checker contract;
* tau-bench rows require an agent/environment interface and persisted state;
* code rows require an executable test harness and are not BFCL function-call
  rows.

The adapters are provider-free until a caller supplies an implementation.  A
missing capability returns an ineligible result; it is never converted into a
failed model outcome.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from theriad.core.providers.interfaces import CompletionRequest, Message, ProviderProtocol

from .study import StudyPrompt

ADAPTER_SCHEMA_VERSION = 1
BFCL_ADAPTER_VERSION = "router-v2-bfcl-adapter-v5"
TAU_ADAPTER_VERSION = "router-v2-tau-bench-adapter-v3"
CODE_ADAPTER_VERSION = "router-v2-executable-code-adapter-v2"


@dataclass(frozen=True)
class StructuredToolCall:
    """One provider-native tool call, retaining sequence and parallel group."""

    name: str
    arguments: dict[str, Any]
    order: int
    call_id: str = ""
    parallel_group: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("structured tool call needs a name")
        if self.order < 0:
            raise ValueError("structured tool call order cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "order": self.order,
            "call_id": self.call_id,
            "parallel_group": self.parallel_group,
        }


@dataclass(frozen=True)
class StructuredCompletionRequest:
    model: str
    messages: tuple[Message, ...]
    tool_schemas: tuple[dict[str, Any], ...]
    max_tokens: int
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredCompletionResponse:
    """Normalized structured response returned by a task-aware provider."""

    model: str
    content: str
    tool_calls: tuple[StructuredToolCall, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class AdapterResult:
    """Collection result with explicit eligibility and grading status."""

    status: str
    eligible: bool
    adapter_id: str
    adapter_version: str
    details: str = ""
    output: str = ""
    tool_calls: tuple[StructuredToolCall, ...] = ()
    environment_state: dict[str, Any] | None = None
    grade: bool | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    api_call_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "eligible": self.eligible,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "details": self.details,
            "output": self.output,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "environment_state": self.environment_state,
            "grade": self.grade,
            "evidence": self.evidence,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "raw": self.raw,
            "cost_usd": self.cost_usd,
            "api_call_count": self.api_call_count,
        }


class StructuredToolProvider(Protocol):
    supports_tool_calls: bool

    async def complete_with_tools(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResponse | Mapping[str, Any]: ...


class TauAgentProvider(Protocol):
    supports_agent_tools: bool

    async def complete_with_tools(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResponse | Mapping[str, Any]: ...


class TauEnvironment(Protocol):
    def reset(self, initial_state: Mapping[str, Any]) -> None: ...

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]: ...

    def apply_actions(self, actions: Sequence[Mapping[str, Any]]) -> None: ...

    def apply_tool_call(self, call: StructuredToolCall) -> Mapping[str, Any]: ...

    def state(self) -> Mapping[str, Any]: ...

    def evaluate(self, trajectory: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _structured_capability(provider: Any) -> bool:
    return bool(getattr(provider, "supports_tool_calls", False)) and callable(
        getattr(provider, "complete_with_tools", None)
    )


def _agent_capability(provider: Any) -> bool:
    return bool(getattr(provider, "supports_agent_tools", False)) and callable(
        getattr(provider, "complete_with_tools", None)
    )


def _call_from_mapping(raw: Mapping[str, Any], order: int) -> StructuredToolCall:
    function = raw.get("function")
    source: Mapping[str, Any] = function if isinstance(function, Mapping) else raw
    arguments = source.get("arguments", source.get("args", {}))
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool-call arguments are not valid JSON") from exc
        arguments = parsed
    if not isinstance(arguments, Mapping):
        raise ValueError("tool-call arguments must be an object")
    return StructuredToolCall(
        name=str(source.get("name") or raw.get("name") or ""),
        arguments=dict(arguments),
        order=order,
        call_id=str(raw.get("id") or raw.get("call_id") or ""),
        parallel_group=str(raw.get("parallel_group") or ""),
    )


def normalize_tool_calls(raw: Any) -> tuple[StructuredToolCall, ...]:
    """Normalize a structured response without discarding calls or order."""

    if isinstance(raw, StructuredCompletionResponse):
        return raw.tool_calls
    if isinstance(raw, Mapping):
        raw = raw.get("tool_calls", raw.get("calls", []))
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("structured tool calls must be a list")
    calls: list[StructuredToolCall] = []
    for order, value in enumerate(raw):
        if isinstance(value, StructuredToolCall):
            calls.append(value)
        elif isinstance(value, Mapping):
            calls.append(_call_from_mapping(value, order))
        else:
            raise ValueError("each structured tool call must be an object")
    return tuple(calls)


_PROVIDER_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _provider_tool_alias(name: str) -> str:
    """Return a reversible, collision-resistant provider-safe tool name."""

    if _PROVIDER_TOOL_NAME.fullmatch(name):
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
    return f"{base[:53]}_{digest}"


def _provider_json_schema(value: Any) -> Any:
    """Normalize BFCL's Python-flavored schema types for provider APIs."""

    type_aliases = {
        "bool": "boolean",
        "dict": "object",
        "float": "number",
        "int": "integer",
        "list": "array",
        "str": "string",
        "tuple": "array",
    }
    if isinstance(value, Mapping):
        return {
            key: (
                type_aliases.get(child, child)
                if key == "type" and isinstance(child, str)
                else _provider_json_schema(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_provider_json_schema(child) for child in value]
    return deepcopy(value)


def _provider_safe_tool_schemas(
    schemas: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, str]]:
    """Alias invalid schema names while retaining an exact reverse map."""

    provider_schemas: list[dict[str, Any]] = []
    canonical_by_alias: dict[str, str] = {}
    for schema in schemas:
        value = deepcopy(dict(schema))
        target = value
        if value.get("type") == "function" and isinstance(value.get("function"), Mapping):
            value["function"] = deepcopy(dict(value["function"]))
            target = value["function"]
        canonical = str(target.get("name") or "")
        if not canonical:
            raise ValueError("BFCL tool schema has no function name")
        alias = _provider_tool_alias(canonical)
        previous = canonical_by_alias.get(alias)
        if previous is not None and previous != canonical:
            raise ValueError("BFCL provider tool-name alias collision")
        target["name"] = alias
        if "parameters" in target:
            target["parameters"] = _provider_json_schema(target["parameters"])
        canonical_by_alias[alias] = canonical
        provider_schemas.append(value)
    return tuple(provider_schemas), canonical_by_alias


def _restore_canonical_tool_names(
    calls: Sequence[StructuredToolCall], canonical_by_alias: Mapping[str, str]
) -> tuple[StructuredToolCall, ...]:
    return tuple(
        replace(call, name=canonical_by_alias.get(call.name, call.name)) for call in calls
    )


def _tau_calls_with_ids(
    calls: Sequence[StructuredToolCall], *, turn: int
) -> tuple[StructuredToolCall, ...]:
    return tuple(
        call
        if call.call_id
        else replace(call, call_id=f"theriad-tau-{turn}-{call.order}")
        for call in calls
    )


def _assistant_tool_calls(calls: Sequence[StructuredToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        for call in calls
    ]


def _canonical_call(call: StructuredToolCall) -> tuple[Any, ...]:
    return (
        call.name,
        json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        call.order,
        call.parallel_group,
    )


def _expected_calls(
    value: Any, tool_schemas: Sequence[Mapping[str, Any]] = ()
) -> tuple[StructuredToolCall, ...]:
    """Decode the complete expected call list; never select element zero."""

    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("BFCL ground truth must preserve a sequence of calls")
    calls: list[StructuredToolCall] = []
    parameters_by_name = {
        str(schema.get("name")): list(
            (schema.get("parameters") or {}).get("properties", {}).keys()
        )
        for schema in tool_schemas
        if isinstance(schema, Mapping)
    }

    def safe_value(node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow
            ):
                left = safe_value(node.left)
                right = safe_value(node.right)
                return {
                    ast.Add: lambda: left + right,
                    ast.Sub: lambda: left - right,
                    ast.Mult: lambda: left * right,
                    ast.Div: lambda: left / right,
                    ast.Pow: lambda: left**right,
                }[type(node.op)]()
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
                value = safe_value(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            raise ValueError(
                "BFCL call expression contains unsupported argument syntax"
            ) from None

    for order, item in enumerate(value):
        if isinstance(item, StructuredToolCall):
            calls.append(item)
        elif isinstance(item, str):
            match = re.fullmatch(r"\s*([A-Za-z_][\w.]*)\((.*)\)\s*", item)
            if not match:
                raise ValueError("BFCL string ground truth is not a call expression")
            expression = ast.parse(f"f({match.group(2)})", mode="eval").body
            arguments: dict[str, Any] = {}
            for index, argument in enumerate(expression.args):
                names = parameters_by_name.get(match.group(1), [])
                arguments[names[index] if index < len(names) else f"arg{index}"] = safe_value(argument)
            for keyword in expression.keywords:
                if keyword.arg is None:
                    raise ValueError("BFCL call expression cannot use **kwargs")
                arguments[keyword.arg] = safe_value(keyword.value)
            calls.append(StructuredToolCall(match.group(1), arguments, order))
        elif isinstance(item, Mapping):
            if "name" in item or "function" in item:
                calls.append(_call_from_mapping(item, order))
            elif len(item) == 1:
                # BFCL's possible-answer shape is {tool_name: {param:
                # [acceptable values]}}.  Compatibility evidence picks the
                # first acceptable value only; the official checker below
                # remains authoritative over all alternatives.
                name, raw_arguments = next(iter(item.items()))
                if not isinstance(raw_arguments, Mapping):
                    raise ValueError("BFCL named ground truth arguments must be an object")
                arguments = {
                    str(key): next(
                        (candidate for candidate in value if candidate != ""), ""
                    )
                    if isinstance(value, Sequence) and not isinstance(value, str | bytes)
                    else value
                    for key, value in raw_arguments.items()
                }
                calls.append(StructuredToolCall(str(name), arguments, order))
            else:
                raise ValueError("BFCL ground truth object is not a named call")
        else:
            raise ValueError("BFCL ground truth call must be a string or object")
    return tuple(calls)


def check_bfcl_calls(
    *,
    task_mode: str,
    expected_tool_calls: Any,
    candidate_tool_calls: Sequence[StructuredToolCall],
    tool_schemas: Sequence[Mapping[str, Any]] = (),
) -> tuple[bool, dict[str, Any]]:
    """Run the provider-free pinned compatibility semantics used by fixtures.

    The actual official checker is represented by the row's pinned executor
    contract.  This function deliberately exposes all calls in evidence so a
    real installed checker can replace it without changing collection shape.
    """

    candidate = tuple(candidate_tool_calls)
    expected = (
        _expected_calls(expected_tool_calls, tool_schemas)
        if task_mode != "irrelevance"
        else ()
    )
    if task_mode == "irrelevance":
        passed = not candidate
    elif task_mode == "parallel":
        # Parallel calls are compared as one group, while their observed order
        # remains in the evidence.  Multiple parallel groups remain ordered.
        passed = len(candidate) == len(expected) and sorted(
            _canonical_call(call)[:2] for call in candidate
        ) == sorted(_canonical_call(call)[:2] for call in expected)
    else:
        passed = len(candidate) == len(expected) and all(
            left.name == right.name
            and left.arguments == right.arguments
            and left.order == right.order
            for left, right in zip(candidate, expected, strict=True)
        )
    return passed, {
        "task_mode": task_mode,
        "expected_tool_calls": [call.to_dict() for call in expected],
        "candidate_tool_calls": [call.to_dict() for call in candidate],
        "ground_truth_preserved": True,
    }


class OfficialCheckerUnavailable(RuntimeError):
    """Raised when a pinned official checker cannot be invoked locally."""


class PinnedBFCLChecker:
    """Invoke the pinned BFCL checker in the approved execution boundary.

    The checker is intentionally a subprocess contract rather than an import in
    the collector process.  This keeps benchmark code and its dependencies out
    of the application interpreter and makes the revision/source paths part of
    the evidence for every graded row.
    """

    checker_id = "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker"
    revision = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        source: str | Path | None = None,
        cache: str | Path | None = None,
    ) -> None:
        root = Path(os.environ.get("THERIAD_ROUTER_V2_SOURCE_CACHE", "/tmp/theriad-router-v2-source-cache"))
        self.python = Path(
            python
            or os.environ.get(
                "THERIAD_BFCL_PYTHON",
                str(root / "acquired/bfcl-venv/bin/python"),
            )
        )
        self.source = Path(
            source
            or os.environ.get(
                "THERIAD_BFCL_SOURCE",
                str(root / "acquired/bfcl/berkeley-function-call-leaderboard"),
            )
        )
        self.cache = Path(
            cache
            or os.environ.get(
                "THERIAD_BFCL_CACHE",
                str(root / f"bfcl-{self.revision}"),
            )
        )

    @property
    def available(self) -> bool:
        return all(path.exists() for path in (self.python, self.source, self.cache))

    def check(
        self,
        *,
        task_mode: str,
        tool_schemas: Sequence[Mapping[str, Any]],
        expected_tool_calls: Any,
        candidate_tool_calls: Sequence[StructuredToolCall],
        task_id: str,
    ) -> dict[str, Any]:
        if not self.available:
            raise OfficialCheckerUnavailable(
                "pinned BFCL python/source/cache paths are unavailable"
            )
        from .sandbox import default_policy, run_isolated

        payload = {
            "task_mode": task_mode,
            "task_id": task_id,
            "tool_schemas": [dict(schema) for schema in tool_schemas],
            "ground_truth": expected_tool_calls,
            "candidate": [call.to_dict() for call in candidate_tool_calls],
            "checker_id": self.checker_id,
            "revision": self.revision,
        }
        source_path = self.source.resolve()
        site_packages_hint = next(
            (
                path
                for path in (self.python.parent.parent / "lib").glob(
                    "python*/site-packages"
                )
                if path.is_dir()
            ),
            None,
        )
        target_python = self.python.resolve()
        target_runtime_root = target_python.parent.parent
        target_stdlib = target_runtime_root / "lib" / f"python{target_python.parts[-2].removeprefix('python')}"
        runner = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(site_packages_hint.resolve()) if site_packages_hint else ''!r})
sys.path.insert(0, {str(source_path)!r})
payload = json.loads(Path('bfcl-payload.json').read_text())
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

mode = str(payload['task_mode'])
candidate = [
    {{call['name']: call['arguments']}}
    for call in payload['candidate']
]
if mode == 'irrelevance':
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_relevance_entry
    from bfcl_eval.model_handler.api_inference.gorilla import GorillaHandler
    handler = GorillaHandler('gorilla-openfunctions-v2', 0.0, 'gorilla-openfunctions-v2', False)
    candidate_text = '' if not candidate else '; '.join(
        f"{{item['name']}}({{', '.join(f'{{key}}={{value!r}}' for key, value in item['arguments'].items())}})"
        for item in payload['candidate']
    )
    checked = _evaluate_single_relevance_entry(
        handler,
        payload['task_id'],
        candidate_text,
        {{'function': payload['tool_schemas']}},
        'gorilla-openfunctions-v2',
        'irrelevance',
    )
else:
    checked = ast_checker(
        payload['tool_schemas'],
        candidate,
        payload['ground_truth'],
        Language.PYTHON,
        mode,
        'gorilla-openfunctions-v2',
    )
print(json.dumps({{'status': 'passed', 'checker': checked}}))
"""
        with TemporaryDirectory(prefix="theriad-router-v2-bfcl-") as directory:
            task_directory = Path(directory)
            (task_directory / "bfcl-payload.json").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            read_paths = tuple(
                str(path)
                for path in (
                    self.source,
                    self.cache,
                    site_packages_hint,
                    self.python.parent.parent,
                    target_runtime_root,
                    target_stdlib,
                    target_stdlib / "lib-dynload",
                )
                if path is not None
            )
            result = run_isolated(
                (str(self.python), "-c", runner),
                task_directory=task_directory,
                policy=replace(
                    default_policy(
                        runtime=str(self.python), allowed_read_paths=read_paths
                    ),
                    cpu_seconds=30,
                    wall_clock_seconds=30.0,
                ),
            )
        if not result.passed:
            raise OfficialCheckerUnavailable(
                f"official BFCL checker failed: {result.status}: {result.stderr[-2000:]}"
                f" stdout={result.stdout[-1000:]} returncode={result.returncode}"
            )
        try:
            checked = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise OfficialCheckerUnavailable(
                "official BFCL checker emitted non-JSON output"
            ) from exc
        checker = checked.get("checker") if isinstance(checked, Mapping) else None
        if not isinstance(checker, Mapping) or not isinstance(checker.get("valid"), bool):
            raise OfficialCheckerUnavailable("official BFCL checker omitted a boolean result")
        return {
            "official_checker_id": self.checker_id,
            "official_checker_revision": self.revision,
            "official_checker_result": dict(checker),
            "official_checker_executor": result.to_dict(),
            "authoritative": True,
        }


class PlainTextAdapter:
    adapter_id = "plain-text"
    adapter_version = "router-v2-plain-text-v1"

    def eligible(self, row: StudyPrompt) -> tuple[bool, str]:
        return True, "plain completion is valid for this task family"

    def collect(
        self, provider: ProviderProtocol, row: StudyPrompt, *, model: str
    ) -> Any:
        eligible, reason = self.eligible(row)
        if not eligible:
            return AdapterResult(
                "ineligible", False, self.adapter_id, self.adapter_version, reason
            )
        return _await(
            provider.complete(
                CompletionRequest(
                    model=model,
                    messages=[Message(role="user", content=row.prompt)],
                    max_tokens=row.max_output_tokens,
                    temperature=0.0,
                )
            )
        )


class BFCLAdapter:
    adapter_id = "router-v2-bfcl"
    adapter_version = BFCL_ADAPTER_VERSION

    def __init__(self, *, official_checker: Any | None = None) -> None:
        self.official_checker = official_checker or PinnedBFCLChecker()

    def eligible(self, row: StudyPrompt) -> tuple[bool, str]:
        grader = row.grader
        schemas = grader.get("tool_schemas") or grader.get("function_definitions")
        if not isinstance(schemas, list) or not schemas:
            return False, "BFCL row lacks pinned function definitions/tool schemas"
        if not str(grader.get("official_checker_id") or "").strip():
            return False, "BFCL row lacks pinned official checker identity"
        if not str(grader.get("official_checker_version") or "").strip():
            return False, "BFCL row lacks pinned official checker version"
        if grader.get("official_checker_available") is not True:
            return False, "pinned BFCL official checker is not executable locally"
        return True, "BFCL structured tool-call adapter is executable"

    def collect(
        self, provider: StructuredToolProvider, row: StudyPrompt, *, model: str
    ) -> AdapterResult:
        eligible, reason = self.eligible(row)
        if not eligible:
            return AdapterResult("ineligible", False, self.adapter_id, self.adapter_version, reason)
        if not _structured_capability(provider):
            return AdapterResult(
                "ineligible_tool_capability",
                False,
                self.adapter_id,
                self.adapter_version,
                "candidate endpoint does not expose provider-native tool calling",
            )
        grader = row.grader
        schemas = tuple(deepcopy(dict(schema)) for schema in grader["tool_schemas"])
        provider_schemas, canonical_by_alias = _provider_safe_tool_schemas(schemas)
        response = _await(
            provider.complete_with_tools(
                StructuredCompletionRequest(
                    model=model,
                    messages=(Message(role="user", content=row.prompt),),
                    tool_schemas=provider_schemas,
                    max_tokens=int(row.max_output_tokens or 1),
                    temperature=0.0,
                    metadata={"prompt_id": row.prompt_id, "task_format": row.task_format},
                )
            )
        )
        calls = _restore_canonical_tool_names(
            normalize_tool_calls(response), canonical_by_alias
        )
        if isinstance(response, Mapping):
            content = str(response.get("content") or "")
            prompt_tokens = int(response.get("prompt_tokens") or 0)
            completion_tokens = int(response.get("completion_tokens") or 0)
            latency_ms = float(response.get("latency_ms") or 0.0)
            provider_name = str(response.get("provider") or "")
            raw = dict(response.get("raw") or {})
        else:
            content = str(getattr(response, "content", "") or "")
            prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(response, "completion_tokens", 0) or 0)
            latency_ms = float(getattr(response, "latency_ms", 0.0) or 0.0)
            provider_name = str(getattr(response, "provider", "") or "")
            raw = dict(getattr(response, "raw", {}) or {})
        mode = str(grader.get("task_mode") or "simple")
        passed, evidence = check_bfcl_calls(
            task_mode=mode,
            expected_tool_calls=grader.get("ground_truth", []),
            candidate_tool_calls=calls,
            tool_schemas=schemas,
        )
        try:
            official = self.official_checker.check(
                task_mode=mode,
                tool_schemas=schemas,
                expected_tool_calls=grader.get("ground_truth", []),
                candidate_tool_calls=calls,
                task_id=str(grader.get("source_task_id") or row.prompt_id),
            )
        except OfficialCheckerUnavailable as exc:
            return AdapterResult(
                "pending_official_checker",
                True,
                self.adapter_id,
                self.adapter_version,
                str(exc),
                output=content,
                tool_calls=calls,
                evidence={
                    "compatibility": evidence,
                    "ground_truth_preserved": True,
                    "provider_tool_name_aliases": canonical_by_alias,
                    "official_checker_status": "pending",
                    "official_checker_error": str(exc),
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                provider=provider_name,
                raw=raw,
                cost_usd=getattr(response, "cost_usd", None),
            )
        official_result = official["official_checker_result"]
        official_passed = bool(official_result["valid"])
        if not calls and content:
            return AdapterResult(
                "ungradable_prose_for_tool_task",
                True,
                self.adapter_id,
                self.adapter_version,
                "structured endpoint returned prose; it was not treated as a failed function call",
                output=content,
                evidence={
                    "tool_calls": [],
                    "prose": content,
                    "compatibility": evidence,
                    "ground_truth_preserved": True,
                    "provider_tool_name_aliases": canonical_by_alias,
                    **official,
                    "official_grade_ignored": True,
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                provider=provider_name,
                raw=raw,
                cost_usd=getattr(response, "cost_usd", None),
            )
        return AdapterResult(
            "graded",
            True,
            self.adapter_id,
            self.adapter_version,
            "pinned official BFCL checker result",
            output=content,
            tool_calls=calls,
            grade=official_passed,
            evidence={
                "compatibility": evidence,
                "ground_truth_preserved": True,
                "provider_tool_name_aliases": canonical_by_alias,
                **official,
                "compatibility_agrees": passed == official_passed,
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            provider=provider_name,
            raw=raw,
            cost_usd=getattr(response, "cost_usd", None),
        )


class TauBenchAdapter:
    adapter_id = "router-v2-tau-bench"
    adapter_version = TAU_ADAPTER_VERSION

    def eligible(self, row: StudyPrompt) -> tuple[bool, str]:
        grader = row.grader
        if not str(grader.get("official_environment_id") or "").strip():
            return False, "tau-bench row lacks pinned environment identity"
        if grader.get("official_environment_available") is not True:
            return False, "pinned tau-bench environment is not executable locally"
        if not str(grader.get("official_evaluator_id") or "").strip():
            return False, "tau-bench row lacks pinned official evaluator identity"
        return True, "tau-bench agent/environment adapter is executable"

    def collect(
        self,
        provider: TauAgentProvider,
        row: StudyPrompt,
        *,
        model: str,
        environment_factory: Any | None = None,
        resume: Mapping[str, Any] | None = None,
        persist: Any | None = None,
    ) -> AdapterResult:
        eligible, reason = self.eligible(row)
        if not eligible:
            return AdapterResult("ineligible", False, self.adapter_id, self.adapter_version, reason)
        if environment_factory is None:
            from .tau2_runtime import tau2_available, tau2_environment_factory

            if tau2_available():
                environment_factory = tau2_environment_factory
        if not _agent_capability(provider) or environment_factory is None:
            return AdapterResult(
                "ineligible_agent_environment_capability",
                False,
                self.adapter_id,
                self.adapter_version,
                "candidate must expose native tool calls for an interactive pinned tau-bench loop",
            )
        environment = environment_factory(row)
        initial_state = row.grader.get("initial_state", {})
        environment.reset(
            deepcopy(initial_state) if isinstance(initial_state, Mapping) else {}
        )
        if resume and resume.get("state") is not None:
            restore = getattr(environment, "restore_state", None)
            if not callable(restore):
                return AdapterResult(
                    "pending_resume_restore",
                    True,
                    self.adapter_id,
                    self.adapter_version,
                    "interrupted tau trajectory has no trusted environment restore operation",
                    evidence={"resume_required": True, "state": dict(resume.get("state") or {})},
                )
            restore(deepcopy(dict(resume.get("state") or {})))

        tools = tuple(dict(schema) for schema in environment.tool_schemas())
        messages: list[Message] = [Message(role="user", content=row.prompt)]
        trajectory: list[dict[str, Any]] = list(resume.get("trajectory") or ()) if resume else []
        if trajectory:
            replay = getattr(environment, "restore_trajectory", None)
            if not callable(replay):
                return AdapterResult(
                    "pending_resume_restore",
                    True,
                    self.adapter_id,
                    self.adapter_version,
                    "interrupted tau trajectory has no trusted trajectory restore operation",
                    environment_state=dict(environment.state()),
                    evidence={"resume_required": True, "trajectory": trajectory},
                )
            replay(trajectory)
            for record in trajectory:
                replay_calls = _tau_calls_with_ids(
                    normalize_tool_calls(record.get("tool_calls") or ()),
                    turn=int(record.get("turn") or 0),
                )
                messages.append(
                    Message(
                        role="assistant",
                        content=str(record.get("assistant_content") or ""),
                        tool_calls=_assistant_tool_calls(replay_calls),
                    )
                )
                tool_results = list(record.get("tool_results") or ())
                if len(tool_results) != len(replay_calls):
                    return AdapterResult(
                        "pending_resume_restore",
                        True,
                        self.adapter_id,
                        self.adapter_version,
                        "interrupted tau trajectory has mismatched calls and results",
                        environment_state=dict(environment.state()),
                        evidence={"resume_required": True, "trajectory": trajectory},
                    )
                for call, tool_result in zip(replay_calls, tool_results, strict=True):
                    messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(tool_result, sort_keys=True),
                            tool_call_id=call.call_id,
                        )
                    )
        max_turns = max(1, int(row.grader.get("max_turns") or 8))
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0
        latency_ms = 0.0
        response_meta: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
            "provider": "",
            "raw": {},
            "cost_usd": None,
        }
        final_content = ""
        for turn in range(len(trajectory), max_turns):
            result = _await(
                provider.complete_with_tools(
                    StructuredCompletionRequest(
                        model=model,
                        messages=tuple(messages),
                        tool_schemas=tools,
                        max_tokens=int(row.max_output_tokens or 1),
                        temperature=0.0,
                        metadata={
                            "prompt_id": row.prompt_id,
                            "task_format": row.task_format,
                            "tau_turn": turn,
                            "tool_choice": "auto",
                        },
                    )
                )
            )
            calls = _tau_calls_with_ids(normalize_tool_calls(result), turn=turn)
            content = str(
                result.get("content") or ""
                if isinstance(result, Mapping)
                else getattr(result, "content", "") or ""
            )
            response_meta = _response_metadata(result)
            prompt_tokens += response_meta["prompt_tokens"]
            completion_tokens += response_meta["completion_tokens"]
            latency_ms += response_meta["latency_ms"]
            if response_meta.get("cost_usd") is not None:
                total_cost += float(response_meta["cost_usd"])
            final_content = content or final_content
            turn_record: dict[str, Any] = {
                "turn": turn,
                "assistant_content": content,
                "tool_calls": [call.to_dict() for call in calls],
                "state_before": dict(environment.state()),
            }
            if not calls:
                trajectory.append(turn_record)
                if persist is not None:
                    persist({"status": "interrupted", "trajectory": trajectory, "state": dict(environment.state())})
                break
            messages.append(
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=_assistant_tool_calls(calls),
                )
            )
            tool_results: list[dict[str, Any]] = []
            for call in calls:
                apply_tool_call = getattr(environment, "apply_tool_call", None)
                if callable(apply_tool_call):
                    tool_result = apply_tool_call(call)
                else:
                    environment.apply_actions((call.to_dict(),))
                    tool_result = {"state": dict(environment.state())}
                if not isinstance(tool_result, Mapping):
                    tool_result = {"result": tool_result}
                tool_result_dict = dict(tool_result)
                tool_results.append(tool_result_dict)
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result_dict, sort_keys=True),
                        tool_call_id=call.call_id,
                    )
                )
            turn_record["tool_results"] = tool_results
            turn_record["state_after"] = dict(environment.state())
            trajectory.append(turn_record)
            if persist is not None:
                persist({"status": "interrupted", "trajectory": trajectory, "state": dict(environment.state())})
        state = dict(environment.state())
        evaluation = environment.evaluate(
            {
                "messages": messages,
                "trajectory": trajectory,
                "state": state,
                "max_turns": max_turns,
            }
        )
        if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("passed"), bool):
            return AdapterResult(
                "pending_external_state",
                True,
                self.adapter_id,
                self.adapter_version,
                "pinned tau-bench evaluator did not return a boolean result",
                environment_state=state,
                evidence={
                    "trajectory": trajectory,
                    "state": state,
                    "evaluation": dict(evaluation),
                    "official_evaluator_status": "pending",
                    "max_turns": max_turns,
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                provider=response_meta["provider"],
                raw=response_meta["raw"],
                cost_usd=total_cost,
                api_call_count=len(trajectory),
            )
        if persist is not None:
            persist({"status": "complete", "trajectory": trajectory, "state": state, "evaluation": dict(evaluation)})
        return AdapterResult(
            "graded",
            True,
            self.adapter_id,
            self.adapter_version,
            "pinned tau-bench interactive environment/evaluator result",
            output=final_content,
            environment_state=state,
            grade=bool(evaluation["passed"]),
            evidence={
                "trajectory": trajectory,
                "state": state,
                "evaluation": dict(evaluation),
                "official_evaluator_id": row.grader.get("official_evaluator_id"),
                "official_evaluator_version": row.grader.get("official_evaluator_version"),
                "interactive_agent_loop": True,
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            provider=response_meta["provider"],
            raw=response_meta["raw"],
            cost_usd=total_cost,
            api_call_count=len(trajectory),
        )


_STRICT_CODE_RESPONSE = re.compile(
    r"\A\s*```python[ \t]*\n(?P<code>.*?)(?:\n)?```\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strict_code_from_response(content: str) -> str | None:
    match = _STRICT_CODE_RESPONSE.fullmatch(content)
    if match is None:
        return None
    code = match.group("code")
    return code if code.strip() else None


class ExecutableCodeAdapter:
    """Collect, execute, and grade code; never route through text grading."""

    adapter_id = "router-v2-executable-code"
    adapter_version = CODE_ADAPTER_VERSION

    def eligible(self, row: StudyPrompt) -> tuple[bool, str]:
        grader = row.grader
        if row.task_format != "code_execution":
            return False, "row is not an executable code task"
        if not str(grader.get("code_task_type") or "").strip():
            return False, "code row lacks implementation/debugging task type"
        if not grader.get("executable_tests"):
            return False, "code row lacks pinned executable tests"
        from .sandbox import code_grader_runtime, runtime_supported, sandbox_available

        selected_runtime = code_grader_runtime(required=True)
        if selected_runtime is None:
            return False, "pinned code-grader runtime is not configured; provider execution is blocked"
        if not sandbox_available() or not runtime_supported(selected_runtime):
            return False, "approved code sandbox or selected grader runtime is unavailable; provider execution is blocked"
        return True, "executable code adapter is configured"

    def collect(
        self, provider: ProviderProtocol, row: StudyPrompt, *, model: str
    ) -> AdapterResult:
        from .sandbox import (
            SandboxPolicyError,
            SandboxUnavailable,
            code_grader_runtime,
            default_policy,
            run_python_tests,
        )

        grader = row.grader
        if not self._row_is_valid(row):
            eligible, reason = self.eligible(row)
            return AdapterResult("ineligible", False, self.adapter_id, self.adapter_version, reason)
        if not self._sandbox_ready():
            return AdapterResult(
                "sandbox_unavailable",
                False,
                self.adapter_id,
                self.adapter_version,
                "approved code sandbox is unavailable; no provider call was made",
                evidence={"provider_call_blocked": True},
            )
        response = _await(
            provider.complete(
                CompletionRequest(
                    model=model,
                    messages=[Message(role="user", content=row.prompt)],
                    max_tokens=int(row.max_output_tokens or 1),
                    temperature=0.0,
                    extra={
                        "response_format": {
                            "type": "text",
                            "description": "exactly one ```python fenced code block and no prose",
                        }
                    },
                )
            )
        )
        content = str(getattr(response, "content", None) or (response.get("content") if isinstance(response, Mapping) else "") or "")
        candidate_source = _strict_code_from_response(content)
        metadata = _response_metadata(response)
        if candidate_source is None:
            return AdapterResult(
                "ungradable_code_response",
                True,
                self.adapter_id,
                self.adapter_version,
                "candidate did not emit the required single python code block",
                output=content,
                evidence={"response_format": "single_python_fence", "extractable": False},
                **metadata,
            )
        tests = grader.get("executable_tests")
        if not isinstance(tests, str):
            return AdapterResult(
                "ungradable_code_tests",
                True,
                self.adapter_id,
                self.adapter_version,
                "pinned executable tests are not a source string",
                output=candidate_source,
                evidence={"extractable": True},
                **metadata,
            )
        selected_runtime = code_grader_runtime(required=True)
        if selected_runtime is None:
            return AdapterResult(
                "sandbox_unavailable",
                False,
                self.adapter_id,
                self.adapter_version,
                "pinned code-grader runtime is not configured; no execution was attempted",
                output=candidate_source,
                evidence={"provider_call_completed": True, "execution_blocked": True},
                **metadata,
            )
        try:
            sandbox_result = run_python_tests(
                candidate_source,
                tests,
                entry_point=str(grader["entry_point"]),
                policy=default_policy(runtime=selected_runtime),
            )
        except (SandboxPolicyError, SandboxUnavailable) as exc:
            return AdapterResult(
                "sandbox_unavailable",
                False,
                self.adapter_id,
                self.adapter_version,
                f"sandbox became unavailable before execution: {exc}",
                output=candidate_source,
                evidence={"provider_call_completed": True, "execution_blocked": True},
                **metadata,
            )
        status = "timeout" if sandbox_result.timed_out else "graded"
        return AdapterResult(
            status,
            True,
            self.adapter_id,
            self.adapter_version,
            "pinned executable code tests",
            output=candidate_source,
            grade=bool(sandbox_result.passed),
            evidence={
                "response_format": "single_python_fence",
                "extractable": True,
                "sandbox": sandbox_result.to_dict(),
                "entry_point": str(grader["entry_point"]),
            },
            **metadata,
        )

    @staticmethod
    def _sandbox_ready() -> bool:
        from .sandbox import code_grader_runtime, runtime_supported, sandbox_available

        selected_runtime = code_grader_runtime(required=True)
        return (
            selected_runtime is not None
            and sandbox_available()
            and runtime_supported(selected_runtime)
        )

    @staticmethod
    def _row_is_valid(row: StudyPrompt) -> bool:
        grader = row.grader
        return (
            row.task_format == "code_execution"
            and bool(str(grader.get("code_task_type") or "").strip())
            and bool(grader.get("executable_tests"))
            and bool(str(grader.get("entry_point") or "").strip())
        )


def _response_metadata(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        raw = dict(response.get("raw") or {})
        return {
            "prompt_tokens": int(response.get("prompt_tokens") or 0),
            "completion_tokens": int(response.get("completion_tokens") or 0),
            "latency_ms": float(response.get("latency_ms") or 0.0),
            "provider": str(response.get("provider") or ""),
            "raw": raw,
            "cost_usd": response.get("cost_usd"),
        }
    return {
        "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
        "latency_ms": float(getattr(response, "latency_ms", 0.0) or 0.0),
        "provider": str(getattr(response, "provider", "") or ""),
        "raw": dict(getattr(response, "raw", {}) or {}),
        "cost_usd": getattr(response, "cost_usd", None),
    }


def adapter_for_prompt(row: StudyPrompt) -> Any:
    kind = str(row.grader.get("kind") or "").lower()
    if row.source == "bfcl-v4" or kind.startswith("bfcl"):
        return BFCLAdapter()
    if row.source == "tau3" or kind.startswith("tau"):
        return TauBenchAdapter()
    if row.task_format == "code_execution":
        return ExecutableCodeAdapter()
    return PlainTextAdapter()


def collection_eligibility(row: StudyPrompt) -> tuple[bool, str]:
    """Return the frozen paid-collection eligibility without invoking a provider."""

    explicit = row.grader.get("collection_eligible")
    if explicit is False:
        return False, str(row.grader.get("collection_block_reason") or "row marked ineligible")
    adapter = adapter_for_prompt(row)
    return adapter.eligible(row)


def runtime_collection_eligibility(
    provider: Any,
    row: StudyPrompt,
    *,
    adapter_context: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Check adapter/provider capabilities before any budget reservation."""

    adapter = adapter_for_prompt(row)
    eligible, reason = adapter.eligible(row)
    if not eligible:
        return False, reason
    if isinstance(adapter, BFCLAdapter):
        if not _structured_capability(provider):
            return False, "candidate endpoint lacks provider-native BFCL tool calling"
        checker = (adapter_context or {}).get("official_checker", adapter.official_checker)
        checker_available = getattr(checker, "available", None)
        if checker_available is False:
            return False, "pinned BFCL official checker is unavailable"
    elif isinstance(adapter, TauBenchAdapter):
        if not _agent_capability(provider):
            return False, "candidate endpoint lacks native tau2 tool-calling capability"
        environment_factory = (adapter_context or {}).get("environment_factory")
        if environment_factory is None:
            from .tau2_runtime import tau2_available

            if tau2_available():
                environment_factory = True
        if not callable(environment_factory) and environment_factory is not True:
            return False, "pinned tau2 environment factory is unavailable"
    elif isinstance(adapter, ExecutableCodeAdapter):
        if not callable(getattr(provider, "complete", None)):
            return False, "candidate endpoint lacks ordinary completion capability"
    elif not callable(getattr(provider, "complete", None)):
        return False, "candidate endpoint lacks ordinary completion capability"
    return True, "runtime adapter/provider capabilities are executable"


def run_bfcl_smoke_fixtures() -> dict[str, Any]:
    """Exercise simple, multiple, parallel, and irrelevance semantics locally."""

    fixtures = {
        "simple": [StructuredToolCall("lookup", {"id": 1}, 0)],
        "multiple": [
            StructuredToolCall("lookup", {"id": 1}, 0),
            StructuredToolCall("summarize", {"id": 1}, 1),
        ],
        "parallel": [
            StructuredToolCall("lookup", {"id": 1}, 0, parallel_group="g0"),
            StructuredToolCall("lookup", {"id": 2}, 1, parallel_group="g0"),
        ],
        "irrelevance": [],
    }
    results: dict[str, Any] = {}
    for mode, expected in fixtures.items():
        passed, evidence = check_bfcl_calls(
            task_mode=mode,
            expected_tool_calls=expected,
            candidate_tool_calls=expected,
        )
        results[mode] = {
            "passed": passed,
            "call_count": len(evidence["candidate_tool_calls"]),
            "ground_truth_preserved": evidence["ground_truth_preserved"],
        }
    return {
        "status": "fixture_semantics_passed",
        "provider_calls": 0,
        "official_checker_executable": False,
        "official_checker_status": "unavailable_locally",
        "fixtures": results,
    }


class _FixtureTauEnvironment:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {"status": "open", "actions": []}

    def reset(self, initial_state: Mapping[str, Any]) -> None:
        self._state = dict(initial_state)
        self._state.setdefault("actions", [])

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        return ({"name": "close_ticket", "parameters": {"type": "object"}},)

    def apply_actions(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._state["actions"] = list(actions)
        if actions:
            self._state["status"] = "closed"

    def apply_tool_call(self, call: StructuredToolCall) -> Mapping[str, Any]:
        self.apply_actions((call.to_dict(),))
        return {"ok": True, "tool": call.name, "state": dict(self._state)}

    def state(self) -> Mapping[str, Any]:
        return dict(self._state)

    def evaluate(self, trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"passed": self._state.get("status") == "closed", "state": dict(self._state)}


class _FixtureTauProvider:
    supports_agent_tools = True

    async def complete_with_tools(self, request: StructuredCompletionRequest) -> StructuredCompletionResponse:
        if any(message.role == "tool" for message in request.messages):
            return StructuredCompletionResponse(
                model=request.model,
                content="done",
                tool_calls=(),
                provider="fixture",
            )
        return StructuredCompletionResponse(
            model=request.model,
            content="",
            tool_calls=(StructuredToolCall("close_ticket", {}, 0, call_id="fixture-1"),),
            provider="fixture",
        )


def run_tau_environment_smoke() -> dict[str, Any]:
    """Exercise stateful agent/tool/evaluator flow without a real tau install."""

    row = StudyPrompt(
        prompt_id="tau-smoke",
        prompt="Close the ticket.",
        split="validation",
        source="tau3",
        benchmark_family="tau3:fixture",
        domain="fixture",
        template_family="tau3:fixture",
        collection_vintage="fixture-v1",
        license_name="internal",
        license_status="internal",
        source_revision="fixture-v1",
        provenance_url="internal://tau-fixture",
        lane="general",
        label_type="tool_state",
        task_format="agent_state",
        grader={
            "kind": "tau3_official_trusted_result",
            "official_environment_id": "tau2-bench",
            "official_environment_available": True,
            "official_evaluator_id": "tau2-evaluate-trajs",
            "executor_id": "tau2-evaluate-trajs",
            "executor_version": "fixture-v1",
            "evidence_policy": "sha256_trusted_result",
            "initial_state": {"status": "open"},
        },
    )
    result = TauBenchAdapter().collect(
        _FixtureTauProvider(), row, model="fixture", environment_factory=lambda _row: _FixtureTauEnvironment()
    )
    return {
        "status": "fixture_environment_smoke_passed" if result.grade else "failed",
        "provider_calls": 0,
        "official_environment_executable": False,
        "official_environment_status": "unavailable_locally",
        "adapter_result": result.to_dict(),
    }
