"""Reproducible, provider-free executor environment manifests.

The study depends on three benchmark-side environments that must not be
implicitly recreated in ``/tmp``.  This module records the exact interpreter,
resolved executable, source revision, frozen dependency set, and sandbox
policy, and verifies that a later preflight is using the same paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .sandbox import SANDBOX_EXECUTOR_ID, SANDBOX_POLICY_VERSION, runtime_details

EXECUTOR_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SOURCE_CACHE = Path("/tmp/theriad-router-v2-source-cache")
_NON_PUBLISHABLE_DISTRIBUTIONS = {
    "code_grader": {"theriad"},
    "bfcl": {"theriad", "bfcl_eval"},
    "tau2": {"theriad", "tau2"},
}


class ExecutorManifestError(RuntimeError):
    """Raised when a pinned executor environment cannot be verified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(source: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _runtime_command(python: Path, args: Iterable[str]) -> tuple[str, str]:
    try:
        result = subprocess.run(
            (str(python), *tuple(args)),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": str(python.parent.parent),
                "PYTHONDONTWRITEBYTECODE": "1",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExecutorManifestError(f"runtime command failed for {python}: {exc}") from exc
    return result.stdout, result.stderr


def _freeze(python: Path) -> tuple[str, ...]:
    try:
        stdout, _ = _runtime_command(python, ("-m", "pip", "freeze", "--all"))
    except ExecutorManifestError:
        # uv-managed virtualenvs intentionally omit pip.  Read the installed
        # distributions through the pinned interpreter instead of silently
        # inspecting whichever pip happens to be on PATH.
        stdout, _ = _runtime_command(
            python,
            (
                "-c",
                "import importlib.metadata as m; "
                "print('\\n'.join(sorted(f'{d.metadata.get(\"Name\")}=={d.version}' "
                "for d in m.distributions() if d.metadata.get('Name'))))",
            ),
        )
    return tuple(line.strip() for line in stdout.splitlines() if line.strip())


def _normalise_frozen(role: str, frozen: Iterable[str]) -> tuple[str, ...]:
    """Exclude project/source distributions from the publishable lock set."""

    excluded = _NON_PUBLISHABLE_DISTRIBUTIONS.get(role, set())
    result: list[str] = []
    for line in frozen:
        name = str(line).split("==", 1)[0].strip().lower().replace("_", "-")
        if name in {value.replace("_", "-") for value in excluded}:
            continue
        result.append(str(line).strip())
    return tuple(sorted(value for value in result if value))


def _dependency_hash(frozen: Iterable[str]) -> str:
    return _sha256_bytes("\n".join(frozen).encode("utf-8"))


def _runtime_identity(python: Path) -> dict[str, Any]:
    stdout, _ = _runtime_command(
        python,
        (
            "-c",
            "import json, os, platform, sys; "
            "print(json.dumps({'sys_version': sys.version, 'sys_version_verbose': sys.version, "
            "'sys_executable': sys.executable, 'resolved_executable': os.path.realpath(sys.executable), "
            "'platform': platform.platform()}, sort_keys=True))",
        ),
    )
    try:
        value = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        raise ExecutorManifestError(f"runtime identity is not JSON for {python}") from exc
    if not isinstance(value, dict):
        raise ExecutorManifestError(f"runtime identity is malformed for {python}")
    return value


def _runtime_details_for(python: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    details = dict(runtime_details(python))
    # runtime_details also serves the active-process sandbox tests.  A
    # manifest, however, must describe the selected interpreter, not the
    # interpreter running the manifest loader.
    details["sys_executable"] = str(identity.get("sys_executable") or "")
    details["sys_version"] = str(identity.get("sys_version") or "")
    details["resolved_executable"] = str(identity.get("resolved_executable") or "")
    details["platform"] = str(identity.get("platform") or details.get("platform") or "")
    return details


def _environment_hash(record: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            key: value
            for key, value in record.items()
            if key not in {"environment_hash", "reproducible_contract_hash"}
        }
    )


def reproducible_contract_hash(record: Mapping[str, Any]) -> str:
    """Hash environment properties that survive a fresh-directory rebuild."""

    directives = []
    for item in record.get("local_source_directives", []) or []:
        if isinstance(item, Mapping):
            directives.append(
                {
                    "kind": str(item.get("kind") or ""),
                    "revision": str(item.get("revision") or ""),
                    "install": str(item.get("install") or ""),
                }
            )
    identity = record.get("python_identity") or {}
    details = record.get("runtime_details") or {}
    contract = {
        "role": record.get("role"),
        "resolved_executable": record.get("resolved_executable"),
        "sys_version_verbose": record.get("sys_version_verbose"),
        "python_version": identity.get("sys_version"),
        "source_revision": record.get("source_revision"),
        "dependency_frozen_set": list(record.get("dependency_frozen_set") or []),
        "dependency_lock_sha256": record.get("dependency_lock_sha256"),
        "local_source_directives": directives,
        "sandbox_policy_version": record.get("sandbox_policy_version"),
        "sandbox_executor_id": SANDBOX_EXECUTOR_ID,
        "runtime_approach": details.get("approach"),
        "platform_system": details.get("platform_system"),
        "python_app_helper_present": bool(details.get("python_app_helper")),
    }
    return _sha256_json(contract)


def _required_paths_for(
    role: str,
    *,
    code_source: Path,
    code_cache: Path,
    bfcl_source: Path,
    bfcl_cache: Path,
    tau_source: Path,
) -> tuple[Path, ...]:
    if role == "code_grader":
        return (code_source / "scripts/validate_router_v2_code_tasks.py", code_cache)
    if role == "bfcl":
        return (
            bfcl_source,
            bfcl_cache,
            bfcl_source / "bfcl_eval/eval_checker/ast_eval/ast_checker.py",
            bfcl_source / "bfcl_eval/eval_checker/eval_runner.py",
        )
    if role == "tau2":
        return (
            tau_source,
            tau_source / "src/tau2/evaluator/evaluator.py",
            tau_source / "src/tau2/domains/mock/environment.py",
        )
    raise ExecutorManifestError(f"unknown executor role {role}")


def _runtime_record(
    *,
    role: str,
    python: Path,
    source: Path,
    source_revision: str,
    rebuild_command: str,
    required_paths: tuple[Path, ...],
    dependency_lock_path: Path,
    local_source_directives: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    requested = str(python.expanduser())
    resolved = python.expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ExecutorManifestError(f"{role} runtime is missing or not executable: {python}")
    version_stdout, _ = _runtime_command(python, ("-VV",))
    identity = _runtime_identity(python)
    frozen = _normalise_frozen(role, _freeze(python))
    details = _runtime_details_for(python, identity)
    record: dict[str, Any] = {
        "role": role,
        "requested_executable": requested,
        "resolved_executable": str(resolved),
        "sys_version_verbose": version_stdout.strip(),
        "identity": [
            str(identity.get("sys_executable") or ""),
            str(identity.get("resolved_executable") or ""),
            str(identity.get("sys_version") or ""),
        ],
        "python_identity": identity,
        "platform": platform.platform(),
        "source": str(source),
        "source_revision": source_revision or _git_revision(source),
        "source_git_head": _git_revision(source),
        "required_paths": [str(path) for path in required_paths],
        "dependency_frozen_set": list(frozen),
        "dependency_lock_sha256": _dependency_hash(frozen),
        "dependency_lock_path": str(dependency_lock_path),
        "dependency_lock_file_sha256": _sha256_path(dependency_lock_path)
        if dependency_lock_path.is_file()
        else "",
        "local_source_directives": [dict(item) for item in local_source_directives],
        "rebuild_command": rebuild_command,
        "sandbox_policy_version": SANDBOX_POLICY_VERSION,
        "runtime_details": details,
    }
    record["environment_hash"] = _environment_hash(record)
    record["reproducible_contract_hash"] = reproducible_contract_hash(record)
    return record


def create_executor_manifest(
    *,
    output: str | Path,
    code_python: str | Path,
    bfcl_python: str | Path,
    tau_python: str | Path,
    bfcl_source: str | Path,
    bfcl_cache: str | Path,
    tau_source: str | Path,
    source_cache: str | Path = DEFAULT_SOURCE_CACHE,
    code_source: str | Path | None = None,
    code_cache: str | Path | None = None,
    expected_bfcl_revision: str = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    expected_tau_revision: str = "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
) -> dict[str, Any]:
    """Inspect existing environments and write a deterministic manifest.

    This function never installs packages or contacts a provider.  A caller
    that needs to rebuild an environment receives the recorded command and a
    non-ready manifest instead of an undocumented temporary fallback.
    """

    output_path = Path(output).expanduser().resolve()
    code_source_path = Path(code_source) if code_source is not None else Path.cwd()
    code_cache_path = (
        Path(code_cache)
        if code_cache is not None
        else Path(source_cache) / "acquired/code-data"
    )
    bfcl_source_path = Path(bfcl_source)
    bfcl_cache_path = Path(bfcl_cache)
    tau_source_path = Path(tau_source)
    lock_dir = (output_path.parent / "executor-locks").resolve()
    lock_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    errors: list[str] = []
    revisions = {
        "code_grader": _git_revision(code_source_path),
        "bfcl": expected_bfcl_revision,
        "tau2": expected_tau_revision,
    }
    required = {
        role: _required_paths_for(
            role,
            code_source=code_source_path,
            code_cache=code_cache_path,
            bfcl_source=bfcl_source_path,
            bfcl_cache=bfcl_cache_path,
            tau_source=tau_source_path,
        )
        for role in ("code_grader", "bfcl", "tau2")
    }
    local_directives = {
        "code_grader": (),
        "bfcl": (
            {
                "kind": "local_source",
                "path": str(bfcl_source_path),
                "revision": revisions["bfcl"],
                "install": "uv pip install --python {python} --no-deps {source}",
            },
        ),
        "tau2": (
            {
                "kind": "local_source",
                "path": str(tau_source_path),
                "revision": revisions["tau2"],
                "install": "uv pip install --python {python} --no-deps {source}",
            },
        ),
    }
    requests = {
        "code_grader": (
            Path(code_python),
            code_source_path,
            revisions["code_grader"],
            f"uv venv --python {Path(code_python).expanduser().resolve()} {Path(code_python).parent.parent} && uv pip sync --python {Path(code_python)} {lock_dir / 'code_grader.freeze.txt'}",
        ),
        "bfcl": (
            Path(bfcl_python),
            bfcl_source_path,
            revisions["bfcl"],
            f"uv venv --python {Path(bfcl_python).expanduser().resolve()} {Path(bfcl_python).parent.parent} && uv pip sync --python {Path(bfcl_python)} {lock_dir / 'bfcl.freeze.txt'} && uv pip install --python {Path(bfcl_python)} --no-deps {bfcl_source_path}",
        ),
        "tau2": (
            Path(tau_python),
            tau_source_path,
            revisions["tau2"],
            f"uv venv --python {Path(tau_python).expanduser().resolve()} {Path(tau_python).parent.parent} && uv pip sync --python {Path(tau_python)} {lock_dir / 'tau2.freeze.txt'} && uv pip install --python {Path(tau_python)} --no-deps {tau_source_path}",
        ),
    }
    for role, (python, source, revision, rebuild) in requests.items():
        try:
            frozen_path = lock_dir / f"{role}.freeze.txt"
            current_frozen = _normalise_frozen(role, _freeze(python))
            frozen_path.write_text("\n".join(current_frozen) + "\n", encoding="utf-8")
            records[role] = _runtime_record(
                role=role,
                python=python,
                source=source,
                source_revision=revision,
                rebuild_command=rebuild,
                required_paths=required[role],
                dependency_lock_path=frozen_path,
                local_source_directives=local_directives[role],
            )
        except ExecutorManifestError as exc:
            errors.append(str(exc))
            records[role] = {
                "role": role,
                "requested_executable": str(python),
                "resolved_executable": str(python.expanduser().resolve()),
                "source": str(source),
                "source_revision": revision,
                "rebuild_command": rebuild,
                "status": "missing_or_unreadable",
            }
    required_paths = [
        path
        for paths in required.values()
        for path in paths
    ]
    errors.extend(
        f"required executor path is missing: {path}"
        for path in required_paths
        if not path.exists()
    )
    sandbox = {
        "executor_id": SANDBOX_EXECUTOR_ID,
        "policy_version": SANDBOX_POLICY_VERSION,
        "network_disabled": True,
        "base_filesystem_read_only": True,
        "task_directory_only_writable": True,
        "credentials_in_environment": False,
    }
    payload: dict[str, Any] = {
        "schema_version": EXECUTOR_MANIFEST_SCHEMA_VERSION,
        "status": "ready" if not errors else "blocked",
        "provider_calls": 0,
        "provider_spend_usd": 0.0,
        "source_cache": str(Path(source_cache)),
        "platform": platform.platform(),
        "sandbox": sandbox,
        "runtimes": records,
        "required_paths": list(dict.fromkeys(str(path) for path in required_paths)),
        "errors": errors,
    }
    payload["manifest_hash"] = _sha256_json(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_executor_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutorManifestError(f"executor manifest is unreadable: {manifest_path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != EXECUTOR_MANIFEST_SCHEMA_VERSION:
        raise ExecutorManifestError("unsupported executor manifest schema")
    expected = _sha256_json({key: value for key, value in raw.items() if key != "manifest_hash"})
    if raw.get("manifest_hash") != expected:
        raise ExecutorManifestError("executor manifest hash does not match its contents")
    if raw.get("status") != "ready":
        raise ExecutorManifestError(
            "executor manifest is not ready: " + "; ".join(str(item) for item in raw.get("errors", []))
        )
    runtimes = raw.get("runtimes")
    if not isinstance(runtimes, Mapping):
        raise ExecutorManifestError("executor manifest has no runtime records")
    for role in ("code_grader", "bfcl", "tau2"):
        record = runtimes.get(role)
        if not isinstance(record, Mapping):
            raise ExecutorManifestError(f"executor manifest is missing {role}")
        executable = Path(str(record.get("requested_executable") or ""))
        resolved = Path(str(record.get("resolved_executable") or ""))
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ExecutorManifestError(f"manifest runtime for {role} is unavailable: {executable}")
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ExecutorManifestError(f"manifest resolved runtime for {role} is unavailable: {resolved}")
        try:
            identity = _runtime_identity(executable)
            verbose, _ = _runtime_command(executable, ("-VV",))
            current_frozen = _normalise_frozen(role, _freeze(executable))
            current_details = _runtime_details_for(executable, identity)
        except ExecutorManifestError as exc:
            raise ExecutorManifestError(f"manifest runtime for {role} cannot be recomputed: {exc}") from exc
        expected_identity = record.get("python_identity")
        if expected_identity != identity:
            raise ExecutorManifestError(f"manifest Python identity drift for {role}")
        if str(record.get("sys_version_verbose") or "") != verbose.strip():
            raise ExecutorManifestError(f"manifest Python version drift for {role}")
        if str(record.get("resolved_executable") or "") != str(executable.expanduser().resolve()):
            raise ExecutorManifestError(f"manifest requested/resolved interpreter drift for {role}")
        if list(record.get("dependency_frozen_set") or []) != list(current_frozen):
            raise ExecutorManifestError(f"manifest dependency frozen set drift for {role}")
        if record.get("dependency_lock_sha256") != _dependency_hash(current_frozen):
            raise ExecutorManifestError(f"manifest dependency lock hash drift for {role}")
        lock_path = Path(str(record.get("dependency_lock_path") or ""))
        if not lock_path.is_file() or _sha256_path(lock_path) != record.get("dependency_lock_file_sha256"):
            raise ExecutorManifestError(f"manifest dependency lock file drift for {role}")
        if record.get("runtime_details") != current_details:
            raise ExecutorManifestError(f"manifest runtime/sandbox details drift for {role}")
        source = Path(str(record.get("source") or ""))
        source_head = _git_revision(source)
        if not source_head or source_head != str(record.get("source_git_head") or ""):
            raise ExecutorManifestError(f"manifest source Git HEAD drift for {role}")
        if source_head != str(record.get("source_revision") or ""):
            raise ExecutorManifestError(f"manifest source revision drift for {role}")
        for required_path in record.get("required_paths", []):
            if not Path(str(required_path)).exists():
                raise ExecutorManifestError(
                    f"manifest required checker/data path is unavailable for {role}: {required_path}"
                )
        if record.get("sandbox_policy_version") != SANDBOX_POLICY_VERSION:
            raise ExecutorManifestError(f"manifest sandbox policy drift for {role}")
        if record.get("environment_hash") != _environment_hash(record):
            raise ExecutorManifestError(f"manifest environment hash drift for {role}")
        if record.get("reproducible_contract_hash") != reproducible_contract_hash(record):
            raise ExecutorManifestError(f"manifest reproducible environment contract drift for {role}")
    for required in raw.get("required_paths", []):
        if not Path(str(required)).exists():
            raise ExecutorManifestError(f"manifest required path is unavailable: {required}")
    sandbox = raw.get("sandbox")
    if not isinstance(sandbox, Mapping) or sandbox.get("policy_version") != SANDBOX_POLICY_VERSION:
        raise ExecutorManifestError("executor manifest sandbox policy is not the pinned policy")
    if sandbox.get("executor_id") != SANDBOX_EXECUTOR_ID:
        raise ExecutorManifestError("executor manifest sandbox executor is not the pinned backend")
    if sandbox.get("network_disabled") is not True or sandbox.get("base_filesystem_read_only") is not True:
        raise ExecutorManifestError("executor manifest sandbox restrictions are weaker than the contract")
    if sandbox.get("task_directory_only_writable") is not True or sandbox.get("credentials_in_environment") is not False:
        raise ExecutorManifestError("executor manifest sandbox isolation fields are invalid")
    return raw


__all__ = [
    "DEFAULT_SOURCE_CACHE",
    "EXECUTOR_MANIFEST_SCHEMA_VERSION",
    "ExecutorManifestError",
    "create_executor_manifest",
    "load_executor_manifest",
    "reproducible_contract_hash",
]
