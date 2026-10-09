"""
Benchmark preflight checks.

Runs before any benchmark to catch resource and configuration problems early.
All checks are read-only and non-destructive.

Usage:
    from theriad.benchmark.preflight import run_preflight, PreflightResult
    result = run_preflight(providers=["openrouter", "ollama"])
    if not result.ok_to_run:
        sys.exit(1)
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import httpx

from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.preflight")

# ── Disk thresholds (bytes) ────────────────────────────────────────────────────
_GB = 1024 ** 3
DISK_BLOCK_LOCAL_GB = 10    # < 10 GB → refuse local models
DISK_WARN_LOCAL_GB  = 20    # < 20 GB → warn, allow OpenRouter only
DISK_ALLOW_SMALL_GB = 20    # >= 20 GB → allow small local models
DISK_ALLOW_BROAD_GB = 40    # >= 40 GB → allow broader local selection

# ── Status codes ──────────────────────────────────────────────────────────────
_OK      = "OK"
_WARN    = "WARN"
_BLOCKED = "BLOCKED"


@dataclass
class CheckResult:
    name: str
    status: str     # OK | WARN | BLOCKED
    value: str      # human-readable measured value
    reason: str     # explanation


@dataclass
class PreflightResult:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok_to_run(self) -> bool:
        return not any(c.status == _BLOCKED for c in self.checks)

    @property
    def local_allowed(self) -> bool:
        """True if local (Ollama) models are permitted given current disk state.
        Local is allowed only when disk is fully OK (>= 20 GB free).
        WARN (10-20 GB) disables local to avoid filling the disk during a run.
        """
        disk = next((c for c in self.checks if c.name == "disk"), None)
        if disk is None:
            return True
        return disk.status == _OK

    def print_report(self, *, file=None) -> None:
        import sys
        fh = file or sys.stdout
        width = 60
        print(f"\n{'─' * width}", file=fh)
        print("  BENCHMARK PREFLIGHT", file=fh)
        print(f"{'─' * width}", file=fh)
        for c in self.checks:
            icon = "✓" if c.status == _OK else ("⚠" if c.status == _WARN else "✗")
            print(f"  {icon} {c.name:<18s} {c.value:<16s} {c.status}", file=fh)
            if c.status != _OK:
                print(f"    → {c.reason}", file=fh)
        print(f"{'─' * width}", file=fh)
        if self.ok_to_run:
            print("  Status: READY TO RUN", file=fh)
        else:
            print("  Status: BLOCKED — fix issues above before running", file=fh)
        print(f"{'─' * width}\n", file=fh)


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_disk(local_providers: bool = True) -> CheckResult:
    """Check free disk space in the current working directory.

    The BLOCKED threshold only applies when local (Ollama) models are involved.
    Cloud-only runs need very little disk space and are never blocked on this check.
    """
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / _GB
        value = f"{free_gb:.1f} GB free"

        if free_gb < DISK_BLOCK_LOCAL_GB:
            if local_providers:
                return CheckResult(
                    name="disk",
                    status=_BLOCKED,
                    value=value,
                    reason=(
                        f"free disk < {DISK_BLOCK_LOCAL_GB} GB — "
                        "local benchmark disabled; free space before running"
                    ),
                )
            # Cloud-only: low disk is a warning, not a blocker
            return CheckResult(
                name="disk",
                status=_WARN,
                value=value,
                reason=(
                    f"free disk < {DISK_BLOCK_LOCAL_GB} GB — "
                    "local (Ollama) models disabled, but cloud providers can still run"
                ),
            )
        if free_gb < DISK_WARN_LOCAL_GB:
            return CheckResult(
                name="disk",
                status=_WARN,
                value=value,
                reason=(
                    f"free disk < {DISK_WARN_LOCAL_GB} GB — "
                    "local (Ollama) benchmark disabled; only OpenRouter/small runs allowed"
                ),
            )
        return CheckResult(name="disk", status=_OK, value=value, reason="")
    except Exception as exc:
        return CheckResult(
            name="disk", status=_WARN,
            value="unknown", reason=f"could not check disk: {exc}",
        )


def _check_memory() -> CheckResult:
    """Check available system memory (cross-platform: psutil preferred, /proc fallback)."""
    available_gb: float | None = None

    # Try psutil first (works on Linux, macOS, Windows)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / _GB
    except ImportError:
        pass

    # Fall back to /proc/meminfo on Linux when psutil is absent
    if available_gb is None:
        try:
            with open("/proc/meminfo") as fh:
                lines = fh.read().splitlines()
            info: dict[str, int] = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    info[key] = int(val)
            available_gb = info.get("MemAvailable", 0) / (1024 ** 2)
        except Exception:
            pass

    if available_gb is None:
        return CheckResult(
            name="memory", status=_WARN,
            value="unknown",
            reason="could not read memory info (install psutil for cross-platform support)",
        )

    value = f"{available_gb:.1f} GB avail"
    if available_gb < 1.0:
        return CheckResult(
            name="memory", status=_BLOCKED,
            value=value,
            reason="< 1 GB RAM available — risk of OOM during benchmark",
        )
    if available_gb < 2.0:
        return CheckResult(
            name="memory", status=_WARN,
            value=value,
            reason="< 2 GB RAM available — avoid large local models",
        )
    return CheckResult(name="memory", status=_OK, value=value, reason="")


def _check_ollama(base_url: str = "http://localhost:11434") -> CheckResult:
    """Check whether Ollama is reachable."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/api/tags")
        if resp.status_code == 200:
            data = resp.json()
            n_models = len(data.get("models", []))
            return CheckResult(
                name="ollama",
                status=_OK,
                value=f"{n_models} model(s)",
                reason="",
            )
        return CheckResult(
            name="ollama", status=_WARN,
            value=f"HTTP {resp.status_code}",
            reason="Ollama returned non-200; local benchmark may fail",
        )
    except Exception:
        return CheckResult(
            name="ollama", status=_WARN,
            value="unreachable",
            reason="Ollama not running — local benchmark will be skipped",
        )


def _check_api_key(provider: str) -> CheckResult:
    """Check that the expected API key env var is present."""
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "groq":       "GROQ_API_KEY",
        "cerebras":   "CEREBRAS_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var is None:
        return CheckResult(
            name=f"{provider}_key", status=_OK,
            value="n/a", reason="",
        )
    present = bool(os.environ.get(env_var, ""))
    if present:
        return CheckResult(
            name=f"{provider}_key", status=_OK,
            value="present", reason="",
        )
    return CheckResult(
        name=f"{provider}_key", status=_BLOCKED,
        value="missing",
        reason=f"{env_var} not set — {provider} benchmark disabled",
    )


def _check_budget(budget_usd: float) -> CheckResult:
    """Sanity-check the budget ceiling."""
    if budget_usd <= 0:
        return CheckResult(
            name="budget", status=_BLOCKED,
            value=f"${budget_usd:.2f}",
            reason="Budget must be positive",
        )
    if budget_usd > 20.0:
        return CheckResult(
            name="budget", status=_WARN,
            value=f"${budget_usd:.2f}",
            reason="Budget > $20 — confirm this is intentional",
        )
    return CheckResult(
        name="budget", status=_OK, value=f"${budget_usd:.2f}", reason="",
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_preflight(
    providers: list[str],
    budget_usd: float = 10.0,
    *,
    check_ollama: bool = True,
    ollama_url: str = "http://localhost:11434",
) -> PreflightResult:
    """
    Run all preflight checks and return a PreflightResult.

    Args:
        providers:    Provider names to be used in the benchmark.
        budget_usd:   Configured spend ceiling.
        check_ollama: Whether to ping Ollama (skip in unit tests).
        ollama_url:   Ollama base URL override.
    """
    checks: list[CheckResult] = []

    local_providers = "ollama" in providers
    checks.append(_check_disk(local_providers=local_providers))
    checks.append(_check_memory())

    # API key checks for paid providers
    paid = [p for p in providers if p != "ollama"]
    for provider in paid:
        checks.append(_check_api_key(provider))

    # Ollama reachability (only if ollama is in the provider list)
    if "ollama" in providers and check_ollama:
        checks.append(_check_ollama(ollama_url))

    checks.append(_check_budget(budget_usd))

    result = PreflightResult(checks=checks)
    logger.info(
        "Preflight complete: %s",
        "READY" if result.ok_to_run else "BLOCKED",
    )
    return result
