"""
theriad.quality — Bounded quality-loop / refinement-pass orchestration.

Design principles
-----------------
- Cost-first: loops only fire when policy + validation justify the spend.
- Validation-driven: prefer objective signals (tests, schema, lint, preview)
  over model-only debate.
- Bounded: hard caps on passes, cost, and latency prevent runaway loops.
- Additive: normal /api/execute is completely unchanged unless the caller
  explicitly enables the loop.
- No hidden-state transfer: each stage is a fresh, independently routed
  model call; KV caches and activations are not shared across models.

Public API
----------
    from theriad.quality import run_quality_loop, QualityLoopRequest, QualityLoopResult
"""
from theriad.quality.loop_engine import run_quality_loop
from theriad.quality.loop_schemas import (
    QualityLoopRequest,
    QualityLoopResult,
    LoopPassDiagnostic,
    ValidationSignal,
    LoopStopReason,
)

__all__ = [
    "run_quality_loop",
    "QualityLoopRequest",
    "QualityLoopResult",
    "LoopPassDiagnostic",
    "ValidationSignal",
    "LoopStopReason",
]
