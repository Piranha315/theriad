"""Static UI contract checks for the unified Analyze experience."""
from pathlib import Path


THERIAD = Path("theriad-ui/src/Theriad.jsx").read_text(encoding="utf-8")


def test_primary_analysis_navigation_has_no_analyze_destination():
    assert "{ id: 'Analyze'" not in THERIAD
    assert "{ id: 'Simulate'" not in THERIAD
    assert "{ id: 'Route'" not in THERIAD
    assert "SimulateScreen" not in THERIAD


def test_analyze_runs_route_and_workflow_analysis_together():
    analyze_start = THERIAD.index("const AnalyzeScreen")
    analyze_end = THERIAD.index("const TIER_LABELS")
    analyze = THERIAD[analyze_start:analyze_end]
    assert "apiFetch('/api/route'" in analyze
    assert "apiFetch('/api/workflow/analyze'" in analyze
    assert "const [routeData, workflowData] = await Promise.all" in analyze


def test_analyze_exposes_execute_and_analysis_metrics():
    assert "Execute" in THERIAD
    assert "Est. Latency" in THERIAD
    assert "Confidence / quality" in THERIAD
    assert "Selected:" in THERIAD
