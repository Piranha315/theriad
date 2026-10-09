"""
J-Space (Joint Space) — shared context blackboard for multi-model agent coordination.

Each model role (orchestrator, executor, evaluator, synthesizer) reads from and
writes to J-Space instead of passing full conversation history. This saves tokens
by compressing observations and keeping only the information each role needs.

Design:
- Lives on the BrowserSession as a plain dict (serializable, no class overhead).
- Factory functions create, read, and update the space.
- Compression is aggressive: raw observations are digested into short summaries.
- The orchestrator's full reasoning never reaches the executor — only the plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """One step in the orchestrator's plan."""
    index: int
    description: str
    expected_action: str  # navigate | search | extract | compare | ...
    success_criteria: str
    done: bool = False
    outcome: str | None = None


@dataclass
class JSpace:
    """
    Shared context blackboard for multi-model agent coordination.

    Token budget per role (approximate targets):
    - Orchestrator sees: goal + progress_summary (~300 tokens)
    - Executor sees: current plan_step + observation_digest + recent_actions (~500 tokens)
    - Evaluator sees: goal + findings + extracted_data (~400 tokens)
    - Synthesizer sees: goal + findings + extracted_data (~400 tokens)
    """
    plan: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    findings: list[str] = field(default_factory=list)
    observation_digest: list[str] = field(default_factory=list)
    extracted_data: list[dict[str, Any]] = field(default_factory=list)
    urls_visited: list[str] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    replan_reason: str | None = None

    @property
    def current_step(self) -> PlanStep | None:
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    def advance_step(self, outcome: str = "completed") -> None:
        step = self.current_step
        if step:
            step.done = True
            step.outcome = outcome
        self.current_step_index += 1

    def add_finding(self, finding: str) -> None:
        self.findings.append(finding)
        if len(self.findings) > 20:
            self.findings = self.findings[-20:]

    def add_observation(self, url: str, digest: str) -> None:
        entry = f"[{url}] {digest}"
        self.observation_digest.append(entry)
        if len(self.observation_digest) > 10:
            self.observation_digest = self.observation_digest[-10:]
        if url not in self.urls_visited:
            self.urls_visited.append(url)

    def record_action(self, action_type: str, target: str | None, result: str) -> None:
        entry = f"{action_type}({target or ''}) → {result}"
        self.recent_actions.append(entry)
        if len(self.recent_actions) > 15:
            self.recent_actions = self.recent_actions[-15:]

    def add_extracted(self, data: dict[str, Any]) -> None:
        self.extracted_data.append(data)
        if len(self.extracted_data) > 10:
            self.extracted_data = self.extracted_data[-10:]

    # ── Prompt formatters — each role gets only what it needs ────────────────

    def format_for_orchestrator(self, goal: str) -> str:
        parts = [f"Goal: {goal}"]
        if self.urls_visited:
            parts.append(f"Pages visited: {', '.join(self.urls_visited[-5:])}")
        if self.findings:
            parts.append("Findings so far:\n" + "\n".join(f"- {f}" for f in self.findings[-8:]))
        if self.plan:
            done = sum(1 for s in self.plan if s.done)
            parts.append(f"Plan progress: {done}/{len(self.plan)} steps completed")
            for s in self.plan:
                mark = "✓" if s.done else "→" if s.index == self.current_step_index else " "
                parts.append(f"  [{mark}] Step {s.index + 1}: {s.description}")
        if self.replan_reason:
            parts.append(f"Replan requested: {self.replan_reason}")
        return "\n".join(parts)

    def format_for_executor(self) -> str:
        parts = []
        step = self.current_step
        if step:
            parts.append(f"Current task (step {step.index + 1}): {step.description}")
            parts.append(f"Expected action: {step.expected_action}")
            parts.append(f"Success criteria: {step.success_criteria}")
        if self.recent_actions:
            parts.append("Recent actions:\n" + "\n".join(f"  {a}" for a in self.recent_actions[-5:]))
        if self.findings:
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in self.findings[-5:]))
        return "\n".join(parts)

    def format_for_evaluator(self, goal: str) -> str:
        parts = [f"Goal: {goal}"]
        if self.findings:
            parts.append("Findings:\n" + "\n".join(f"- {f}" for f in self.findings))
        if self.extracted_data:
            parts.append(f"Extracted data entries: {len(self.extracted_data)}")
            for d in self.extracted_data[-5:]:
                parts.append(f"  {d}")
        if self.urls_visited:
            parts.append(f"Pages visited ({len(self.urls_visited)}): {', '.join(self.urls_visited[-5:])}")
        if self.plan:
            done = sum(1 for s in self.plan if s.done)
            parts.append(f"Plan: {done}/{len(self.plan)} steps completed")
        return "\n".join(parts)

    def format_for_synthesizer(self, goal: str) -> str:
        parts = [f"Goal: {goal}"]
        if self.findings:
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in self.findings))
        if self.extracted_data:
            parts.append("Extracted data:")
            for d in self.extracted_data:
                parts.append(f"  {d}")
        if self.urls_visited:
            parts.append(f"Sources: {', '.join(self.urls_visited)}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> JSpace:
        plan_data = data.pop("plan", [])
        plan = [PlanStep(**s) for s in plan_data]
        return cls(plan=plan, **data)
