"""
Workflow detector: classifies a prompt as multi-step workflow vs single v1 call.

Detection pipeline (in order):
1. RuleGuardrailDetector — keyword heuristics; cheap, no model needed.
   High-confidence rule decisions are final (≥ 0.75).
2. MatryoshkaEmbeddingDetector — progressive cosine similarity at 64/128/256/full
   embedding dims. Stops at the first dim that produces a decisive margin.
3. WorkflowDetector (orchestrator) — runs 1 then 2 if needed, then falls back
   to project/tool signal heuristics for still-ambiguous prompts.

Design:
- No provider calls, no API key required.
- SentenceTransformerBackend is injected (lazy model download on first use).
  Tests inject MockEmbeddingBackend to avoid network access.
- WorkflowDetector accepts an optional embedding_backend so callers and tests
  can inject any EmbeddingBackend without subclassing.
"""
from __future__ import annotations

import numpy as np

from theriad.layers.layer2_embedding.backends import (
    EmbeddingBackend,
    SentenceTransformerBackend,
)

from .types import WorkflowDetection

# ── Rule sets ─────────────────────────────────────────────────────────────────

_WORKFLOW_VERBS: frozenset[str] = frozenset({
    "build", "design", "implement", "deploy", "refactor", "debug",
    "document", "create", "develop", "configure", "integrate", "fix",
    "migrate", "scaffold", "generate", "add", "update", "restructure",
    "ship", "launch", "release", "install", "upgrade", "optimize",
    "bootstrap", "initialize", "init", "set",
})

_WORKFLOW_TECH: frozenset[str] = frozenset({
    "frontend", "backend", "database", "db", "api", "docker", "ci", "cd",
    "tests", "repo", "repository", "auth", "authentication",
    "endpoint", "server", "client", "service", "module", "package",
    "app", "application", "pipeline", "file", "files", "script", "scripts",
    "codebase", "microservice", "lambda", "function", "container",
})

_SEQUENCE_WORDS: frozenset[str] = frozenset({
    "then", "after", "next", "finally", "subsequently", "afterwards",
    "first", "second", "third", "lastly",
})

_SINGLE_VERBS: frozenset[str] = frozenset({
    "summarize", "summarise", "translate", "rewrite", "extract", "explain",
    "classify", "describe", "list", "answer", "convert", "paraphrase",
    "define", "identify", "analyze", "analyse", "review", "compare",
    "proofread", "edit",
})

# Signals whose presence near an ambiguous prompt forces workflow escalation
_PROJECT_SIGNALS: frozenset[str] = frozenset({
    "repo", "repository", "codebase", "project", "file", "files",
    "script", "scripts", "module",
})

_TOOL_SIGNALS: frozenset[str] = frozenset({
    "pytest", "npm", "docker", "ci", "jest", "ruff", "mypy",
    "pipeline", "makefile", "dockerfile", "git",
})

# Matryoshka dims to try, smallest first
_MRL_DIMS: list[int | None] = [64, 128, 256, None]  # None = full dim

# Min |margin| required to treat an embedding decision as confident
_MARGIN_THRESHOLD: float = 0.08

# High-confidence threshold for rule guardrail — decisions above this are final
_RULE_CONFIDENCE_THRESHOLD: float = 0.75

# Prototype corpus for the embedding classifier
_WORKFLOW_PROTOTYPES: list[str] = [
    "Build a full app with frontend, backend, tests, and documentation.",
    "Refactor this project, fix bugs, add tests, and update docs.",
    "Turn this codebase into a production-ready service.",
    "Implement the feature, run tests, debug failures, and summarize changes.",
    "Create a web app with authentication, database, API routes, and deployment.",
    "Help me get this repo ready to ship.",
    "Design, build, test, and deploy a React app.",
    "Set up CI/CD, add linting, fix type errors, and write a README.",
    "Scaffold a new project with backend, database migrations, and auth.",
    "Refactor the backend, add unit tests, fix failing routes, and document the API.",
]

_SINGLE_PROTOTYPES: list[str] = [
    "Summarize this article in three sentences.",
    "Translate this paragraph into French.",
    "Explain this concept briefly.",
    "Rewrite this paragraph more clearly.",
    "Extract all email addresses from this text.",
    "Answer this question briefly.",
    "Classify this text into one of three categories.",
    "Explain how to build a React app.",
    "Describe what this function does.",
    "Convert this JSON structure to YAML.",
    "List the key arguments in this document.",
    "Proofread this email and suggest improvements.",
]


def _detect_workflow_type(tokens: set[str]) -> str | None:
    """Infer workflow template from tech signal tokens."""
    build_signals = {"frontend", "backend", "database", "api", "docker", "auth", "authentication"}
    if tokens & build_signals:
        return "software_build"
    if tokens & {"pipeline", "etl", "data"}:
        return "software_build"  # only template available for now
    return "software_build"  # default to software_build for all detected workflows


class RuleGuardrailDetector:
    """
    Keyword-based classifier. Fast, no model required.

    Returns high-confidence decisions for clear-cut prompts and low-confidence
    (ambiguous) signals for prompts that need semantic analysis.
    """

    def detect(self, prompt: str) -> WorkflowDetection:
        # Strip punctuation so "build," matches "build" in the frozen sets
        import re
        tokens = re.sub(r"[^\w\s]", " ", prompt.lower()).split()
        token_set = set(tokens)

        wf_verb_count = len(_WORKFLOW_VERBS & token_set)
        tech_count = len(_WORKFLOW_TECH & token_set)
        seq_count = len(_SEQUENCE_WORDS & token_set)
        sp_verb_count = len(_SINGLE_VERBS & token_set)

        starts_with_sp_verb = bool(tokens) and tokens[0] in _SINGLE_VERBS

        # ── Clear single-prompt: starts with an imperative single-verb ─────────
        # "Explain how to build X" — even if "build" appears, the intent is explanation
        if starts_with_sp_verb and wf_verb_count <= 1:
            sp_signals = list(token_set & _SINGLE_VERBS)
            return WorkflowDetection(
                is_workflow=False,
                confidence=0.90,
                workflow_type=None,
                decision_source="rule_guardrail",
                reason=(
                    f"Prompt starts with single-task verb '{tokens[0]}'; "
                    f"workflow verb count ({wf_verb_count}) too low for workflow."
                ),
                signals=sp_signals,
            )

        # ── Clear single-prompt: single-verb, no workflow tech or verbs ────────
        if sp_verb_count >= 1 and wf_verb_count == 0 and tech_count == 0:
            sp_signals = list(token_set & _SINGLE_VERBS)
            return WorkflowDetection(
                is_workflow=False,
                confidence=0.82,
                workflow_type=None,
                decision_source="rule_guardrail",
                reason="Single-task verb with no workflow tech or action signals.",
                signals=sp_signals,
            )

        # ── Strong workflow: 3+ distinct action verbs ─────────────────────────
        if wf_verb_count >= 3:
            wf_signals = list(token_set & _WORKFLOW_VERBS)
            return WorkflowDetection(
                is_workflow=True,
                confidence=min(0.95, 0.88 + (wf_verb_count - 3) * 0.02),
                workflow_type=_detect_workflow_type(token_set),
                decision_source="rule_guardrail",
                reason=f"Found {wf_verb_count} distinct workflow action verbs.",
                signals=wf_signals,
            )

        # ── Strong workflow: 2+ verbs with connectors or tech ─────────────────
        if wf_verb_count >= 2 and (seq_count >= 1 or tech_count >= 2):
            wf_signals = list((token_set & _WORKFLOW_VERBS) | (token_set & _SEQUENCE_WORDS))
            return WorkflowDetection(
                is_workflow=True,
                confidence=0.87,
                workflow_type=_detect_workflow_type(token_set),
                decision_source="rule_guardrail",
                reason=(
                    f"{wf_verb_count} workflow verbs with "
                    f"{seq_count} sequence word(s) and {tech_count} tech artifact(s)."
                ),
                signals=wf_signals,
            )

        # ── Workflow: 2+ action verbs (catch-all for multi-verb prompts) ─────────
        if wf_verb_count >= 2:
            wf_signals = list(token_set & _WORKFLOW_VERBS)
            return WorkflowDetection(
                is_workflow=True,
                confidence=0.80,
                workflow_type=_detect_workflow_type(token_set),
                decision_source="rule_guardrail",
                reason=f"{wf_verb_count} distinct workflow verbs found.",
                signals=wf_signals,
            )

        # ── Probable workflow: single verb + strong tech cluster ───────────────
        if wf_verb_count >= 1 and tech_count >= 3:
            wf_signals = list((token_set & _WORKFLOW_VERBS) | (token_set & _WORKFLOW_TECH))
            return WorkflowDetection(
                is_workflow=True,
                confidence=0.82,
                workflow_type=_detect_workflow_type(token_set),
                decision_source="rule_guardrail",
                reason=f"Workflow verb with strong tech cluster ({tech_count} artifacts).",
                signals=wf_signals[:8],
            )

        # ── Ambiguous: pass to semantic layer ─────────────────────────────────
        mixed = list((token_set & _WORKFLOW_VERBS) | (token_set & _SINGLE_VERBS))
        return WorkflowDetection(
            is_workflow=False,
            confidence=0.50,
            workflow_type=None,
            decision_source="rule_guardrail",
            reason="Mixed or insufficient signals — needs semantic classification.",
            signals=mixed[:8],
        )


class MatryoshkaEmbeddingDetector:
    """
    Progressive cosine-similarity classifier using Matryoshka embeddings.

    Embeds the prompt and prototype corpora once at full dimension, then tries
    truncated dims [64, 128, 256, full] in order. Returns a decision at the
    first dim that produces abs(margin) ≥ _MARGIN_THRESHOLD.

    If all dims are ambiguous, returns a low-confidence result.

    Args:
        backend: EmbeddingBackend to use. Defaults to SentenceTransformerBackend.
                 Inject MockEmbeddingBackend in tests.
    """

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self._backend: EmbeddingBackend = backend or SentenceTransformerBackend("all-MiniLM-L6-v2")
        self._wf_proto_cache: np.ndarray | None = None
        self._sp_proto_cache: np.ndarray | None = None

    def _load_prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        """Embed prototype corpora (cached after first call)."""
        if self._wf_proto_cache is None:
            self._wf_proto_cache = self._backend.embed_batch(_WORKFLOW_PROTOTYPES)
            self._sp_proto_cache = self._backend.embed_batch(_SINGLE_PROTOTYPES)
        return self._wf_proto_cache, self._sp_proto_cache

    @staticmethod
    def _truncate(arr: np.ndarray, dims: int | None) -> np.ndarray:
        """Slice last axis to first `dims` columns and renormalize each row."""
        if dims is None or dims >= arr.shape[-1]:
            return arr
        sliced = arr[..., :dims].copy()
        if sliced.ndim == 1:
            n = float(np.linalg.norm(sliced))
            return sliced / n if n > 0 else sliced
        norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
        return np.where(norms > 0, sliced / norms, sliced)

    def detect(self, prompt: str) -> WorkflowDetection:
        full_prompt_emb = self._backend.embed(prompt)
        wf_full, sp_full = self._load_prototypes()

        decisive_margin: float | None = None
        decisive_dims: int | None = None

        for dims in _MRL_DIMS:
            p_emb = self._truncate(full_prompt_emb, dims)
            wf_embs = self._truncate(wf_full, dims)
            sp_embs = self._truncate(sp_full, dims)

            wf_sim = float(np.mean(wf_embs @ p_emb))
            sp_sim = float(np.mean(sp_embs @ p_emb))
            margin = wf_sim - sp_sim

            actual_dims = dims if (dims is not None and dims < full_prompt_emb.shape[0]) else int(full_prompt_emb.shape[0])

            if abs(margin) >= _MARGIN_THRESHOLD:
                decisive_margin = margin
                decisive_dims = actual_dims
                break

        if decisive_margin is not None:
            is_wf = decisive_margin > 0
            confidence = min(0.95, 0.5 + abs(decisive_margin) * 3.0)
            return WorkflowDetection(
                is_workflow=is_wf,
                confidence=round(confidence, 4),
                workflow_type="software_build" if is_wf else None,
                decision_source="mrl_embedding",
                reason=(
                    f"MRL embedding margin={decisive_margin:.4f} at dim={decisive_dims}. "
                    f"{'Workflow' if is_wf else 'Single-prompt'} prototype cluster closer."
                ),
                signals=["embedding_decisive"],
                semantic_margin=round(decisive_margin, 6),
                embedding_model=self._backend.model_name,
                embedding_dims_used=decisive_dims,
            )

        # Indecisive at all dims — return low-confidence result
        last_dims = int(full_prompt_emb.shape[0])
        p_emb = full_prompt_emb
        wf_sim = float(np.mean(wf_full @ p_emb))
        sp_sim = float(np.mean(sp_full @ p_emb))
        last_margin = wf_sim - sp_sim

        return WorkflowDetection(
            is_workflow=False,
            confidence=0.52,
            workflow_type=None,
            decision_source="mrl_embedding",
            reason=(
                f"Embedding margin too narrow at all MRL dims (max |margin|<{_MARGIN_THRESHOLD}). "
                "Passing to default fallback."
            ),
            signals=["embedding_ambiguous"],
            semantic_margin=round(last_margin, 6),
            embedding_model=self._backend.model_name,
            embedding_dims_used=last_dims,
        )


class WorkflowDetector:
    """
    Orchestrates rule guardrail → MRL embedding → project/tool-signal default.

    Usage:
        detector = WorkflowDetector()
        result = detector.detect("Build a React app with auth and tests.")
        if result.is_workflow:
            planner.plan(result, prompt)

    Args:
        embedding_backend: Inject a custom EmbeddingBackend (e.g. MockEmbeddingBackend
                           in tests). If None, SentenceTransformerBackend is used.
    """

    def __init__(self, embedding_backend: EmbeddingBackend | None = None) -> None:
        self._rule = RuleGuardrailDetector()
        self._embedding = MatryoshkaEmbeddingDetector(backend=embedding_backend)

    def detect(self, prompt: str) -> WorkflowDetection:
        # 1. Rule guardrail — fast, no model
        rule_result = self._rule.detect(prompt)
        if rule_result.confidence >= _RULE_CONFIDENCE_THRESHOLD:
            return rule_result

        # 2. Embedding detector — semantic cosine similarity
        emb_result = self._embedding.detect(prompt)
        if emb_result.confidence >= _RULE_CONFIDENCE_THRESHOLD:
            # A deterministic project/tool signal is stronger evidence of an
            # actionable workflow than a negative prototype match. This also
            # keeps decisions stable when a hash-backed test embedding changes
            # across interpreter processes.
            if not emb_result.is_workflow:
                signal_result = self._default_decision(prompt, emb_result)
                if signal_result.is_workflow:
                    return signal_result
            return emb_result

        # 3. Default: project/tool signal heuristics
        return self._default_decision(prompt, emb_result)

    def _default_decision(self, prompt: str, emb_result: WorkflowDetection) -> WorkflowDetection:
        import re
        token_set = set(re.sub(r"[^\w\s]", " ", prompt.lower()).split())
        matched = token_set & (_PROJECT_SIGNALS | _TOOL_SIGNALS)

        if matched:
            return WorkflowDetection(
                is_workflow=True,
                confidence=0.65,
                workflow_type="software_build",
                decision_source="default",
                reason=f"Project/tool signals detected after ambiguous embedding: {sorted(matched)}.",
                signals=sorted(matched),
                semantic_margin=emb_result.semantic_margin,
                embedding_model=emb_result.embedding_model,
                embedding_dims_used=emb_result.embedding_dims_used,
            )

        combined_signals = list(dict.fromkeys(emb_result.signals + list(token_set & (_WORKFLOW_VERBS | _SINGLE_VERBS))))
        return WorkflowDetection(
            is_workflow=False,
            confidence=0.60,
            workflow_type=None,
            decision_source="default",
            reason="No strong workflow signals found. Routing to v1 single-call path.",
            signals=combined_signals[:8],
            semantic_margin=emb_result.semantic_margin,
            embedding_model=emb_result.embedding_model,
            embedding_dims_used=emb_result.embedding_dims_used,
        )
