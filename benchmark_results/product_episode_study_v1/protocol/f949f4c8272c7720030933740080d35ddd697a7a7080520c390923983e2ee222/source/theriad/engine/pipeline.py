"""
TheriadPipeline — the top-level orchestrator.

Responsibilities:
1. Run routing layers in order; stop at first high-confidence decision.
2. Resolve the chosen tier to a concrete model+provider via PolicyResolver.
3. Execute the chosen provider with retry + fallback.
4. Return the CompletionResponse (cost accounting is handled by callers).

Design decisions:
- Pipeline does NOT own providers; they are injected (testability, Principle 2).
- Layers are also injected in order. Order = [layer0, layer1, layer2, layer3?].
- PolicyResolver is optional: when omitted the pipeline uses the default registry
  with the 'balanced' policy, so it works out-of-the-box with no extra config.
- On routing failure (all layers abstain), fall back to fallback_tier.
- On execution failure after all retries, try fallback chain before raising.
- asyncio.wait_for provides timeout per attempt.
- Principle 1: never spend more on routing than executing. The pipeline enforces
  this by running cheap layers first and short-circuiting immediately.
"""

from __future__ import annotations

import asyncio

from theriad.core.logging.logger import get_logger
from theriad.core.providers.interfaces import (
    CompletionRequest,
    CompletionResponse,
    ProviderError,
    ProviderProtocol,
)
from theriad.core.routing.interfaces import RoutingDecision, RoutingLayer, RoutingResult

from .policies import ExecutionPolicy

logger = get_logger("engine.pipeline")


class TheriadPipeline:
    """
    Orchestrates routing layers → tier resolution → provider execution → fallback.

    Args:
        layers:           Routing layers in priority order.
        providers:        Map of provider_name → ProviderProtocol.
        fallback_tier:    Tier to use if all routing layers abstain.
        fallback_provider:Provider to use when no PolicyResolver is set.
        policy:           Retry/fallback/timeout policy.
        policy_resolver:  Optional tier→model resolver. If None, uses default registry.
        active_policy:    Policy name ("balanced", "aggressive_savings",
                          "quality_first", "free_moe_experimental").
    """

    def __init__(
        self,
        layers: list[RoutingLayer],
        providers: dict[str, ProviderProtocol],
        fallback_tier: str = "cheap_bulk",
        fallback_provider: str = "openrouter",
        policy: ExecutionPolicy | None = None,
        policy_resolver=None,  # PolicyResolver | None — avoid circular import
        active_policy: str = "balanced",
    ):
        self._layers = layers
        self._providers = providers
        self._fallback_tier = fallback_tier
        self._fallback_provider = fallback_provider
        self._policy = policy or ExecutionPolicy()
        self._policy_resolver = policy_resolver
        self._active_policy = active_policy

    def route(self, prompt: str) -> RoutingResult:
        """Run routing layers and return the best tier decision."""
        context: dict = {}
        for layer in self._layers:
            result = layer.route(prompt, context)
            if result is not None:
                logger.info(
                    "routing decided",
                    extra={
                        "layer": result.layer_name,
                        "tier": result.decision.tier,
                        "confidence": result.decision.confidence,
                    },
                )
                return result

        # All layers abstained — use fallback tier
        fallback_decision = RoutingDecision(
            tier=self._fallback_tier,
            confidence=0.0,
            reason="all_layers_abstained: using config fallback_tier",
        )
        logger.info("routing fallback", extra={"tier": self._fallback_tier})
        return RoutingResult(decision=fallback_decision, layer_name="fallback", features=context)

    def _resolve_tier(
        self,
        tier: str,
        confidence: float,
        message: str = "",
        task_type: str | None = None,
    ) -> tuple[str, str, list[tuple[str, str]]]:
        """
        Resolve a tier label to (provider, model, fallback_chain).

        Uses PolicyResolver when available; falls back to default registry otherwise.
        ``task_type`` comes from RoutingDecision.signals and lets the resolver
        score quality on the relevant capability axis; None is safe and
        reproduces the previous behaviour.
        Returns: (provider_name, model_slug, [(provider, model), …])
        """
        if self._policy_resolver is not None:
            resolution = self._policy_resolver.resolve(
                tier,
                self._active_policy,
                confidence,
                message=message,
                task_type=task_type,
            )
            return (
                resolution.provider,
                resolution.model,
                resolution.fallback_tuples(),
            )

        # No resolver: use default registry with balanced policy
        from theriad.models.registry import DEFAULT_BASELINE_MODEL, DEFAULT_REGISTRY

        candidate = DEFAULT_REGISTRY.primary_for_tier(tier)
        if candidate is None:
            # Ultimate fallback — should not happen with valid tier names.
            return (self._fallback_provider, DEFAULT_BASELINE_MODEL, [])
        fallback_candidates = DEFAULT_REGISTRY.fallback_chain_for_tier(tier)
        fallback_chain = [(c.provider, c.model) for c in fallback_candidates]
        return (candidate.provider, candidate.model, fallback_chain)

    async def execute(
        self,
        messages: list,
        model: str | None = None,
        provider_name: str | None = None,
        prompt_for_routing: str | None = None,
    ) -> CompletionResponse:
        """
        Full pipeline: route → resolve tier → execute with retry → fallback chain.

        Args:
            messages:           List of Message objects.
            model:              Override model slug (bypasses tier resolution if provided).
            provider_name:      Override provider name.
            prompt_for_routing: Text used for routing. Defaults to last user message.
        """
        if prompt_for_routing is None:
            user_msgs = [m for m in messages if m.role == "user"]
            # .text ignores image blocks — routing classifies prose, not pixels.
            prompt_for_routing = user_msgs[-1].text if user_msgs else ""

        if model is None:
            routing_result = self.route(prompt_for_routing)
            tier = routing_result.decision.tier
            confidence = routing_result.decision.confidence
            resolved_provider, model, tier_fallback_chain = self._resolve_tier(
                tier,
                confidence,
                prompt_for_routing,
                task_type=routing_result.decision.signals.get("task_type"),
            )
            chosen_provider = provider_name or resolved_provider
            # Build execution fallback chain from tier config (overrides static policy chain)
            execution_fallback = tier_fallback_chain
        else:
            chosen_provider = provider_name or self._fallback_provider
            execution_fallback = []

        from theriad.storage.prompt_cache import prompt_cache_extra

        request = CompletionRequest(
            model=model,
            messages=messages,
            extra=prompt_cache_extra(chosen_provider),
        )

        # Primary execution with retries
        try:
            return await self._execute_with_retry(chosen_provider, request)
        except ProviderError as primary_exc:
            # Non-retryable 4xx (except 429): skip fallback chain entirely
            if (
                primary_exc.status_code
                and 400 <= primary_exc.status_code < 500
                and primary_exc.status_code != 429
            ):
                raise
            logger.warning(
                "primary execution failed",
                extra={"provider": chosen_provider, "error": str(primary_exc)},
            )
        except Exception as primary_exc:
            logger.warning(
                "primary execution failed",
                extra={"provider": chosen_provider, "error": str(primary_exc)},
            )

        # Tier-based fallback chain first, then static policy fallback
        combined_fallback = execution_fallback + list(self._policy.fallback.chain)
        for fb_provider, fb_model in combined_fallback:
            fallback_req = CompletionRequest(
                model=fb_model,
                messages=messages,
                extra=prompt_cache_extra(fb_provider),
            )
            try:
                logger.info("trying fallback", extra={"provider": fb_provider, "model": fb_model})
                return await self._execute_with_retry(fb_provider, fallback_req)
            except Exception as exc:
                logger.warning(
                    "fallback failed", extra={"provider": fb_provider, "error": str(exc)}
                )

        raise ProviderError(chosen_provider, "all providers exhausted (primary + fallback chain)")

    async def _execute_with_retry(
        self, provider_name: str, request: CompletionRequest
    ) -> CompletionResponse:
        policy = self._policy
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ProviderError(provider_name, f"provider '{provider_name}' not configured")

        last_exc: Exception | None = None
        from theriad.runtime.circuit_breaker import get_circuit_registry, model_circuit_key

        model_key = model_circuit_key(provider_name, request.model)
        for attempt in range(policy.retry.max_attempts):
            try:
                response = await asyncio.wait_for(
                    provider.complete(request),
                    timeout=policy.timeout_seconds,
                )
                get_circuit_registry().record_success(model_key)
                return response
            except asyncio.TimeoutError:
                last_exc = ProviderError(provider_name, f"timeout on attempt {attempt + 1}")
                get_circuit_registry().record_failure(model_key, last_exc)
                logger.warning("attempt timeout", extra={"attempt": attempt + 1})
            except ProviderError as exc:
                last_exc = exc
                transient = exc.status_code is None or exc.status_code in {408, 409, 425, 429} or (
                    isinstance(exc.status_code, int) and exc.status_code >= 500
                )
                get_circuit_registry().record_failure(model_key, exc, transient=transient)
                # 4xx errors (except 429) are non-retryable
                if exc.status_code and 400 <= exc.status_code < 500 and exc.status_code != 429:
                    logger.error("non-retryable error", extra={"status": exc.status_code})
                    raise
                logger.warning("retryable error", extra={"attempt": attempt + 1, "error": str(exc)})

            if attempt < policy.retry.max_attempts - 1:
                wait = min(
                    policy.retry.backoff_base_seconds * (2**attempt),
                    policy.retry.backoff_cap_seconds,
                )
                await asyncio.sleep(wait)

        raise last_exc or ProviderError(provider_name, "max retries exceeded")
