"""Bounded input-path diagnostics on the canonical episode collector and budget."""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import httpx
from .episode_hooks import EpisodeRoutingSession, JsonlAttemptSink, current_episode, episode_routing, EpisodeRoutingError
from .episode_study import EpisodeOutcome, ProductStudyBudget, _collector_lock, append_event, digest
from .episode_protocol import verify_protocol

class RequestArchiveTransport(httpx.AsyncBaseTransport):
    """Archive synthetic/development JSON bodies at the actual HTTP boundary.

    Headers, cookies, URLs and credentials are deliberately never archived.
    Only explicitly enabled diagnostics attach this transport.
    """
    def __init__(self, path, inner=None):
        self.path = Path(path)
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport()
    async def handle_async_request(self, request):
        body = await request.aread()
        payload = json.loads(body)
        session = current_episode()
        starts = [e for e in session.events if e.get('event') == 'attempt_started'] if session else []
        append_event(self.path, {'episode_id': session.episode_id if session else None,
            'policy_id': session.policy_id if session else None,
            'attempt_id': starts[-1]['attempt_id'] if starts else None,
            'body_sha256': hashlib.sha256(body).hexdigest(), 'body': payload})
        return await self.inner.handle_async_request(request)
    async def aclose(self):
        await self.inner.aclose()

def diagnostic_provider(directory, provider, timeout):
    from theriad.core.providers.factory import build_provider
    result = build_provider(provider, timeout=timeout)
    if getattr(result, 'name', None) != 'openrouter':
        raise ValueError('HTTP archive only supports this approved OpenRouter diagnostic')
    result._transport = RequestArchiveTransport(Path(directory)/'requests.jsonl', getattr(result, '_transport', None))
    return result

class InputProbeRuntime:
    def __init__(self, directory, metadata, protocol):
        self.directory, self.metadata, self.study_protocol = Path(directory), metadata, protocol
    def maximum_cost(self, row, policy):
        q = self.metadata[policy['endpoint']]
        return (row.max_input_tokens*q['reservation_input_per_1m'] + row.max_output_tokens*q['reservation_output_per_1m']) / 1e6
    async def __call__(self, row, policy_id, policy):
        from theriad.core.providers.interfaces import CompletionRequest, Message
        from theriad.agents.model_telemetry import complete_with_telemetry
        model=policy['model']; quote=self.metadata[policy['endpoint']]
        session=EpisodeRoutingSession(row.episode_id, policy_id=policy_id, model_metadata={model:quote},
            max_output_tokens=row.max_output_tokens, max_input_tokens=row.max_input_tokens, max_model_calls=1,
            sink=JsonlAttemptSink(self.directory/'attempts.jsonl'),
            response_sink=lambda e: append_event(self.directory/'responses.jsonl',e))
        started=time.perf_counter(); response=None; error=None
        with episode_routing(session):
            try:
                provider=diagnostic_provider(self.directory,'openrouter',row.max_seconds)
                response=await complete_with_telemetry(provider,CompletionRequest(model,
                    [Message('system',row.fixture['system']),Message('user',row.prompt)],
                    max_tokens=row.max_output_tokens,extra=row.fixture.get('extra',{})),role='input_integrity')
            except Exception as exc:
                error=type(exc).__name__
        success=False
        if response:
            try: success=json.loads(response.content)==row.grader['answer'] and response.finish_reason=='stop'
            except (ValueError,TypeError): pass
        accounting=session.summary()
        return EpisodeOutcome(row.episode_id,row.group_id,row.family,policy_id,row.content_hash,success,
            'provider_failed' if error else 'completed',accounting['total_cost_usd'],
            (time.perf_counter()-started)*1000,len(session.attempts),
            evidence={'output': response.content if response else None,'error':error,'accounting':accounting,
                      'synthetic_diagnostic':True,'promotion_eligible':False})

async def run_input_probes(parent, directory):
    from .episode_recovery import study_cli
    from .episode_study import Episode
    from .episode_engineering import require_paid_authorization
    cli=study_cli(); parent=Path(parent); directory=Path(directory)
    plan=cli.read_json(directory/'plan.json'); auth=cli.read_json(directory/'authorization.json')
    protocol=cli.read_json(directory/'protocol.json');verify_protocol(protocol)
    contract=cli.read_json(parent/'contract.json');require_paid_authorization(parent,contract,phase='development')
    if not directory.resolve().is_relative_to(parent.resolve()) or auth.get('approved') is not True or auth.get('plan_hash')!=digest(plan) or auth.get('protocol_hash')!=protocol['manifest_hash'] or not auth.get('user_authorization_reference') or plan.get('parent_contract_hash')!=digest(contract):
        raise ValueError('probe authorization/provenance mismatch')
    rows=[Episode(**r) for r in plan['rows']]
    if len(rows)!=2 or len(plan['policies'])!=2 or any(r.split!='train' or r.max_calls!=1 or r.max_seconds>180 for r in rows):
        raise ValueError('probe exceeds the four-completion scope')
    runtime=InputProbeRuntime(directory,plan['metadata'],protocol)
    maximum=sum(runtime.maximum_cost(r,p) for r in rows for p in plan['policies'].values())
    if maximum*1.2>plan['cap_usd'] or plan['cap_usd']>0.5:
        raise ValueError('probe exceeds its half-dollar bound')
    budget=ProductStudyBudget(parent/'budget')
    if budget.stage_caps['development']+budget.stage_caps['pilot']>14.46:
        raise ValueError('original Phase A cap changed')
    with _collector_lock(budget):
        for row in rows:
            await cli._collect_row(row,plan['policies'],stage='development',runtime=runtime,directory=directory,
                contract={'profile':'input_integrity_diagnostic','plan_hash':digest(plan),'protocol_hash':protocol['manifest_hash']},budget_override=budget)
    return cli.outcomes(directory/'development-outcomes.jsonl')
