"""Versioned development retries using the canonical runtime and parent budget.

Original outcomes are never repaired in place. This produces diagnostic trials,
with full local response archives, and cannot fit or promote a router.
"""
from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

from .episode_study import ProductStudyBudget, _collector_lock, atomic_json, digest, read_events
from .episode_protocol import verify_protocol
from .episode_engineering import require_paid_authorization, verify_admission
from .episode_runtime import EpisodeRuntime

LUNA_RETRY_CELLS = {'gsm8k:train/2997': 'train', 'gsm8k:train/314': 'calibration'}


def recovery_splits(plan):
    return ['train', 'calibration'] if plan.get('version') == 'luna-two-call-override-v1' else ['train']


def validate_luna_retry_override(plan, authorization):
    """The user's duplicate-risk exception is limited to these two cells."""
    if (plan.get('model') != 'openai/gpt-5.6-luna'
            or authorization.get('retry_unreconciled_404s') is not True
            or plan.get('preserve_historical_unknown_charges') is not True
            or len(plan.get('rows', [])) != 2
            or {r.get('episode_id'): r.get('original_split') for r in plan['rows']} != LUNA_RETRY_CELLS
            or plan.get('max_output_tokens') != 128000
            or set(plan.get('metadata', {})) != {'openai/gpt-5.6-luna'}):
        raise ValueError('Luna override permits only the two named original cells, with historical charges retained')
    from theriad.core.providers.openrouter_capabilities import compatible_payload, SNAPSHOT_KEY
    quote = plan['metadata'][plan['model']]
    snapshot = quote.get(SNAPSHOT_KEY)
    if quote.get('provider_only') != 'openai/flex' or not isinstance(snapshot, dict) or snapshot.get('endpoint_tag') != 'openai/flex':
        raise ValueError('Luna override requires the reviewed exact OpenAI/flex endpoint')
    compatible_payload({'model': plan['model'], 'messages': [], 'temperature': 0,
                        'max_tokens': 128000}, {SNAPSHOT_KEY: snapshot})


def study_cli():
    source = Path(__file__).resolve().parents[2] / 'scripts/run_router_episode_study.py'
    spec = importlib.util.spec_from_file_location('recovery_study_cli', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_recovery(parent, directory):
    cli = study_cli()
    parent, directory = Path(parent).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(parent) or directory == parent:
        raise ValueError('recovery must have a separate directory under its parent study')
    plan = cli.read_json(directory / 'plan.json')
    authorization = cli.read_json(directory / 'authorization.json')
    protocol = cli.read_json(directory / 'protocol.json')
    verify_protocol(protocol)
    splits = recovery_splits(plan)
    luna_override = plan.get('version') == 'luna-two-call-override-v1'
    contract, _, controls = cli.validate_contract(parent, splits)
    require_paid_authorization(parent, contract, phase='development')
    if (plan.get('parent_contract_hash') != digest(contract)
            or plan.get('parent_outcomes_hash') != digest(read_events(parent / 'development-outcomes.jsonl'))
            or authorization.get('approved') is not True
            or authorization.get('plan_hash') != digest(plan)
            or authorization.get('protocol_hash') != protocol['manifest_hash']
            or not authorization.get('user_authorization_reference')):
        raise ValueError('recovery approval, source or original observations changed')
    if luna_override:
        validate_luna_retry_override(plan, authorization)
    if ((not luna_override and plan.get('model') != 'deepseek/deepseek-v4-flash-0731')
            or type(plan.get('max_output_tokens')) is not int or plan['max_output_tokens'] <= 0
            or type(plan.get('max_seconds')) is not int or not 1 <= plan['max_seconds'] <= 600
            or type(plan.get('retry_cap_usd')) not in (int, float) or not 0 < plan['retry_cap_usd'] <= 4
            or not 1 <= len(plan.get('rows', [])) <= 5):
        raise ValueError('recovery exceeds the bounded DeepSeek development retry scope')
    model = plan['model']
    quote = plan['metadata'][model]
    if plan['max_output_tokens'] != plan['published_max_completion_tokens']:
        raise ValueError('recovery must bind the published completion limit')
    original = {r.episode_id: r for r in cli.load_rows(parent, splits)}
    observed = {(r.episode_id, r.policy_id): r for r in cli.outcomes(parent / 'development-outcomes.jsonl')}
    selected, seen = [], set()
    for item in plan['rows']:
        key = item['episode_id']
        row = original.get(key)
        before = observed.get((key, model))
        if (key in seen or row is None or before is None or before.success is True
                or row.content_hash != item['original_row_hash']
                or digest(asdict(before)) != item['original_outcome_hash']):
            raise ValueError('recovery requires unique failed original training observations')
        seen.add(key)
        if luna_override and row.split != LUNA_RETRY_CELLS[key]:
            raise ValueError('Luna retry cannot move an original exposure split')
        selected.append(row)
    verify_admission(parent, selected, contract)
    retry_rows = [replace(r, max_output_tokens=plan['max_output_tokens'], max_seconds=plan['max_seconds']) for r in selected]
    if any(r.max_input_tokens + r.max_output_tokens > quote['context_window'] for r in retry_rows):
        raise ValueError('recovery exceeds the frozen model context')
    runtime = EpisodeRuntime(directory=directory, metadata=plan['metadata'], baseline_plans=controls,
                             use_full_output_limit=True, archive_responses=True)
    maximum = sum(runtime.maximum_cost(r, {'kind': 'model', 'model': model}) for r in retry_rows)
    if maximum * 1.2 > plan['retry_cap_usd'] + 1e-12:
        raise ValueError('recovery worst-case price plus contingency exceeds its cap')
    budget = ProductStudyBudget(parent / 'budget')
    caps = budget.stage_caps
    if caps['pilot'] + caps['development'] > 14.46 + 1e-12:
        raise ValueError('original Phase A cap changed')
    return cli, plan, protocol, retry_rows, controls, budget


async def run_recovery(parent, directory):
    cli, plan, protocol, rows, controls, budget = validate_recovery(parent, directory)
    from theriad.core.providers.factory import build_provider
    from theriad.runtime.circuit_breaker import get_circuit_registry
    # Same product provider factory/circuit breaker; longer timeout only for
    # these explicitly versioned recovery trials, never production settings.
    def provider_factory(provider, **kwargs):
        if plan.get('archive_requests'):
            from .episode_diagnostics import diagnostic_provider
            return diagnostic_provider(directory, provider, plan['max_seconds'])
        return build_provider(provider, timeout=plan['max_seconds'])
    runtime = EpisodeRuntime(directory=directory, metadata=plan['metadata'], baseline_plans=controls,
        provider_factory=provider_factory, use_full_output_limit=True, archive_responses=True)
    runtime.study_protocol = protocol
    cli.preflight(directory)
    identity_contract = {'profile': 'development_recovery', 'plan_hash': digest(plan),
                         'protocol_hash': protocol['manifest_hash']}
    policies = {plan['model']: {'kind': 'model', 'model': plan['model']}}
    with _collector_lock(budget):
        if plan.get('version') == 'luna-two-call-override-v1' and not read_events(directory / 'development-outcomes.jsonl'):
            available = min(14.46 - budget.summary()['committed_usd'],
                            budget.summary()['stages']['development']['remaining_usd'])
            if plan['retry_cap_usd'] > available:
                raise ValueError('Both Luna retries must fit remaining authorization with old reservations retained')
        for row in rows:
            # Preserve provider outage protection; wait out an actual cooldown
            # instead of resetting the breaker or submitting doomed requests.
            for state in get_circuit_registry().snapshot('llm:openrouter'):
                if state['status'] == 'open':
                    await asyncio.sleep(min(31, state['retry_after_seconds'] + .05))
            await cli._collect_row(row, policies, stage='development', runtime=runtime,
                directory=directory, contract=identity_contract, allow_bounded_unknown=True,
                budget_override=budget)
            recovery_report(parent, directory)
    recovery_report(parent, directory)


def recovery_report(parent, directory):
    cli = study_cli()
    plan = cli.read_json(directory / 'plan.json')
    observed = cli.outcomes(directory / 'development-outcomes.jsonl')
    terminal = [e for e in read_events(directory / 'attempts.jsonl') if e.get('event') == 'attempt_finished']
    originals = {r.episode_id: r for r in cli.load_rows(parent, recovery_splits(plan))}
    report = {'status': 'completed' if len(observed) == len(plan['rows']) else 'running',
        'diagnostic_only': True, 'promotion_eligible': False, 'plan_hash': digest(plan),
        'outcomes': [asdict(r) for r in observed],
        'known_retry_cost_usd': sum(e['cost_usd'] for e in terminal if e.get('cost_usd') is not None),
        'unknown_retry_charges': sum(e.get('cost_usd') is None for e in terminal),
        'truncated_attempts': sum(bool(e.get('truncated')) for e in terminal),
        'parent_budget': ProductStudyBudget(parent / 'budget').summary(),
        'note': 'Failure-selected retries with changed output/time limits; not held-out router evidence. Original grades and billing holes remain untouched.'}
    atomic_json(directory / 'report.json', report)
    lines = ['# Development retry review', '', report['note'], '',
             'Full per-call content and reasoning: [responses.jsonl](responses.jsonl).', '']
    for value in observed:
        lines.extend([f'## {value.episode_id}', '', f'Status: {value.status}; original grader success: {value.success}; cost: {value.cost_usd}',
                      '', '### Task', '', originals[value.episode_id].prompt, '', '### Saved final response', '',
                      value.evidence.get('output') or '(No final response; inspect the per-call archive and task trajectory.)', ''])
    (directory / 'REVIEW.md').write_text('\n'.join(lines))
