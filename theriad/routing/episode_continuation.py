"""Source-versioned, bounded engineering batches on the existing collector."""
import asyncio
from pathlib import Path
from dataclasses import replace
from .episode_recovery import study_cli
from .episode_protocol import verify_protocol
from .episode_engineering import require_paid_authorization,verify_admission
from .episode_runtime import EpisodeRuntime
from .episode_diagnostics import diagnostic_provider
from .episode_study import ProductStudyBudget,_collector_lock,digest,read_events,atomic_json

def validate_continuation(parent,directory):
    cli=study_cli();parent=Path(parent).resolve();directory=Path(directory).resolve()
    if directory==parent or not directory.is_relative_to(parent):raise ValueError('separate child batch directory required')
    plan=cli.read_json(directory/'plan.json');auth=cli.read_json(directory/'authorization.json');protocol=cli.read_json(directory/'protocol.json')
    verify_protocol(protocol)
    contract,_,controls=cli.validate_contract(parent,cli.DEVELOPMENT_SPLITS)
    require_paid_authorization(parent,contract,phase='development')
    deferral=cli.read_json(parent/'deferred-model-authorization.json')
    if auth.get('approved') is not True or not auth.get('user_authorization_reference') or auth.get('plan_hash')!=digest(plan) or auth.get('protocol_hash')!=protocol['manifest_hash'] or plan.get('parent_contract_hash')!=digest(contract) or plan.get('original_outcomes_hash')!=digest(read_events(parent/'development-outcomes.jsonl')) or plan.get('deferral_hash')!=digest(deferral):
        raise ValueError('continuation authorization or foundation changed')
    model='deepseek/deepseek-v4-flash-0731'
    if plan.get('active_policies')!={'baseline':{'kind':'current'},model:{'kind':'model','model':model}} or plan.get('deferred_models')!=['openai/gpt-5.6-luna']:
        raise ValueError('continuation must preserve the deferred Luna scope')
    original={r.episode_id:r for r in cli.load_rows(parent,cli.DEVELOPMENT_SPLITS)}
    entries=plan['rows'];selected=[]
    if len(entries)!=5 or len({x['episode_id'] for x in entries})!=5:raise ValueError('initial continuation is exactly five paired groups')
    for item in entries:
        row=original[item['episode_id']]
        if row.content_hash!=item['original_row_hash'] or row.split!='train':raise ValueError('first batch requires unchanged training group identities')
        selected.append(row)
    if {r.family for r in selected}!=set(cli.FAMILIES):raise ValueError('first batch must retain all five families')
    verify_admission(parent,selected,contract)
    if plan.get('max_seconds')!=600 or type(plan.get('max_output_tokens')) is not int or not 0<plan['max_output_tokens']<=943718:
        raise ValueError('unapproved task limits')
    rows=[replace(r,max_output_tokens=plan['max_output_tokens'],max_seconds=plan['max_seconds']) for r in selected]
    runtime=EpisodeRuntime(directory=directory,metadata=plan['metadata'],baseline_plans=controls,use_full_output_limit=True,archive_responses=True)
    maximum=sum(runtime.maximum_cost(r,p) for r in rows for p in plan['active_policies'].values())
    if maximum*1.2>plan['batch_cap_usd'] or plan['batch_cap_usd']>8.75:raise ValueError('batch exceeds frozen conservative bound')
    budget=ProductStudyBudget(parent/'budget')
    if budget.stage_caps['pilot']+budget.stage_caps['development']>14.46:raise ValueError('Phase A cap changed')
    return cli,plan,protocol,rows,controls,budget

async def run_continuation(parent,directory):
    directory=Path(directory);cli,plan,protocol,rows,controls,budget=validate_continuation(parent,directory)
    runtime=EpisodeRuntime(directory=directory,metadata=plan['metadata'],baseline_plans=controls,
        provider_factory=lambda provider,**kw:diagnostic_provider(directory,provider,plan['max_seconds']),
        use_full_output_limit=True,archive_responses=True)
    runtime.study_protocol=protocol;cli.preflight(directory)
    with _collector_lock(budget):
        if not read_events(directory/'development-outcomes.jsonl') and budget.summary()['committed_usd']+plan['batch_cap_usd']>14.46:
            raise ValueError('remaining Phase A allowance does not cover the entire initial batch')
        for row in rows:
            from theriad.runtime.circuit_breaker import get_circuit_registry
            for state in get_circuit_registry().snapshot('llm:openrouter'):
                if state['status']=='open':
                    await asyncio.sleep(min(31,state['retry_after_seconds']+.05))
            await cli._collect_row(row,plan['active_policies'],stage='development',runtime=runtime,directory=directory,
                contract={'profile':'postrepair_engineering_batch','plan_hash':digest(plan),'protocol_hash':protocol['manifest_hash']},
                budget_override=budget,allow_bounded_unknown=True)
            report_continuation(parent,directory)
    report_continuation(parent,directory)

def report_continuation(parent,directory):
    cli=study_cli();directory=Path(directory);plan=cli.read_json(directory/'plan.json')
    values=cli.outcomes(directory/'development-outcomes.jsonl');attempts=read_events(directory/'attempts.jsonl')
    start={e['attempt_id'] for e in attempts if e.get('event')=='attempt_started'};finish={e['attempt_id'] for e in attempts if e.get('event')=='attempt_finished'}
    terminals=[e for e in attempts if e.get('event')=='attempt_finished']
    observed=[{'episode_id':v.episode_id,'family':v.family,'policy_id':v.policy_id,'success':v.success,'status':v.status,'cost_usd':v.cost_usd,'latency_ms':v.latency_ms} for v in values]
    report={'status':'completed' if len(values)==10 and not start-finish else 'running','observations':observed,
        'known_new_cost_usd':sum(e['cost_usd'] for e in terminals if e.get('cost_usd') is not None),'unknown_attempt_costs':sum(e.get('cost_usd') is None for e in terminals),
        'pending_attempts':len(start-finish),'truncated_attempts':sum(bool(e.get('truncated')) for e in terminals),'deferred_models':plan['deferred_models'],
        'original_candidate_pool_complete':False,'fitting_permitted':False,'promotion_eligible':False,
        'parent_budget':ProductStudyBudget(Path(parent)/'budget').summary(),
        'note':'Five prospectively selected engineering training pairs on the repaired source. Old results and failure-selected diagnostics are not pooled. Luna and confirmatory study remain deferred.'}
    atomic_json(directory/'report.json',report)
    return report
