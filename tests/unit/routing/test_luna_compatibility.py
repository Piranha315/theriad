from dataclasses import replace
import pytest

from theriad.core.providers.openrouter_capabilities import LUNA
from theriad.routing.episode_luna_compatibility import missing_luna_cells
from theriad.routing.episode_luna_compatibility import bounded_luna_schedule, require_luna_billing_clearance, require_luna_batch_budget
from theriad.routing.episode_study import Episode, EpisodeOutcome


def row(name, split="train"):
    return Episode(name, "group-"+name, "chat", split, "hello", {}, {"kind": "exact", "answer": "yes"})


def outcome(r, status="completed", success=False):
    return EpisodeOutcome(r.episode_id, r.group_id, r.family, LUNA, r.content_hash,
                          success, status, None if status == "execution_error" else .01, 1, 1)


def test_only_missing_cells_selected_and_original_splits_retained():
    rows = [row("train"), row("calibration", "calibration"), row("threshold", "threshold_selection"), row("observed-failure"), row("404")]
    failure = {"event": "attempt_finished", "model": LUNA, "episode_id": "404", "attempt_id": "a",
               "provider_status_code": 404, "cost_usd": None, "recorded_at": "2026-09-13T04:13:36Z"}
    result = missing_luna_cells(rows, [outcome(rows[3]), outcome(rows[4], "execution_error")], [failure])
    assert [(r["episode_id"], r["split"]) for r in result["missing"]] == [(r.episode_id, r.split) for r in rows[:3]]
    assert result["retained"][0]["episode_id"] == "observed-failure"
    assert result["billing_blocked"][0]["episode_id"] == "404"
    assert result["collection_allowed"] is False


@pytest.mark.parametrize("split", ["test", "pilot", "smoke"])
def test_later_or_other_splits_rejected(split):
    with pytest.raises(ValueError, match="development splits"):
        missing_luna_cells([row("outside", split)], [], [])


def test_pending_or_unfinalized_attempts_cannot_be_replayed():
    started = {"event": "attempt_started", "model": LUNA, "episode_id": "r", "attempt_id": "a"}
    with pytest.raises(ValueError, match="pending"):
        missing_luna_cells([row("r")], [], [started])
    with pytest.raises(ValueError, match="needs recovery"):
        missing_luna_cells([row("r")], [], [started, {**started, "event": "attempt_finished"}])


def test_duplicate_outcomes_or_rows_rejected():
    r = row("r")
    with pytest.raises(ValueError, match="duplicate Luna"):
        missing_luna_cells([r], [outcome(r), outcome(r)], [])
    with pytest.raises(ValueError, match="duplicate original"):
        missing_luna_cells([r, r], [], [])


def test_schedule_is_deterministic_split_preserving_and_bounded():
    rows = [row(str(i), split) for i, split in enumerate(["train"]*6+["calibration"]*3+["threshold_selection"]*2)]
    inventory = missing_luna_cells(rows, [], [])
    costs = {r.episode_id: .1 for r in rows}
    schedule = bounded_luna_schedule(inventory, costs, .61)
    reverse = {**inventory, "missing": list(reversed(inventory["missing"]))}
    assert schedule == bounded_luna_schedule(reverse, costs, .61)
    assert schedule["first_batch_fits"]
    assert [b["split"] for b in schedule["batches"]] == ["train", "train", "calibration", "threshold_selection"]
    assert len([c for b in schedule["batches"] for c in b["cells"]]) == 11
    assert all(b["cap_usd"] >= b["maximum_usd"]*1.2 for b in schedule["batches"])
    assert schedule["collection_allowed"] is False


@pytest.mark.parametrize("cost", [float('nan'), float('inf'), -1, True])
def test_invalid_cost_cannot_authorize_a_batch(cost):
    r=row("r");inventory=missing_luna_cells([r],[],[])
    with pytest.raises(ValueError):bounded_luna_schedule(inventory,{"r":cost},1)


def test_budget_gate_preserves_old_reservations_and_stage_cap():
    summary={"committed_usd": 14.0,"stages":{"development":{"remaining_usd": .2}}}
    with pytest.raises(ValueError):require_luna_batch_budget({"cap_usd": .21},summary)
    require_luna_batch_budget({"cap_usd": .2},summary)
    summary["committed_usd"]=14.4
    with pytest.raises(ValueError):require_luna_batch_budget({"cap_usd": .2},summary)


def test_billing_gate_rejects_unknown_or_unsupported_zero(tmp_path):
    r=row("404")
    event={"event":"attempt_finished","model":LUNA,"episode_id":"404","attempt_id":"a","provider_status_code":404,"cost_usd":None,"recorded_at":"2026-09-13T04:13:36Z"}
    old=outcome(r,"execution_error")
    inventory=missing_luna_cells([r],[old],[event])
    with pytest.raises(ValueError,match="unresolved"):require_luna_billing_clearance(inventory,[old],tmp_path)
    with pytest.raises(ValueError,match="provider-evidenced"):require_luna_billing_clearance(inventory,[replace(old,cost_usd=0)],tmp_path)


def test_billing_gate_binds_both_attempts_and_evidence_bytes(tmp_path):
    import hashlib
    from theriad.routing.episode_study import digest
    from dataclasses import asdict
    rows=[row("404a"),row("404b")];originals=[outcome(r,"execution_error") for r in rows]
    events=[{"event":"attempt_finished","model":LUNA,"episode_id":r.episode_id,"attempt_id":r.episode_id,"provider_status_code":404,"cost_usd":None,"recorded_at":"2026-09-13T04:13:36Z"} for r in rows]
    inventory=missing_luna_cells(rows,originals,events)
    evidence=b'OFFLINE FIXTURE: explicit provider billing confirmation'
    (tmp_path/'receipt.txt').write_bytes(evidence)
    amended=[]
    for old,event in zip(originals,events):
        proof={"original_outcome_hash":digest(asdict(old)),"attempt_terminal_hashes":{event['attempt_id']:digest(event)},"basis":"provider_explicit_billing_confirmation","provider_evidence_sha256":hashlib.sha256(evidence).hexdigest(),"provider_evidence_path":"receipt.txt","verified_cost_usd":0}
        amended.append(replace(old,cost_usd=0,evidence={"provider_billing_reconciliation":proof}))
    with pytest.raises(ValueError):require_luna_billing_clearance(inventory,[amended[0],originals[1]],tmp_path)
    require_luna_billing_clearance(inventory,amended,tmp_path)
    (tmp_path/'receipt.txt').write_bytes(b'altered')
    with pytest.raises(ValueError,match="absent or changed"):require_luna_billing_clearance(inventory,amended,tmp_path)


def test_unattempted_continuation_requires_unchanged_full_historical_reservations(tmp_path):
    from theriad.routing.episode_luna_compatibility import verify_retained_history
    from theriad.routing.episode_study import atomic_json, digest
    entry={'reservation_id':'old','actual_cost_usd':None,'maximum_cost_usd':1.23,'terminal_unknown_evidence_hash':'unresolved'}
    atomic_json(tmp_path/'budget'/'development.json',{'entries':[entry]})
    p={'historical_reservations':[{'stage':'development','reservation_id':'old','entry_hash':digest(entry)}]}
    auth={'collect_only_unattempted_with_history_reserved':True}
    verify_retained_history(tmp_path,p,auth)
    with pytest.raises(ValueError):verify_retained_history(tmp_path,p,{})
    for changed in ({**entry,'actual_cost_usd':0},{**entry,'maximum_cost_usd':.01},{**entry,'terminal_unknown_evidence_hash':None}):
        atomic_json(tmp_path/'budget'/'development.json',{'entries':[changed]})
        with pytest.raises(ValueError):verify_retained_history(tmp_path,p,auth)


@pytest.mark.asyncio
async def test_canonical_batch_is_billing_gated_idempotent_and_cannot_move_to_new_directory(tmp_path, monkeypatch):
    import hashlib
    from datetime import datetime, timezone
    from dataclasses import asdict
    from theriad.routing.episode_luna_compatibility import run_missing_luna_batch
    from theriad.routing.episode_recovery import study_cli
    from theriad.routing.episode_study import atomic_json, append_event, read_events, digest, ProductStudyBudget
    from theriad.core.providers.openrouter_capabilities import SNAPSHOT_KEY, catalog_digest
    cli=study_cli();parent=tmp_path/'parent';parent.mkdir();child=parent/'batch';child.mkdir()
    original=[row('missing'),row('404a'),row('404b')];old=[outcome(r,'execution_error') for r in original[1:]]
    contract={'fixture':'offline-only'}
    events=[]
    for r in original[1:]:
        event={'event':'attempt_finished','model':LUNA,'episode_id':r.episode_id,'attempt_id':r.episode_id,'provider_status_code':404,'cost_usd':None,'recorded_at':'2026-09-13T04:13:36Z'}
        events.append(event);append_event(parent/'attempts.jsonl',event)
    for value in old:append_event(parent/'development-outcomes.jsonl',{'event':'episode_completed','outcome':asdict(value)})
    inventory=missing_luna_cells(original,old,events)
    catalog={'data':{'id':LUNA,'endpoints':[{'tag':'openai/flex','status':0,'max_completion_tokens':128000,'supported_parameters':['max_tokens','tools','response_format']}]}}
    snap={'catalog':catalog,'catalog_digest':catalog_digest(catalog),'endpoint_tag':'openai/flex','retrieved_at':datetime.now(timezone.utc).isoformat()}
    batch=bounded_luna_schedule(inventory,{'missing':.1},1)['batches'][0]
    protocol={'manifest_hash':'fixture'}
    calls=[]
    class Runtime:
        def __init__(self,**kwargs):pass
        def maximum_cost(self,row,policy):return .1
        async def __call__(self,row,policy_id,policy):
            calls.append(row.episode_id)
            assert row.split=='train' and row.max_output_tokens==128000
            return EpisodeOutcome(row.episode_id,row.group_id,row.family,policy_id,row.content_hash,False,'completed',0,1,1)
    monkeypatch.setattr('theriad.routing.episode_protocol.verify_protocol',lambda *_a,**_k:None)
    monkeypatch.setattr('theriad.routing.episode_engineering.require_paid_authorization',lambda *_a,**_k:None)
    monkeypatch.setattr('theriad.routing.episode_engineering.verify_admission',lambda *_a,**_k:None)
    monkeypatch.setattr('theriad.routing.episode_runtime.EpisodeRuntime',Runtime)
    monkeypatch.setattr(cli,'validate_contract',lambda *_a:(contract,{},{}))
    monkeypatch.setattr(cli,'load_rows',lambda *_a:original)
    monkeypatch.setattr(cli,'preflight',lambda *_a:None)
    monkeypatch.setattr('theriad.routing.episode_recovery.study_cli',lambda:cli)
    atomic_json(parent/'budget-config.json',{'stage_caps_usd':{'pilot':.21,'development':14.25,'smoke':2.42,'test':0},'hard_cap_usd':16.88})
    def prepare(path):
        plan={'parent_contract_hash':digest(contract),'parent_outcomes_hash':digest(read_events(parent/'development-outcomes.jsonl')),'parent_attempts_hash':digest(read_events(parent/'attempts.jsonl')),'max_output_tokens':128000,'max_seconds':600,'metadata':{LUNA:{'provider_only':'openai/flex',SNAPSHOT_KEY:snap}},'batch_index':1,'batch':batch}
        atomic_json(path/'execution-plan.json',plan);atomic_json(path/'protocol.json',protocol)
        atomic_json(path/'authorization.json',{'approved':True,'user_authorization_reference':'OFFLINE TEST ONLY','plan_hash':digest(plan),'protocol_hash':'fixture'})
    prepare(child)
    with pytest.raises(ValueError,match='unresolved'):await run_missing_luna_batch(parent,child)
    assert not calls and not (parent/'luna-compatibility-claims.jsonl').exists()
    receipt=b'OFFLINE FIXTURE ONLY';(parent/'receipt.txt').write_bytes(receipt)
    for value,event in zip(old,events):
        proof={'original_outcome_hash':digest(asdict(value)),'attempt_terminal_hashes':{event['attempt_id']:digest(event)},'basis':'provider_explicit_billing_confirmation','provider_evidence_sha256':hashlib.sha256(receipt).hexdigest(),'provider_evidence_path':'receipt.txt','verified_cost_usd':0}
        amended=replace(value,cost_usd=0,evidence={'provider_billing_reconciliation':proof})
        append_event(parent/'development-outcomes.jsonl',{'event':'episode_cost_reconciled','original_outcome_hash':digest(asdict(value)),'outcome':asdict(amended)})
    prepare(child)
    await run_missing_luna_batch(parent,child);await run_missing_luna_batch(parent,child)
    assert calls==['missing']
    other=parent/'duplicate';other.mkdir();prepare(other)
    with pytest.raises(ValueError,match='already belongs'):await run_missing_luna_batch(parent,other)
    assert calls==['missing']


def test_no_dispatch_recovery_requires_complete_immutable_evidence(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import asdict
    from theriad.routing.episode_luna_compatibility import verified_no_dispatch_inventory
    from theriad.routing.episode_study import atomic_json, append_event, digest
    monkeypatch.setattr('theriad.routing.episode_protocol.verify_protocol', lambda *_a, **_k: None)
    r = row('recover', 'calibration'); source = tmp_path/'old'; source.mkdir()
    cell = {'episode_id':r.episode_id,'original_row_hash':r.content_hash}
    atomic_json(source/'execution-plan.json', {'batch':{'cells':[cell]}})
    atomic_json(source/'protocol.json', {})
    old = asdict(replace(outcome(r, 'execution_error'), cost_usd=0))
    append_event(source/'development-outcomes.jsonl', {'event':'episode_completed','outcome':old})
    start = {'event':'attempt_started','attempt_id':'a','episode_id':r.episode_id,'model':LUNA}
    terminal = {**start,'event':'attempt_finished','request_dispatched':False,'cost_usd':0}
    append_event(source/'attempts.jsonl',start); append_event(source/'attempts.jsonl',terminal)
    proof = {**cell,'source_batch':'old','split':'calibration','original_outcome_hash':digest(old),
             'model_terminal_hashes':[digest(terminal)]}
    def bind():
        proof['file_hashes'] = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() if (source/name).exists() else None
            for name in ['execution-plan.json','protocol.json','development-outcomes.jsonl','attempts.jsonl','requests.jsonl']}
    bind()
    inventory, owners = verified_no_dispatch_inventory(tmp_path,[r],[proof])
    assert inventory['missing'][0]['split']=='calibration' and owners[r.episode_id]['directory']==str(source)
    with pytest.raises(ValueError): verified_no_dispatch_inventory(tmp_path,[replace(r,split='test')],[proof])
    with pytest.raises(ValueError): verified_no_dispatch_inventory(tmp_path,[r],[proof,proof])
    append_event(source/'requests.jsonl',{'episode_id':r.episode_id})
    with pytest.raises(ValueError,match='changed'): verified_no_dispatch_inventory(tmp_path,[r],[proof])
    bind()
    with pytest.raises(ValueError,match='no-dispatch proof'): verified_no_dispatch_inventory(tmp_path,[r],[proof])
    (source/'requests.jsonl').unlink()
    for changed in ({**terminal,'cost_usd':None},{**terminal,'request_dispatched':True}):
        (source/'attempts.jsonl').unlink()
        append_event(source/'attempts.jsonl',start);append_event(source/'attempts.jsonl',changed)
        proof['model_terminal_hashes']=[digest(changed)];bind()
        with pytest.raises(ValueError,match='no-dispatch proof'):verified_no_dispatch_inventory(tmp_path,[r],[proof])
    (source/'attempts.jsonl').unlink();append_event(source/'attempts.jsonl',start);bind()
    with pytest.raises(ValueError,match='no-dispatch proof'):verified_no_dispatch_inventory(tmp_path,[r],[proof])


@pytest.mark.asyncio
async def test_cooldown_waits_until_ready_without_reset_or_dispatch(monkeypatch):
    from unittest.mock import Mock, AsyncMock
    from theriad.routing.episode_luna_compatibility import wait_for_openrouter_cooldown
    registry=Mock(); registry.snapshot.side_effect=[
        [{'status':'open','retry_after_seconds':61}],
        [{'status':'open','retry_after_seconds':31}],
        [{'status':'open','retry_after_seconds':1}],
        [{'status':'half_open','retry_after_seconds':0,'half_open_in_flight':False}]]
    sleep=AsyncMock()
    monkeypatch.setattr('theriad.runtime.circuit_breaker.get_circuit_registry',lambda:registry)
    monkeypatch.setattr('asyncio.sleep',sleep)
    await wait_for_openrouter_cooldown()
    assert [v.args[0] for v in sleep.call_args_list]==[30,30,1.05]
    registry.reset.assert_not_called()
    registry.snapshot.side_effect=None;registry.snapshot.return_value=[{'half_open_in_flight':True}]
    with pytest.raises(RuntimeError,match='in flight'):await wait_for_openrouter_cooldown()
