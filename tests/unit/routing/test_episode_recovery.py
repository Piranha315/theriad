from dataclasses import asdict

import pytest

from theriad.core.providers.interfaces import CompletionRequest, CompletionResponse, Message
from theriad.routing.episode_hooks import EpisodeRoutingSession, episode_routing
from theriad.routing.episode_recovery import study_cli
from theriad.routing.episode_study import Episode, EpisodeOutcome, ProductStudyBudget


@pytest.mark.asyncio
async def test_full_output_override_and_response_archive_are_recovery_only():
    from theriad.agents.model_telemetry import complete_with_telemetry
    records = []
    content = 'answer' * 10000
    class Provider:
        name = 'test'
        async def complete(self, request):
            assert request.max_tokens == 943718
            return CompletionResponse(model='test/model', content=content, reasoning='reasoning' * 10000,
                prompt_tokens=2, completion_tokens=20, latency_ms=1, provider='test',
                cost_usd=.01, finish_reason='stop', raw={'secret-header': 'DO NOT ARCHIVE'})
    request = CompletionRequest('test/model', [Message('user', 'hello')], max_tokens=220)
    ordinary = EpisodeRoutingSession(episode_id='ordinary', max_output_tokens=943718)
    assert ordinary.guard_request(request).max_tokens == 220
    recovery = EpisodeRoutingSession(episode_id='retry', policy_id='model', max_output_tokens=943718,
        use_full_output_limit=True, response_sink=records.append)
    with episode_routing(recovery):
        response = await complete_with_telemetry(Provider(), request)
    assert response.content == content
    assert records[0]['content'] == content
    assert records[0]['reasoning'] == 'reasoning' * 10000
    assert records[0]['request_max_tokens'] == 943718
    assert 'DO NOT ARCHIVE' not in str(records)


@pytest.mark.asyncio
async def test_local_circuit_rejection_is_recorded_as_no_dispatch():
    from theriad.agents.model_telemetry import complete_with_telemetry
    from theriad.runtime.circuit_breaker import CircuitOpenError
    class Provider:
        name = 'test'
        async def complete(self, request):
            raise CircuitOpenError('llm:test', 10)
    session = EpisodeRoutingSession(episode_id='retry')
    with episode_routing(session), pytest.raises(CircuitOpenError):
        await complete_with_telemetry(Provider(), CompletionRequest('test/model', []))
    terminal = session.attempts[-1]
    assert terminal['cost_usd'] == 0
    assert terminal['cost_basis'] == 'local_circuit_rejection_no_dispatch'
    assert terminal['request_dispatched'] is False


@pytest.mark.asyncio
async def test_recovery_collection_uses_parent_budget_and_never_replays(tmp_path):
    cli = study_cli()
    parent = tmp_path / 'parent'
    directory = parent / 'recovery'
    directory.mkdir(parents=True)
    budget = ProductStudyBudget(parent / 'budget')
    row = Episode('task', 'group', 'chat', 'train', 'hello', {}, {'kind': 'exact', 'answer': 'yes'})
    class Runtime:
        calls = 0
        def maximum_cost(self, row, policy):
            return .1
        async def __call__(self, row, policy_id, policy):
            self.calls += 1
            return EpisodeOutcome(row.episode_id, row.group_id, row.family, policy_id,
                row.content_hash, True, 'completed', .01, 1, 1)
    runtime = Runtime()
    for _ in range(2):
        await cli._collect_row(row, {'model': {'kind': 'model', 'model': 'test'}}, stage='development',
            runtime=runtime, directory=directory, contract={'profile': 'development_recovery'},
            allow_bounded_unknown=True, budget_override=budget)
    assert runtime.calls == 1
    assert budget.summary()['realized_cost_usd'] == .01
    assert not (directory / 'budget').exists()
