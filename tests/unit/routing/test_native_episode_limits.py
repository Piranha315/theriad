import pytest
from theriad.routing.episode_hooks import EpisodeRoutingSession,EpisodeRoutingError
from theriad.core.providers.interfaces import CompletionRequest,Message

def test_native_allowance_preserves_context_and_regular_caps():
    quote={'context_window':32768,'native_output_tokens':32768}
    request=CompletionRequest('m',[Message('user','x'*1000)],max_tokens=700)
    regular=EpisodeRoutingSession('regular',model_metadata={'m':quote},max_output_tokens=943718)
    assert regular.guard_request(request).max_tokens==700
    native=EpisodeRoutingSession('native',model_metadata={'m':quote},max_output_tokens=943718,use_full_output_limit=True)
    result=native.guard_request(request)
    assert 31000<result.max_tokens<32768
    assert result.max_tokens+1000<32768
    with pytest.raises(EpisodeRoutingError):
        native.guard_request(CompletionRequest('m',[Message('user','x'*32768)]))

@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_native_cap_fails_before_dispatch(value):
    session=EpisodeRoutingSession('native',model_metadata={'m':{'native_output_tokens':value}},use_full_output_limit=True)
    with pytest.raises(EpisodeRoutingError):session.guard_request(CompletionRequest('m',[]))
