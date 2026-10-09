import pytest
from theriad.routing.episode_hooks import EpisodeRoutingSession, EpisodeRoutingError
from theriad.core.providers.interfaces import CompletionRequest

def test_pinning_overrides_fallback_and_rejects_wrong_backend():
    s=EpisodeRoutingSession('probe',model_metadata={'m':{'provider_only':'open-inference/fp8','expected_backend':'OpenInference'}})
    req=s.guard_request(CompletionRequest('m',[],extra={'provider':{'allow_fallbacks':True,'order':['relace']}}))
    assert req.extra['provider']=={'only':['open-inference/fp8'],'order':['open-inference/fp8'],'allow_fallbacks':False}
    s.verify_response_backend('m','OpenInference')
    for value in [None,'Relace']:
        with pytest.raises(EpisodeRoutingError):s.verify_response_backend('m',value)
