from copy import deepcopy
from datetime import datetime, timezone
import pytest

from theriad.core.providers.openrouter_capabilities import LUNA, SNAPSHOT_KEY, catalog_digest
from theriad.routing.episode_recovery import LUNA_RETRY_CELLS, recovery_splits, validate_luna_retry_override


def plan():
    catalog={'data':{'id':LUNA,'endpoints':[{'tag':'openai/flex','status':0,'max_completion_tokens':128000,'supported_parameters':['max_tokens']}]}}
    snapshot={'catalog':catalog,'catalog_digest':catalog_digest(catalog),'retrieved_at':datetime.now(timezone.utc).isoformat(),'endpoint_tag':'openai/flex'}
    return {'version':'luna-two-call-override-v1','model':LUNA,'preserve_historical_unknown_charges':True,
            'rows':[{'episode_id':k,'original_split':v} for k,v in LUNA_RETRY_CELLS.items()],
            'max_output_tokens':128000,'metadata':{LUNA:{'provider_only':'openai/flex',SNAPSHOT_KEY:snapshot}}}


def test_explicit_override_preserves_original_train_and_calibration_splits():
    p=plan();before=deepcopy(p)
    validate_luna_retry_override(p,{'retry_unreconciled_404s':True})
    assert p==before and recovery_splits(p)==['train','calibration']
    assert recovery_splits({'version':'old-deepseek-retry'})==['train']


@pytest.mark.parametrize('change',['consent','extra_row','wrong_row','split','model','history','endpoint'])
def test_override_cannot_expand_scope_or_discard_history(change):
    p=plan();auth={'retry_unreconciled_404s':True}
    if change=='consent':auth.clear()
    if change=='extra_row':p['rows'].append(dict(p['rows'][0]))
    if change=='wrong_row':p['rows'][0]['episode_id']='different'
    if change=='split':p['rows'][1]['original_split']='train'
    if change=='model':p['model']='different'
    if change=='history':p['preserve_historical_unknown_charges']=False
    if change=='endpoint':p['metadata'][LUNA]['provider_only']='openai'
    with pytest.raises(ValueError):validate_luna_retry_override(p,auth)
