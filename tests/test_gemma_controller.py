import json
import types
import pytest
from experiments.rmct_restart_20260928.gemma_controller import command
from experiments.rmct_restart_20260928.qwen_progress import progress,next_slice
from experiments.rmct_restart_20260928.gemma_production_plan import build,REVISION
from experiments.gemma4_methods.validation import saved_response


def plan():
    return build(repo='/source',python='/venv/bin/python',model='/model/'+REVISION,
        targets=['model.language_model.layers.0.self_attn.q_proj'],data='/data/pool',manifest='/data/manifest',
        commit='a'*40,run_name='fresh',approval_reference='explicit',validation_sha256='b'*64)


def test_rmct_native_slice_uses_absolute_attempt_cursor_not_segment_offset():
    before=progress(70,62)
    selected=next_slice(before,64)
    parent={'progress':before,'checkpoint':'/checkpoint','files':{}}
    argv=command(plan(),before,selected,parent)
    load=json.loads(argv[argv.index('--load-config')+1])
    assert load=={'n_datapoints':4,'segment_index':4,'batch_offset':70}
    assert argv[argv.index('--n-datapoints')+1]=='4'
    assert argv[argv.index('--resume-from')+1]=='file:///checkpoint'


def test_fresh_has_no_parent_and_never_changes_rollout_recipe():
    p=plan();argv=command(p,progress(0,0),next_slice(progress(0,0),64),None)
    assert '--resume-from' not in argv
    for key in ('n-ref-rollouts','n-train-rollouts','n-consistency-rollouts'):
        assert argv[argv.index('--'+key)+1]=='96'
    with pytest.raises(ValueError,match='Missing'):
        command(p,progress(16,12),next_slice(progress(16,12),64),None)


def test_validation_retains_length_outputs_and_rejects_unknown_or_missing_eos():
    processor=types.SimpleNamespace(decode=lambda ids,**kwargs:'raw')
    request={'sample_id':'id'}
    assert saved_response(request,types.SimpleNamespace(token_ids=[1,2],finish_reason='length'),processor,[99],2)['finish_reason']=='length'
    for tokens,reason in (([1,2],'stop'),([1,99],'unknown'),([True,99],'stop')):
        with pytest.raises(ValueError):
            saved_response(request,types.SimpleNamespace(token_ids=tokens,finish_reason=reason),processor,[99],2)
