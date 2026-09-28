"""Identity aliases do not waive checkpoint ownership of model and tokenizer."""
import json

import pytest

from ctm.evals.local_model import local_checkpoint_model


@pytest.mark.parametrize('requested', ['alias', 'different', 'missing'])
def test_checkpoint_model_uses_actual_snapshot_identity(tmp_path, monkeypatch, requested):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(snapshot, target_is_directory=True)
    (tmp_path / 'different').mkdir()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'manifest.json').write_text(json.dumps({
        'backend': 'local', 'model': str(snapshot), 'lora': True,
    }))
    (checkpoint / 'adapter_config.json').write_text('{}')
    captured = []
    monkeypatch.setattr('inspect_ai.model.get_model',
                        lambda name, **kwargs: captured.append(name) or name)
    if requested == 'alias':
        result = local_checkpoint_model(checkpoint, base_model=str(alias),
                                        model_args={'provider': 'vllm'})
        assert result == f'vllm/{snapshot}:{checkpoint}'
        assert captured == [result]
    else:
        with pytest.raises(ValueError, match='base model mismatch'):
            local_checkpoint_model(checkpoint, base_model=str(tmp_path / requested),
                                   model_args={'provider': 'vllm'})
        assert captured == []
