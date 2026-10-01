from unittest.mock import patch
from experiments.gemma4_methods import train


def test_control_probe_does_not_use_thinking_on_only_alignment_wrapper():
    official, unified, alignment = object(), object(), object()
    with patch('ctm.evals.local_model.Gemma4UnifiedTextProcessor',return_value=unified) as wrap, \
         patch('ctm.backends.gemma_thinking.attest_thinking',return_value={'probe':'verified'}) as attest, \
         patch.object(train,'alignment_processor',return_value=alignment) as align:
        processor,evidence=train.training_processors(official)
    wrap.assert_called_once_with(official)
    attest.assert_called_once_with(unified)
    align.assert_called_once_with(official)
    assert processor is alignment
    assert evidence=={'probe':'verified'}
