"""Gemma fresh-start policy. Not a launcher or a historical continuation."""
MODEL = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'


def token_cap(dataset):
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError('A canonical dataset name is required')
    return 20480 if dataset in {'logiqa', 'hellaswag'} else 65536


def fresh_config(*, source_commit, approval_reference, validation_manifest_sha256):
    for name, value, length in [('source_commit', source_commit, 40),
                                ('validation_manifest_sha256', validation_manifest_sha256, 64)]:
        if len(value) != length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid {name}')
    if not approval_reference.strip():
        raise ValueError('Explicit new-run user approval reference required')
    return {
        'schema': 'gemma-rmct-fresh-thinking-v1',
        'model': {'repo_id': MODEL, 'revision': REVISION},
        'source_commit': source_commit, 'approval_reference': approval_reference,
        'initialization': {'weights': 'original_base', 'optimizer': 'fresh',
                           'resume_from': None, 'reuse_target_cache': False},
        'chat_template_kwargs': {'enable_thinking': True},
        'generation': {'caps_include_reasoning': True,
                       'dataset_caps': {'logiqa': 20480, 'hellaswag': 20480},
                       'other_dataset_cap': 65536,
                       'unknown_termination': 'fail_closed',
                       'length_termination': 'exclude_from_gradients_and_rate_estimates'},
        'validation': {'manifest_sha256': validation_manifest_sha256,
                       'question_ids': 200, 'native_prompts': 600,
                       'every_encountered_qids': 256, 'counts_no_update_batches': True, 'metric': 'TBSR',
                       'patience': 2, 'min_delta': 0, 'improvement': 'strict_decrease',
                       'diagnostics_select_or_stop': False, 'history': 'new'},
        'execution': {'nodes': 1, 'gpus': 4, 'trainer_gpus': 1,
                      'rollout_workers': 3, 'sampler': 'vllm',
                      'multiprocessing': 'spawn'},
    }
