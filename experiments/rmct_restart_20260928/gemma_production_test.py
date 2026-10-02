import unittest
from experiments.rmct_restart_20260928.gemma_production_plan import build, REVISION


class ProductionPlanTests(unittest.TestCase):
    def test_native_fresh_recipe(self):
        result = build(repo='/deployment', python='/venv/bin/python', model='/models/'+REVISION,
            targets=['model.language_model.layers.0.self_attn.q_proj'], data='/data/pool',
            manifest='/data/manifest', one_bias_manifest='/data/one-bias.json',
            one_bias_manifest_sha256='c'*64, commit='a'*40, run_name='fresh-gemma',
            approval_reference='user-approval', validation_sha256='b'*64)
        args = result['argument_map']
        self.assertEqual(args['max_new_tokens'], 20480)
        self.assertEqual(args['local_rollout_gpus'], '1,2,3')
        self.assertEqual(args['local_device'], 'cuda:0')
        self.assertFalse(any(k.startswith('resume') for k in args))
        self.assertNotIn('local_phase_shared', args)
        self.assertNotIn('local_qwen35_rollout_parity_attestation', args)
        self.assertEqual(args['load_config'], {'n_datapoints': 16, 'attempt_offset': 0})
        self.assertEqual(args['batch_size'], 4)
        self.assertEqual(args['setting_config']['one_bias_manifest_sha256'], 'c'*64)
        self.assertEqual(result['controller_contract']['encountered_qids_per_validation'], 256)
        self.assertEqual(args['n_train_rollouts'], 96)
        self.assertEqual(args['n_ref_rollouts'], 96)
        self.assertEqual(args['lr'], .0001)
        self.assertFalse(result['ready_to_launch'])
        self.assertIn('--max-new-tokens', result['argv'])
