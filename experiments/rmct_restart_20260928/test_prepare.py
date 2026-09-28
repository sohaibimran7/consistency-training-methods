"""CPU-only preparation checks; do not certify deployed runtime behavior."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('restart_prepare', HERE/'prepare.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.baseline=json.loads((HERE/'fixtures/original_recipe.json').read_text())
        self.kw=dict(repo='/canonical/repo',python='/canonical/venv/bin/python',
                     commit='a'*40,run_name='clean-first',data='/data/pool',
                     manifest='/data/manifest',attestation='/new/preflight.json')

    def test_fresh_only(self):
        import hashlib
        self.assertEqual(hashlib.sha256((HERE/'fixtures/original_recipe.json').read_bytes()).hexdigest(),module.BASELINE_SHA)
        before=json.dumps(self.baseline,sort_keys=True)
        plan=module.build(self.baseline,**self.kw)
        self.assertFalse(plan['ready_to_launch'])
        self.assertFalse(any(s.startswith('--resume') for s in plan['argv']))
        self.assertEqual(plan['initial_optimizer_step'],0)
        self.assertEqual(plan['first_segment_end'],16)
        self.assertEqual(plan['first_validation_step'],64)
        self.assertEqual(before,json.dumps(self.baseline,sort_keys=True))

    def test_policy_and_paths(self):
        plan=module.build(self.baseline,**self.kw)
        argv=plan['argv']
        self.assertEqual(argv[:2],['/canonical/venv/bin/python','/canonical/repo/scripts/train_rlct.py'])
        self.assertEqual(argv[argv.index('--max-new-tokens')+1],'20480')
        self.assertEqual(argv[argv.index('--local-qwen35-rollout-parity-attestation')+1],'/new/preflight.json')
        self.assertEqual(plan['validation']['history'],'new-campaign-only')
        self.assertEqual(plan['validation']['patience'],2)

    def test_reject_unpinned_commit(self):
        self.kw['commit']='main'
        with self.assertRaises(ValueError): module.build(self.baseline,**self.kw)

    def test_reject_relative_deployment(self):
        self.kw['repo']='old-worktree'
        with self.assertRaises(ValueError): module.build(self.baseline,**self.kw)


if __name__=='__main__': unittest.main()
