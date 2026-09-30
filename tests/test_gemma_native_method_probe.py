import unittest
from pathlib import Path
from experiments.gemma4_methods.native_method_probe import same_state
import json
import sys
import tempfile
from unittest.mock import patch


class NativeProbeTests(unittest.TestCase):
    def test_native_consumer_rejects_model_or_dependency_drift(self):
        from experiments.gemma4_methods import native_method_probe as p
        from experiments.gemma4_methods.selection_adapter import file_identity
        import types
        with tempfile.TemporaryDirectory() as folder:
            model=Path(folder)/p.train.REVISION
            model.mkdir()
            weights=model/'model.safetensors'
            weights.write_text('publisher-fixture')
            root=Path(p.__file__).resolve().parents[2]
            receipt=Path(folder)/'cpu.json'
            receipt.write_text(json.dumps({'schema':'rmct-restart-cpu-v1','status':'cpu_checks_passed',
                'source_commit':'source','source_root':str(root),'python':sys.executable,
                'sources':[],'model_files':[file_identity(weights)],'dependencies':{'torch':'pinned'}}))
            args=types.SimpleNamespace(repository=root,commit='source',model=str(model),cpu_receipt=receipt)
            with patch.object(p,'check_source',return_value=root), \
                 patch.dict('os.environ',{'SLURM_JOB_ID':'123'}), \
                 patch.object(p.importlib.metadata,'version',return_value='pinned'):
                p.verify_context(args)
                weights.write_text('changed-after-native-probe')
                with self.assertRaisesRegex(ValueError,'model bytes changed'):
                    p.verify_context(args)
            weights.write_text('publisher-fixture')
            with patch.object(p,'check_source',return_value=root), \
                 patch.dict('os.environ',{'SLURM_JOB_ID':'123'}), \
                 patch.object(p.importlib.metadata,'version',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'dependency changed'):
                    p.verify_context(args)

    def test_optimizer_tensor_values_not_flags_determine_readback(self):
        import torch
        state={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor([.1,.2])}},'param_groups':[{'lr':1e-4}]}
        import copy
        self.assertTrue(same_state(state,copy.deepcopy(state)))
        changed=copy.deepcopy(state)
        changed['state'][0]['exp_avg'][1]=.3
        self.assertFalse(same_state(state,changed))
        self.assertFalse(same_state(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64)))

    def test_probe_wrappers_never_invoke_historical_probe_or_runtime(self):
        for name in ('preflight.sbatch','online_preflight.sbatch'):
            text=Path('experiments/gemma4_methods',name).read_text()
            self.assertIn('native_method_probe',text)
            self.assertIn('for gemma_probe_stage in update restore',text)
            self.assertNotIn('20260916-a10',text)
            self.assertNotIn('max_tokens=20000',text)
