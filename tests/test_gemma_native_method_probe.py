import unittest
from pathlib import Path
from experiments.gemma4_methods.native_method_probe import same_state


class NativeProbeTests(unittest.TestCase):
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
