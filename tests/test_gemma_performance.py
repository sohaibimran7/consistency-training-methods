import unittest
import tempfile
from pathlib import Path
from experiments.gemma4_methods.performance import summary,worker_benchmark_plan


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def row(self,step=1):
        from experiments.gemma4_methods.reference.train import checkpoint_identity
        name=f'checkpoints/step-{step:06d}'
        directory=self.root/'runs'/'opct'/name
        directory.mkdir(parents=True,exist_ok=True)
        for file in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','manifest.json'):
            (directory/file).write_text('fixture')
        return {'run_root':str(self.root),'method':'opct','step':step,'checkpoint_saved':True,
                'checkpoint':name,'checkpoint_files':checkpoint_identity(directory),
                'stage_wall_seconds':{'sampling':8.,'prepare_score_backward':4.,
                                      'optimizer':1.,'checkpoint':3.}}

    def test_timing_proportions_require_saved_distinct_updates(self):
        result=summary([self.row(),self.row(2)])
        self.assertEqual(result['saved_updates'],2)
        self.assertEqual(result['stage_fractions']['sampling'],.5)
        self.assertEqual(result['total_measured_stage_seconds'],32)
        with self.assertRaises(ValueError):
            summary([self.row(),self.row()])
        row=self.row()
        row['checkpoint_saved']=False
        with self.assertRaises(ValueError):
            summary([row])

    def test_invalid_timings_are_rejected(self):
        for value in (-1,float('inf'),float('nan')):
            row=self.row()
            row['stage_wall_seconds']['sampling']=value
            with self.assertRaises(ValueError):
                summary([row])

    def test_saved_flag_does_not_replace_checkpoint_byte_verification(self):
        row=self.row()
        (self.root/'runs'/'opct'/row['checkpoint']/'optimizer.pt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'checkpoint bytes'):
            summary([row])

    def test_worker_plan_does_not_claim_speedup(self):
        for method,sequences in (('bct',1),('opct',16),('rmct',96)):
            result=worker_benchmark_plan(method)
            self.assertEqual(result['concurrent_sequences_per_request'],sequences)
            self.assertIsNone(result['measured_speedup'])
            self.assertEqual(result['worker_counts'],[1,2,3])
