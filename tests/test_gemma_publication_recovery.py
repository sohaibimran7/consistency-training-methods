import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.gemma4_methods import checkpoint as c
from experiments.gemma4_methods.reference import train as helpers

RNG={'schema':'ctm.rl_runtime_rng.v1','python_random_state':[],
     'torch_cpu_rng_state_base64':'cpu','torch_cuda_rng_state_base64':'cuda','torch_cuda_coordinator_device':0}


class Backend:
    async def save_checkpoint(self,**kwargs):
        root=Path(kwargs['log_dir'])/'checkpoints'/kwargs['name']
        root.mkdir(parents=True)
        for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt'):
            (root/name).write_text('native-fixture')
        (root/'manifest.json').write_text(json.dumps({'kind':'both','model':'model','loop_state':kwargs['loop_state']}))


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.state={'step':1,'attempts':1,'decision':'continue','pending':[.2],
                    'last_update_question_ids':['a','b']}
        self.row={'step':1,'attempt':0,'loss':.2,'question_ids':['a','b']}

    def seal(self):
        with patch.object(c,'capture_runtime_rng_state',return_value=RNG):
            return asyncio.run(c.seal_checkpoint(Backend(),run_dir=self.root,method='act',
                state=self.state,plan_hash='plan',window_metrics=[self.row]))

    def test_crash_after_metrics_before_checkpoint_preserves_tail_and_retries_once(self):
        original=json.dumps(self.row)+'\n'
        (self.root/'metrics.jsonl').write_text(original)
        self.assertIsNone(c.recover_publication(self.root,'plan','act','m'*64))
        self.assertEqual((self.root/'metrics.jsonl').read_text(),'')
        self.assertEqual(next((self.root/'recovery').glob('metrics-*')).read_text(),original)
        self.seal()
        c.recover_publication(self.root,'plan','act','m'*64)
        rows=[json.loads(x) for x in (self.root/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(rows,[self.row])

    def test_crash_after_pointer_before_external_progress_is_rebuilt(self):
        self.seal()
        self.assertTrue((self.root/'state.json').exists())
        self.assertFalse((self.root/'progress'/'step-000001.json').exists())
        c.recover_publication(self.root,'plan','act','m'*64)
        progress=json.loads((self.root/'progress'/'step-000001.json').read_text())
        self.assertEqual((progress['actual_optimizer_step'],progress['next_attempt_index']),(1,1))
        self.assertEqual(helpers.load_resume(self.root,'plan','act')[0],self.state)

    def test_crash_after_checkpoint_rename_before_receipt_recovers_native_publication(self):
        original=helpers.plan.immutable_json
        def inject(path,value):
            if 'receipts' in Path(path).parts:
                raise RuntimeError('injected publication crash')
            return original(path,value)
        with patch.object(helpers.plan,'immutable_json',side_effect=inject):
            with self.assertRaises(RuntimeError):
                self.seal()
        self.assertTrue((self.root/'checkpoints'/'step-000001').exists())
        self.assertFalse((self.root/'state.json').exists())
        c.recover_publication(self.root,'plan','act','m'*64)
        self.assertEqual(helpers.load_resume(self.root,'plan','act')[0],self.state)

    def test_duplicate_logged_retry_is_replaced_by_authoritative_immutable_row(self):
        self.seal()
        original=json.dumps({**self.row,'loss':99})+'\n'+json.dumps(self.row)+'\n'
        (self.root/'metrics.jsonl').write_text(original)
        c.recover_publication(self.root,'plan','act','m'*64)
        self.assertEqual(len((self.root/'metrics.jsonl').read_text().splitlines()),1)
        self.assertEqual(next((self.root/'recovery').glob('metrics-*')).read_text(),original)


def test_recovery_progress_builder_carries_one_bias_manifest_identity():
    from experiments.gemma4_methods.checkpoint import make_progress
    receipt = {'plan_sha256': 'p', 'checkpoint': 'checkpoints/step-000001', 'checkpoint_files': {},
               'convergence': {'step': 1, 'attempts': 3, 'last_update_question_ids': ['a', 'b', 'c', 'd']}}
    value = make_progress(receipt, 'f' * 64)
    assert value['one_bias_manifest_sha256'] == 'f' * 64
    assert (value['consumed_qid_position'], value['encountered_qid_bias_examples']) == (12, 12)
