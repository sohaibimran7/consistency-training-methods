"""CPU safety checks; mocks here are not native runtime clearance."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.gemma4_methods import selection_adapter as a


class Processor:
    text = '<|turn>system\n<|think|>\n<|turn>model\n'

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        assert kwargs['add_generation_prompt'] is True
        return [1, 2] if kwargs['tokenize'] else self.text

    def encode(self, text, **kwargs):
        return [1, 2]


class EvidenceTests(unittest.TestCase):
    def test_incomplete_rng_is_rejected(self):
        from experiments.gemma4_methods.checkpoint import require_coordinator_rng
        with self.assertRaisesRegex(ValueError,'Complete CUDA'):
            require_coordinator_rng({'schema':'ctm.rl_runtime_rng.v1',
                                     'python_random_state':[]})

    def test_checkpoint_injects_rng_before_sealing_without_modifying_caller(self):
        import asyncio
        from experiments.gemma4_methods import checkpoint as c
        rng = {'schema':'ctm.rl_runtime_rng.v1','python_random_state':[],
               'torch_cpu_rng_state_base64':'cpu','torch_cuda_rng_state_base64':'cuda',
               'torch_cuda_coordinator_device':0}
        calls=[]
        class Backend:
            async def save_checkpoint(self, **kwargs):
                calls.append(kwargs)
                return 'saved'
        original={'step':64}
        with patch.object(c,'capture_runtime_rng_state',return_value=rng):
            result=asyncio.run(c.RNGCheckpointBackend(Backend()).save_checkpoint(loop_state=original))
        self.assertEqual(result,'saved')
        self.assertEqual(original,{'step':64})
        self.assertEqual(calls[0]['loop_state']['runtime_rng'],rng)
        self.assertIs(calls[0]['loop_state']['rollout_worker_rng_serialized'],False)

    def test_rng_restore_requires_actual_readback(self):
        from experiments.gemma4_methods import checkpoint as c
        rng = {'schema':'ctm.rl_runtime_rng.v1','python_random_state':[],
               'torch_cpu_rng_state_base64':'cpu','torch_cuda_rng_state_base64':'cuda',
               'torch_cuda_coordinator_device':0}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'manifest.json').write_text(json.dumps({'loop_state':{'runtime_rng':rng}}))
            with patch.object(c,'restore_runtime_rng_state') as restore, \
                 patch.object(c,'capture_runtime_rng_state',return_value={**rng,'python_random_state':[1]}):
                with self.assertRaisesRegex(ValueError,'differs'):
                    c.restore_coordinator_rng(folder)
                restore.assert_called_once_with(rng,require_torch=True)

    def test_native_prompt_uses_true_and_checks_token_parity(self):
        p = Processor()
        self.assertEqual(a.native_prompt(p, []), [1, 2])
        with patch.object(p, 'encode', return_value=[2, 1]):
            with self.assertRaisesRegex(ValueError, 'prompt mismatch'):
                a.native_prompt(p, [])

    def test_closed_thought_prompt_fails(self):
        p = Processor()
        p.text += '<|channel>thought\n<channel|>'
        with self.assertRaises(ValueError):
            a.native_prompt(p, [])

    def test_immutable_artifact_read_checks_actual_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'record.json'
            p.write_text(json.dumps({'step': 64}))
            record = a.file_identity(p)
            self.assertEqual(a.read_verified(record), {'step': 64})
            p.write_text(json.dumps({'step': 65}))
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                a.read_verified(record)

    def test_saved_status_cannot_replace_native_verifier(self):
        with self.assertRaises(TypeError):
            a.GemmaVerifiers(processor=Processor(),
                             verify_runtime_checkpoint=True, verify_scheduler=True)

    def test_bootstrap_delegates_all_evidence_to_shared_api(self):
        module_name = 'experiments.rmct_restart_20260928.validation_selection'
        shared = types.ModuleType(module_name)
        calls = []
        def bootstrap(contract, start, **kwargs):
            calls.append((contract,start,kwargs))
            return 64
        shared.bootstrap_budget = bootstrap
        verify = lambda start, contract: start
        with patch.dict(sys.modules, {module_name:shared}):
            self.assertEqual(a.verified_bootstrap('contract','start',100,verify),64)
        self.assertEqual(calls, [('contract','start',
                                 {'requested_updates':100,'verify_start':verify})])

    def test_native_hooks_require_start_receipt_and_verifier(self):
        module_name = '_gemma_mock_hooks'
        module = types.ModuleType(module_name)
        module.factory = lambda args: types.SimpleNamespace(
            adapter=a.GemmaVerifiers(processor=Processor(),
                                    verify_runtime_checkpoint=lambda p,c:p,
                                    verify_scheduler=lambda s,c:True),
            normalized_progress=lambda *args:None,
            verify_fresh_initialization=lambda *args:True)
        with patch.dict(sys.modules, {module_name:module}):
            with self.assertRaisesRegex(TypeError,'start_record'):
                a.load_native_hooks(module_name+':factory',None)
