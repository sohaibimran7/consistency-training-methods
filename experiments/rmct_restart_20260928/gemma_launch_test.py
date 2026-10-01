import unittest
import tempfile
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from experiments.rmct_restart_20260928.gemma_launch_parity import score_records, verify_module_origins


class FakeSampler:
    def __init__(self, mutation=lambda results: None):
        self._api = NS(SamplingParams=lambda **kw: NS(**kw), TokensPrompt=lambda **kw: kw)
        self.engine = NS(generate=self.generate)
        self.mutation = mutation
        self.awake = False

    def wake_up(self):
        self.awake = True

    def _policy_lora_request(self):
        return 'nonzero-adapter'

    def generate(self, prompts, params, **kwargs):
        assert self.awake and params.max_tokens == 1 and params.prompt_logprobs == 0
        self.request = kwargs['lora_request']
        rows = [NS(prompt_token_ids=p['prompt_token_ids'],
                   prompt_logprobs=[{token: NS(logprob=-float(i))} for i, token in enumerate(p['prompt_token_ids'])],
                   outputs=[NS(finish_reason='stop', token_ids=[106])]) for p in prompts]
        self.mutation(rows)
        return rows


class ParityTests(unittest.TestCase):
    def test_module_origin_and_hash_gate(self):
        path = Path(__file__).resolve()
        receipt = {'source_root': str(path.parent), 'sources': [
            {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
        self.assertEqual(verify_module_origins(receipt, [path]), [str(path)])
        receipt['sources'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_module_origins(receipt, [path])

    def test_external_module_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'not in CPU-attested'):
                verify_module_origins({'source_root': root, 'sources': []}, [__file__])

    def run_case(self, mutation=lambda rows: None, base=True):
        sampler = FakeSampler(mutation)
        termination = []
        scores = score_records(sampler, [{'prompt': [2, 3], 'completion': [4, 5]}],
                               use_base=base, termination=termination)
        return sampler, scores, termination

    def test_base_and_policy_exact_scores(self):
        for base in (True, False):
            sampler, scores, termination = self.run_case(base=base)
            self.assertEqual(scores, [[-2., -3.]])
            self.assertEqual(sampler.request, None if base else 'nonzero-adapter')
            self.assertEqual(termination[0]['generated_tokens'], 1)

    def test_bad_termination_fails(self):
        for reason in (None, 'unknown', 'abort'):
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                self.run_case(lambda rows: setattr(rows[0].outputs[0], 'finish_reason', reason))

    def test_unused_length_tail_does_not_change_fixed_token_scores(self):
        _, scores, termination = self.run_case(
            lambda rows: setattr(rows[0].outputs[0], 'finish_reason', 'length'))
        self.assertEqual(scores, [[-2., -3.]])
        self.assertEqual(termination[0]['cap'], 1)
        self.assertEqual(termination[0]['generated_token_ids'], [106])
        self.assertFalse(termination[0]['used_for_scores_or_training'])

    def test_token_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'token mismatch'):
            self.run_case(lambda rows: setattr(rows[0], 'prompt_token_ids', [9]))

    def test_missing_score_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'teacher-forced'):
            self.run_case(lambda rows: rows[0].prompt_logprobs.__setitem__(2, None))

    def test_nonfinite_score_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'teacher-forced'):
            self.run_case(lambda rows: setattr(rows[0].prompt_logprobs[2][4], 'logprob', float('nan')))

    def test_incomplete_batch_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'batch incomplete'):
            self.run_case(lambda rows: rows.clear())

    def test_cap_overrun_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_case(lambda rows: setattr(rows[0].outputs[0], 'token_ids', [1, 2]))

    def test_empty_or_invalid_tail_fails(self):
        for tokens in ([], [-1], [True]):
            with self.subTest(tokens=tokens), self.assertRaises(RuntimeError):
                self.run_case(lambda rows: setattr(rows[0].outputs[0], 'token_ids', tokens))
