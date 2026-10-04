import unittest
from experiments.gemma4_methods.train import recipe as _recipe, CAP, validate_completion
from types import SimpleNamespace
from experiments.gemma4_methods.reference import plan
from ctm_data.adapters.mcq_bias import shared_qid_one_bias as one
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import DATUM_SCHEMA, SCHEMA_VERSION


def _manifest():
    def msg(text):
        return [{'role': 'user', 'content': text}]
    rows = [{'datum_schema': DATUM_SCHEMA, 'schema_version': SCHEMA_VERSION, 'question_id': f'q{i}',
             'source_dataset': ('logiqa', 'hellaswag')[i % 2], 'question': 'q', 'ground_truth': 'A',
             'prompt_style': 'none', 'clean_messages': msg('clean'),
             'biased_options': {b: 'B' for b in plan.BIASES},
             'variants': {b: {'messages': msg(b), 'biased_option': 'B', 'biasing_text': b} for b in plan.BIASES},
             'provenance': {'wrong_argument_source_line_number': i + 1}} for i in range(8)]
    return one.build_manifest(rows, seed=42, source={'pool_sha256': 'x'})


def recipe():
    return _recipe(_manifest())


class GemmaMethodsTests(unittest.TestCase):
    def test_termination_uses_model_and_renderer(self):
        backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9)))
        renderer = SimpleNamespace(get_stop_sequences=lambda: [8])
        for tokens, reason in [([1, 9], 'model_eos'), ([1, 8], 'model_eos'), ([1, 2, 3], 'length')]:
            self.assertEqual(validate_completion(SimpleNamespace(tokens=tokens), backend,
                renderer=renderer, max_tokens=3)[1], reason)
        for tokens in [[], [1], [1, 2, 3, 4]]:
            with self.assertRaises(ValueError):
                validate_completion(SimpleNamespace(tokens=tokens), backend, renderer=renderer, max_tokens=3)

    def test_exact_cap_and_exclusion(self):
        self.assertEqual(CAP, 20480)
        self.assertEqual(recipe()['generation']['output_token_cap'], 20480)
        self.assertIn('before_any_backward', recipe()['generation']['length_stop_policy'])

    def test_stopping_delegated_to_shared_validation(self):
        self.assertEqual(recipe()['convergence']['metric'], 'TBSR')
        self.assertFalse(recipe()['convergence']['loss_selects_or_stops'])

    def test_tiny_strict_improvement_resets_patience(self):
        state = {'step': 0, 'pending': [], 'decision': 'continue'}
        for step in range(1, 129):
            state = plan.observe(state, step=step, loss=1.)
        self.assertEqual(state['nonimproving'], 7)
        for step in range(129, 145):
            state = plan.observe(state, step=step, loss=0.999999)
        self.assertEqual(state['nonimproving'], 0)
        self.assertEqual(state['decision'], 'continue')

    def test_eight_complete_stale_windows(self):
        state = {'step': 0, 'pending': [], 'decision': 'continue'}
        for step in range(1, 144):
            state = plan.observe(state, step=step, loss=1.)
        self.assertEqual(state['decision'], 'continue')
        state = plan.observe(state, step=144, loss=1.)
        self.assertEqual(state['decision'], 'plateau')

    def test_scientific_estimator_unchanged(self):
        for key in ['optimizer', 'opct', 'loss_options', 'data']:
            self.assertEqual(recipe()[key], plan.contract()[key])
        # One-bias protocol (user-approved 2026-10-02): 4 distinct QIDs x 1 assigned bias.
        batch = recipe()['batch']
        self.assertEqual((batch['qids_per_update'], batch['biases_per_qid'], batch['paired_rows_per_update']), (4, 1, 4))
        self.assertFalse(batch['repeat_after_pool_exhaustion'])
        bct = recipe()['bct']
        self.assertEqual(bct['supervised_bias'], 'assigned_bias_only')
        self.assertEqual(bct['target_representation'], plan.contract()['bct']['target_representation'])
        exposure = recipe()['exposure']
        self.assertFalse(exposure['cycling_allowed'])
        self.assertTrue(exposure['skipped_batches_consume_encounters'])
        self.assertEqual(recipe()['convergence']['every_encountered_qid_bias_examples'], 256)
        self.assertEqual(recipe()['execution']['training_gpus']['opct'], 4)


if __name__ == '__main__':
    unittest.main()


class ReasoningCompletionTests(unittest.TestCase):
    def setUp(self):
        ids = {'<|channel>': 100, '<channel|>': 101}
        tokenizer = SimpleNamespace(convert_tokens_to_ids=ids.get, unk_token_id=3)
        self.renderer = SimpleNamespace(get_stop_sequences=lambda: [106], tokenizer=tokenizer)
        self.backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1)))

    def finish(self, tokens):
        return validate_completion(SimpleNamespace(tokens=tokens), self.backend,
                                   renderer=self.renderer, max_tokens=50)[1]

    def test_unclosed_reasoning_with_eos_is_rejected(self):
        self.assertEqual(self.finish([100, 7, 8, 106]), 'unclosed_reasoning')
        self.assertEqual(self.finish([100, 7, 101, 9, 100, 5, 1]), 'unclosed_reasoning')  # reopened, not closed

    def test_closed_reasoning_and_no_reasoning_accepted(self):
        self.assertEqual(self.finish([100, 7, 101, 9, 106]), 'model_eos')
        self.assertEqual(self.finish([9, 9, 106]), 'model_eos')  # never opened: unchanged policy

    def test_length_cap_still_reported(self):
        self.assertEqual(self.finish([100] + [7] * 49), 'length')

    def test_incomplete_set_and_skip_reason(self):
        from experiments.gemma4_methods.train import INCOMPLETE, TruncatedGroup
        self.assertEqual(INCOMPLETE, ('length', 'unclosed_reasoning'))
        self.assertEqual(TruncatedGroup('incomplete_reasoning_group').reason, 'incomplete_reasoning_group')
        self.assertEqual(TruncatedGroup().reason, 'incomplete_generation_group')
