import unittest
from experiments.gemma4_methods.train import recipe, CAP, validate_completion
from types import SimpleNamespace
from experiments.gemma4_methods.reference import plan


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
        for key in ['batch', 'optimizer', 'opct', 'bct', 'loss_options', 'data']:
            self.assertEqual(recipe()[key], plan.contract()[key])
        self.assertEqual(recipe()['execution']['training_gpus']['opct'], 4)


if __name__ == '__main__':
    unittest.main()
