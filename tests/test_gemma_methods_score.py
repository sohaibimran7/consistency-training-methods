import unittest
from types import SimpleNamespace as S
from experiments.gemma4_methods.score import select_pairs, valid_completion


class SelectionTests(unittest.TestCase):
    def test_truncation_excluded(self):
        for reason in ['stop', 'length', 'max_tokens']:
            sample = S(error=None, output=S(choices=[S(stop_reason=reason)]))
            self.assertEqual(valid_completion(sample), reason == 'stop')

    def test_unknown_and_errors_fail_closed(self):
        for sample in [S(error='bad', output=None), S(error=None, output=S(choices=[])),
                       S(error=None, output=S(choices=[S(stop_reason='unknown')]))]:
            with self.assertRaises(ValueError):
                valid_completion(sample)

    def test_missing_clean_does_not_exclude_luna_source(self):
        clean = {'a': object(), 'c': object()}
        biased = {'a': object(), 'b': object()}
        self.assertEqual(select_pairs(clean, biased), ['a'])
        self.assertEqual(set(biased), {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
