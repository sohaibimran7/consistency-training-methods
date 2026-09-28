from types import SimpleNamespace

import unittest

from experiments.rmct_two_bias_eval.step352 import validate_generation


def log(ids, status="success"):
    return SimpleNamespace(status=status, samples=[SimpleNamespace(id=i) for i in ids])


class GenerationValidationTests(unittest.TestCase):
    def test_completion_order_is_not_selection_order(self):
        validate_generation(log(["b", "a"]), ["a", "b"])

    def test_reject_missing_duplicate_foreign_or_incomplete(self):
        for ids, status in [(["a", "a"], "success"), (["a", "c"], "success"), (["a"], "success"), (["a", "b"], "started")]:
            with self.subTest(ids=ids, status=status), self.assertRaises(ValueError):
                validate_generation(log(ids, status), ["a", "b"])
