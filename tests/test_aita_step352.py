from types import SimpleNamespace
import unittest

from experiments.elephant_aita_ntaflip import no_cap_hf
from experiments.elephant_aita_ntaflip.step352 import partition_ids


class AITA352Tests(unittest.TestCase):
    def test_pair_partitions_have_exact_coverage(self):
        for count in (397, 398):
            source = list(range(count))
            parts = [partition_ids(source, index) for index in range(4)]
            self.assertEqual(sorted(pair for part in parts for pair in part), source)
            self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_transformers_provenance_is_not_decoding(self):
        saved = {"_from_model_config": True, "eos_token_id": [1, 2], "temperature": .6}
        model = SimpleNamespace(generation_config=SimpleNamespace(to_diff_dict=lambda: saved))
        self.assertEqual(no_cap_hf._saved_generation_config(model), saved)

    def test_unknown_processors_and_caps_still_fail(self):
        for saved in ({"max_length": 20}, {"forced_eos_token_id": 1}, {"_from_model_config": "true"}):
            with self.subTest(saved=saved), self.assertRaises(no_cap_hf.NoTokenCapRuntimeError):
                no_cap_hf._saved_generation_config(SimpleNamespace(generation_config=saved))
