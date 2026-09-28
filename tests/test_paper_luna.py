"""Offline tests; no provider access or private artifacts required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('paper_luna_common', Path(__file__).resolve().parents[1] / 'scripts/paper_luna/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

class PaperLunaTests(unittest.TestCase):
    def test_joint_target(self):
        for clean in (False, True):
            for biased in (False, True):
                r = dict(method='base', dataset='d', bias='b', qid=1, score=.5,
                         already_matched=clean, biased_matches=biased,
                         target=int(not clean and biased))
                common.validate_rows([r])
                with self.assertRaises(ValueError):
                    common.validate_rows([dict(r, target=1-r['target'])])

    def test_missing_and_duplicate_rejected(self):
        r = dict(method='base', dataset='d', bias='b', qid=1, score=.5, target=0)
        for bad in (None, float('nan'), -1, 2):
            with self.assertRaises(ValueError):
                common.validate_rows([dict(r, score=bad)])
        with self.assertRaises(ValueError):
            common.validate_rows([r, r])

    def test_shared_question_bootstrap(self):
        rows = [dict(dataset=d, qid=i) for d in ('a', 'b') for i in range(3)]
        clusters, lookup, weights = common.cluster_weights(rows + rows, 100, 7)
        self.assertEqual(len(clusters), 6)
        np.testing.assert_array_equal(weights, common.cluster_weights(rows, 100, 7)[2])
        for d in ('a', 'b'):
            np.testing.assert_array_equal(weights[:, [lookup[d, i] for i in range(3)]].sum(axis=1), 3)

    def test_icl_corrected_labels_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = {m: [dict(dataset='d', bias='b', qid=i, u=0 if i==1 else 1,
                                b=1, option=1, ack=0) for i in (1, 2)] for m in common.METHODS}
            (root/'samples.json').write_text(json.dumps(samples))
            labels=[]; results=[]
            for m in common.METHODS:
                for c in common.CONDITIONS:
                    for i in (1, 2):
                        rid=f'{m}:{c}:{i}'
                        labels.append(dict(method=m, condition=c, request_id=rid, case_id=str(i),
                                           dataset='d', bias='b', qid=i, target=0))
                        results.append(dict(request_id=rid, ok=True, score=75))
            (root/'full-private-labels.json').write_text(json.dumps(labels))
            (root/'luna-results.jsonl').write_text('\n'.join(map(json.dumps, results)))
            state=common.load_icl(SimpleNamespace(samples=root/'samples.json', ledger_dir=[root], bootstrap=10, seed=7))
            self.assertEqual(state['common'], ['1'])
            self.assertEqual(state['maps']['base','both']['1']['target'], 1)
            self.assertEqual(state['maps']['base','both']['1']['score'], .75)

if __name__ == '__main__':
    unittest.main()
