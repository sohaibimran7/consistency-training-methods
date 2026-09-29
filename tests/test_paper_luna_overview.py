import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('luna_overview',Path(__file__).resolve().parents[1]/'scripts/paper_luna/overview.py')
overview=importlib.util.module_from_spec(spec)
spec.loader.exec_module(overview)

class OverviewTests(unittest.TestCase):
    def test_relabel_and_exclude(self):
        samples={'bct':[dict(dataset='d',bias='b',qid=str(i),u=u,b=b,option=1,clean_verified=True)
                        for i,(u,b) in enumerate([(0,1),(1,1),(None,1),(0,None),(0,0)])]}
        labels=[dict(method='bct',dataset='d',bias='b',qid=str(i),target=0) for i in range(6)]
        rows,audit=overview.correct_labels(labels,samples)
        self.assertEqual([r['target'] for r in rows],[1,0])
        self.assertEqual(audit['excluded'],3)
        self.assertEqual(audit['missing_source'],1)
        self.assertEqual(audit['changed_eligible_targets'],1)

    def test_preserves_view_effort_request_identity(self):
        label=dict(method='rmct352',dataset='d',bias='b',qid='q',target=1,
                   effort='minimal',view='prompt_only',request_id='shared',case_id='c')
        sample=dict(dataset='d',bias='b',qid='q',u=0,b=0,option=1,clean_verified=False)
        rows,audit=overview.correct_labels([label],{'rmct352':[sample]})
        self.assertEqual(rows[0]['target'],0)
        for key in ('effort','view','request_id','case_id'):
            self.assertEqual(rows[0][key],label[key])
        self.assertEqual(audit['unverified_base_clean_rows'],1)

if __name__=='__main__':unittest.main()
