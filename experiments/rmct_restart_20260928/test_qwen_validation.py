import unittest
from experiments.rmct_restart_20260928 import qwen_validation as v


def row(step,n=1,d=10,**kw):
    return dict(step=step,towards_switches=n,eligible_pairs=d,tbsr=n/d,
                campaign_id='fresh',validation_sha256=v.PROMPT_SHA,settings_sha256=v.settings_sha(),
                verified=True,job_completed=True,response_count=600,**kw)


class ControllerTests(unittest.TestCase):
    def test_ties_stop_earliest(self):
        r=v.replay([row(64),row(128,2,20),row(192)],campaign_id='fresh')
        self.assertTrue(r['stopped']);self.assertEqual(r['best_checkpoint']['step'],64)
    def test_improvement_resets(self):
        r=v.replay([row(64),row(128),row(192,0)],campaign_id='fresh')
        self.assertFalse(r['stopped']);self.assertEqual(r['best_checkpoint']['step'],192)
    def test_duplicates_missing_and_old_campaign_fail(self):
        for rows in ([row(64),row(64)],[row(128)],[row(64),row(192)]):
            with self.assertRaises(ValueError):v.replay(rows,campaign_id='fresh')
        with self.assertRaises(ValueError):v.replay([row(64)],campaign_id='old')
    def test_incomplete_does_not_consume_patience(self):
        for field,val in [('verified',False),('job_completed',False),('response_count',599)]:
            r=row(64);r[field]=val
            with self.assertRaises(ValueError):v.replay([r],campaign_id='fresh')
    def test_no_post_stop_or_old_settings(self):
        with self.assertRaises(ValueError):v.replay([row(s) for s in (64,128,192,256)],campaign_id='fresh')
        r=row(64);r['settings_sha256']='historical65536'
        with self.assertRaises(ValueError):v.replay([r],campaign_id='fresh')


if __name__=='__main__':unittest.main()
