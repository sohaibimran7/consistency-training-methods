"""Offline regression tests; no model, network, fonts or image dependencies."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from recovery import overlay

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.original=[dict(model='m',id=str(i),qid=str(i),dataset='d',condition='c',
                            ground_truth='A',biased_option='B',error={'message':'timeout'}) for i in range(2)]
        self.good={**self.original[0],'error':None,'scores':{'parsed':0,'accuracy':0},'stop_reason':'max_tokens'}

    def test_partial_recovery_preserves_failed_and_unparsed(self):
        before=copy.deepcopy(self.original)
        merged,recovered,unresolved=overlay(self.original,[self.good,self.original[1]])
        self.assertEqual(self.original,before)
        self.assertEqual(len(recovered),1)
        self.assertEqual(merged[0]['scores']['parsed'],0)
        self.assertEqual(merged[1],self.original[1])
        self.assertEqual(unresolved[0]['reason'],'retry_failed')

    def test_missing_retry_explicit(self):
        self.assertEqual(len(overlay(self.original,[])[2]),2)

    def test_reject_duplicate_success(self):
        with self.assertRaises(ValueError):overlay(self.original,[self.good,self.good])

    def test_reject_changed_identity(self):
        with self.assertRaises(ValueError):overlay(self.original,[{**self.good,'qid':'wrong'}])

    def test_reject_unrequested_and_duplicate_original(self):
        with self.assertRaises(ValueError):overlay(self.original,[{**self.good,'id':'new'}])
        with self.assertRaises(ValueError):overlay(self.original*2,[])

    def test_reject_missing_score(self):
        with self.assertRaises(ValueError):overlay(self.original,[{**self.good,'scores':{}}])

    def test_offline_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'collected').mkdir()
            source=root/'collected/rows.json';source.write_text(json.dumps(self.original))
            before=source.read_bytes()
            retry=root/'retry.json';retry.write_text(json.dumps([self.good,self.original[1]]))
            output=root/'output'
            command=[sys.executable,str(Path(__file__).with_name('collect_recovered.py')),
                     '--root',str(root),'--retry-rows',str(retry),'--output',str(output)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(json.loads((output/'recovered-rows.json').read_text())),1)
            self.assertEqual(len(json.loads((output/'unresolved-manifest.json').read_text())),1)
            self.assertEqual(source.read_bytes(),before)
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main()
