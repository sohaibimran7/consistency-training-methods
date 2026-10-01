import unittest
from unittest.mock import patch
from experiments.gemma4_methods.native_hooks import scheduler_complete


class NativeHookTests(unittest.TestCase):
    def test_scheduler_success_requires_exact_actual_job_and_exitcode(self):
        with patch('subprocess.check_output',return_value='123|COMPLETED|0:0\n'):
            self.assertTrue(scheduler_complete({'job_id':'123'},{}))
        for output in ('124|COMPLETED|0:0\n','123|RUNNING|0:0\n','123|COMPLETED|1:0\n',''):
            with patch('subprocess.check_output',return_value=output):
                with self.assertRaises(ValueError):
                    scheduler_complete({'job_id':'123'}, {})

    def test_status_flag_and_shell_inputs_are_not_scheduler_proof(self):
        for receipt in ({'passed':True},{'job_id':'123;exit'}, {'job_id':'123','status':'COMPLETED'}):
            with self.assertRaises(ValueError):
                scheduler_complete(receipt,{})
