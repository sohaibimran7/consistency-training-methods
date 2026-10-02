import unittest
from experiments.rmct_restart_20260928.gemma_config import fresh_config, token_cap


class GemmaFreshTests(unittest.TestCase):
    def test_dataset_caps(self):
        self.assertEqual(token_cap('logiqa'), 20480)
        self.assertEqual(token_cap('hellaswag'), 20480)
        self.assertEqual(token_cap('hle-text-mc'), 65536)
        with self.assertRaises(ValueError):
            token_cap('')

    def test_fresh_mode_and_validation(self):
        c = fresh_config(source_commit='a'*40, approval_reference='user-message-reference',
                         validation_manifest_sha256='b'*64)
        self.assertIsNone(c['initialization']['resume_from'])
        self.assertEqual(c['initialization']['optimizer'], 'fresh')
        self.assertTrue(c['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(c['validation']['every_encountered_qids'], 256)
        self.assertTrue(c['validation']['counts_no_update_batches'])
        self.assertEqual(c['validation']['patience'], 2)
        self.assertEqual(c['validation']['improvement'], 'strict_decrease')
        self.assertFalse(c['validation']['diagnostics_select_or_stop'])

    def test_missing_provenance_rejected(self):
        with self.assertRaises(ValueError):
            fresh_config(source_commit='HEAD', approval_reference='user', validation_manifest_sha256='b'*64)
        with self.assertRaises(ValueError):
            fresh_config(source_commit='a'*40, approval_reference='', validation_manifest_sha256='b'*64)
