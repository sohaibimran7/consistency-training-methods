"""Synthetic offline contract tests only; receipts are not real runtime proof."""
import copy
import unittest
from scripts.prepare_eval_gaming_64k import digest,validate

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(target_config={'max_tokens':65536},no_fallback=True,reuse_historical_8192=False,
            requested_context_limit=70000,models={'base':['model','revision']},
            rows=[dict(source_id='one',condition='B',messages_sha256='a'*64)])
        self.proof=dict(campaign_sha256=digest(self.config),roles={'base':dict(model='model',revision='revision',
            configured_context_limit=70000,effective_context_limit=70000,long_context_configuration_verified=True,
            attestation_sha256='b'*64,tokenizer_configuration_sha256='c'*64,model_configuration_sha256='d'*64)},
            token_counts=[dict(role='base',source_id='one',condition='B',messages_sha256='a'*64,
                tokenizer_configuration_sha256='c'*64,prompt_tokens=4464,
                render_contract={'enable_thinking':True,'add_generation_prompt':True,'tools':None})])
    def test_exact_boundary(self):self.assertEqual(validate(self.config,self.proof)['minimum_headroom'],0)
    def test_reject_overflow(self):
        self.proof['token_counts'][0]['prompt_tokens']=4465
        with self.assertRaisesRegex(ValueError,'Context overflow'):validate(self.config,self.proof)
    def test_reject_missing_prompt(self):
        self.proof['token_counts']=[]
        with self.assertRaisesRegex(ValueError,'Incomplete'):validate(self.config,self.proof)
    def test_reject_legacy_context(self):
        self.proof['roles']['base']['effective_context_limit']=32768
        with self.assertRaisesRegex(ValueError,'context mismatch'):validate(self.config,self.proof)
    def test_reject_short_output(self):
        self.config['target_config']['max_tokens']=8192
        with self.assertRaisesRegex(ValueError,'Output policy'):validate(self.config,self.proof)
    def test_reject_unverified_rope(self):
        self.proof['roles']['base']['long_context_configuration_verified']=False
        with self.assertRaisesRegex(ValueError,'unverified'):validate(self.config,self.proof)
    def test_reject_render_mismatch(self):
        self.proof['token_counts'][0]['render_contract']['enable_thinking']=False
        with self.assertRaisesRegex(ValueError,'rendered'):validate(self.config,self.proof)
    def test_reject_wrong_revision(self):
        self.proof['roles']['base']['revision']='other'
        with self.assertRaisesRegex(ValueError,'revision'):validate(self.config,self.proof)
    def test_reject_reuse(self):
        self.config['reuse_historical_8192']=True
        with self.assertRaisesRegex(ValueError,'reuse forbidden'):validate(self.config,self.proof)

if __name__=='__main__':unittest.main()
