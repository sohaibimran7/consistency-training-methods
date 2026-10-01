"""Actual600response verifier replay with CPU fixture tokens, not GPU proof."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments.gemma4_methods.selection_adapter import GemmaVerifiers,file_identity


class Processor:
    def apply_chat_template(self,messages,**kwargs):
        return [1] if kwargs['tokenize'] else '<|turn>system\n<|think|>\n<|turn>model\n'
    def encode(self,text,**kwargs):
        return [1]
    def decode(self,tokens,**kwargs):
        if tokens==[99]:
            return '<turn|>'
        return '<|channel>thought\nr<channel|>'+('B' if tokens[0]==2 else 'A')+'<turn|>'


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.processor=Processor()
        self.adapter=GemmaVerifiers(processor=self.processor,verify_runtime_checkpoint=lambda p,c:p,
                                     verify_scheduler=lambda s,c:True)
        self.contract={'campaign_id':'fixture','method':'act','model':'model','source_commit':'source',
            'settings':{'enable_thinking':True,'max_tokens':20480,'temperature':1.,'top_p':.95,
                        'top_k':20,'stop_token_ids':[99]}}
        self.progress={'actual_optimizer_step':64,'checkpoint_files':{}}
        rows=[]
        samples={}
        for dataset in ('logiqa','hellaswag'):
            for qid in range(100):
                for condition in ('clean','wrong_argument','suggested_answer'):
                    sid=f'{dataset}:{qid}:{condition}'
                    row={'sample_id':sid,'dataset':dataset,'question_id':str(qid),'condition':condition,
                         'messages':[{'role':'user','content':'Question'}],'biased_option':'B'}
                    rows.append(row)
                    request=self.save(sid+'-request',{'sample_id':sid,'checkpoint_files':{},
                        'prompt_token_ids':[1],'settings':self.contract['settings'],'model':'model'})
                    tokens=[2 if condition=='wrong_argument' else 3,99]
                    response=self.save(sid+'-response',{'sample_id':sid,'request':request,
                        'generated_token_ids':tokens,'raw_text':self.processor.decode(tokens),'finish_reason':'stop'})
                    samples[sid]={'request':request,'response':response}
        self.rows=rows
        self.contract['population']=self.save('population',{'rows':rows})
        self.validation={**{k:self.contract[k] for k in ('campaign_id','method','model','source_commit')},
            'schema':'gemma-native-validation-evidence-v1','checkpoint_files':{},
            'scheduler':{'job_id':'fixture'},'samples':samples}

    def save(self,name,value):
        path=self.root/(name+'.json')
        path.write_text(json.dumps(value))
        return file_identity(path)

    def verify(self):
        return self.adapter.verify_validation(self.save('validation',self.validation),self.progress,self.contract)

    def test_all600_responses_reproduce400_eligible_pairs(self):
        result=self.verify()
        self.assertEqual((result['response_count'],result['towards_switches'],result['eligible_pairs']),(600,200,400))
        self.assertEqual(result['tbsr'],.5)

    def test_malformed_generated_tokens_fail_not_denominator_drop(self):
        sid=next(iter(self.validation['samples']))
        reference=self.validation['samples'][sid]['response']
        response=json.loads(Path(reference['path']).read_text())
        for invalid in ([True,99],['2',99],[-1,99],{'token':2}):
            self.validation['samples'][sid]['response']=self.save('bad-response',{
                **response,'generated_token_ids':invalid})
            with self.assertRaisesRegex(ValueError,'generation termination'):
                self.verify()

    def test_invalid_promoted_option_and_missing_sample_fail(self):
        self.rows[1]['biased_option']='BB'
        self.contract['population']=self.save('population',{'rows':self.rows})
        with self.assertRaisesRegex(ValueError,'single A-D'):
            self.verify()
        self.rows[1]['biased_option']='B'
        self.contract['population']=self.save('population',{'rows':self.rows})
        self.validation['samples'].pop(next(iter(self.validation['samples'])))
        with self.assertRaisesRegex(ValueError,'coverage'):
            self.verify()
