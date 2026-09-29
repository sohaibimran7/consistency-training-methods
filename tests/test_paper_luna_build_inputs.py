import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('luna_builder', Path(__file__).resolve().parents[1] / 'scripts/paper_luna/build_inputs.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BuilderTests(unittest.TestCase):
    def test_identity_and_unicode(self):
        messages = b.messages_for('résist', 'system')
        expected = hashlib.sha256(json.dumps(['openai/gpt-5.6-luna', 'xhigh', messages], sort_keys=True).encode()).hexdigest()
        self.assertEqual(b.request_id(messages), expected)
        self.assertIn('résist', messages[1]['content'])
        self.assertIsNone(b.reasoning('unclosed reasoning'))
        self.assertIsNone(b.reasoning('</think></think>'))
        self.assertIsNone(b.reasoning('<think> </think> answer'))
        self.assertEqual(b.reasoning('<think>evidence</think> answer'), 'evidence')

    def test_template_boundary(self):
        template = [{'content':'s'}, {'content':'bank\n'+b.MARKER+'placeholder'}]
        self.assertEqual(b.messages_for('cot','s',template)[1]['content'], 'bank\n'+json.dumps({'reasoning':'cot'}))
        with self.assertRaises(ValueError):
            b.messages_for('cot', 's', [{'content':'s'},{'content':'no marker'}])

    def test_remapped_recovery_log_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log.eval';path.write_text('fixture')
            row=dict(biased_source='/old/log.eval', dataset='d', bias='b', qid='q')
            s=NS(id=9, metadata={'cap64k_source':{'dataset':'d','bias':'b','qid':'q'}}, output=NS(completion='<think>r</think> A'))
            inputs=b.Inputs({'/old/log.eval':str(path)}, reader=lambda p:NS(samples=[s]))
            self.assertIs(inputs.sample(row),s)
            self.assertEqual(len(inputs.files),1)
            bad=b.Inputs({'/old/log.eval':str(path)}, reader=lambda p:NS(samples=[s,s]))
            with self.assertRaises(ValueError):bad.sample(row)

    def test_filter_missing_not_negative(self):
        r=dict(dataset='d',bias='b',qid='q',u=0,b=1,option=1,biased_source='p')
        samples={m:[] for m in b.METHODS};samples['base']=[r]
        inputs=NS(sample=lambda r:NS(output=NS(completion='<think>r</think>A')))
        rows,missing,stats=b.build_filter(inputs,samples,'s',{})
        self.assertEqual(rows,[])
        self.assertEqual(stats['base']['missing_filtered'],1)
        rid=b.request_id(b.messages_for('r','s'))
        rows,_,_=b.build_filter(inputs,samples,'s',{rid:{'score':75}})
        self.assertEqual(rows[0]['target'],1)
        self.assertEqual(rows[0]['score'],.75)

    def test_rare_membership_requires_all_scores(self):
        templates={m:{c:[{'content':'s'},{'content':c+b.MARKER}] for c in b.CONFIGS[1:]} for m in b.METHODS}
        row=dict(method='base',case_id='c',dataset='d',bias='b',qid='q',ack=0,target=1)
        inputs=NS(sample=lambda r:NS(output=NS(completion='<think>r</think>A')))
        ids={c:b.request_id(b.messages_for('r','s',None if c=='zero_shot' else templates['base'][c])) for c in b.CONFIGS}
        cache={rid:{'score':50} for rid in ids.values()}
        rows,counts=b.build_rare(inputs,[row],[row],templates,cache,set())
        self.assertFalse(rows[0]['baseline']);self.assertTrue(rows[0]['rare'])
        self.assertEqual(counts['base']['quadrants']['0,1']['selected'],1)
        with self.assertRaises(ValueError):b.build_rare(inputs,[row],[row],templates,{},set())

    def test_ledger_order_and_invalid_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.jsonl'
            path.write_text('\n'.join(json.dumps(dict(request_id='r',ok=ok,score=score)) for ok,score in [(True,20),(False,None),(True,30)]))
            self.assertEqual(b.Inputs().scores([path])['r']['score'],30)
            path.write_text(json.dumps(dict(request_id='r',ok=True,score=None)))
            with self.assertRaises(ValueError):b.Inputs().scores([path])

if __name__ == '__main__':unittest.main()
