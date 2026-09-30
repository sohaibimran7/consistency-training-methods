"""Launcher fail-closed tests, not deployed runtime attestation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.gemma4_methods import launch_guard as g


class GuardTests(unittest.TestCase):
    def test_real_prompt_probe_covers_all_five_method_views(self):
        from experiments.gemma4_methods.native_prompt_probe import probe
        class Processor:
            def apply_chat_template(self,messages,**kwargs):
                if kwargs['tokenize']:
                    return [1] if kwargs['enable_thinking'] else [2]
                return '<|turn>system\n<|think|>\n<|turn>model\n'
            def encode(self,text,**kwargs):
                return [1]
        rows=[]
        for dataset in ('logiqa','hellaswag'):
            rows.append({'question_id':dataset+':1','source_dataset':dataset,
                'clean_messages':[{'role':'user','content':'Question'}],
                'variants':{bias:{'messages':[{'role':'user','content':'Cue Question'}],
                                 'biasing_text':'Cue'}
                            for bias in ('wrong_argument','suggested_answer')}})
        result=probe(Processor(),rows)
        self.assertEqual(len(result),40)
        self.assertEqual({r['method'] for r in result},{'bct','opct','act','attct','mlpct'})
        internal=[r for r in result if r['method']=='act' and r['bias']=='suggested_answer'
                  and r['side']=='variant_messages']
        self.assertTrue(all(r['messages'][0]['content']=='Cue\n\nQuestion' for r in internal))

    def test_missing_environment_fails_before_runtime(self):
        result = subprocess.run(['bash','-c',
            'source experiments/gemma4_methods/deployment_env.sh'],env={'PATH':'/usr/bin:/bin'},
            text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('GEMMA_DEPLOY_REPO',result.stderr)

    def test_runtime_manifest_hash_and_interpreter_are_mandatory(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pins.json'
            path.write_text(json.dumps({'schema':'gemma-reviewed-runtime-v1',
                                       'python':'/not/this/python'}))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(g,'check_source',return_value=Path(folder)):
                with self.assertRaisesRegex(ValueError,'bytes changed'):
                    g.verify(folder,'commit',path,'wrong')
                with self.assertRaisesRegex(ValueError,'interpreter differs'):
                    g.verify(folder,'commit',path,digest)

    def test_imported_repository_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            packages={name:{'version':'pinned','module_file':str(root/(name+'.py'))}
                      for name in ('torch','transformers','vllm','inspect_ai')}
            path=root/'pins.json'
            path.write_text(json.dumps({'schema':'gemma-reviewed-runtime-v1',
                'python':sys.executable,'python_version':sys.version,'packages':packages}))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            def module(name):
                return types.SimpleNamespace(__file__=packages[name]['module_file']
                    if name in packages else '/historical-overlay/engine.py')
            with patch.object(g,'check_source',return_value=root), \
                 patch.object(g.importlib,'import_module',side_effect=module), \
                 patch.object(g.importlib.metadata,'version',return_value='pinned'):
                with self.assertRaisesRegex(ValueError,'shadowed by overlay'):
                    g.verify(root,'commit',path,digest)

    def test_gpu_recipe_and_selection_flags_are_explicit(self):
        train=Path('experiments/gemma4_methods/train.sbatch').read_text()
        evaluation=Path('experiments/gemma4_methods/evaluate.sbatch').read_text()
        score=Path('experiments/gemma4_methods/score.sbatch').read_text()
        for flag in ('--selection-contract','--selection-folder','--verifier-factory'):
            self.assertIn(flag,train)
        self.assertIn('bct:4|opct:4|act:1|attct:1|mlpct:1',train)
        self.assertIn('#SBATCH --array=0-3',evaluation)
        for text in (train,evaluation,score):
            self.assertNotIn('client-overlay',text)
            self.assertNotIn('20260916-a10',text)
            self.assertNotIn('20260920-r4',text)
        self.assertIn('GEMMA_LAUNCH_PROFILE=grading',score)
        self.assertIn('GEMMA_GRADING_ENV_FILE',score)
