import json
from pathlib import Path
import unittest
import subprocess
from unittest.mock import patch
from experiments.rmct_restart_20260928.qwen_train_window import command,gates,execute_segment
from experiments.rmct_restart_20260928.qwen_validation_producer import scheduler_complete


class ControllerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).parent/'fixtures/original_recipe.json').read_text())
        a=self.plan['argv'];i=a.index('--resume-from');del a[i:i+2]
        a.remove('--resume-with-optimizer');a.remove('--resume-state-required')
    def test_first_segment_no_parent(self):
        a=command(self.plan,0,None)
        self.assertFalse(any(x.startswith('--resume') for x in a))
        self.assertEqual(json.loads(a[a.index('--load-config')+1])['segment_index'],0)
    def test_later_segment_strict_clean_parent(self):
        a=command(self.plan,1,dict(step=16,checkpoint='/fresh/checkpoint'))
        self.assertEqual(a[a.index('--resume-from')+1],'file:///fresh/checkpoint')
        self.assertIn('--resume-state-required',a)
        self.assertIn('--resume-with-optimizer',a)
    def test_wrong_or_missing_parent(self):
        for i,parent in [(0,dict(step=16)),(1,None),(2,dict(step=16))]:
            with self.assertRaises(ValueError):command(self.plan,i,parent)
    def test_no_failed_or_running_score(self):
        for state in ('RUNNING|0:0','FAILED|1:0','COMPLETED|1:0',''):
            with patch('subprocess.check_output',return_value=state):
                with self.assertRaises(ValueError):scheduler_complete('123')
        with patch('subprocess.check_output',return_value='COMPLETED|0:0\n'):
            self.assertEqual(scheduler_complete('123'),'COMPLETED|0:0')

    def test_absent_incorporation_rejected(self):
        with patch('experiments.rmct_restart_20260928.qwen_train_window.source_check'), patch('subprocess.run',side_effect=subprocess.CalledProcessError(1,'git')):
            with self.assertRaises(subprocess.CalledProcessError):
                gates(dict(incorporated_commit='a'*40),'plan','preflight','rl','regression',Path('/repo'))

    def test_native_gate_wrong_source_rejected(self):
        prefix='experiments.rmct_restart_20260928.qwen_train_window.'
        with patch(prefix+'source_check'),patch('subprocess.run'),patch('subprocess.check_output',return_value='a'*40),patch(prefix+'sha',return_value='hash'),patch(prefix+'read',side_effect=[dict(source_commit='a'*40,plan_sha256='hash'),dict(schema='rmct-native-rl-gate-v1',status='passed',source_commit='b'*40)]):
            with self.assertRaises(ValueError):gates(dict(incorporated_commit='a'*40),'plan','preflight','rl','regression',Path('/repo'))

    def parent(self):
        a=self.plan['argv'];campaign=a[a.index('--run-name')+1];experiment=a[a.index('--experiment-name')+1]
        name=f'{campaign}-s001'
        return dict(schema='rmct-clean-checkpoint-v1',campaign_id=campaign,step=16,
                    checkpoint=str(Path('/repo/logs')/experiment/name/'checkpoints'/f'{experiment}_{name}'),
                    files={},parent=None,command=command(self.plan,0,None))

    def test_wrong_campaign_or_path_never_starts_child(self):
        for field,value in [('campaign_id','wrong'),('checkpoint','/old/checkpoint'),('step',32)]:
            parent=self.parent();parent[field]=value
            with patch('subprocess.run') as child:
                with self.assertRaises(ValueError):execute_segment(self.plan,1,parent,Path('/repo'),{})
                child.assert_not_called()

    def test_changed_parent_seal_never_starts_child(self):
        parent=self.parent()
        with patch('experiments.rmct_restart_20260928.qwen_train_window.seal',return_value=dict(parent,files={'changed':True})),patch('subprocess.run') as child:
            with self.assertRaises(ValueError):execute_segment(self.plan,1,parent,Path('/repo'),{})
            child.assert_not_called()

    def test_invalid_strict_resume_state_never_starts_child(self):
        with patch('experiments.rmct_restart_20260928.qwen_train_window.seal',side_effect=ValueError('invalid optimizer/RNG state')),patch('subprocess.run') as child:
            with self.assertRaises(ValueError):execute_segment(self.plan,1,self.parent(),Path('/repo'),{})
            child.assert_not_called()

    def test_valid_parent_revalidated_before_child(self):
        events=[];parent=self.parent()
        def checked(*a,**kw):events.append('seal');return parent
        def launched(*a,**kw):events.append('child')
        with patch('experiments.rmct_restart_20260928.qwen_train_window.seal',side_effect=checked),patch('subprocess.run',side_effect=launched):
            execute_segment(self.plan,1,parent,Path('/repo'),{})
        self.assertEqual(events,['seal','child'])


if __name__=='__main__':unittest.main()
