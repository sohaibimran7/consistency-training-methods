import unittest
from experiments.rmct_restart_20260928.qwen_checkpoint import validate_metadata
from experiments.rmct_restart_20260928.qwen_validation_server import check_executable


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(backend='local',kind='both',loop_state=dict(global_step=16,optimizer_step=16,step=16,final=True,accumulated_grads=0))
        self.r=dict(checkpoint_kind='both',world_size=4,train_logical_indices=[0,1,2,3],process_group_backend='nccl',device_type='cuda',rng_state_file='replicated_training_rng.pt')
    def test_valid_metadata(self):validate_metadata(self.m,self.r,16)
    def test_partial_and_wrong_step(self):
        for key,val in [('final',False),('accumulated_grads',1),('optimizer_step',15)]:
            m=dict(self.m,loop_state=dict(self.m['loop_state'],**{key:val}))
            with self.assertRaises(ValueError):validate_metadata(m,self.r,16)
    def test_missing_rank_state(self):
        for key,val in [('world_size',1),('rng_state_file',None),('checkpoint_kind','adapter')]:
            with self.assertRaises(ValueError):validate_metadata(self.m,dict(self.r,**{key:val}),16)
    def test_wrong_vllm_environment(self):
        check_executable('/new/venv/bin/vllm','/new/venv')
        for path in (None,'/old/venv/bin/vllm','/usr/bin/vllm'):
            with self.assertRaises(ValueError):check_executable(path,'/new/venv')


if __name__=='__main__':unittest.main()
