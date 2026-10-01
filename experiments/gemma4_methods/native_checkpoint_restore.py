"""Actually restore a production checkpoint in an isolated GPU process.

No optimizer update or rollout is performed. Private vLLM RNG is not restored.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.gemma4_methods import train
from experiments.gemma4_methods.native_method_probe import same_state,save
from experiments.gemma4_methods.selection_adapter import file_identity
from experiments.gemma4_methods.launch_guard import check_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','commit','checkpoint','model','method','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    root=check_source(a.repository,a.commit)
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Allocated native restoration required')
    checkpoint=Path(a.checkpoint).resolve()
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    if manifest['model']!=a.model or manifest['loop_state']['method']!=a.method:
        raise ValueError('Checkpoint method/model mismatch')
    import torch
    from transformers import AutoModelForImageTextToText
    from ctm.backends.local.engine import LocalBackend
    from ctm.core.config import LoRAConfig,AdamConfig
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from experiments.gemma4_methods.checkpoint import restore_coordinator_rng
    online=a.method in ('bct','opct')
    model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,local_files_only=True,
        **({'attn_implementation':'eager'} if not online else {}))
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
        and 'language_model' in n.split('.') and
        (('self_attn' in n.split('.') or 'mlp' in n.split('.')) if online else n.rsplit('.',1)[-1] in ('q_proj','v_proj'))]
    backend=LocalBackend(device='cuda:0',dtype=torch.bfloat16,model_instance=model,sampler='hf',gradient_checkpointing=True)
    backend.setup(model=a.model,lora=LoRAConfig(rank=8,alpha=16,dropout=0,train_mlp=online,
        train_attn=online,train_unembed=False,seed=42,target_modules=targets),
        resume_from='file://'+str(checkpoint),resume_with_optimizer=True)
    try:
        expected=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        if not same_state(expected,backend._pending_optimizer_state):
            raise ValueError('Pending optimizer differs from actual saved tensors')
        adam=AdamConfig(**train.reference.contract()['optimizer'])
        backend._optimizer=torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad],
            lr=1e-4,betas=(adam.beta1,adam.beta2),eps=adam.eps,weight_decay=adam.weight_decay)
        backend._optimizer.load_state_dict(backend._pending_optimizer_state)
        backend._pending_optimizer_state=None
        if not same_state(expected,backend._optimizer.state_dict()):
            raise ValueError('Materialized optimizer readback differs')
        actual=get_peft_model_state_dict(backend.model)
        if not same_state(load_file(str(checkpoint/'adapter_model.safetensors')),actual):
            raise ValueError('Native adapter readback differs')
        rng=restore_coordinator_rng(checkpoint)
        save(a.output,{'schema':'gemma-native-checkpoint-restore-v1','source_commit':a.commit,
            'repository':str(root),'model':a.model,'method':a.method,'python':sys.executable,
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'checkpoint':str(checkpoint),
            'checkpoint_files':{str(f.relative_to(checkpoint)):file_identity(f) for f in checkpoint.rglob('*') if f.is_file()},
            'optimizer_state_count':len(backend._optimizer.state_dict()['state']),
            'adapter_tensor_count':len(actual),'coordinator_rng':rng,'optimizer_updates':0,
            'rollout_worker_rng_serialized':False})
    finally:
        backend.shutdown()


if __name__=='__main__':
    main()
