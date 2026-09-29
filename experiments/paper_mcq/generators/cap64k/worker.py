"""Use the original frozen evaluator and attested adapter, with a larger budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
RUNTIME=Path('/scratch/a5v/sohaib.a5v/ctm/methods-three-terminal-eval-20260913-compat-r2')
REPO=RUNTIME/'repo'
index=int(os.environ['SLURM_PROCID'])
shard=ROOT/f'shard-{index:02d}.json'
data=json.loads(shard.read_text())
cache=Path(tempfile.mkdtemp(prefix=f'cap64k-{index}-'))
for key in ('TMPDIR','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
    p=cache/key;p.mkdir();os.environ[key]=str(p)
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(RUNTIME))
os.environ['CTM_MEETING_EVAL_ROOT']=str(RUNTIME)
from evaluate import verify_runtime
verify_runtime()
from transformers import AutoTokenizer
snapshot='/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
for sample in data['samples']:
    messages=sample['input']
    assert isinstance(messages,list)
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
    n=len(ids)
    assert n+65536<=98304,(sample['id'],n)
model=data['model'].removeprefix('vllm/')
args=dict(data['model_args'])
command=[sys.executable,str(REPO/'scripts/run_evals.py'),'--task-factory','retry_tasks:tasks',
    '--task-args',json.dumps({'shard':str(shard)}),'--generation-config',json.dumps(data['generation_config']),
    '--log-dir',str(ROOT/'raw'/f'shard-{index:02d}'),'--max-tasks','1',
    '--isolate-tasks','--persistent-vllm-server','--yes']
if ':' in model:
    base,adapter=model.split(':',1)
    args['provider']='vllm'
    command+=['--local-checkpoint',adapter,'--base-model',base]
else:
    command+=['--model',data['model']]
command+=['--model-args',json.dumps(args)]
(ROOT/f'command-{index:02d}.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,cwd=REPO,check=True)
