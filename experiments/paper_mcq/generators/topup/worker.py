"""Generate missing questions only, using the existing attested RMCT adapter."""
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
cache=Path(tempfile.mkdtemp(prefix=f'rmct-topup-{index}-'))
for key in ('TMPDIR','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
    p=cache/key;p.mkdir();os.environ[key]=str(p)
sys.path.insert(0,str(REPO));sys.path.insert(0,str(RUNTIME))
os.environ['CTM_MEETING_EVAL_ROOT']=str(RUNTIME)
from evaluate import verify_runtime
verify_runtime()
model=data['model'].removeprefix('vllm/')
base,adapter=model.split(':',1)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
for sample in data['samples']:
    encoded=tokenizer.apply_chat_template(sample['input'],tokenize=True,add_generation_prompt=True)
    ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
    if ids and isinstance(ids[0],list):ids=ids[0]
    assert len(ids)+data['generation_config']['max_tokens']<=data['model_args']['max_model_len']
args=dict(data['model_args'],provider='vllm')
command=[sys.executable,str(REPO/'scripts/run_evals.py'),'--task-factory','topup_tasks:tasks',
    '--task-args',json.dumps({'shard':str(shard)}),'--generation-config',json.dumps(data['generation_config']),
    '--local-checkpoint',adapter,'--base-model',base,'--model-args',json.dumps(args),
    '--log-dir',str(ROOT/'raw'/f'shard-{index:02d}'),'--max-tasks','1',
    '--isolate-tasks','--persistent-vllm-server','--yes']
(ROOT/f'command-{index:02d}.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,cwd=REPO,check=True)
