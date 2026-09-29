"""Pin final checkpoint identities and construct the 16-worker submission."""
import argparse
from pathlib import Path
from full_suite import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
q=read(root/'source/qwen-final.json');g=read(root/'source/gemma-final.json')
models={};workers=[]
for method in ['base','act','attct','mlpct','bct','opct','rmct']:
    cp=q['checkpoints'][method];key='qwen-'+method
    models[key]={'family':'qwen','snapshot':'/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a','checkpoint':cp['path'] if cp else None,'step':cp['step'] if cp else 0,'weights_sha256':cp['files']['adapter_model.safetensors'] if cp else None,'config_sha256':cp['files']['adapter_config.json'] if cp else None}
    workers += [{'key':key,'rank':r,'shards':2} for r in range(2)]
for method in ['base','rmct']:
    cp=g['checkpoint'] if method=='rmct' else None;key='gemma-'+method
    models[key]={'family':'gemma','snapshot':g['snapshot'],'checkpoint':cp['path'] if cp else None,'step':192 if cp else 0,'weights_sha256':cp['files']['adapter_model.safetensors']['sha256'] if cp else None,'config_sha256':cp['adapter_config']['sha256'] if cp else None}
    workers.append({'key':key,'rank':0,'shards':1})
write(root/'launch-plan.json',{'models':models,'workers':workers,'manifest_sha256':sha(root/'manifest.json'),'code_sha256':{n:sha(root/n) for n in ['full_suite.py','full_worker.py','full.sbatch']},'target_count':300,'generations_per_model':1200,'total_generations':10800,'output_cap':65536,'context':131072})
