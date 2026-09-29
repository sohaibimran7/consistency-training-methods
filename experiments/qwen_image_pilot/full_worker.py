"""One isolated vLLM image-suite worker; immutable checkpoints and raw logs."""
import argparse,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
from full_suite import read,write,sha,CONTEXT

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
root=a.root;plan=read(root/'launch-plan.json');entry=plan['workers'][a.index];key=entry['key']
port=18200+a.index
model=plan['models'][key];out=root/'runs'/key/f"rank-{entry['rank']}";out.mkdir(parents=True,exist_ok=True)
assert sha(root/'manifest.json')==plan['manifest_sha256']
for name,h in plan['code_sha256'].items():assert sha(root/name)==h
adapter=None
if model['checkpoint']:
    cp=Path(model['checkpoint']);assert sha(cp/'adapter_model.safetensors')==model['weights_sha256']
    assert sha(cp/'adapter_config.json')==model['config_sha256']
    if model['family']=='qwen':
        from safetensors.torch import load_file,save_file
        import torch
        raw=load_file(cp/'adapter_model.safetensors');translated={}
        for k,v in raw.items():
            assert k.startswith('base_model.model.model.layers.'),k
            translated[k.replace('base_model.model.model.layers.','base_model.model.model.language_model.layers.',1)]=v
        assert len(raw)==len(translated)
        original_translated=dict(translated)
        zero_keys=[]
        if key=='qwen-act':
            # vLLM 0.21's packed qkv/z loader dereferences a missing z entry.
            # ACT did not train z: A=B=0 is exactly the same zero update.
            config=read(Path(model['snapshot'])/'config.json')['text_config']
            z_rows=config['linear_num_value_heads']*config['linear_value_head_dim']
            for k,v in list(translated.items()):
                if k.endswith('.linear_attn.in_proj_qkv.lora_A.weight'):
                    ak=k.replace('in_proj_qkv','in_proj_z');bk=ak.replace('lora_A','lora_B')
                    assert ak not in translated and bk not in translated
                    translated[ak]=torch.zeros_like(v)
                    translated[bk]=torch.zeros((z_rows,v.shape[0]),dtype=v.dtype)
                    zero_keys.extend([ak,bk])
            assert zero_keys
        adapter=out/'adapter';adapter.mkdir(exist_ok=True)
        save_file(translated,adapter/'adapter_model.safetensors')
        adapter_config=read(cp/'adapter_config.json')
        if zero_keys:adapter_config['target_modules']=sorted(set(adapter_config['target_modules'])|{'in_proj_z'})
        write(adapter/'adapter_config.json',adapter_config)
        reread=load_file(adapter/'adapter_model.safetensors')
        assert all(torch.equal(v,reread[k]) for k,v in translated.items())
        assert all(torch.equal(v,reread[k]) for k,v in original_translated.items())
        assert all(torch.count_nonzero(reread[k]).item()==0 for k in zero_keys)
        assert any(torch.count_nonzero(v).item() for k,v in raw.items() if 'lora_B' in k)
        write(out/'adapter-check.json',{'source':str(cp),'source_sha256':model['weights_sha256'],'translated_sha256':sha(adapter/'adapter_model.safetensors'),'tensor_count':len(raw),'original_tensors_equal':True,'zero_update_padding':zero_keys})
    else:adapter=cp
command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',model['snapshot'],'--served-model-name',key if not adapter else 'base',
         '--host','127.0.0.1','--port',str(port),'--max-model-len',str(CONTEXT),'--max-num-seqs','8','--max-num-batched-tokens','8192',
         '--gpu-memory-utilization','0.90','--limit-mm-per-prompt',json.dumps({'image':6,'video':0,**({'audio':0} if model['family']=='gemma' else {})}),'--generation-config','auto']
if model['family']=='qwen':command+=['--additional-config','{"gdn_prefill_backend":"triton"}','--reasoning-parser','qwen3']
if adapter:
    rank=read(adapter/'adapter_config.json')['r']
    command+=['--enable-lora','--max-lora-rank',str(rank),'--lora-modules',key+'='+str(adapter)]
write(out/'server-command.json',command)
with (out/'server.log').open('w') as log:
    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(360):
            if server.poll() is not None:raise RuntimeError('vLLM exited; see server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:assert r.status==200
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=2) as r:
                    served=json.load(r)
                assert key in [m['id'] for m in served['data']]
                write(out/'served-models.json',served)
                break
            except Exception:time.sleep(5)
        else:raise RuntimeError('vLLM startup timeout')
        # Full-suite calls have an explicit 65,536 output allowance. vLLM rejects
        # over-context inputs instead of silently reducing that allowance.
        client_python='/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python'
        subprocess.run([client_python,str(root/'full_suite.py'),'evaluate','--root',str(root),'--key',key,'--rank',str(entry['rank']),'--shards',str(entry['shards']),'--url',f'http://127.0.0.1:{port}/v1'],check=True)
    finally:
        server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill();server.wait()
