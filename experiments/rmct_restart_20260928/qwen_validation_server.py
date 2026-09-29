"""One-GPU canonical server lifecycle for a prepared checkpoint contract."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from experiments.rmct_restart_20260928.qwen_validation_executor import read,check_contract,worker,write


def check_executable(executable,prefix):
    if executable is None or Path(executable).absolute().parent!=Path(prefix).absolute()/'bin':
        raise ValueError('vLLM executable is not from the selected environment')


def run(folder,manifest,rank,port):
    from experiments.act_repair_gate import runtime_parity as backend
    root=Path(__file__).resolve().parents[2]
    if not Path(backend.__file__).resolve().is_relative_to(root):raise ValueError('Noncanonical server helper')
    check_executable(shutil.which('vllm'),sys.prefix)
    if not 1024<=port<=65535:raise ValueError('Invalid port')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('One GPU per server required')
    folder=Path(folder).resolve();contract=read(folder/'contract.json');check_contract(contract,root)
    write(folder/f'server-{rank}-claim.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),port=port))
    caches={name:folder/f'server-{rank}-cache'/leaf for name,leaf in backend.PARALLEL_VLLM_CACHE_ENVIRONMENT}
    for path in caches.values():path.mkdir(parents=True,exist_ok=False)
    log=folder/f'server-{rank}.log'
    process=backend._start_vllm(model=contract['model'],port=port,max_model_len=98304,
        gpu_memory_utilization=.90,max_loras=1,log_path=log,enforce_eager=True,
        gdn_prefill_backend='triton',device_token=devices[0],cache_directories=caches)
    endpoint=f'http://127.0.0.1:{port}/v1'
    try:
        if backend._await_server(process,endpoint,log)!=contract['model']:raise ValueError('Wrong served model')
        backend._load_vllm_adapters(endpoint,{contract['adapter']:contract['adapter']})
        worker(folder,manifest,rank,endpoint)
    finally:backend._stop_process(process)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();run(a.folder,a.manifest,a.rank,a.port)
