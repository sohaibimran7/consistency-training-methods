"""Produce verified scores only after completed generation; never launch jobs."""
import argparse
from pathlib import Path
import subprocess
from experiments.rmct_restart_20260928 import qwen_validation as v
from experiments.rmct_restart_20260928.qwen_validation_executor import read,write,check_contract


def scheduler_complete(job):
    if not str(job).isdigit():raise ValueError('Numeric Slurm job required')
    state=subprocess.check_output(['sacct','-X','-j',str(job),'--noheader','--format=State,ExitCode','-P'],text=True).strip()
    if state!='COMPLETED|0:0':raise ValueError('Generation job not successfully completed')
    return state


def produce(folder,manifest,job):
    from transformers import AutoTokenizer
    folder=Path(folder).resolve();contract=read(folder/'contract.json')
    root=Path(__file__).resolve().parents[2];check_contract(contract,root)
    state=scheduler_complete(job)
    for rank in range(4):
        claim=read(folder/f'worker-{rank}-claim.json');done=read(folder/f'worker-{rank}-complete.json')
        if claim['job_id']!=str(job) or done['count']!=150 or done['contract_sha256']!=v.sha(folder/'contract.json'):
            raise ValueError('Worker/job/contract mismatch')
    records=[read(p) for p in sorted((folder/'responses').glob('*.json'))]
    metrics=v.score(manifest,records)
    tokenizer=AutoTokenizer.from_pretrained(contract['model'],local_files_only=True)
    history=[]
    for step in range(64,contract['step'],64):
        earlier=folder.parent/str(step)
        v.verify_saved(earlier,manifest,tokenizer)
        scheduler_complete(read(earlier/'completed-job.json')['job_id'])
        history.append(read(earlier/'score.json'))
    row=dict(metrics,campaign_id=contract['campaign_id'],step=contract['step'],
             validation_sha256=v.PROMPT_SHA,settings_sha256=v.settings_sha(),
             verified=True,job_completed=True,response_count=len(records),
             contract_sha256=v.sha(folder/'contract.json'),
             response_hashes={r['sample_id']:v.sha(folder/'responses'/f"{r['sample_id']}.json") for r in records})
    decision=v.replay(history+[row],campaign_id=contract['campaign_id'])
    write(folder/'score.json',row)
    v.verify_saved(folder,manifest,tokenizer)
    write(folder/'completed-job.json',dict(job_id=str(job),scheduler_state=state))
    write(folder/'decision.json',decision)
    write(folder/'verified-complete.json',dict(score_sha256=v.sha(folder/'score.json'),
          decision_sha256=v.sha(folder/'decision.json'),contract_sha256=v.sha(folder/'contract.json')))
    return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--job',required=True)
    a=p.parse_args();print(produce(a.folder,a.manifest,a.job))
