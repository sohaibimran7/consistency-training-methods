"""Verify the multimodal namespace translation preserves every LoRA tensor."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.root/'source/rmct-plan.json').read_text())
raw=Path(plan['checkpoint'])/'adapter_model.safetensors'
assert hashlib.sha256(raw.read_bytes()).hexdigest()=='214c97e0b1b4af027495797a3cddd7f3e1fcb1a600a0968904aa7d125429f921'
src=load_file(str(raw));dst=load_file(str(a.adapter/'adapter_model.safetensors'))
assert len(src)==len(dst)
for k,v in src.items():
    translated=k.replace('base_model.model.model.layers.','base_model.model.model.language_model.layers.',1)
    assert translated in dst and torch.equal(v,dst[translated]),k
assert any(torch.count_nonzero(v).item() for k,v in dst.items() if 'lora_B' in k)
receipt={'raw_path':str(raw),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'adapter':str(a.adapter),'translated_sha256':hashlib.sha256((a.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),'tensor_count':len(src),'all_tensors_identical_after_namespace_translation':True,'nonzero_lora_B':True,'limitation':'tensor audit; runtime activation is checked from logged model outputs'}
(a.root/'adapter-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
