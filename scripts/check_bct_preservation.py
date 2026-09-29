"""CPU-only real-tokenizer synthetic preservation probe; not historical audit."""
import hashlib,json,sys,traceback,os,socket,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctm.backends.renderers import get_renderer_and_tokenizer
from ctm.training.bct_response import decode_bct_response,verify_supervised_preservation

model=Path(sys.argv[1]);family=sys.argv[2]
receipt=dict(model=str(model),family=family,synthetic=True,model_inference=False)
helper = Path(__file__).resolve().parents[1] / 'ctm/training/bct_response.py'
receipt['helper_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
receipt.update(hostname=socket.gethostname(),pid=os.getpid(),started=time.time())
import ctm.training.bct_response as bct_module
receipt['helper_sha256']=hashlib.sha256(Path(bct_module.__file__).read_bytes()).hexdigest()
print(json.dumps({'phase':'start',**receipt}),file=sys.stderr,flush=True)
try:
    renderer,tokenizer=get_renderer_and_tokenizer(str(model),source='hf')
    prompt=[dict(role='user',content='PROMPT_SENTINEL')]
    prefix=renderer.build_generation_prompt(prompt)
    if family=='qwen':raw='REASONING_SENTINEL\n</think>\n\nFINAL_SENTINEL<|im_end|>'
    elif family=='gptoss':raw='<|channel|>analysis<|message|>REASONING_SENTINEL<|end|><|start|>assistant<|channel|>final<|message|>FINAL_SENTINEL<|return|>'
    elif family=='gemma':raw='<|channel>thought\nREASONING_SENTINEL\n<channel|>FINAL_SENTINEL<turn|>'
    else:raise ValueError('Family-specific native completion protocol must be inspected first')
    tokens=tokenizer.encode(raw,add_special_tokens=False)
    assistant=decode_bct_response(renderer,tokenizer,tokens,prompt=prefix)
    full,weights=renderer.build_supervised_example([*prompt,assistant])
    supervised=tokenizer.decode([t for t,w in zip(full.to_ints(),weights.tolist(),strict=True) if w>0],skip_special_tokens=False)
    receipt.update(assistant=assistant,supervised_text=supervised,prompt_text=tokenizer.decode(prefix.to_ints()),
                   full_supervised_rendering=tokenizer.decode(full.to_ints(),skip_special_tokens=False),
                   token_weights=[dict(id=t,text=tokenizer.decode([t],skip_special_tokens=False),weight=float(w))
                                  for t,w in zip(full.to_ints(),weights.tolist(),strict=True)])
    verify_supervised_preservation(renderer,tokenizer,[*prompt,assistant])
    assert 'REASONING_SENTINEL' in supervised and 'FINAL_SENTINEL' in supervised and 'PROMPT_SENTINEL' not in supervised
    receipt.update(passed=True,tokenizer_class=str(type(tokenizer)),renderer_class=str(type(renderer)),
                   assistant=assistant,prompt_text=tokenizer.decode(prefix.to_ints()),supervised_text=supervised)
except Exception as e:receipt.update(passed=False,error=repr(e),traceback=traceback.format_exc())
receipt['asset_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()
                         if p.name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','processor_config.json')}
receipt['finished']=time.time()
print(json.dumps(receipt),flush=True)

sys.exit(0 if receipt.get('passed') else 1)
