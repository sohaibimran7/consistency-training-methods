"""Immutable terminal-method campaign using the frozen RMCT176 task protocol."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(os.environ["CTM_MEETING_EVAL_ROOT"]).resolve()
REPO = ROOT / "repo"
TRAIN = Path('/scratch/a5v/sohaib.a5v/ctm/methods-two-bias-convergence-20260910-prefix-v2/runs')
STEPS = {'act': 992, 'attct': 880, 'mlpct': 736}
sys.path.insert(0, str(REPO))
from infra.isambard import run_qwen35_rmct_convergence_r4_two_bias_evals as ref
from ctm.evals.qwen35_vllm_scope import SCOPE


def read(path):
    return json.loads(path.read_text())


def verify_runtime():
    receipt = read(ROOT / 'runtime-patch.json')
    for relative, digest in receipt['overlay_sha256'].items():
        if ref._sha256_file(ROOT / relative) != digest:
            raise ValueError(f'patched source changed after deployment: {relative}')
    module = ROOT / 'vendor/vllm/lora/layers/column_parallel_linear.py'
    if ref._sha256_file(module) != receipt['patched_vllm_module_sha256']:
        raise ValueError('isolated vLLM sparse-loader patch changed')
    spec = importlib.util.find_spec('vllm')
    if spec is None or Path(spec.origin).resolve().parent != ROOT / 'vendor/vllm':
        raise ValueError('Python did not select the isolated, patched vLLM package')
    return ref._identity(ROOT / 'runtime-patch.json', label='isolated runtime repair')


def prepare(method):
    patch = verify_runtime()
    state_path = TRAIN / method / 'state.json'
    state = read(state_path)
    if state['method'] != method or state['convergence']['step'] != STEPS[method] or state['convergence']['decision'] != 'plateau':
        raise ValueError('terminal training state differs from the authorized checkpoint')
    checkpoint = (state_path.parent / state['checkpoint']).resolve()
    for name, digest in state['checkpoint_files'].items():
        if ref._sha256_file(checkpoint / name) != digest:
            raise ValueError(f'checkpoint payload changed: {name}')
    paths = ref._launch_paths(ROOT / method)
    source = REPO / 'artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json'
    ref.materialize_deployment_manifest(source, artifact_root=source.parent, output=paths.deployment_manifest)
    parity = REPO / 'artifacts/stage1-iid-diagnostic-none-20260801-source'
    launch = {
        'schema': 'ctm-terminal-method-rmct176-protocol-v2',
        'condition': f'{method}-terminal-step{STEPS[method]}-compat-r2',
        'method': method, 'optimizer_step': STEPS[method],
        'protocol_reference': 'rmct-convergence-r4-s011-two-bias-v1-r005',
        'adapter_scope': SCOPE if method in ('attct', 'mlpct') else None,
        'runtime_amendment': patch,
        'training_state': ref._identity(state_path, label='terminal training state'),
        'checkpoint': {'path': str(checkpoint), **{
            label: ref._identity(checkpoint / name, label=label)
            for label, name in [('adapter_model', 'adapter_model.safetensors'), ('adapter_config', 'adapter_config.json'), ('manifest', 'manifest.json')]
        }},
        'critical_sources': ref._critical_source_identities(),
        'coordinator': ref._identity(Path(__file__), label='method coordinator'),
        'deployment_manifest': ref._identity(paths.deployment_manifest, label='deployed tasks'),
        'parity_data': ref._parity_data(parity / 'train-eval-n200.jsonl', parity / 'manifest.json'),
        'runtime': {'profile': 'vllm', 'model_args': ref.VLLM_MODEL_ARGS,
                    'generation_config': ref.GENERATION_CONFIG,
                    'sampler': ref._vllm_sampler_runtime(vllm_device_tokens=os.environ['CUDA_VISIBLE_DEVICES'].split(','))},
        'sample_limit': 100, 'task_count': 21,
    }
    ref._write_immutable_json(paths.contract, launch, label='method launch contract')
    runtime = ref.ensure_attested_vllm_runtime(launch=launch, paths=paths, python=sys.executable)
    receipt = {'schema': 'ctm-terminal-method-evaluation-receipt-v2',
               'condition': launch['condition'], 'method': method,
               'launch_contract': ref._identity(paths.contract, label='launch'),
               'runtime_amendment': patch, 'runtime': runtime, 'sample_limit': 100,
               'generation_config': ref.GENERATION_CONFIG,
               'deployment_manifest': launch['deployment_manifest']}
    ref._write_immutable_json(paths.evaluation_receipt, receipt, label='method evaluation receipt')
    print(json.dumps({'prepared': method, 'runtime': runtime}), flush=True)


def worker(wave):
    patch = verify_runtime()
    rank = int(os.environ['SLURM_PROCID'])
    cells = [(m, i) for i in range(1, 22) for m in STEPS]
    offset = wave * 16 + rank
    if offset >= len(cells):
        return
    method, index = cells[offset]
    paths = ref._launch_paths(ROOT / method)
    launch = read(paths.contract)
    if launch['runtime_amendment'] != patch:
        raise ValueError('generation runtime differs from parity runtime')
    runtime = read(paths.runtime / 'runtime-receipt.json')
    tokens = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(tokens) != 1:
        raise ValueError('each evaluation worker must have exactly one Slurm GPU')
    cache = Path(tempfile.mkdtemp(prefix=f'ctm-methods-{method}-{index}-', dir='/tmp'))
    for name in ['TMPDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH']:
        target = cache / name
        target.mkdir()
        os.environ[name] = str(target)
    result = ref._run_group(gpu=tokens[0], task_indices=[index], launch=launch,
        runtime=runtime, paths=paths, python=sys.executable,
        launch_contract_sha256=ref._sha256_file(paths.contract),
        evaluation_receipt_sha256=ref._sha256_file(paths.evaluation_receipt), phase=f'wave-{wave}')
    print(json.dumps({'method': method, 'task': index, 'result': result}), flush=True)


def finalize():
    verify_runtime()
    for method in STEPS:
        paths = ref._launch_paths(ROOT / method)
        receipts = [ref._load_task_receipt(paths, task_index=i,
            launch_contract_sha256=ref._sha256_file(paths.contract),
            evaluation_receipt_sha256=ref._sha256_file(paths.evaluation_receipt)) for i in range(1,22)]
        if not all(receipts):
            raise ValueError(f'incomplete task receipts for {method}')
        ref._write_immutable_json(paths.completion, {'schema': 'ctm-terminal-method-completion-v2',
            'method': method, 'step': STEPS[method], 'tasks': receipts}, label='completion')


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2])
    elif sys.argv[1] == 'worker':
        worker(int(sys.argv[2]))
    elif sys.argv[1] == 'finalize':
        finalize()
    else:
        raise ValueError(sys.argv)
