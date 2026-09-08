"""Static contracts for the RMCT-convergence worker-parity fastpath.

These checks intentionally do not import torch/vLLM or create CUDA contexts.
The fastpath itself is the GPU-side proof; this test protects its immutable
handoff ABI and prevents a later launcher edit from quietly dropping a
production worker option.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
HELPER = ROOT / "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
GENERIC_PREFLIGHT = ROOT / "infra/vastai/preflight_qwen35_phase_shared.py"


def _module():
    spec = importlib.util.spec_from_file_location("rmct_convergence_worker_parity_test_module", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fastpath_has_an_immutable_global_sidecar_and_read_only_resume_interface(tmp_path):
    module = _module()
    args = module._parse_args(
        ["--output-dir", str(tmp_path / "evidence"), "--model-snapshot", str(tmp_path / "snapshot")]
    )

    assert args.resume is False
    assert module.WORKER_PARITY_ATTESTATION == "qwen35-rollout-worker-parity-attestation.json"
    assert module.RESULT == "result.json"
    assert module.SUCCESS == "SUCCESS"
    source = HELPER.read_text(encoding="utf-8")
    assert "qwen35_rollout_parity_bootstrap=True" in source
    assert "use --resume only for a completed receipt" in source
    assert "training marker" in source
    assert "rm -rf" not in source


def test_fastpath_binds_every_production_worker_option_including_dtype():
    module = _module()
    expected = module._expected_worker_engine_kwargs()

    assert expected == {
        "gpu_memory_utilization": 0.34,
        "dtype": "bfloat16",
        "enable_sleep_mode": True,
        "language_model_only": True,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "max_model_len": 32768,
        "gdn_prefill_backend": "triton",
        "seed": 42,
        "logprobs_mode": "processed_logprobs",
        "tensor_parallel_size": 1,
    }
    source = HELPER.read_text(encoding="utf-8")
    assert "expected_model=str(snapshot)" in source
    assert "expected_worker_engine_kwargs=expected_worker_kwargs" in source
    assert "worker_count=len(workers)" in source
    assert "adapter_version != 2" in source
    assert "count=2 * GPU_COUNT" in source


def test_generic_phase_shared_preflight_attests_an_explicit_worker_dtype_too():
    source = GENERIC_PREFLIGHT.read_text(encoding="utf-8")

    assert '"--worker-dtype"' in source
    assert '"dtype": args.worker_dtype' in source
    assert '"worker_dtype": args.worker_dtype' in source
