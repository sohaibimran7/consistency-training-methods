"""GPU uniqueness is about CUDA UUIDs, not process-local device numbering."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_PATH = Path(__file__).resolve().parents[1] / "infra/isambard/gemma_gpu_binding.py"
_SPEC = importlib.util.spec_from_file_location("gemma_gpu_binding", _PATH)
binding = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(binding)


def records():
    return [{"schema": binding.SCHEMA, "job_id": "123", "step_id": "0",
             "rank": rank, "local_rank": rank % 4, "hostname": f"node-{rank // 4}",
             "gpu_uuid": f"GPU-{rank}", "visible_device_count": 1,
             "cuda_visible_devices": "0"} for rank in range(16)]


def test_local_device_zero_is_valid_when_actual_uuids_are_distinct():
    result = binding.validate_bindings(records())
    assert result["status"] == "passed"
    assert result["gpu_count"] == 16
    assert result["node_count"] == 4


def test_collocated_workers_fail_despite_one_visible_device_each():
    inputs = records()
    for item in inputs:
        item["gpu_uuid"] = f"GPU-node-{item['rank'] // 4}"
    with pytest.raises(ValueError, match="share a GPU UUID"):
        binding.validate_bindings(inputs)


@pytest.mark.parametrize("change,match", [
    (lambda items: items.pop(), "exactly 16"),
    (lambda items: items[1].update(rank=0), "exactly 0 through 15"),
    (lambda items: items[0].update(visible_device_count=4), "exactly one"),
    (lambda items: items[0].update(job_id="124"), "one Slurm job step"),
    (lambda items: items[0].update(hostname="unexpected"), "four workers"),
    (lambda items: items[0].update(local_rank=1), "local ranks"),
    (lambda items: items[0].update(gpu_uuid=""), "must be present"),
])
def test_incomplete_or_inconsistent_mapping_fails(change, match):
    inputs = records()
    change(inputs)
    with pytest.raises(ValueError, match=match):
        binding.validate_bindings(inputs)


def test_receipt_publication_never_overwrites_evidence(tmp_path):
    target = tmp_path / "rank-00.json"
    binding._write_exclusive_json(target, {"first": True})
    with pytest.raises(FileExistsError):
        binding._write_exclusive_json(target, {"second": True})
    assert json.loads(target.read_text()) == {"first": True}
    assert list(tmp_path.iterdir()) == [target]


def test_worker_attests_before_generation():
    worker = (_PATH.parent / "run_gemma4_12b_base_two_bias_evals_16gpu_worker.sh").read_text()
    assert worker.index("gemma_gpu_binding.py") < worker.index('exec "$python_bin" "$launcher" worker')


def test_same_identity_probe_is_used_in_smoke_and_barrier(monkeypatch):
    cuda = SimpleNamespace(device_count=lambda: 1,
                           get_device_properties=lambda index: SimpleNamespace(uuid="GPU-test", name="H100"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert binding.local_cuda_identity()["gpu_uuid"] == "GPU-test"
    cuda.get_device_properties = lambda index: SimpleNamespace(name="H100")
    with pytest.raises(ValueError, match="did not report"):
        binding.local_cuda_identity()
    cuda.device_count = lambda: 4
    with pytest.raises(ValueError, match="instead of one"):
        binding.local_cuda_identity()
    smoke = (_PATH.parent / "run_gemma4_12b_base_two_bias_smoke.sbatch").read_text()
    assert smoke.index("--probe-single") < smoke.index('"$python_bin" "$launcher" discarded-smoke')
