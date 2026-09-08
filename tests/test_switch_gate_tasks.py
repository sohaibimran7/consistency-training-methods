import functools
import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Dataset(list):
    def filter(self, predicate, name=None):
        del name
        return _Dataset(sample for sample in self if predicate(sample))


class _Task:
    def __init__(self, dataset, *, scorer=None, metadata=None):
        self.dataset = _Dataset(dataset)
        self.scorer = list(scorer or [])
        self.metadata = metadata
        self.registry_name = None
        self.task_args = None


def _decorator(function=None, *, name=None, **_attributes):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            result.registry_name = name or fn.__name__
            result.task_args = dict(bound.arguments)
            return result

        return wrapped

    return decorate(function) if function is not None else decorate


def _load_samples(path: str | Path, variant: str) -> _Dataset:
    samples = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = {
            "source_dataset": row["source_dataset"],
            "prompt_style": row["prompt_style"],
            "variant": variant,
        }
        if variant == "biased":
            metadata["bias_type"] = row["bias_type"]
        samples.append(SimpleNamespace(id=row["question_id"], metadata=metadata))
    return _Dataset(samples)


@pytest.fixture
def task_module(monkeypatch):
    calls = {"unbiased": [], "biased": []}

    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.Task = _Task
    inspect_module.task = _decorator

    mcq_package = types.ModuleType("mcq_bias")
    mcq_package.__path__ = []
    mcq_tasks = types.ModuleType("mcq_bias.tasks")

    def unbiased_task_from_frozen(path, metadata=None):
        calls["unbiased"].append({"path": str(path), "metadata": metadata})
        return _Task(_load_samples(path, "unbiased"), scorer=["mcq", "options"], metadata=metadata)

    def task_from_frozen(path, **kwargs):
        calls["biased"].append({"path": str(path), **kwargs})
        scorers = ["mcq", "options"]
        if kwargs.get("include_bias_acknowledged", True):
            scorers.append("bias_acknowledged")
        if kwargs.get("unbiased_log"):
            scorers.append("switch")
        return _Task(_load_samples(path, "biased"), scorer=scorers, metadata=kwargs.get("metadata"))

    mcq_tasks.unbiased_task_from_frozen = unbiased_task_from_frozen
    mcq_tasks.task_from_frozen = task_from_frozen
    mcq_package.tasks = mcq_tasks
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "mcq_bias", mcq_package)
    monkeypatch.setitem(sys.modules, "mcq_bias.tasks", mcq_tasks)
    sys.modules.pop("experiments.switch_gate.tasks", None)
    module = importlib.import_module("experiments.switch_gate.tasks")
    yield module, calls
    sys.modules.pop("experiments.switch_gate.tasks", None)


def _pair_row(dataset: str, question_id: str, bias_type: str = "wrong_argument") -> dict:
    return {
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": "clean"}],
        "biased_messages": [{"role": "user", "content": "biased"}],
        "bias_type": bias_type,
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "frozen bias",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return payload


def _write_training_manifest(tmp_path: Path) -> Path:
    rows = [_pair_row("logiqa", "l1"), _pair_row("hellaswag", "h1")]
    frozen = tmp_path / "screen.jsonl"
    payload = _write_jsonl(frozen, rows)
    manifest = {
        "schema_version": 1,
        "kind": "switch_gate_split_manifest",
        "screen": {
            "path": str(frozen),
            "row_count": 2,
            "counts_by_dataset": {"logiqa": 1, "hellaswag": 1},
            "question_ids": ["l1", "h1"],
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "confirmation": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_training_factory_orders_baselines_and_preserves_pairing_headers(tmp_path, task_module):
    module, calls = task_module
    manifest = _write_training_manifest(tmp_path)

    tasks = module.training_tasks(
        manifest=str(manifest),
        split="screen",
        unbiased_log="logs/switch-gate/screen",
    )

    assert [task.registry_name for task in tasks] == [
        "switch_gate_unbiased",
        "switch_gate_unbiased",
        "switch_gate_biased",
        "switch_gate_biased",
    ]
    assert [task.task_args["dataset"] for task in tasks] == ["logiqa", "hellaswag", "logiqa", "hellaswag"]
    assert all(task.task_args["prompt_style"] == "none" for task in tasks)
    assert all(task.dataset for task in tasks)
    for task in tasks:
        assert all(sample.metadata["source_dataset"] == task.task_args["dataset"] for sample in task.dataset)
        assert all(sample.metadata["prompt_style"] == task.task_args["prompt_style"] for sample in task.dataset)

    assert len(calls["unbiased"]) == 2
    assert len(calls["biased"]) == 2
    assert all(call["unbiased_log"] == "logs/switch-gate/screen" for call in calls["biased"])
    assert all(call["include_bias_acknowledged"] is False for call in calls["biased"])
    assert all("bias_acknowledged" not in task.scorer for task in tasks)
    assert all("switch" in task.scorer for task in tasks[2:])

    # These are the exact fields used by mcq_bias's directory resolver (with
    # the eval model checked separately by Inspect at runtime).
    for unbiased_task in tasks[:2]:
        assert "unbiased" in unbiased_task.registry_name
        assert set(unbiased_task.task_args) >= {"dataset", "prompt_style"}


def test_hle_factory_uses_explicit_files_and_omits_model_grader(tmp_path, task_module):
    module, calls = task_module
    unbiased = tmp_path / "hle-unbiased.jsonl"
    _write_jsonl(
        unbiased,
        [
            {
                "question_id": "q1",
                "source_dataset": "hle-text-mc",
                "prompt_style": "none",
                "unbiased_messages": [{"role": "user", "content": "clean"}],
                "ground_truth": "A",
            }
        ],
    )
    suggested = tmp_path / "hle-suggested.jsonl"
    wrong_argument = tmp_path / "hle-wrong-argument.jsonl"
    _write_jsonl(suggested, [_pair_row("hle-text-mc", "q1", "suggested_answer")])
    _write_jsonl(wrong_argument, [_pair_row("hle-text-mc", "q1", "wrong_argument")])

    tasks = module.hle_tasks(
        unbiased_file=str(unbiased),
        bias_files={"suggested_answer": str(suggested), "wrong_argument": str(wrong_argument)},
        unbiased_log="logs/switch-gate/hle",
    )

    assert [task.registry_name for task in tasks] == [
        "switch_gate_unbiased",
        "switch_gate_biased",
        "switch_gate_biased",
    ]
    assert [task.task_args.get("bias_type") for task in tasks] == [None, "suggested_answer", "wrong_argument"]
    assert all(task.task_args["dataset"] == "hle-text-mc" for task in tasks)
    assert all(task.task_args["prompt_style"] == "none" for task in tasks)
    assert [call["path"] for call in calls["biased"]] == [str(suggested), str(wrong_argument)]
    assert all(call["include_bias_acknowledged"] is False for call in calls["biased"])
    assert all(call["unbiased_log"] == "logs/switch-gate/hle" for call in calls["biased"])
    assert all("bias_acknowledged" not in task.scorer for task in tasks)
    assert all(
        sample.metadata["source_dataset"] == task.task_args["dataset"] for task in tasks for sample in task.dataset
    )
