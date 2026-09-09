from __future__ import annotations

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

from experiments.stage2_ood_hle import materialize as prepare


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


def _row(dataset: str, question_id: str, bias_type: str | None = None) -> dict:
    row = {
        "question": f"question {question_id}",
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": f"clean {question_id}"}],
        "ground_truth": "A",
    }
    if bias_type is not None:
        row.update(
            {
                "bias_type": bias_type,
                "biased_messages": [{"role": "user", "content": f"{bias_type} {question_id}"}],
                "biased_option": "B",
                "biasing_text": f"bias {bias_type}",
            }
        )
    return row


def _write_artifact(path: Path, rows: list[dict]) -> dict:
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "question_ids": [row["question_id"] for row in rows],
    }


def _document(tmp_path: Path) -> dict:
    in_rows = [_row("logiqa", "l-1"), _row("logiqa", "l-2"), _row("hellaswag", "h-1"), _row("hellaswag", "h-2")]
    hle_rows = [_row("hle-text-mc", "hle-1")]
    in_artifacts = {"unbiased": _write_artifact(tmp_path / "in-clean.jsonl", in_rows)}
    hle_artifacts = {"unbiased": _write_artifact(tmp_path / "hle-clean.jsonl", hle_rows)}
    for bias_type in (prepare.TRAINING_BIAS, *prepare.HELDOUT_BIASES):
        in_artifacts[bias_type] = _write_artifact(
            tmp_path / f"in-{bias_type}.jsonl", [_row(row["source_dataset"], row["question_id"], bias_type) for row in in_rows]
        )
        hle_artifacts[bias_type] = _write_artifact(
            tmp_path / f"hle-{bias_type}.jsonl", [_row("hle-text-mc", "hle-1", bias_type)]
        )
    return {
        "populations": {
            "in_domain": {"artifacts": in_artifacts},
            "hle": {"artifacts": hle_artifacts, "source_dataset": "hle-text-mc"},
        },
        "regimes": {name: {} for name in prepare.REGIMES},
        "regime_order": list(prepare.REGIMES),
    }


@pytest.fixture
def task_module(monkeypatch):
    calls = {"unbiased": [], "biased": []}
    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.Task = _Task
    inspect_module.task = _decorator

    mcq_package = types.ModuleType("mcq_bias")
    mcq_package.__path__ = []
    mcq_tasks = types.ModuleType("mcq_bias.tasks")
    mcq_scorers = types.ModuleType("mcq_bias.scorers")

    def unbiased_task_from_frozen(path, metadata=None):
        calls["unbiased"].append({"path": str(path), "metadata": metadata})
        return _Task(_load_samples(path, "unbiased"), scorer=["mcq", "options"], metadata=metadata)

    def task_from_frozen(
        path,
        metadata=None,
        unbiased_log=None,
        grader_model=None,
        include_bias_acknowledged=True,
        question_ids_from=None,
        source_dataset=None,
        source_identity_digest=None,
    ):
        calls["biased"].append(
            {
                "path": str(path),
                "metadata": metadata,
                "unbiased_log": unbiased_log,
                "grader_model": grader_model,
                "include_bias_acknowledged": include_bias_acknowledged,
                "question_ids_from": question_ids_from,
                "source_dataset": source_dataset,
                "source_identity_digest": source_identity_digest,
            }
        )
        return _Task(_load_samples(path, "biased"), scorer=["mcq", "switch"], metadata=metadata)

    mcq_tasks.unbiased_task_from_frozen = unbiased_task_from_frozen
    mcq_tasks.task_from_frozen = task_from_frozen
    mcq_scorers.switch_values = lambda *_args, **_kwargs: {"towards_bias_switch": None}
    for name in ("mcq_bias_scorer", "options_considered_scorer", "bias_acknowledged_scorer", "switch_scorer"):
        setattr(mcq_scorers, name, lambda *_args, **_kwargs: None)
    mcq_package.tasks = mcq_tasks
    mcq_package.scorers = mcq_scorers
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "mcq_bias", mcq_package)
    monkeypatch.setitem(sys.modules, "mcq_bias.tasks", mcq_tasks)
    monkeypatch.setitem(sys.modules, "mcq_bias.scorers", mcq_scorers)
    sys.modules.pop("experiments.stage2_ood_hle.tasks", None)
    module = importlib.import_module("experiments.stage2_ood_hle.tasks")
    yield module, calls
    sys.modules.pop("experiments.stage2_ood_hle.tasks", None)


def test_factory_creates_fresh_three_clean_eighteen_biased_matrix(tmp_path, monkeypatch, task_module):
    module, calls = task_module
    document = _document(tmp_path)
    monkeypatch.setattr(module, "validate_manifest", lambda _manifest: document)

    tasks = module.ood_tasks(tmp_path / "manifest.json", "logs/ood-condition")

    assert len(tasks) == 21
    assert [task.registry_name for task in tasks] == [
        *("stage2_ood_unbiased",) * 3,
        *("stage2_ood_biased",) * 18,
    ]
    assert [task.task_args["regime"] for task in tasks[:3]] == [
        prepare.IID,
        prepare.IID,
        prepare.HELDOUT_DATASET,
    ]
    assert [len(task.dataset) for task in tasks[:3]] == [2, 2, 1]
    assert [task.task_args["bias_type"] for task in tasks[3:6]] == [
        prepare.TRAINING_BIAS,
        prepare.TRAINING_BIAS,
        prepare.TRAINING_BIAS,
    ]
    assert all(task.task_args["unbiased_log"] == "logs/ood-condition" for task in tasks[3:])
    assert all(call["include_bias_acknowledged"] is False for call in calls["biased"])
    assert all(call["grader_model"] is None for call in calls["biased"])
    assert len(calls["unbiased"]) == 3
    assert len(calls["biased"]) == 18

    in_domain_digest = document["populations"]["in_domain"]["artifacts"]["unbiased"]["content_sha256"]
    hle_digest = document["populations"]["hle"]["artifacts"]["unbiased"]["content_sha256"]
    for task in tasks:
        expected = in_domain_digest if task.task_args["population"] == "in_domain" else hle_digest
        assert task.task_args["source_identity_digest"] == f"stage2-ood-hle-2x2:{expected}"
    assert [task.task_args["regime"] for task in tasks[6:16]] == [prepare.HELDOUT_BIAS] * 10
    assert [task.task_args["regime"] for task in tasks[16:]] == [prepare.HELDOUT_DATASET_AND_BIAS] * 5


def test_factory_rejects_wrong_prompt_style_and_unpaired_grader(task_module):
    module, _ = task_module
    with pytest.raises(ValueError, match="prompt_style"):
        module.ood_tasks("unused", "logs", prompt_style="encourage_cot")
    with pytest.raises(ValueError, match="include_bias_acknowledged"):
        module.ood_tasks("unused", "logs", grader_model="openrouter/example")


def test_factory_installs_explicit_generic_hf_eos_only_runtime(tmp_path, monkeypatch, task_module):
    from ctm.evals import hf_eos_only

    module, _ = task_module
    document = _document(tmp_path)
    monkeypatch.setattr(module, "validate_manifest", lambda _manifest: document)
    installs = []
    monkeypatch.setattr(hf_eos_only, "install_native_hf_eos_only_sampling", lambda: installs.append(True))

    result = module.ood_tasks(
        tmp_path / "manifest.json",
        "logs/ood-condition",
        hf_eos_only_no_token_cap=True,
    )

    assert len(result) == 21
    assert installs == [True]
    with pytest.raises(ValueError, match="must be boolean"):
        module.ood_tasks("unused", "logs", hf_eos_only_no_token_cap="true")


def test_task_specs_recheck_artifact_bytes_after_manifest_validation(tmp_path, monkeypatch, task_module):
    module, _ = task_module
    document = _document(tmp_path)
    path = Path(document["populations"]["in_domain"]["artifacts"]["unbiased"]["path"])
    path.write_text("{}\n")
    monkeypatch.setattr(module, "validate_manifest", lambda _manifest: document)
    with pytest.raises(ValueError, match="bytes changed"):
        module.ood_task_specs(tmp_path / "manifest.json")
