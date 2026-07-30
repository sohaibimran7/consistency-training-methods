from __future__ import annotations

from types import SimpleNamespace

import pytest

import ctm_data.inspect as inspect_helpers
from ctm_data.inspect import build_inspect_task, build_memory_dataset


class FakeMemoryDataset:
    def __init__(self, *, samples, name, location, shuffled):
        self.samples = samples
        self.name = name
        self.location = location
        self.shuffled = shuffled


class FakeTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_memory_dataset_only_imports_inspect_when_called(monkeypatch):
    imports = []

    def fake_import(name):
        imports.append(name)
        assert name == "inspect_ai.dataset"
        return SimpleNamespace(MemoryDataset=FakeMemoryDataset)

    monkeypatch.setattr(inspect_helpers, "import_module", fake_import)
    samples = [object(), object()]

    dataset = build_memory_dataset(samples, name="rows", location="local.jsonl", shuffled=True)

    assert imports == ["inspect_ai.dataset"]
    assert dataset.samples == samples
    assert dataset.samples is not samples
    assert dataset.name == "rows"
    assert dataset.location == "local.jsonl"
    assert dataset.shuffled is True


def test_build_inspect_task_passes_caller_policy_through_unchanged(monkeypatch):
    imports = []

    def fake_import(name):
        imports.append(name)
        modules = {
            "inspect_ai.dataset": SimpleNamespace(MemoryDataset=FakeMemoryDataset),
            "inspect_ai": SimpleNamespace(Task=FakeTask),
        }
        return modules[name]

    monkeypatch.setattr(inspect_helpers, "import_module", fake_import)
    samples = [object()]
    scorer = object()
    solver = object()
    metadata = {"condition": "caller-owned"}

    task = build_inspect_task(
        samples,
        scorer=scorer,
        solver=solver,
        task_name="explicit-task",
        dataset_name="explicit-dataset",
        dataset_location="source.jsonl",
        task_options={"metadata": metadata, "tags": ["caller-tag"], "version": "v2"},
    )

    assert imports == ["inspect_ai.dataset", "inspect_ai"]
    assert task.kwargs == {
        "dataset": task.kwargs["dataset"],
        "scorer": scorer,
        "solver": solver,
        "name": "explicit-task",
        "metadata": metadata,
        "tags": ["caller-tag"],
        "version": "v2",
    }
    assert task.kwargs["dataset"].samples == samples
    assert task.kwargs["dataset"].name == "explicit-dataset"
    assert task.kwargs["dataset"].location == "source.jsonl"
    assert task.kwargs["dataset"].shuffled is False


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [("scorer", {"scorer": None, "solver": object()}), ("solver", {"scorer": object(), "solver": None})],
)
def test_build_inspect_task_requires_explicit_scorer_and_solver(field, kwargs):
    with pytest.raises(ValueError, match=field):
        build_inspect_task([], **kwargs)


def test_task_options_cannot_replace_structural_arguments():
    with pytest.raises(ValueError, match="dataset"):
        build_inspect_task(
            [],
            scorer=object(),
            solver=object(),
            task_options={"dataset": object()},
        )


@pytest.mark.parametrize(
    ("samples", "error"),
    [
        ({"not": "samples"}, "iterable of Inspect Sample"),
        ("not samples", "iterable of Inspect Sample"),
        (None, "iterable of Inspect Sample"),
    ],
)
def test_memory_dataset_rejects_ambiguous_sample_containers(samples, error):
    with pytest.raises(TypeError, match=error):
        build_memory_dataset(samples)  # type: ignore[arg-type]
