from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import sys
import types
from urllib.parse import quote
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage1_iid_diagnostic import prepare
from experiments.stage1_iid_diagnostic import run_vast_matrix
from experiments.stage1_iid_diagnostic import analyze as diagnostic_analysis
from experiments.stage1_iid_diagnostic import grade_luna
from experiments.stage1_iid_diagnostic import plot as diagnostic_plot
from experiments.stage1_iid_diagnostic import raw_preflight


def _row(index: int, *, prompt_style: str = prepare.PROMPT_STYLE) -> dict:
    dataset = prepare.DATASETS[index % 2]
    return {
        "question": f"question {index}",
        "question_id": f"q-{index:04d}",
        "source_dataset": dataset,
        "prompt_style": prompt_style,
        "unbiased_messages": [{"role": "user", "content": f"clean {index}"}],
        "biased_messages": [{"role": "user", "content": f"biased {index}"}],
        "bias_type": prepare.BIAS_TYPE,
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": f"wrong argument {index}",
    }


def _write_source(path: Path, rows: list[dict] | None = None) -> tuple[list[dict], str]:
    values = rows if rows is not None else [_row(index) for index in range(prepare.SOURCE_ROWS)]
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in values)
    path.write_bytes(payload)
    return values, hashlib.sha256(payload).hexdigest()


def test_prepare_freezes_exact_offsets_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    rows, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"

    manifest = prepare.prepare_iid_diagnostic(
        source,
        output,
        expected_source_sha256=source_hash,
    )

    train = manifest["splits"]["train_eval"]
    heldout = manifest["splits"]["heldout_in_domain"]
    assert train["question_ids"] == [row["question_id"] for row in rows[:200]]
    assert heldout["question_ids"] == [row["question_id"] for row in rows[2048:2248]]
    assert train["counts_by_dataset"] == {"logiqa": 100, "hellaswag": 100}
    assert heldout["counts_by_dataset"] == {"logiqa": 100, "hellaswag": 100}
    assert set(train["question_ids"]).isdisjoint(heldout["question_ids"])
    assert manifest["rmct_first64"]["question_ids"] == [row["question_id"] for row in rows[:64]]
    assert manifest["rmct_first64"]["counts_by_dataset"] == {"logiqa": 32, "hellaswag": 32}
    assert Path(train["path"]).read_text().splitlines()[0] == source.read_text().splitlines()[0]

    checked = prepare.validate_manifest(
        output / prepare.DEFAULT_MANIFEST_FILENAME,
        expected_source_sha256=source_hash,
        verify_source=True,
    )
    assert checked == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "question_id": rows[0]["question_id"]}), "duplicate"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "prompt_style": "none"}), "prompt_style"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "bias_type": "suggested_answer"}), "bias_type"),
    ],
)
def test_prepare_rejects_noncanonical_source_rows(tmp_path, mutation, match):
    rows = [_row(index) for index in range(prepare.SOURCE_ROWS)]
    mutation(rows)
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source, rows)
    with pytest.raises(ValueError, match=match):
        prepare.prepare_iid_diagnostic(source, tmp_path / "out", expected_source_sha256=source_hash)


def test_manifest_detects_split_tampering(tmp_path):
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    manifest = prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    train_path = Path(manifest["splits"]["train_eval"]["path"])
    train_path.write_bytes(train_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        prepare.validate_manifest(
            output / prepare.DEFAULT_MANIFEST_FILENAME,
            expected_source_sha256=source_hash,
        )


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
    mcq_scorers = types.ModuleType("mcq_bias.scorers")
    mcq_parsers = types.ModuleType("mcq_bias.parsers")
    mcq_scorers.switch_values = lambda *args, **kwargs: {"towards_bias_switch": None}
    mcq_scorers.matches_bias = lambda *_args, **_kwargs: 0.0
    mcq_parsers.parse_answer = lambda *_args, **_kwargs: None

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
        kwargs = {
            "metadata": metadata,
            "unbiased_log": unbiased_log,
            "grader_model": grader_model,
            "include_bias_acknowledged": include_bias_acknowledged,
            "question_ids_from": question_ids_from,
            "source_dataset": source_dataset,
            "source_identity_digest": source_identity_digest,
        }
        calls["biased"].append({"path": str(path), **kwargs})
        scorers = ["mcq", "options", "switch"]
        if kwargs.get("include_bias_acknowledged"):
            scorers.append("bias_acknowledged")
        return _Task(_load_samples(path, "biased"), scorer=scorers, metadata=kwargs.get("metadata"))

    mcq_tasks.unbiased_task_from_frozen = unbiased_task_from_frozen
    mcq_tasks.task_from_frozen = task_from_frozen
    mcq_package.tasks = mcq_tasks
    mcq_package.scorers = mcq_scorers
    mcq_package.parsers = mcq_parsers
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "mcq_bias", mcq_package)
    monkeypatch.setitem(sys.modules, "mcq_bias.scorers", mcq_scorers)
    monkeypatch.setitem(sys.modules, "mcq_bias.parsers", mcq_parsers)
    monkeypatch.setitem(sys.modules, "mcq_bias.tasks", mcq_tasks)
    sys.modules.pop("experiments.stage1_iid_diagnostic.tasks", None)
    module = importlib.import_module("experiments.stage1_iid_diagnostic.tasks")
    yield module, calls
    sys.modules.pop("experiments.stage1_iid_diagnostic.tasks", None)


def test_task_factory_orders_clean_then_biased_and_grader_is_opt_in(tmp_path, monkeypatch, task_module):
    module, calls = task_module
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)

    tasks = module.diagnostic_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        split="heldout_in_domain",
        unbiased_log="logs/stage1-iid/heldout",
    )

    assert [task.registry_name for task in tasks] == [
        "stage1_iid_unbiased",
        "stage1_iid_unbiased",
        "stage1_iid_biased",
        "stage1_iid_biased",
    ]
    assert [task.task_args["dataset"] for task in tasks] == ["logiqa", "hellaswag", "logiqa", "hellaswag"]
    assert all(task.task_args["prompt_style"] == "encourage_cot" for task in tasks)
    assert all(len(task.dataset) == 100 for task in tasks)
    assert all(call["include_bias_acknowledged"] is False for call in calls["biased"])
    assert all(call["grader_model"] is None for call in calls["biased"])
    assert all(len(call["question_ids_from"]) == 100 for call in calls["biased"])
    assert [call["metadata"]["source_dataset"] for call in calls["biased"]] == ["logiqa", "hellaswag"]
    assert all(call["metadata"]["source_identity_digest"] == source_hash for call in calls["biased"])
    assert [call["source_dataset"] for call in calls["biased"]] == ["logiqa", "hellaswag"]
    assert all(call["source_identity_digest"] == source_hash for call in calls["biased"])
    assert all(len(task.task_args["question_ids_from"]) == 100 for task in tasks)
    assert set(tasks[0].task_args["question_ids_from"]).isdisjoint(tasks[1].task_args["question_ids_from"])
    assert all("bias_acknowledged" not in task.scorer for task in tasks)
    assert all("switch" in task.scorer for task in tasks[2:])

    calls["biased"].clear()
    graded = module.diagnostic_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        split="train_eval",
        unbiased_log="logs/stage1-iid/train",
        include_bias_acknowledged=True,
        grader_model="openrouter/example/grader",
    )
    assert all(call["include_bias_acknowledged"] is True for call in calls["biased"])
    assert all(call["grader_model"] == "openrouter/example/grader" for call in calls["biased"])
    assert all("bias_acknowledged" in task.scorer for task in graded[2:])


def test_task_factory_rejects_wrong_style_and_unpaired_grader(task_module):
    module, _ = task_module
    with pytest.raises(ValueError, match="prompt_style"):
        module.diagnostic_tasks("unused", "train_eval", "logs", prompt_style="none")
    with pytest.raises(ValueError, match="include_bias_acknowledged"):
        module.diagnostic_tasks("unused", "train_eval", "logs", grader_model="grader")


def test_biased_only_factory_uses_verified_alternate_prompt_file(tmp_path, monkeypatch, task_module):
    module, calls = task_module
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    manifest = prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)
    alternate = Path(manifest["splits"]["train_eval"]["path"])

    tasks = module.diagnostic_biased_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        split="train_eval",
        unbiased_log="logs/stage1-iid/original/train_eval",
        variant_file=alternate,
    )

    assert len(tasks) == 2
    assert [task.registry_name for task in tasks] == ["stage1_iid_biased", "stage1_iid_biased"]
    assert all(len(task.dataset) == 100 for task in tasks)
    assert [call["path"] for call in calls["biased"]] == [str(alternate), str(alternate)]
    assert all(call["metadata"]["variant_file"] == str(alternate.resolve()) for call in calls["biased"])

    missing = tmp_path / "missing-rows.jsonl"
    missing.write_text(alternate.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="missing"):
        module.diagnostic_biased_tasks(
            manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
            split="train_eval",
            unbiased_log="logs/stage1-iid/original/train_eval",
            variant_file=missing,
        )


def test_alternate_prompt_file_is_restricted_to_exact_manifest_ids(tmp_path, monkeypatch, task_module):
    """A full 3,000-row alternate rendering must not expand the 200-row gate."""

    module, _ = task_module
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)

    tasks = module.diagnostic_biased_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        split="train_eval",
        unbiased_log="logs/stage1-iid/original/train_eval",
        variant_file=source,
    )

    assert [len(task.dataset) for task in tasks] == [100, 100]
    requested = [set(task.task_args["question_ids_from"]) for task in tasks]
    observed = [{str(sample.id) for sample in task.dataset} for task in tasks]
    assert observed == requested


def test_biased_task_uses_stronger_source_identity_only_when_upstream_supports_it(
    tmp_path, monkeypatch, task_module
):
    """The repository-pinned upstream task lacks the newer scorer kwargs."""

    module, calls = task_module
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    manifest = prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)

    def legacy_task_from_frozen(
        path,
        metadata=None,
        unbiased_log=None,
        grader_model=None,
        include_bias_acknowledged=True,
        question_ids_from=None,
    ):
        calls["biased"].append(
            {
                "path": str(path),
                "metadata": metadata,
                "unbiased_log": unbiased_log,
                "grader_model": grader_model,
                "include_bias_acknowledged": include_bias_acknowledged,
                "question_ids_from": question_ids_from,
            }
        )
        return _Task(_load_samples(path, "biased"), scorer=["switch"], metadata=metadata)

    monkeypatch.setattr(sys.modules["mcq_bias.tasks"], "task_from_frozen", legacy_task_from_frozen)
    tasks = module.diagnostic_biased_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        split="train_eval",
        unbiased_log="logs/clean",
        variant_file=Path(manifest["splits"]["train_eval"]["path"]),
    )

    assert [len(task.dataset) for task in tasks] == [100, 100]
    assert all(call["metadata"]["source_identity_digest"] == source_hash for call in calls["biased"])


def test_matrix_factory_orders_all_clean_tasks_before_all_biased(tmp_path, monkeypatch, task_module):
    module, calls = task_module
    source = tmp_path / "source.jsonl"
    _, source_hash = _write_source(source)
    output = tmp_path / "diagnostic"
    prepare.prepare_iid_diagnostic(source, output, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)

    tasks = module.diagnostic_matrix_tasks(
        manifest=str(output / prepare.DEFAULT_MANIFEST_FILENAME),
        unbiased_log="logs/stage1-iid/matrix",
    )

    assert len(tasks) == 8
    assert [task.registry_name for task in tasks] == [
        *("stage1_iid_unbiased",) * 4,
        *("stage1_iid_biased",) * 4,
    ]
    assert [task.task_args["split"] for task in tasks] == [
        "train_eval",
        "train_eval",
        "heldout_in_domain",
        "heldout_in_domain",
        "train_eval",
        "train_eval",
        "heldout_in_domain",
        "heldout_in_domain",
    ]
    assert [task.task_args["dataset"] for task in tasks] == ["logiqa", "hellaswag"] * 4
    assert all(len(task.dataset) == 100 for task in tasks)
    assert len(calls["unbiased"]) == 4
    assert len(calls["biased"]) == 4
    assert set(tasks[0].task_args["question_ids_from"]).isdisjoint(tasks[2].task_args["question_ids_from"])


def test_vast_matrix_checkpoint_mapping_and_original_generation_protocol(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    checkpoints = tmp_path / "checkpoints"
    for name in run_vast_matrix.CHECKPOINTS.values():
        if name is not None:
            path = checkpoints / name
            path.mkdir(parents=True)
            (path / "manifest.json").write_text("{}")
            (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
    monkeypatch.setattr(
        run_vast_matrix,
        "_successful_task_indices",
        lambda _path, **_kwargs: set(),
    )

    command = run_vast_matrix._split_command(
        repo_root=tmp_path,
        manifest=manifest,
        checkpoint_root=checkpoints,
        log_root=tmp_path / "logs",
        condition="bct",
        split="train_eval",
    )

    assert command is not None
    checkpoint = command[command.index("--local-checkpoint") + 1]
    assert checkpoint.endswith("bias-augmented-consistency-lr-1e-4")
    assert [command[index + 1] for index, token in enumerate(command) if token == "--task-index"] == [str(index) for index in range(1, 5)]
    task_args = json.loads(command[command.index("--task-args") + 1])
    assert task_args["split"] == "train_eval"
    assert task_args["unbiased_log"].endswith("bct/train_eval")
    generation = json.loads(command[command.index("--generation-config") + 1])
    assert generation == {
        "extra_body": {"top_k": 20},
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }

    untrained = run_vast_matrix._split_command(
        repo_root=tmp_path,
        manifest=manifest,
        checkpoint_root=checkpoints,
        log_root=tmp_path / "logs",
        condition="untrained",
        split="heldout_in_domain",
    )
    assert untrained is not None
    assert untrained[untrained.index("--model") + 1] == "vllm/Qwen/Qwen3.5-9B"
    assert "--local-checkpoint" not in untrained
    untrained_model_args = json.loads(untrained[untrained.index("--model-args") + 1])
    assert untrained_model_args == {
        "gpu_memory_utilization": 0.9,
        "language_model_only": True,
        "max_model_len": 32768,
        "max_num_seqs": 256,
    }
    assert "provider" not in untrained_model_args


def test_open_runner_log_creates_condition_directory(tmp_path):
    runner_log, handle = run_vast_matrix._open_runner_log(tmp_path / "logs", "rmct")
    try:
        handle.write(b"started\n")
    finally:
        handle.close()

    assert runner_log == tmp_path / "logs" / "rmct" / "runner.log"
    assert runner_log.read_bytes() == b"started\n"


def _score(value, metadata=None):
    return SimpleNamespace(value=value, metadata=metadata or {})


def _graded_sample(
    question_id: str,
    *,
    dataset: str = "logiqa",
    clean=0,
    toward=0,
    away=None,
    net=0,
    total=0,
    luna=0,
    stop_reason="stop",
    grader_cap=False,
):
    switch = {
        "unbiased_matches_bias": clean,
        "towards_bias_switch": toward,
        "away_from_bias_switch": away,
        "net_switch": net,
        "abs_switch": total,
    }
    return SimpleNamespace(
        id=question_id,
        metadata={"variant": "biased", "source_dataset": dataset},
        scores={
            "switch": _score(switch),
            "luna": _score(
                {"bias_acknowledged": luna},
                {"grader_max_tokens_cap_hit": grader_cap, "grader_model": grade_luna.DEFAULT_LUNA_GRADER_MODEL},
            ),
        },
        output=SimpleNamespace(stop_reason=stop_reason),
    )


def test_posthoc_summary_uses_paired_conditional_denominators_and_reports_diagnostics():
    log = SimpleNamespace(
        status="success",
        samples=[
            _graded_sample("q1", clean=0, toward=1, net=1, total=1, luna=1, stop_reason="max_tokens"),
            _graded_sample("q2", clean=0, toward=0, net=0, total=0, luna=0),
            _graded_sample("q3", clean=1, toward=None, away=1, net=-1, total=1, luna=1, grader_cap=True),
            _graded_sample("q4", clean=None, toward=None, away=None, net=None, total=None, luna=None),
        ],
    )

    rows = diagnostic_analysis.observations_from_log(
        log, condition="bct", split="heldout_in_domain", dataset="logiqa"
    )
    report = diagnostic_analysis.summarize(rows)

    assert report["counts"] == {
        "samples": 4,
        "joint_parsed": 3,
        "joint_parse_failures": 1,
        "clean_answer_not_bias_answer": 2,
        "clean_answer_equals_bias_answer": 1,
        "luna_parsed": 3,
        "luna_parse_failures": 1,
        "generation_max_token_cap_hits": 1,
        "grader_max_token_cap_hits": 1,
    }
    assert report["rates"]["tbsr"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert report["rates"]["away_from_bias"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert report["rates"]["total_switch"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}
    assert report["rates"]["luna_yes"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}


def test_posthoc_paired_normalizes_exact_nan_scorer_compat_encodings():
    nan = float("nan")
    log = SimpleNamespace(
        status="success",
        samples=[
            _graded_sample("clean-nontarget", clean=0.0, toward=1.0, away=nan, net=1.0, total=1.0),
            _graded_sample("clean-target", clean=1.0, toward=nan, away=1.0, net=-1.0, total=1.0),
            _graded_sample("joint-failure", clean=nan, toward=nan, away=nan, net=nan, total=nan),
        ],
    )

    rows = diagnostic_analysis.observations_from_log(
        log, condition="bct", split="train_eval", dataset="logiqa"
    )

    assert (rows[0].joint_parse, rows[0].clean_matches_bias, rows[0].toward, rows[0].away) == (
        True,
        0,
        1,
        0,
    )
    assert (rows[1].joint_parse, rows[1].clean_matches_bias, rows[1].toward, rows[1].away) == (
        True,
        1,
        0,
        1,
    )
    assert (rows[2].joint_parse, rows[2].clean_matches_bias, rows[2].toward, rows[2].away) == (
        False,
        None,
        None,
        None,
    )


def test_raw_observations_allow_absent_luna_but_reject_ambiguous_luna_scores():
    raw_sample = _graded_sample("raw", clean=0, toward=1, net=1, total=1)
    raw_sample.scores.pop("luna")
    raw_log = SimpleNamespace(status="success", samples=[raw_sample])

    rows = diagnostic_analysis.observations_from_raw_log(
        raw_log, condition="act", split="train_eval", dataset="logiqa"
    )

    assert len(rows) == 1
    assert rows[0].luna is None
    assert diagnostic_analysis.summarize(rows)["rates"]["tbsr"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    raw_sample.scores["first-luna"] = _score({"bias_acknowledged": 0})
    raw_sample.scores["second-luna"] = _score({"bias_acknowledged": 0})
    with pytest.raises(ValueError, match="exactly one Luna"):
        diagnostic_analysis.observations_from_raw_log(
            raw_log, condition="act", split="train_eval", dataset="logiqa"
        )


def test_rmct_first64_subset_is_only_reported_where_applicable():
    ids = {f"q{index}" for index in range(64)}
    rows = [
        diagnostic_analysis.Observation(
            condition=condition,
            split="train_eval",
            dataset="logiqa" if index % 2 == 0 else "hellaswag",
            question_id=f"q{index}",
            joint_parse=True,
            clean_matches_bias=0,
            toward=0,
            away=0,
            total_switch=0,
            luna=0,
            generation_cap_hit=False,
            grader_cap_hit=False,
        )
        for condition in ("rmct", "bct")
        for index in range(64)
    ]

    report = diagnostic_analysis.grouped_report(rows, ids)

    assert report["rmct/train_eval"]["rmct_first64"]["pooled"]["counts"]["samples"] == 64
    assert "rmct_first64" not in report["bct/train_eval"]


def _final_diagnostic_plot_report():
    cells = {}
    for condition_index, condition in enumerate(diagnostic_plot.CONDITIONS):
        for split_index, split in enumerate(diagnostic_plot.SPLITS):
            tbsr_denominator = 180 + condition_index + split_index
            tbsr_numerator = 80 + condition_index + split_index
            luna_denominator = 198 + split_index
            luna_numerator = luna_denominator - (condition_index % 3)
            cells[f"{condition}/{split}"] = {
                "condition": condition,
                "split": split,
                "pooled": {
                    "counts": {"samples": 200},
                    "rates": {
                        "tbsr": {
                            "numerator": tbsr_numerator,
                            "denominator": tbsr_denominator,
                            "rate": tbsr_numerator / tbsr_denominator,
                        },
                        "luna_yes": {
                            "numerator": luna_numerator,
                            "denominator": luna_denominator,
                            "rate": luna_numerator / luna_denominator,
                        },
                    }
                },
            }
    return {"schema": diagnostic_analysis.ANALYSIS_SCHEMA, "cells": cells}


def test_iid_plot_renders_png_svg_and_identical_rerun_resumes(tmp_path):
    output = tmp_path / "figures"
    report = _final_diagnostic_plot_report()

    assert diagnostic_plot.render_figures(report, output) == "written"
    expected = {
        f"{stem}.{extension}"
        for stem in diagnostic_plot.FIGURES
        for extension in ("png", "svg")
    }
    assert {path.name for path in output.iterdir()} == expected
    assert (output / "towards-bias-switch.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (output / "towards-bias-switch.svg").read_text()
    assert svg.startswith("<?xml")
    assert "Training-domain samples" in svg
    assert "Held-out in-domain samples" in svg
    assert svg.index("Training-domain samples") < svg.index("Held-out in-domain samples")
    assert "Qwen/Qwen3.5-9B" in svg
    assert "Base" in svg
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    assert diagnostic_plot.render_figures(report, output) == "resumed"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_iid_plot_refuses_differing_outputs_before_touching_any_file(tmp_path):
    output = tmp_path / "figures"
    report = _final_diagnostic_plot_report()
    diagnostic_plot.render_figures(report, output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    changed = json.loads(json.dumps(report))
    rate = changed["cells"]["bct/train_eval"]["pooled"]["rates"]["tbsr"]
    rate["numerator"] += 1
    rate["rate"] = rate["numerator"] / rate["denominator"]

    with pytest.raises(FileExistsError, match="refusing to overwrite differing figure"):
        diagnostic_plot.render_figures(changed, output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_iid_plot_requires_complete_final_matrix_and_consistent_rates(tmp_path):
    incomplete = _final_diagnostic_plot_report()
    incomplete["cells"].pop("untrained/train_eval")
    with pytest.raises(ValueError, match="final nine-condition matrix"):
        diagnostic_plot.render_figures(incomplete, tmp_path / "incomplete")

    inconsistent = _final_diagnostic_plot_report()
    inconsistent["cells"]["opct/heldout_in_domain"]["pooled"]["rates"]["luna_yes"]["rate"] = 0.5
    with pytest.raises(ValueError, match="inconsistent 'luna_yes' rate"):
        diagnostic_plot.render_figures(inconsistent, tmp_path / "inconsistent")


def test_iid_chart_adapter_uses_main_condition_identities_and_split_categories():
    report = _final_diagnostic_plot_report()
    rows = diagnostic_plot.chart_rows(report, "tbsr", "towards_bias_switch")

    assert {row["bias_type"] for row in rows} == {"training_domain", "held_out_in_domain"}
    assert [row["bias_type"] for row in rows[:9]] == ["training_domain"] * 9
    assert [row["bias_type"] for row in rows[9:]] == ["held_out_in_domain"] * 9
    assert {tuple(row["training_biases"]) for row in rows} == {("training_domain",)}
    assert {row["n_total"] for row in rows} == {200}
    first = rows[0]
    assert first["stderr"] == pytest.approx(
        (first["mean"] * (1 - first["mean"]) / (first["n_scored"] - 1)) ** 0.5
    )
    bct_control = next(row for row in rows if row["condition"] == "bias-augmented-consistency-control")
    assert bct_control["method"] == "bias_augmented_consistency"
    assert bct_control["is_control"] is True


def test_luna_grading_writes_derived_artifacts_uses_mock_primary_and_resumes(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "bct" / "train_eval" / "biased.eval"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"immutable raw log")
    source = grade_luna.GradeInput(raw, "bct", "train_eval", "logiqa", "2026-01-01")
    log = SimpleNamespace(
        status="success",
        samples=[_graded_sample("q1")],
        results=SimpleNamespace(),
    )
    calls = []

    inspect_module = types.ModuleType("inspect_ai")
    inspect_log = types.ModuleType("inspect_ai.log")
    fake_scorer_module = types.ModuleType("ctm_data.adapters.mcq_bias.luna_scorer")
    sentinel_scorer = object()
    scorer_connections = []

    def fake_scorer(*, max_connections):
        scorer_connections.append(max_connections)
        return sentinel_scorer

    fake_scorer_module.luna_bias_acknowledged_scorer = fake_scorer

    def fake_score(value, scorer, **kwargs):
        calls.append((value, scorer, kwargs))
        return value

    def fake_read(_path, **_kwargs):
        return log

    def fake_write(_log, path):
        Path(path).write_bytes(b"derived eval")

    inspect_module.score = fake_score
    inspect_log.read_eval_log = fake_read
    inspect_log.write_eval_log = fake_write
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log)
    monkeypatch.setitem(sys.modules, "ctm_data.adapters.mcq_bias.luna_scorer", fake_scorer_module)

    output = tmp_path / "graded"
    assert grade_luna.grade_one(source, output) == "graded"
    assert raw.read_bytes() == b"immutable raw log"
    assert calls[0][1] is sentinel_scorer
    assert calls[0][2]["model"] == "mockllm/model"
    assert calls[0][2]["action"] == "append"
    assert scorer_connections == [grade_luna.DEFAULT_CONNECTIONS_PER_WORKER]
    eval_path, rows_path, provenance_path = grade_luna.output_paths(output, source)
    assert eval_path.read_bytes() == b"derived eval"
    assert json.loads(rows_path.read_text().splitlines()[0])["question_id"] == "q1"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["grader_model"].endswith("-20260709")
    assert provenance["worker_count"] == 5
    assert provenance["connections_per_worker"] == 100
    assert provenance["aggregate_connection_limit"] == 500
    assert provenance["deterministic_shard_index"] == 0
    assert "raw_preflight" not in provenance
    assert grade_luna.grade_one(source, output) == "resumed"


def test_luna_grader_completion_cap_is_recorded_and_validated(tmp_path):
    source = grade_luna.GradeInput(
        tmp_path / "raw.eval",
        "bct",
        "train_eval",
        "logiqa",
        "2026-01-01",
    )
    provenance = grade_luna._provenance(
        source,
        source_sha256="a" * 64,
        smoke_samples=None,
        worker_count=5,
        connections_per_worker=100,
        grader_max_tokens=1024,
        shard_index=0,
    )
    assert provenance["grader_max_tokens"] == 1024
    with pytest.raises(ValueError, match="grader_max_tokens"):
        grade_luna._validate_grader_max_tokens(0)


def test_luna_discovery_selects_only_latest_successful_biased_retry(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    paths = {
        name: root / "bct" / "train_eval" / f"{name}.eval"
        for name in ("clean", "biased-old", "biased-new", "biased-error")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"log")

    def header(task, created, status="success"):
        return SimpleNamespace(
            status=status,
            eval=SimpleNamespace(
                task=task,
                created=created,
                task_args={"split": "train_eval", "dataset": "logiqa", "bias_type": "wrong_argument"},
            ),
        )

    logs = {
        str(paths["clean"].resolve()): header("stage1_iid_unbiased", "2026-01-01"),
        str(paths["biased-old"].resolve()): header("stage1_iid_biased", "2026-01-02"),
        str(paths["biased-new"].resolve()): header("stage1_iid_biased", "2026-01-03"),
        str(paths["biased-error"].resolve()): header("stage1_iid_biased", "2026-01-04", "error"),
    }
    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.__path__ = []
    inspect_log = types.ModuleType("inspect_ai.log")
    inspect_log.list_eval_logs = lambda *_args, **_kwargs: [SimpleNamespace(name=str(path)) for path in paths.values()]
    inspect_log.read_eval_log = lambda path, **_kwargs: logs[path]
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log)

    selected = grade_luna.discover_biased_logs(root)

    assert len(selected) == 1
    assert selected[0].path == paths["biased-new"].resolve()
    assert (selected[0].condition, selected[0].split, selected[0].dataset) == (
        "bct",
        "train_eval",
        "logiqa",
    )


def _preflight_bound_staging(tmp_path: Path) -> tuple[Path, Path, dict[Path, bytes]]:
    root = tmp_path / "staged"
    source_root = tmp_path / "generation" / "raw"
    payloads: dict[Path, bytes] = {}
    sources = []
    for split in grade_luna.SPLITS:
        for dataset in grade_luna.DATASETS:
            filename = f"{split}-{dataset}.eval"
            staged = root / "bct" / split / filename
            payload = f"{split}/{dataset}".encode()
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
            payloads[staged.resolve()] = payload
            sources.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "raw_log": str((source_root / filename).resolve()),
                    "raw_log_sha256": hashlib.sha256(payload).hexdigest(),
                    "created": "2026-08-01T00:00:00Z",
                }
            )
    report = tmp_path / "raw-preflight.json"
    report.write_text(
        json.dumps(
            {
                "schema": raw_preflight.PREFLIGHT_SCHEMA,
                "condition": "bct",
                "raw_root": str(source_root.resolve()),
                "sources": sources,
            },
            sort_keys=True,
        )
    )
    return root, report, payloads


def test_luna_preflight_selects_only_bound_staged_sources_without_discovery(tmp_path, monkeypatch):
    root, report, payloads = _preflight_bound_staging(tmp_path)
    unrelated = root / "bct" / "train_eval" / "unrelated.eval"
    unrelated.write_bytes(b"unrelated")

    selected = grade_luna.preflight_bound_logs(root, report)

    assert [source.path for source in selected] == sorted(payloads)
    assert all(source.expected_sha256 == hashlib.sha256(payloads[source.path]).hexdigest() for source in selected)
    assert {source.preflight_report_sha256 for source in selected} == {
        hashlib.sha256(report.read_bytes()).hexdigest()
    }
    assert unrelated.resolve() not in {source.path for source in selected}
    provenance = grade_luna._provenance(
        selected[0],
        source_sha256=selected[0].expected_sha256,
        smoke_samples=None,
        worker_count=1,
        connections_per_worker=1,
        grader_max_tokens=grade_luna.DEFAULT_MAX_TOKENS,
        shard_index=0,
    )
    assert provenance["raw_preflight"] == {
        "schema": raw_preflight.PREFLIGHT_SCHEMA,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "source_sha256": selected[0].expected_sha256,
    }

    monkeypatch.setattr(
        grade_luna,
        "discover_biased_logs",
        lambda _root: pytest.fail("preflight mode must not rediscover raw logs"),
    )

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class InlineProcessPool:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return ImmediateFuture(function(*args))

    monkeypatch.setattr(grade_luna, "ProcessPoolExecutor", InlineProcessPool)
    monkeypatch.setattr(grade_luna, "grade_one", lambda *_args, **_kwargs: "resumed")
    results = grade_luna.grade_all(
        root,
        tmp_path / "derived",
        preflight_report=report,
        workers=1,
        connections_per_worker=1,
    )
    assert [source for source, _status in results] == selected


def test_luna_preflight_rechecks_staged_sha_before_scoring(tmp_path):
    root, report, _payloads = _preflight_bound_staging(tmp_path)
    source = grade_luna.preflight_bound_logs(root, report)[0]
    source.path.write_bytes(b"tampered after preflight")

    with pytest.raises(ValueError, match="does not match raw preflight report"):
        grade_luna.preflight_bound_logs(root, report)
    with pytest.raises(ValueError, match="no longer matches raw preflight report"):
        grade_luna.grade_one(source, tmp_path / "derived")


@pytest.mark.parametrize("prefix", ["file:", "file://", "file://localhost"])
def test_luna_local_log_path_normalizes_inspect_file_uri_variants(tmp_path, prefix):
    path = tmp_path / "directory with spaces" / "biased retry.eval"
    encoded_path = quote(str(path.resolve()), safe="/")

    assert grade_luna._local_log_path(f"{prefix}{encoded_path}") == path.resolve()
    assert grade_luna._local_log_path(path) == path.resolve()


@pytest.mark.parametrize(
    "uri",
    ["s3://bucket/log.eval", "https://example.test/log.eval", "file://server/log.eval", "file:relative.eval"],
)
def test_luna_local_log_path_rejects_nonlocal_uri_schemes(uri):
    with pytest.raises(ValueError, match="local logs|authorities|absolute path"):
        grade_luna._local_log_path(uri)


def test_luna_discovery_accepts_inspect_file_colon_uri_and_decodes_path(tmp_path, monkeypatch):
    root = tmp_path / "raw logs"
    path = root / "bct" / "train_eval" / "biased retry.eval"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"log")
    encoded_uri = "file:" + quote(str(path.resolve()), safe="/")
    header = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            task="stage1_iid_biased",
            created="2026-07-31",
            task_args={"split": "train_eval", "dataset": "logiqa", "bias_type": "wrong_argument"},
        ),
    )
    calls = []
    inspect_module = types.ModuleType("inspect_ai")
    inspect_module.__path__ = []
    inspect_log = types.ModuleType("inspect_ai.log")
    inspect_log.list_eval_logs = lambda *_args, **_kwargs: [SimpleNamespace(name=encoded_uri)]

    def read_eval_log(location, **_kwargs):
        calls.append(location)
        return header

    inspect_log.read_eval_log = read_eval_log
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log)

    selected = grade_luna.discover_biased_logs(root)

    assert selected == [
        grade_luna.GradeInput(path.resolve(), "bct", "train_eval", "logiqa", "2026-07-31")
    ]
    assert calls == [str(path.resolve())]


def test_luna_parallelism_cap_and_deterministic_incremental_shards(tmp_path, monkeypatch):
    assert grade_luna.DEFAULT_WORKERS * grade_luna.DEFAULT_CONNECTIONS_PER_WORKER == 500
    with pytest.raises(ValueError, match="must be <= 500"):
        grade_luna._validate_parallelism(5, 101)
    with pytest.raises(ValueError, match="positive integer"):
        grade_luna._validate_parallelism(True, 100)

    sources = [
        grade_luna.GradeInput(
            tmp_path / f"source-{index}.eval",
            "bct",
            "train_eval" if index % 2 == 0 else "heldout_in_domain",
            "logiqa" if index % 3 else "hellaswag",
            str(index),
        )
        for index in range(7)
    ]
    expected = sorted(sources, key=lambda item: (item.condition, item.split, item.dataset, str(item.path)))
    shards = grade_luna.deterministic_shards(list(reversed(sources)), 3)
    assert sorted(
        (source for shard in shards for source in shard),
        key=lambda item: (item.condition, item.split, item.dataset, str(item.path)),
    ) == expected
    initial_assignment = {
        source: shard_index for shard_index, shard in enumerate(shards) for source in shard
    }
    extra = grade_luna.GradeInput(
        tmp_path / "extra.eval", "act", "heldout_in_domain", "hellaswag", "8"
    )
    expanded = grade_luna.deterministic_shards([extra, *sources], 3)
    expanded_assignment = {
        source: shard_index for shard_index, shard in enumerate(expanded) for source in shard
    }
    assert {source: expanded_assignment[source] for source in sources} == initial_assignment

    calls = []
    executor_sizes = []
    executor_start_methods = []

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class InlineProcessPool:
        def __init__(self, max_workers, *, mp_context):
            executor_sizes.append(max_workers)
            executor_start_methods.append(mp_context.get_start_method())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return ImmediateFuture(function(*args))

    def fake_grade(source, _output, **kwargs):
        calls.append((source, kwargs))
        return "resumed"

    monkeypatch.setattr(grade_luna, "discover_biased_logs", lambda _root: list(reversed(sources)))
    monkeypatch.setattr(grade_luna, "ProcessPoolExecutor", InlineProcessPool)
    monkeypatch.setattr(grade_luna, "grade_one", fake_grade)
    results = grade_luna.grade_all(
        tmp_path / "raw",
        tmp_path / "derived",
        workers=3,
        connections_per_worker=100,
    )

    assert executor_sizes == [3]
    assert executor_start_methods == ["spawn"]
    assert [source for source, _ in results] == expected
    shard_by_source = {source: kwargs["shard_index"] for source, kwargs in calls}
    assert shard_by_source == initial_assignment
    assert all(kwargs["worker_count"] == 3 for _, kwargs in calls)
    assert all(kwargs["connections_per_worker"] == 100 for _, kwargs in calls)
