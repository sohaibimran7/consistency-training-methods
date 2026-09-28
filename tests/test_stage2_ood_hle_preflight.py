from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import crosshost_merge, grade_luna, raw_preflight, stage_luna
from experiments.stage2_ood_hle.materialize import (
    HELDOUT_BIAS,
    HELDOUT_BIASES,
    HELDOUT_DATASET,
    HELDOUT_DATASET_AND_BIAS,
    HLE_DATASET,
    IID,
    IN_DOMAIN_DATASETS,
    TRAINING_BIAS,
)
from experiments.stage2_ood_hle.tasks import OODTaskSpec


def _specs(tmp_path: Path) -> list[OODTaskSpec]:
    """Minimal files/IDs with the real immutable 3-clean/18-biased topology."""

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    files: dict[tuple[str, str], Path] = {}
    for population, biases in (
        ("in_domain", ("unbiased", TRAINING_BIAS, *HELDOUT_BIASES)),
        ("hle", ("unbiased", TRAINING_BIAS, *HELDOUT_BIASES)),
    ):
        for bias in biases:
            path = frozen / f"{population}-{bias}.jsonl"
            path.write_text(f"{population}/{bias}\n", encoding="utf-8")
            files[(population, bias)] = path

    iid_ids = {dataset: (f"iid-{dataset}-1",) for dataset in IN_DOMAIN_DATASETS}
    hle_ids = ("hle-1",)
    iid_identity = f"stage2-ood-hle-2x2:{hashlib.sha256(b'iid').hexdigest()}"
    hle_identity = f"stage2-ood-hle-2x2:{hashlib.sha256(b'hle').hexdigest()}"
    specs: list[OODTaskSpec] = []
    for dataset in IN_DOMAIN_DATASETS:
        specs.append(
            OODTaskSpec(
                "unbiased",
                IID,
                "in_domain",
                dataset,
                None,
                str(files[("in_domain", "unbiased")]),
                iid_ids[dataset],
                iid_identity,
            )
        )
    specs.append(
        OODTaskSpec(
            "unbiased",
            HELDOUT_DATASET,
            "hle",
            HLE_DATASET,
            None,
            str(files[("hle", "unbiased")]),
            hle_ids,
            hle_identity,
        )
    )
    for dataset in IN_DOMAIN_DATASETS:
        specs.append(
            OODTaskSpec(
                "biased",
                IID,
                "in_domain",
                dataset,
                TRAINING_BIAS,
                str(files[("in_domain", TRAINING_BIAS)]),
                iid_ids[dataset],
                iid_identity,
            )
        )
    specs.append(
        OODTaskSpec(
            "biased",
            HELDOUT_DATASET,
            "hle",
            HLE_DATASET,
            TRAINING_BIAS,
            str(files[("hle", TRAINING_BIAS)]),
            hle_ids,
            hle_identity,
        )
    )
    for bias in HELDOUT_BIASES:
        for dataset in IN_DOMAIN_DATASETS:
            specs.append(
                OODTaskSpec(
                    "biased",
                    HELDOUT_BIAS,
                    "in_domain",
                    dataset,
                    bias,
                    str(files[("in_domain", bias)]),
                    iid_ids[dataset],
                    iid_identity,
                )
            )
    for bias in HELDOUT_BIASES:
        specs.append(
            OODTaskSpec(
                "biased",
                HELDOUT_DATASET_AND_BIAS,
                "hle",
                HLE_DATASET,
                bias,
                str(files[("hle", bias)]),
                hle_ids,
                hle_identity,
            )
        )
    assert len(specs) == 21
    return specs


def _fake_logs(tmp_path: Path, specs: list[OODTaskSpec]) -> tuple[Path, dict[Path, SimpleNamespace]]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    clean_paths: dict[tuple[str, str], Path] = {}
    for index, spec in enumerate(specs):
        if spec.kind != "unbiased":
            continue
        path = raw_root / f"{index:02d}-{spec.dataset}-clean.eval"
        path.write_bytes(f"clean:{index}".encode())
        clean_paths[(spec.population, spec.dataset)] = path

    logs: dict[Path, SimpleNamespace] = {}
    for index, spec in enumerate(specs):
        if spec.kind == "unbiased":
            path = clean_paths[(spec.population, spec.dataset)]
        else:
            path = raw_root / f"{index:02d}-{spec.regime}-{spec.dataset}-{spec.bias_type}.eval"
            path.write_bytes(f"biased:{index}".encode())
        task = "stage2_ood_unbiased" if spec.kind == "unbiased" else "stage2_ood_biased"
        args = {
            "frozen_file": spec.frozen_file,
            "dataset": spec.dataset,
            "regime": spec.regime,
            "population": spec.population,
            "bias_type": spec.bias_type,
            "question_ids_from": list(spec.question_ids),
            "source_identity_digest": spec.source_identity_digest,
            "prompt_style": "none",
            "include_bias_acknowledged": False,
            "grader_model": None,
        }
        if spec.kind == "biased":
            args["unbiased_log"] = str(raw_root)
        metadata = dict(args)
        metadata["source_dataset"] = spec.dataset
        samples = []
        for question_id in spec.question_ids:
            sample_metadata = {
                "variant": "unbiased" if spec.kind == "unbiased" else "biased",
                "source_dataset": spec.dataset,
                "prompt_style": "none",
            }
            scores = {}
            if spec.kind == "biased":
                sample_metadata.update({"bias_type": spec.bias_type, "biasing_text": "Frozen intervention."})
                scores["switch"] = SimpleNamespace(
                    value={
                        "unbiased_matches_bias": 0,
                        "towards_bias_switch": 0,
                        "away_from_bias_switch": 0,
                        "net_switch": 0,
                        "abs_switch": 0,
                    },
                    metadata={"unbiased_log": str(clean_paths[(spec.population, spec.dataset)])},
                )
            samples.append(SimpleNamespace(id=question_id, metadata=sample_metadata, scores=scores))
        logs[path] = SimpleNamespace(
            status="success",
            eval=SimpleNamespace(
                task=task,
                task_args=args,
                metadata=metadata,
                created=f"2026-08-02T00:{index:02d}:00Z",
            ),
            samples=samples,
        )
    return raw_root, logs


def _install_fake_inspect(monkeypatch, logs: dict[Path, SimpleNamespace]) -> None:
    monkeypatch.setattr(raw_preflight, "_discover_eval_log_paths", lambda _root: sorted(logs))
    monkeypatch.setattr(raw_preflight, "_read_eval_log", lambda path, *, header_only: logs[path])
    monkeypatch.setattr(
        raw_preflight,
        "_assert_runtime",
        lambda _path, *, runtime: ("vllm/Qwen/Qwen3.5-9B", {"profile": runtime["profile"]}),
    )


def _preflight(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    specs = _specs(tmp_path)
    raw_root, logs = _fake_logs(tmp_path, specs)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _install_fake_inspect(monkeypatch, logs)
    monkeypatch.setattr(raw_preflight, "validate_manifest", lambda _path: {})
    monkeypatch.setattr(raw_preflight, "ood_task_specs", lambda _path: specs)
    return (
        raw_preflight.preflight_raw_logs(
            raw_root,
            manifest,
            condition="act-vllm-compat",
            runtime_profile="vllm",
        ),
        raw_root,
    )


def test_raw_preflight_binds_exact_21_task_matrix_and_clean_pair_hashes(tmp_path, monkeypatch):
    report, raw_root = _preflight(tmp_path, monkeypatch)

    assert report["schema"] == raw_preflight.PREFLIGHT_SCHEMA
    assert report["raw_root"] == str(raw_root.resolve())
    assert report["contract"]["task_count"] == 21
    assert report["contract"]["clean_task_count"] == 3
    assert report["contract"]["biased_task_count"] == 18
    assert len(report["sources"]) == 21
    biased = [source for source in report["sources"] if source["kind"] == "biased"]
    assert len(biased) == 18
    assert all(source["paired_clean"]["raw_log_sha256"] for source in biased)
    assert all(source["unbiased_log"] == str(raw_root.resolve()) for source in biased)
    assert raw_preflight.validate_preflight_report(report) == report

    output = tmp_path / "preflight.json"
    assert raw_preflight.write_report(output, report) == "written"
    assert raw_preflight.write_report(output, report) == "resumed"


def test_raw_preflight_rejects_missing_task_and_wrong_resolved_clean_log(tmp_path, monkeypatch):
    specs = _specs(tmp_path)
    raw_root, logs = _fake_logs(tmp_path, specs)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _install_fake_inspect(monkeypatch, logs)
    monkeypatch.setattr(raw_preflight, "validate_manifest", lambda _path: {})
    monkeypatch.setattr(raw_preflight, "ood_task_specs", lambda _path: specs)
    missing = dict(logs)
    missing.pop(next(path for path, log in logs.items() if log.eval.task == "stage2_ood_biased"))
    monkeypatch.setattr(raw_preflight, "_discover_eval_log_paths", lambda _root: sorted(missing))
    with pytest.raises(ValueError, match="incomplete"):
        raw_preflight.preflight_raw_logs(raw_root, manifest, condition="base-vllm", runtime_profile="vllm")

    _install_fake_inspect(monkeypatch, logs)
    biased = next(log for log in logs.values() if log.eval.task == "stage2_ood_biased")
    biased.samples[0].scores["switch"].metadata["unbiased_log"] = str(raw_root / "wrong-clean.eval")
    with pytest.raises(ValueError, match="wrong clean log"):
        raw_preflight.preflight_raw_logs(raw_root, manifest, condition="base-vllm", runtime_profile="vllm")


def test_luna_staging_accepts_only_18_report_hashed_biased_logs(tmp_path, monkeypatch):
    report, _raw_root = _preflight(tmp_path, monkeypatch)
    report_path = tmp_path / "preflight.json"
    raw_preflight.write_report(report_path, report)
    staged_root = tmp_path / "staged"
    staged = stage_luna.stage_from_preflight(report_path, staged_root)
    assert len(staged) == 18
    assert {status for _, status in staged} == {"staged"}
    assert {status for _, status in stage_luna.stage_from_preflight(report_path, staged_root)} == {"resumed"}

    selected = grade_luna.preflight_bound_logs(staged_root, report_path)
    assert len(selected) == 18
    assert {source.bias_type for source in selected} == {TRAINING_BIAS, *HELDOUT_BIASES}
    assert all(source.preflight_report_sha256 for source in selected)
    eval_path, rows_path, provenance_path = grade_luna.output_paths(tmp_path / "derived", selected[0])
    assert "act-vllm-compat" in str(eval_path)
    assert eval_path.suffix == ".eval"
    assert rows_path.suffix == ".jsonl"
    assert provenance_path.name.endswith(".provenance.json")

    corrupted = selected[0].path
    corrupted.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        grade_luna.preflight_bound_logs(staged_root, report_path)


def test_preflight_report_rejects_tampered_paired_clean_binding(tmp_path, monkeypatch):
    report, _raw_root = _preflight(tmp_path, monkeypatch)
    altered = copy.deepcopy(report)
    biased = next(source for source in altered["sources"] if source["kind"] == "biased")
    biased["paired_clean"]["raw_log"] = "/different/clean.eval"
    with pytest.raises(ValueError, match="paired_clean"):
        raw_preflight.validate_preflight_report(altered)

    altered = copy.deepcopy(report)
    altered["sources"][0]["model"] = "vllm/Qwen/Qwen3.5-4B"
    with pytest.raises(ValueError, match="wrong base vLLM model"):
        raw_preflight.validate_preflight_report(altered)


def test_luna_parallelism_stays_within_the_500_connection_cap():
    from ctm_data.adapters.mcq_bias.luna_scorer import DEFAULT_LUNA_GRADER_MODEL

    assert grade_luna.DEFAULT_LUNA_GRADER_MODEL == DEFAULT_LUNA_GRADER_MODEL
    assert grade_luna.DEFAULT_WORKERS * grade_luna.DEFAULT_CONNECTIONS_PER_WORKER == 500
    with pytest.raises(ValueError, match="<= 500"):
        grade_luna._validate_parallelism(5, 101)


def test_multi_condition_luna_grading_uses_one_global_worker_pool(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    derived = tmp_path / "derived"
    staged.mkdir()

    def source(condition: str) -> grade_luna.GradeInput:
        return grade_luna.GradeInput(
            path=staged / f"{condition}.eval",
            condition=condition,
            regime=IID,
            population="in_domain",
            dataset="logiqa",
            bias_type=TRAINING_BIAS,
            created="2026-08-02T00:00:00Z",
            expected_sha256="a" * 64,
            preflight_report_sha256="b" * 64,
            manifest_sha256="c" * 64,
            paired_clean={},
        )

    by_report = {"first": [source("act")], "second": [source("attct")]}
    monkeypatch.setattr(grade_luna, "preflight_bound_logs", lambda _root, report: by_report[str(report)])
    monkeypatch.setattr(
        grade_luna,
        "_grade_shard",
        lambda _index, sources, *_args: [(item, "resumed") for item in sources],
    )
    observed: list[int] = []

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class InlinePool:
        def __init__(self, *, max_workers, mp_context):
            observed.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            return Future(function(*args))

    monkeypatch.setattr(grade_luna, "ProcessPoolExecutor", InlinePool)
    results = grade_luna.grade_many(
        staged,
        derived,
        preflight_reports=("first", "second"),
        workers=5,
        connections_per_worker=100,
    )
    assert observed == [5]
    assert [item.condition for item, _ in results] == ["act", "attct"]


def test_luna_grader_appends_with_the_pinned_scorer_and_hash_bound_provenance(tmp_path, monkeypatch):
    raw = tmp_path / "biased.eval"
    raw.write_bytes(b"raw-bytes")
    digest = raw_preflight._sha256_file(raw)
    source = grade_luna.GradeInput(
        path=raw,
        condition="act-vllm-compat",
        regime=IID,
        population="in_domain",
        dataset="logiqa",
        bias_type=TRAINING_BIAS,
        created="2026-08-02T00:00:00Z",
        expected_sha256=digest,
        preflight_report_sha256="a" * 64,
        manifest_sha256="b" * 64,
        paired_clean={
            "raw_log": "/generation/clean.eval",
            "raw_log_sha256": "c" * 64,
            "question_ids_sha256": "d" * 64,
            "source_identity_digest": f"stage2-ood-hle-2x2:{'e' * 64}",
        },
    )
    raw_log = SimpleNamespace(status="success", samples=[SimpleNamespace(id="q", scores={})])
    scored = SimpleNamespace(
        status="success",
        samples=[
            SimpleNamespace(
                id="q",
                scores={
                    "luna": SimpleNamespace(
                        value={"bias_acknowledged": 1.0},
                        metadata={
                            "grader_model": grade_luna.DEFAULT_LUNA_GRADER_MODEL,
                            "grader_response": "Answer: YES",
                            "grader_usage": {"total_tokens": 3},
                            "grader_stop_reason": "stop",
                            "grader_max_tokens_cap_hit": False,
                        },
                    )
                },
            )
        ],
    )
    import inspect_ai
    import inspect_ai.log
    import ctm_data.adapters.mcq_bias.luna_scorer as luna_scorer

    calls: list[dict] = []
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda _path: raw_log)
    monkeypatch.setattr(inspect_ai.log, "write_eval_log", lambda _log, path: Path(path).write_bytes(b"scored"))
    monkeypatch.setattr(
        inspect_ai,
        "score",
        lambda _raw, scorer, **kwargs: calls.append({"scorer": scorer, **kwargs}) or scored,
    )
    monkeypatch.setattr(luna_scorer, "luna_bias_acknowledged_scorer", lambda **kwargs: {"luna": kwargs})

    assert grade_luna.grade_one(source, tmp_path / "derived", worker_count=1, connections_per_worker=17) == "graded"
    assert calls == [
        {
            "scorer": {"luna": {"max_connections": 17}},
            "model": grade_luna.INSPECT_RESCORE_MODEL,
            "action": "append",
            "display": "none",
            "copy": True,
        }
    ]
    eval_path, rows_path, provenance_path = grade_luna.output_paths(tmp_path / "derived", source)
    assert eval_path.read_bytes() == b"scored"
    rows = [json.loads(row) for row in rows_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["bias_acknowledged"] == 1.0
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["raw_preflight"]["source_sha256"] == digest
    assert provenance["aggregate_connection_limit"] == 17


def test_crosshost_merge_imports_only_missing_biased_logs_and_preflights_atomically(tmp_path, monkeypatch):
    """A handoff may add bias logs, but never relocates a paired clean log."""

    specs = _specs(tmp_path)
    canonical_root, logs = _fake_logs(tmp_path, specs)
    incoming_root = tmp_path / "incoming"
    incoming_root.mkdir()
    payload_to_log: dict[bytes, SimpleNamespace] = {}
    for source_path, log in logs.items():
        destination = incoming_root / source_path.relative_to(canonical_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        payload_to_log[source_path.read_bytes()] = log

    # The source handoff has a complete raw tree (including the three clean
    # files that prove pair-byte equality), while the canonical root initially
    # has only its clean logs.  Archive test fixtures rather than deleting
    # them so the test also mirrors the helper's preservation guarantee.
    archived = tmp_path / "archived-source-fixtures"
    archived.mkdir()
    for path, log in logs.items():
        if log.eval.task.endswith("stage2_ood_biased"):
            path.rename(archived / path.name)

    def discover(root):
        return sorted(Path(root).rglob("*.eval"))

    def read(path, *, header_only):
        del header_only
        return payload_to_log[Path(path).read_bytes()]

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(raw_preflight, "_discover_eval_log_paths", discover)
    monkeypatch.setattr(raw_preflight, "_read_eval_log", read)
    monkeypatch.setattr(
        raw_preflight,
        "_assert_runtime",
        lambda _path, *, runtime: ("vllm/Qwen/Qwen3.5-9B", {"profile": runtime["profile"]}),
    )
    monkeypatch.setattr(raw_preflight, "validate_manifest", lambda _path: {})
    monkeypatch.setattr(raw_preflight, "ood_task_specs", lambda _path: specs)
    monkeypatch.setattr(crosshost_merge, "ood_task_specs", lambda _path: specs)

    preflight_path = tmp_path / "preflight.json"
    handoff_path = tmp_path / "handoff.json"
    result = crosshost_merge.merge_crosshost_raw_logs(
        canonical_root,
        [incoming_root],
        manifest,
        condition="rmct-hf-peft",
        runtime_profile="vllm",
        handoff_id="h200-20260803",
        preflight_output=preflight_path,
        handoff_output=handoff_path,
    )

    assert result.preflight_status == "written"
    assert result.handoff_status == "written"
    assert len(result.copy_statuses) == 18
    assert {status for _, status in result.copy_statuses} == {"copied"}
    assert all(
        path.is_relative_to(canonical_root / crosshost_merge.TRANSACTION_DIRECTORY) for path, _ in result.copy_statuses
    )
    report = raw_preflight.validate_preflight_report(preflight_path)
    assert len(report["sources"]) == 21
    assert all(Path(source["raw_log"]).is_relative_to(canonical_root) for source in report["sources"])
    receipt = crosshost_merge.validate_handoff_report(handoff_path)
    assert len(receipt["incoming_sources"]) == 21
    assert len([source for source in receipt["incoming_sources"] if source["identity"]["kind"] == "biased"]) == 18

    # A repeat neither creates a second copy nor changes either immutable
    # report; it sees the exact transaction files as canonical candidates.
    resumed = crosshost_merge.merge_crosshost_raw_logs(
        canonical_root,
        [incoming_root],
        manifest,
        condition="rmct-hf-peft",
        runtime_profile="vllm",
        handoff_id="h200-20260803",
        preflight_output=preflight_path,
        handoff_output=handoff_path,
    )
    assert resumed.copy_statuses == ()
    assert resumed.preflight_status == "resumed"
    assert resumed.handoff_status == "resumed"
