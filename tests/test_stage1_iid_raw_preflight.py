from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage1_iid_diagnostic import gate_analysis, raw_preflight
from experiments.stage1_iid_diagnostic.prepare import (
    BIAS_TYPE,
    DATASETS,
    HELDOUT_COUNTS,
    MANIFEST_KIND,
    PROMPT_STYLE,
    SCHEMA_VERSION,
    SOURCE_COUNTS,
    SOURCE_ROWS,
    SOURCE_SHA256,
    TRAIN_EVAL_COUNTS,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _split(tmp_path: Path, split: str) -> tuple[Path, dict, dict[str, tuple[str, ...]]]:
    rows = []
    ids_by_dataset: dict[str, tuple[str, ...]] = {}
    for dataset in DATASETS:
        ids = tuple(f"{split}-{dataset}-{index:03d}" for index in range(100))
        ids_by_dataset[dataset] = ids
        rows.extend(
            {
                "question_id": question_id,
                "source_dataset": dataset,
                "bias_type": BIAS_TYPE,
                "prompt_style": PROMPT_STYLE,
            }
            for question_id in ids
        )
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path = tmp_path / f"{split}.jsonl"
    path.write_bytes(payload)
    counts = TRAIN_EVAL_COUNTS if split == "train_eval" else HELDOUT_COUNTS
    entry = {
        "content_sha256": _digest(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "counts_by_dataset": counts,
        "question_ids": [row["question_id"] for row in rows],
    }
    return path, entry, ids_by_dataset


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[tuple[str, str], tuple[str, ...]]]:
    train_path, train_entry, train_ids = _split(tmp_path, "train_eval")
    heldout_path, heldout_entry, heldout_ids = _split(tmp_path, "heldout_in_domain")
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "source": {
            "content_sha256": SOURCE_SHA256,
            "row_count": SOURCE_ROWS,
            "counts_by_dataset": SOURCE_COUNTS,
            "bias_type": BIAS_TYPE,
            "prompt_style": PROMPT_STYLE,
        },
        "splits": {"train_eval": train_entry, "heldout_in_domain": heldout_entry},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, sort_keys=True))
    ids = {
        **{("train_eval", dataset): values for dataset, values in train_ids.items()},
        **{("heldout_in_domain", dataset): values for dataset, values in heldout_ids.items()},
    }
    return path, {"train_eval": train_path, "heldout_in_domain": heldout_path}, ids


def _loaded(tmp_path: Path, split: str, dataset: str, ids: tuple[str, ...]) -> gate_analysis.LoadedLog:
    path = tmp_path / "raw" / split / f"{dataset}.eval"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{split}/{dataset}".encode())
    header = gate_analysis.LogHeader(
        prompt_variant="native",
        split=split,
        dataset=dataset,
        question_ids=ids,
        variant_file=None,
        unbiased_log=f"/clean/{split}/{dataset}.eval",
        prompt_style=PROMPT_STYLE,
        source_identity_digest=SOURCE_SHA256,
        created="2026-08-01T00:00:00Z",
    )
    return gate_analysis.LoadedLog(
        header=header,
        path=path,
        sha256=_digest(path.read_bytes()),
        rows=tuple(object() for _ in ids),
    )


def _loaded_cells(tmp_path: Path, ids: dict[tuple[str, str], tuple[str, ...]]) -> dict[tuple[str, str], gate_analysis.LoadedLog]:
    return {(split, dataset): _loaded(tmp_path, split, dataset, ids[(split, dataset)]) for split in gate_analysis.SPLITS for dataset in DATASETS}


def test_raw_preflight_binds_native_logs_to_verified_frozen_splits(tmp_path, monkeypatch):
    manifest, split_files, ids = _manifest(tmp_path)
    loaded = _loaded_cells(tmp_path, ids)
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    report = raw_preflight.preflight_raw_logs(
        tmp_path / "raw",
        manifest,
        split_files=split_files,
        condition="attct",
    )

    assert report["schema"] == raw_preflight.PREFLIGHT_SCHEMA
    assert report["condition"] == "attct"
    assert "runtime_profile" not in report["contract"]
    assert len(report["sources"]) == 4
    assert {source["sample_count"] for source in report["sources"]} == {100}
    assert all(source["variant_file"] is None for source in report["sources"])
    output = tmp_path / "preflight.json"
    assert gate_analysis.write_report(output, report) == "written"
    assert gate_analysis.write_report(output, report) == "resumed"


def test_raw_preflight_rejects_header_ids_not_in_frozen_split(tmp_path, monkeypatch):
    manifest, split_files, ids = _manifest(tmp_path)
    loaded = _loaded_cells(tmp_path, ids)
    key = ("train_eval", "logiqa")
    item = loaded[key]
    altered_ids = ("not-a-frozen-id", *item.header.question_ids[1:])
    loaded[key] = _loaded(tmp_path, key[0], key[1], altered_ids)
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    with pytest.raises(ValueError, match="question IDs do not match the frozen split"):
        raw_preflight.preflight_raw_logs(
            tmp_path / "raw",
            manifest,
            split_files=split_files,
            condition="mlpct",
        )


def test_raw_preflight_rejects_tampered_staged_split(tmp_path, monkeypatch):
    manifest, split_files, ids = _manifest(tmp_path)
    loaded = _loaded_cells(tmp_path, ids)
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)
    split_files["heldout_in_domain"].write_text("tampered\n")

    with pytest.raises(ValueError, match="different SHA-256"):
        raw_preflight.preflight_raw_logs(
            tmp_path / "raw",
            manifest,
            split_files=split_files,
            condition="mlpct",
        )


def test_expected_model_accepts_inspect_generate_config_object(tmp_path, monkeypatch):
    """Inspect records generation settings as GenerateConfig, not a dict."""

    from inspect_ai.model import GenerateConfig
    import inspect_ai.log

    checkpoint = tmp_path / "verified-vllm-compat-adapter"
    evaluation = SimpleNamespace(
        model=f"vllm/Qwen/Qwen3.5-9B:{checkpoint}",
        model_generate_config=GenerateConfig(
            max_tokens=20480,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            extra_body={"top_k": 20},
        ),
        model_args={
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "language_model_only": True,
            "max_num_seqs": 256,
        },
    )
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda *_args, **_kwargs: SimpleNamespace(eval=evaluation),
    )

    observed = raw_preflight._assert_expected_model(
        tmp_path / "biased.eval",
        base_model="Qwen/Qwen3.5-9B",
        checkpoint=str(checkpoint),
    )

    assert observed == f"vllm/Qwen/Qwen3.5-9B:{checkpoint}"


def test_expected_model_accepts_bare_native_vllm_base_model(tmp_path, monkeypatch):
    """The unadapted vLLM baseline has no ``:adapter`` suffix."""

    from inspect_ai.model import GenerateConfig
    import inspect_ai.log

    evaluation = SimpleNamespace(
        model="vllm/Qwen/Qwen3.5-9B",
        model_generate_config=GenerateConfig(
            max_tokens=20480,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            extra_body={"top_k": 20},
        ),
        model_args={
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "language_model_only": True,
            "max_num_seqs": 256,
        },
    )
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda *_args, **_kwargs: SimpleNamespace(eval=evaluation),
    )

    observed = raw_preflight._assert_expected_model(
        tmp_path / "biased.eval",
        base_model="Qwen/Qwen3.5-9B",
        checkpoint=None,
    )

    assert observed == "vllm/Qwen/Qwen3.5-9B"


def _hf_peft_evaluation(*, checkpoint: str, max_connections: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        model="hf/Qwen/Qwen3.5-9B",
        model_generate_config={
            "max_tokens": 20480,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_connections": max_connections,
        },
        model_args={
            "device": "cuda:0",
            "dtype": "bfloat16",
        },
        metadata={
            "checkpoint": checkpoint,
            "checkpoint_backend": "local",
            "base_model": "Qwen/Qwen3.5-9B",
            "model_args": {
                "provider": "hf",
                "device": "cuda:0",
                "dtype": "bfloat16",
            },
        },
    )


def _install_header_reader(monkeypatch, evaluation: SimpleNamespace) -> None:
    import inspect_ai.log

    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda *_args, **_kwargs: SimpleNamespace(eval=evaluation),
    )


def test_hf_peft_preflight_binds_raw_checkpoint_and_runtime(tmp_path, monkeypatch):
    manifest, split_files, ids = _manifest(tmp_path)
    loaded = _loaded_cells(tmp_path, ids)
    checkpoint = "/workspace/checkpoints/bct-control-raw"
    _install_header_reader(monkeypatch, _hf_peft_evaluation(checkpoint=checkpoint))
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    report = raw_preflight.preflight_raw_logs(
        tmp_path / "raw",
        manifest,
        split_files=split_files,
        condition="bct-control",
        expected_base_model="Qwen/Qwen3.5-9B",
        expected_checkpoint=checkpoint,
        runtime_profile="hf-peft",
        expected_max_connections=1,
    )

    assert report["contract"]["runtime_profile"] == "hf-peft"
    assert report["contract"]["expected_max_connections"] == 1
    assert {source["model"] for source in report["sources"]} == {"hf/Qwen/Qwen3.5-9B"}
    assert {source["runtime"]["checkpoint"] for source in report["sources"]} == {checkpoint}
    assert {source["runtime"]["max_connections"] for source in report["sources"]} == {1}
    assert {source["runtime"]["dtype"] for source in report["sources"]} == {"bfloat16"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda evaluation: setattr(evaluation, "model", "hf/Qwen/Qwen3.5-4B"), "evaluator model"),
        (lambda evaluation: evaluation.metadata.__setitem__("checkpoint", "/other/raw-checkpoint"), "metadata.checkpoint"),
        (lambda evaluation: evaluation.metadata.__setitem__("checkpoint_backend", "vllm"), "metadata.checkpoint_backend"),
        (lambda evaluation: evaluation.metadata.__setitem__("base_model", "Qwen/Qwen3.5-4B"), "metadata.base_model"),
        (lambda evaluation: evaluation.metadata["model_args"].__setitem__("provider", "vllm"), "metadata.model_args.provider"),
        (lambda evaluation: evaluation.model_args.__setitem__("dtype", "float16"), "model_args.dtype"),
        (lambda evaluation: evaluation.model_generate_config.__setitem__("max_connections", 2), "max_connections decode"),
    ],
)
def test_hf_peft_preflight_rejects_runtime_mismatch(tmp_path, monkeypatch, mutate, message):
    manifest, split_files, ids = _manifest(tmp_path)
    loaded = _loaded_cells(tmp_path, ids)
    checkpoint = "/workspace/checkpoints/bct-control-raw"
    evaluation = _hf_peft_evaluation(checkpoint=checkpoint)
    mutate(evaluation)
    _install_header_reader(monkeypatch, evaluation)
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    with pytest.raises(ValueError, match=message):
        raw_preflight.preflight_raw_logs(
            tmp_path / "raw",
            manifest,
            split_files=split_files,
            condition="bct-control",
            expected_base_model="Qwen/Qwen3.5-9B",
            expected_checkpoint=checkpoint,
            runtime_profile="hf-peft",
            expected_max_connections=1,
        )


def test_hf_peft_preflight_requires_explicit_safe_batch_size(tmp_path):
    with pytest.raises(ValueError, match="requires --expected-max-connections >= 1"):
        raw_preflight.preflight_raw_logs(
            tmp_path / "unused-raw",
            tmp_path / "unused-manifest.json",
            split_files={},
            condition="bct-control",
            expected_base_model="Qwen/Qwen3.5-9B",
            expected_checkpoint="/workspace/checkpoints/bct-control-raw",
            runtime_profile="hf-peft",
        )
