"""Offline guards for the receipt-selected Step 16 Luna adapter."""

from __future__ import annotations

from pathlib import Path

from experiments.rmct_two_bias_eval import luna_bridge as luna
from experiments.rmct_two_bias_eval import step16_luna_bridge as step16


def test_step16_luna_policy_is_exactly_five_times_one_hundred():
    step16._assert_parallelism()
    assert step16.WORKERS == 5
    assert step16.CONNECTIONS_PER_WORKER == 100
    assert step16.WORKERS * step16.CONNECTIONS_PER_WORKER == 500


def test_step16_matrix_has_the_expected_final_18_cells_and_1200_samples():
    biased = [step16._TASK_MATRIX[index] for index in step16.EXPECTED_BIASED_TASKS]

    assert len(biased) == 18
    assert sum(row[-1] for row in biased) == 1200
    assert tuple(step16.REPAIRED_TASKS) == (4, 6, 7, 9, 11, 13)


def test_portable_capture_copy_compares_destination_content_not_source_path(tmp_path: Path):
    source = tmp_path / "source" / "receipt.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"receipt bytes\n")
    identity = step16._file_identity(source, label="source")
    destination = tmp_path / "portable" / "custody" / "receipt.json"

    assert step16._copy_once(source, destination, expected=identity, label="receipt") == "staged"
    assert step16._copy_once(source, destination, expected=identity, label="receipt") == "resumed"
    assert destination.read_bytes() == source.read_bytes()


def test_step16_profile_emits_a_dedicated_staging_schema(tmp_path: Path):
    raw = tmp_path / "published.eval"
    staged = tmp_path / "staged.eval"
    preflight = tmp_path / "preflight.json"
    evaluation = tmp_path / "evaluation.json"
    receipt = tmp_path / "receipt.json"
    clean = tmp_path / "clean.eval"
    for path in (raw, preflight, evaluation, receipt, clean):
        path.write_bytes(path.name.encode("utf-8"))
    raw_identity = step16._file_identity(raw, label="raw")
    staged.write_bytes(raw.read_bytes())
    source = luna.GradeInput(
        task_index=4,
        condition=step16.CONDITION,
        regime="iid",
        population="in_domain",
        dataset="logiqa",
        bias_type="wrong_argument",
        evaluation_bias_status="seen",
        sample_count=50,
        raw_path=raw,
        raw_sha256=str(raw_identity["sha256"]),
        raw_size_bytes=int(raw_identity["size_bytes"]),
        staged_path=staged,
        preflight_path=preflight,
        preflight_sha256=step16._file_identity(preflight, label="preflight")["sha256"],
        evaluation_receipt_path=evaluation,
        evaluation_receipt_sha256=step16._file_identity(evaluation, label="evaluation")["sha256"],
        task_receipt_path=receipt,
        task_receipt_sha256=step16._file_identity(receipt, label="receipt")["sha256"],
        paired_clean={"raw_log": str(clean), "raw_log_sha256": step16._file_identity(clean, label="clean")["sha256"]},
    )

    original = {
        name: getattr(luna, name)
        for name in ("BRIDGE_SCHEMA", "STAGING_RECEIPT_SCHEMA", "ATTEMPT_RECEIPT_SCHEMA", "DERIVED_PROVENANCE_SCHEMA", "DERIVED_NAMESPACE")
    }
    try:
        step16._profile_luna_bridge()
        payload = luna._staging_receipt(source)

        assert payload["schema"] == step16.STAGING_RECEIPT_SCHEMA
        assert payload["bridge_schema"] == step16.BRIDGE_SCHEMA
        assert payload["source"]["raw_log"]["sha256"] == raw_identity["sha256"]
    finally:
        for name, value in original.items():
            setattr(luna, name, value)
