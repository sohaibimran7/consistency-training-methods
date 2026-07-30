"""Strict, fully offline tests for the adapter-owned BCT target chain."""

from __future__ import annotations

import json
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ctm.artifacts import artifact_manifest_path
from ctm.training.sft import load_samples
from scripts.irpan_2510_27062.artifacts import (
    MANIFEST_SHA256_FIELD,
    producer_identity,
    write_artifact,
)
from scripts.irpan_2510_27062.bct_targets import (
    BCTTargetError,
    import_bct_target_results,
    make_external_generator_identity,
    make_fixture_generator_identity,
    materialize_bct_target_requests,
    materialize_bct_training_data,
    read_bct_target_requests,
    read_bct_targets,
    read_bct_training_data,
)
from scripts.irpan_2510_27062.schema import (
    canonical_json,
    make_derived_record,
    make_source_record,
    sha256_bytes,
    sha256_json,
    sha256_text,
)
from scripts.irpan_2510_27062.sycophancy import (
    PROMPT_PAIR_ARTIFACT_KIND,
    build_sycophancy_pairs,
    normalize_arc_rows,
)
from scripts.irpan_2510_27062.training_views import (
    JAILBREAK_DOMAIN,
    SYCOPHANCY_DOMAIN,
    materialize_training_view,
    read_training_view,
)


def _write_source(path: Path, *, domain: str) -> None:
    if domain == SYCOPHANCY_DOMAIN:
        normalized = normalize_arc_rows(
            [
                {
                    "id": identifier,
                    "question": f"Which symbol is {position}?",
                    "choices": {"label": ["A", "B", "C"], "text": ["alpha", "beta", "gamma"]},
                    "answerKey": answer,
                }
                for identifier, position, answer in (
                    ("arc-1", "first", "A"),
                    ("arc-2", "second", "B"),
                )
            ],
            subset="fixture",
            split="train",
            revision="fixture-v1",
        )
        rows = build_sycophancy_pairs(normalized, wrong_option_seed=29)
        kind = PROMPT_PAIR_ARTIFACT_KIND
    else:
        source = make_source_record(
            record_type="harmbench_prompt",
            source="harmbench",
            source_key="fixture",
            payload={"prompt": "Summarize a fictional safety review."},
        )
        core = source["payload"]["prompt"]
        suffix = "[Shared instruction] Answer concisely."
        clean = f"{core}\n\n{suffix}"
        rows = []
        for wrapper_id, prefix in (("role_v1", "Role-play audit: "), ("tag_v1", "<audit>")):
            candidate_id = f"{source['example_id']}:wrapper:{wrapper_id}"
            wrapped = f"{prefix}{core}\n\n{suffix}"
            rows.append(
                make_derived_record(
                    record_type="act_training_export",
                    example_id=f"{candidate_id}:training:act:fixture",
                    source="harmbench",
                    source_key=f"fixture::{wrapper_id}::act",
                    payload={
                        "source_id": candidate_id,
                        "candidate_id": candidate_id,
                        "reference_messages": [{"role": "user", "content": clean}],
                        "variant_messages": [{"role": "user", "content": wrapped}],
                        "alignment_text": core,
                        "alignment_text_sha256": sha256_text(core),
                        "clean_prompt_sha256": sha256_text(clean),
                        "wrapped_prompt_sha256": sha256_text(wrapped),
                        "training_export_version": "fixture_v1",
                    },
                    parent_hashes=[source["content_sha256"]],
                )
            )
        kind = "act_training_exports"
    write_artifact(
        path,
        rows,
        artifact_kind=kind,
        role="training",
        producer=producer_identity("bct-target-fixture", __file__),
        config={"domain": domain},
    )


def _make_view(tmp_path: Path, *, domain: str = SYCOPHANCY_DOMAIN, stem: str = "main") -> Path:
    source = tmp_path / f"{stem}.{domain}.source.jsonl"
    view = tmp_path / f"{stem}.{domain}.view.jsonl"
    _write_source(source, domain=domain)
    materialize_training_view(source, view, domain=domain)
    return view


def _make_requests(
    tmp_path: Path,
    *,
    domain: str = SYCOPHANCY_DOMAIN,
    stem: str = "main",
    generator: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    view = _make_view(tmp_path, domain=domain, stem=stem)
    requests = tmp_path / f"{stem}.{domain}.requests.jsonl"
    materialize_bct_target_requests(
        view,
        requests,
        generator_identity=generator or make_fixture_generator_identity(f"{domain}-{stem}-generator"),
    )
    return view, requests


def _request_generator(request_path: Path) -> dict[str, Any]:
    rows, _ = read_bct_target_requests(request_path)
    return rows[0]["expected_generator_identity"]


def _results(request_path: Path) -> list[dict[str, Any]]:
    requests, _ = read_bct_target_requests(request_path)
    return [
        {
            "pair_id": row["pair_id"],
            "clean_prompt_sha256": row["clean_prompt_sha256"],
            "reference_messages_sha256": row["reference_messages_sha256"],
            "request_record_sha256": row["request_record_sha256"],
            "response": f"Fixture response for {row['pair_id'][-8:]}",
            "response_sha256": sha256_text(f"Fixture response for {row['pair_id'][-8:]}"),
            "metadata": {"fixture": True},
        }
        for row in reversed(requests)
    ]


def _make_targets(
    tmp_path: Path,
    *,
    domain: str = SYCOPHANCY_DOMAIN,
    stem: str = "main",
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    view, requests = _make_requests(tmp_path, domain=domain, stem=stem)
    targets = tmp_path / f"{stem}.{domain}.targets.jsonl"
    generator = _request_generator(requests)
    decoding = {"temperature": 0, "max_tokens": 64}
    import_bct_target_results(
        requests,
        _results(requests),
        targets,
        generator_identity=generator,
        decoding_parameters=decoding,
    )
    return view, requests, targets, generator, decoding


def _rewrite_artifact(
    path: Path,
    *,
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if mutate_rows is not None:
        mutate_rows(rows)
    payload = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    path.write_bytes(payload)
    sidecar = artifact_manifest_path(path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["row_count"] = len(rows)
    manifest["content_sha256"] = sha256_bytes(payload)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    unsigned = {key: value for key, value in manifest.items() if key != MANIFEST_SHA256_FIELD}
    manifest[MANIFEST_SHA256_FIELD] = sha256_json(unsigned)
    sidecar.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("domain", [SYCOPHANCY_DOMAIN, JAILBREAK_DOMAIN])
def test_full_bct_chain_is_deterministic_loader_native_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("BCT artifact pipeline attempted a live call")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    if domain == JAILBREAK_DOMAIN:
        generator = make_external_generator_identity(
            generator_id="fixture-jailbreak-generator",
            provider="fixture-provider",
            model="fixture/model",
            revision="fixture-revision-1",
            date="2026-07-29",
        )
    else:
        generator = make_fixture_generator_identity("sycophancy-v1")
    view, requests = _make_requests(tmp_path, domain=domain, generator=generator)
    request_rows, request_manifest = read_bct_target_requests(requests)
    assert request_rows == sorted(request_rows, key=lambda row: row["pair_id"])
    assert request_manifest["provenance"]["role"] == "training"
    assert all("variant_messages" not in row and "wrapped_prompt" not in row for row in request_rows)
    assert all(row["source_id"] == row["pair_id"] for row in request_rows)
    assert all(row["expected_generator_identity"] == generator for row in request_rows)
    decoding = {"max_tokens": 64, "temperature": 0}
    results = _results(requests)
    target_a = tmp_path / f"{domain}.targets-a.jsonl"
    target_b = tmp_path / f"{domain}.targets-b.jsonl"
    manifest_a = import_bct_target_results(
        requests,
        results,
        target_a,
        generator_identity=generator,
        decoding_parameters=decoding,
    )
    manifest_b = import_bct_target_results(
        requests,
        list(reversed(results)),
        target_b,
        generator_identity=generator,
        decoding_parameters=decoding,
    )
    targets, _ = read_bct_targets(
        target_a,
        expected_generator_identity=generator,
        expected_decoding_parameters=decoding,
    )
    assert manifest_a["content_sha256"] == manifest_b["content_sha256"]
    assert targets == read_bct_targets(target_b)[0]
    assert [row["pair_id"] for row in targets] == [row["pair_id"] for row in request_rows]
    assert all(row["generator_identity_sha256"] == generator["identity_sha256"] for row in targets)

    output = tmp_path / f"{domain}.bct-training.jsonl"
    output_manifest = materialize_bct_training_data(view, target_a, output)
    training_rows, verified_manifest = read_bct_training_data(output)
    assert output_manifest["content_sha256"] == verified_manifest["content_sha256"]
    assert verified_manifest["provenance"]["role"] == "training"
    assert load_samples(output) == training_rows  # The real public BCT loader reads these exact bytes.
    pair_by_id = {row["pair_id"]: row for row in read_training_view(view)[0]}
    target_by_id = {row["pair_id"]: row for row in targets}
    for row in training_rows:
        pair = pair_by_id[row["pair_id"]]
        target = target_by_id[row["pair_id"]]
        assert row["messages"][:-1] == pair["variant_messages"]
        assert row["messages"][-1] == {"role": "assistant", "content": target["response"]}
        assert row["response_sha256"] == sha256_text(row["messages"][-1]["content"])
        assert row["metadata"]["generator_identity"] == generator

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_bct_training_data(view, target_a, output)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate"])
def test_offline_target_import_requires_an_exact_unique_pair_id_join(tmp_path: Path, case: str) -> None:
    _, requests = _make_requests(tmp_path)
    results = _results(requests)
    if case == "missing":
        results = results[:-1]
        match = "ID mismatch"
    elif case == "extra":
        results.append({"pair_id": "extra-pair", "response": "Extra response."})
        match = "ID mismatch"
    else:
        results.append(dict(results[0]))
        match = "duplicate offline BCT target result"
    with pytest.raises(BCTTargetError, match=match):
        import_bct_target_results(
            requests,
            results,
            tmp_path / f"{case}.targets.jsonl",
            generator_identity=_request_generator(requests),
            decoding_parameters={"temperature": 0},
        )


def test_offline_target_import_rejects_prompt_and_response_hash_mismatch(tmp_path: Path) -> None:
    _, requests = _make_requests(tmp_path)
    results = _results(requests)
    bad_prompt = [dict(row) for row in results]
    bad_prompt[0]["clean_prompt_sha256"] = "0" * 64
    with pytest.raises(BCTTargetError, match="clean_prompt_sha256 mismatch"):
        import_bct_target_results(
            requests,
            bad_prompt,
            tmp_path / "bad-prompt.jsonl",
            generator_identity=_request_generator(requests),
            decoding_parameters={},
        )

    bad_response = [dict(row) for row in results]
    bad_response[0]["response_sha256"] = "1" * 64
    with pytest.raises(BCTTargetError, match="response_sha256 mismatch"):
        import_bct_target_results(
            requests,
            bad_response,
            tmp_path / "bad-response.jsonl",
            generator_identity=_request_generator(requests),
            decoding_parameters={},
        )


def test_offline_target_import_rejects_a_generator_other_than_the_request_identity(tmp_path: Path) -> None:
    _, requests = _make_requests(tmp_path)

    with pytest.raises(BCTTargetError, match="identity pinned by the target request"):
        import_bct_target_results(
            requests,
            _results(requests),
            tmp_path / "wrong-generator.jsonl",
            generator_identity=make_fixture_generator_identity("different-model-revision"),
            decoding_parameters={"temperature": 0},
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda rows: rows[0].__setitem__("clean_prompt_sha256", "0" * 64), "clean_prompt_sha256 mismatch"),
        (lambda rows: rows[0].__setitem__("response", "Tampered response."), "response_sha256 mismatch"),
        (
            lambda rows: rows[0]["generator_identity"].__setitem__("model", "tampered/model"),
            "generator identity hash mismatch",
        ),
        (
            lambda rows: rows[0]["decoding_parameters"].__setitem__("temperature", 0.9),
            "decoding parameters differ",
        ),
    ],
)
def test_target_reader_rejects_semantic_tampering_after_transport_hashes_are_updated(
    tmp_path: Path,
    tamper,
    message: str,
) -> None:
    _, _, targets, _, _ = _make_targets(tmp_path)
    _rewrite_artifact(targets, mutate_rows=tamper)
    with pytest.raises(BCTTargetError, match=message):
        read_bct_targets(targets)


def test_target_reader_and_export_reject_provenance_tampering_and_bare_pair_target(tmp_path: Path) -> None:
    view, _, targets, generator, decoding = _make_targets(tmp_path)

    with pytest.raises(BCTTargetError, match="generator identity differs"):
        read_bct_targets(targets, expected_generator_identity=make_fixture_generator_identity("different"))
    with pytest.raises(BCTTargetError, match="decoding parameters differ"):
        read_bct_targets(targets, expected_decoding_parameters={"temperature": 0.5})
    with pytest.raises(BCTTargetError, match="schema"):
        materialize_bct_training_data(view, view, tmp_path / "bare-pair-output.jsonl")

    def alter_view_lineage(manifest: dict[str, Any]) -> None:
        manifest["provenance"]["training_view_content_sha256"] = "f" * 64

    _rewrite_artifact(targets, mutate_manifest=alter_view_lineage)
    with pytest.raises(BCTTargetError, match="training-view lineage"):
        read_bct_targets(targets)

    assert generator["identity_sha256"]
    assert sha256_json(decoding)


def test_target_reader_rejects_self_consistent_generation_prompt_provenance_tampering(tmp_path: Path) -> None:
    _, _, targets, _, _ = _make_targets(tmp_path)

    def alter_generation_prompt_hash(manifest: dict[str, Any]) -> None:
        generation = manifest["provenance"]["generation_provenance"]
        generation["prompt_template_sha256"] = "f" * 64

    _rewrite_artifact(targets, mutate_manifest=alter_generation_prompt_hash)
    with pytest.raises(BCTTargetError, match="prompt_template_sha256"):
        read_bct_targets(targets)


def test_export_rejects_target_artifact_from_a_different_view(tmp_path: Path) -> None:
    view_a, _, targets_a, _, _ = _make_targets(tmp_path, stem="a")
    view_b = _make_view(tmp_path, stem="b")
    assert read_training_view(view_a)[1]["content_sha256"] == read_training_view(view_b)[1]["content_sha256"]

    # Change the second view's reconstruction seed by building a distinct domain
    # artifact; content identity, not paths, is the join authority.
    different_view = _make_view(tmp_path, domain=JAILBREAK_DOMAIN, stem="different")
    with pytest.raises(BCTTargetError, match="different training view"):
        materialize_bct_training_data(different_view, targets_a, tmp_path / "mismatched.jsonl")


def test_export_rejects_self_consistent_prompt_mismatch_and_missing_target_id(tmp_path: Path) -> None:
    view, _, targets, _, _ = _make_targets(tmp_path)

    mismatched_prompt = tmp_path / "mismatched-prompt.targets.jsonl"
    shutil.copyfile(targets, mismatched_prompt)
    shutil.copyfile(artifact_manifest_path(targets), artifact_manifest_path(mismatched_prompt))

    def replace_clean_prompt(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        row["clean_prompt"] = "A different clean prompt."
        row["reference_messages"][-1]["content"] = row["clean_prompt"]
        row["clean_prompt_sha256"] = sha256_text(row["clean_prompt"])
        row["reference_messages_sha256"] = sha256_json(row["reference_messages"])
        row["target_record_sha256"] = sha256_json(
            {key: value for key, value in row.items() if key != "target_record_sha256"}
        )

    _rewrite_artifact(mismatched_prompt, mutate_rows=replace_clean_prompt)
    with pytest.raises(BCTTargetError, match="prompt_template_sha256"):
        read_bct_targets(mismatched_prompt)
    with pytest.raises(BCTTargetError, match="prompt_template_sha256"):
        materialize_bct_training_data(view, mismatched_prompt, tmp_path / "prompt-mismatch-output.jsonl")

    missing_target = tmp_path / "missing-target.targets.jsonl"
    shutil.copyfile(targets, missing_target)
    shutil.copyfile(artifact_manifest_path(targets), artifact_manifest_path(missing_target))
    _rewrite_artifact(missing_target, mutate_rows=lambda rows: rows.pop())
    with pytest.raises(BCTTargetError, match="prompt_template_sha256|example manifest|ID mismatch"):
        materialize_bct_training_data(view, missing_target, tmp_path / "missing-target-output.jsonl")


def test_copying_verified_target_artifacts_does_not_weaken_pair_identity(tmp_path: Path) -> None:
    view, _, targets, _, _ = _make_targets(tmp_path)
    copied = tmp_path / "copied-targets.jsonl"
    shutil.copyfile(targets, copied)
    shutil.copyfile(artifact_manifest_path(targets), artifact_manifest_path(copied))
    assert read_bct_targets(copied)[0] == read_bct_targets(targets)[0]
    output = tmp_path / "copied-output.jsonl"
    materialize_bct_training_data(view, copied, output)
    assert read_bct_training_data(output)[0]
