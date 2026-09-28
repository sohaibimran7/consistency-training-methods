from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctm.artifacts import read_verified_artifact_manifest
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import (
    ANSWER_FORMAT_INSTRUCTION,
    materialize_shared_qid_two_bias,
    reconstruct_suggested_answer,
)
from experiments.act_expanded import materialize_arguments as arguments
from experiments.act_expanded import selection


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(values: list[str]) -> str:
    return _sha256("\n".join(values).encode("utf-8"))


def _candidate(dataset: str, rank: int) -> dict[str, object]:
    canonical = selection._render_canonical_input(
        f"Fresh {dataset} question {rank}?", ["correct", "distractor", "other"]
    )
    ground_truth = "A"
    labels = selection._answer_labels(canonical)
    fingerprint = selection._content_fingerprint(canonical)
    return {
        "biased_option": selection._deterministic_biased_option(canonical, ground_truth, labels),
        "candidate_id": _sha256(f"{dataset}\0{fingerprint}".encode("utf-8")),
        "canonical_input": canonical,
        "content_fingerprint": fingerprint,
        "ground_truth": ground_truth,
        "question_id": hashlib.sha1(canonical.encode("utf-8")).hexdigest(),
        "selection_rank": rank,
        "selection_role": "homogeneous_gemma_generation_candidate",
        "source_dataset": dataset,
        "source_origin": "pinned_hf_train_question",
        "source_question_id": hashlib.sha1(canonical.encode("utf-8")).hexdigest(),
        "source_row_index": rank,
    }


def _selection_fixture(
    tmp_path: Path, *, per_dataset: int = 1
) -> tuple[Path, Path, list[dict[str, object]]]:
    rows = [
        _candidate(dataset, rank)
        for rank in range(per_dataset)
        for dataset in ("logiqa", "hellaswag")
    ]
    payload = selection._canonical_jsonl(rows)
    data_sha = _sha256(payload)
    data = tmp_path / selection.CANDIDATE_FILENAME.format(sha256=data_sha)
    data.write_bytes(payload)
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    ids_sha = selection._values_sha256(candidate_ids)
    manifest = {
        "schema": selection.SELECTION_SCHEMA,
        "schema_version": selection.SCHEMA_VERSION,
        "kind": selection.SELECTION_KIND,
        "prompt_style": selection.PROMPT_STYLE,
        "bias_type": selection.BIAS_TYPE,
        "canonical_pair_transform": selection.CANONICAL_PAIR_TRANSFORM,
        "candidate_source_mode": "fresh_only",
        "requested_balanced_n_total": len(rows),
        "question_candidates": {
            "filename": data.name,
            "content_sha256": data_sha,
            "row_count": len(rows),
            "counts_by_dataset": {"logiqa": per_dataset, "hellaswag": per_dataset},
            "counts_by_source_origin": {"pinned_hf_train_question": len(rows)},
            "counts_by_dataset_and_source_origin": {
                "logiqa": {"pinned_hf_train_question": per_dataset},
                "hellaswag": {"pinned_hf_train_question": per_dataset},
            },
            "candidate_ids_sha256": ids_sha,
        },
        "homogeneous_argument_generation_contract": {
            "provider": "openrouter",
            "model": "google/gemma-4-31b-it",
            "required_for_every_candidate_id_sha256": ids_sha,
            "must_generate_one_new_wrong_argument_for_every_selected_question": True,
            "legacy_biasing_text_is_not_a_valid_input_or_output": True,
        },
    }
    manifest_payload = selection._canonical_json(manifest)
    manifest_path = tmp_path / selection.SELECTION_MANIFEST_FILENAME.format(
        sha256=_sha256(manifest_payload)
    )
    manifest_path.write_bytes(manifest_payload)
    return data, manifest_path, rows


def _native_protected_row(dataset: str) -> dict[str, object]:
    question = selection._render_canonical_input(f"Protected {dataset}?", ["yes", "no"])
    biased_option = "B"
    return {
        "question": question,
        "question_id": f"protected-{dataset}",
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": question + ANSWER_FORMAT_INSTRUCTION}],
        "biased_messages": [{"role": "user", "content": "argument\n" + question + ANSWER_FORMAT_INSTRUCTION}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": biased_option,
        "biasing_text": "argument for B",
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _artifact_entry(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    payload = path.read_bytes()
    ids = [str(row["question_id"]) for row in rows]
    return {
        "path": str(path),
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
    }


def _protected_manifests(tmp_path: Path) -> tuple[Path, Path]:
    iid_ids = ["iid-logiqa", "iid-hellaswag"]
    iid = tmp_path / "iid.manifest.json"
    iid.write_text(
        json.dumps(
            {
                "splits": {
                    "train_eval": {
                        "row_count": len(iid_ids),
                        "question_ids": iid_ids,
                        "question_ids_sha256": _ids_sha256(iid_ids),
                    }
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    wrong_rows = [_native_protected_row(dataset) for dataset in ("logiqa", "hellaswag")]
    clean_rows = [
        {
            key: row[key]
            for key in (
                "question",
                "question_id",
                "source_dataset",
                "prompt_style",
                "unbiased_messages",
                "ground_truth",
            )
        }
        for row in wrong_rows
    ]
    suggested_rows = []
    for row in wrong_rows:
        rebuilt = reconstruct_suggested_answer(str(row["question"]), str(row["biased_option"]))
        suggested_rows.append(
            {
                **row,
                "bias_type": "suggested_answer",
                "biased_messages": rebuilt["messages"],
                "biasing_text": rebuilt["biasing_text"],
            }
        )
    paths: dict[str, Path] = {}
    for name, rows in (("unbiased", clean_rows), ("wrong_argument", wrong_rows), ("suggested_answer", suggested_rows)):
        path = tmp_path / f"protected-{name}.jsonl"
        path.write_bytes(_jsonl(rows))
        paths[name] = path
    stage2 = tmp_path / "stage2.manifest.json"
    stage2.write_text(
        json.dumps(
            {
                "populations": {
                    "in_domain": {
                        "artifacts": {
                            "unbiased": _artifact_entry(paths["unbiased"], clean_rows),
                            "wrong_argument": _artifact_entry(paths["wrong_argument"], wrong_rows),
                            "suggested_answer": _artifact_entry(paths["suggested_answer"], suggested_rows),
                        }
                    }
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    return iid, stage2


def test_cap_preflight_normalizes_nested_aliases_and_request_is_exact() -> None:
    row = _candidate("logiqa", 0)
    request = arguments.build_generation_request(row)
    assert sorted(request) == ["messages", "model"]
    assert request["model"] == "google/gemma-4-31b-it"
    for key in (
        "max_tokens",
        "maxCompletionTokens",
        "maxOutputTokens",
        "maxGenerationTokens",
        "generationMaxTokens",
        "completionMaxTokens",
        "outputMaxTokens",
        "generationMaxLength",
        "maxLength",
        "reasoningTokens",
        "generation_max_tokens",
        "completion_max_tokens",
        "output_max_tokens",
        "generation_max_length",
    ):
        with pytest.raises(ValueError, match="must not set output-token field"):
            arguments.assert_no_output_token_cap({"nested": {key: 1}})


def test_generation_is_durable_resumable_and_rejects_length_stops(tmp_path: Path) -> None:
    candidates, manifest, rows = _selection_fixture(tmp_path)
    journal = tmp_path / "attempts.jsonl"
    calls: list[dict[str, object]] = []

    async def generate(request: dict[str, object]) -> arguments.GenerationResponse:
        calls.append(request)
        target = request["messages"][0]["content"].split("justify the answer ", 1)[1][0]
        return arguments.GenerationResponse(
            completion=f"A plausible case. Therefore, the best answer is: ({target}).",
            response_id=f"response-{len(calls)}",
            reported_model="google/gemma-4-31b-it",
            finish_reason="stop",
        )

    progress = asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=manifest,
            journal=journal,
            n_total=2,
            max_connections=2,
            max_candidates_this_run=1,
            generate_call=generate,
        )
    )
    assert progress.accepted_after == 1
    assert progress.scheduled_this_run == 1
    assert len(journal.read_text().splitlines()) == 1

    resumed_remaining = asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=manifest,
            journal=journal,
            n_total=2,
            max_connections=2,
            generate_call=generate,
        )
    )
    assert resumed_remaining.accepted_before == 1
    assert resumed_remaining.accepted_after == 2
    assert resumed_remaining.scheduled_this_run == 1
    assert len(journal.read_text().splitlines()) == 2

    async def should_not_run(request: dict[str, object]) -> arguments.GenerationResponse:
        raise AssertionError(f"accepted candidate was queried again: {request}")

    resumed = asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=manifest,
            journal=journal,
            n_total=2,
            generate_call=should_not_run,
        )
    )
    assert resumed.accepted_before == resumed.accepted_after == 2
    assert resumed.attempts_appended == 0
    assert resumed.scheduled_this_run == 0

    accepted, _ = arguments._validate_journal(
        [json.loads(line) for line in journal.read_text().splitlines()], rows
    )
    length_row = dict(next(iter(accepted.values())))
    length_row["finish_reason"] = "length"
    length_row["accepted"] = False
    length_row["acceptance_reason"] = "unrequested_length_limit_stop"
    length_row["natural_termination"] = False
    length_row["candidate_id"] = rows[0]["candidate_id"]
    length_row["question_id"] = rows[0]["question_id"]
    length_row["source_dataset"] = rows[0]["source_dataset"]
    length_row["biased_option"] = rows[0]["biased_option"]
    length_row["attempt_index"] = 1
    replayed, _ = arguments._validate_journal([length_row], [rows[0]])
    assert replayed == {}


def test_transport_failures_are_redacted_journalled_and_do_not_cancel_peers(tmp_path: Path) -> None:
    candidates, manifest, rows = _selection_fixture(tmp_path)
    journal = tmp_path / "transport-attempts.jsonl"

    async def generate(request: dict[str, object]) -> arguments.GenerationResponse:
        user = str(request["messages"][1]["content"])
        if "Fresh logiqa" in user:
            raise OSError("secret provider text that must never be journalled")
        target = str(request["messages"][0]["content"]).split("justify the answer ", 1)[1][0]
        return arguments.GenerationResponse(
            completion=f"A case. Therefore, the best answer is: ({target}).",
            reported_model="google/gemma-4-31b-it",
            finish_reason="stop",
        )

    progress = asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=manifest,
            journal=journal,
            n_total=2,
            attempts_per_candidate=2,
            generate_call=generate,
        )
    )
    assert progress.accepted_after == 1
    assert progress.transport_errors == 2
    journal_text = journal.read_text()
    assert "secret provider text" not in journal_text
    decoded = [json.loads(line) for line in journal_text.splitlines()]
    failed = [row for row in decoded if row["finish_reason"] == "transport_error"]
    assert len(failed) == 2
    assert all(row["completion"] == "" for row in failed)
    assert all(row["generation_error_class"] == "builtins.OSError" for row in failed)
    accepted, attempts = arguments._validate_journal(decoded, rows)
    assert len(accepted) == 1
    assert sorted(attempts.values()) == [1, 2]


def test_openai_sdk_wire_body_is_exact_and_runtime_is_attested() -> None:
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    row = _candidate("logiqa", 0)
    request = arguments.build_generation_request(row)
    observed_bodies: list[dict[str, object]] = []

    async def handler(http_request: object) -> object:
        raw = await http_request.aread()
        observed_bodies.append(json.loads(raw))
        return httpx.Response(
            200,
            json={
                "id": "mock-id",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemma-4-31b-it",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "A case. Therefore, the best answer is: (B).",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def exercise() -> tuple[arguments.GenerationResponse, dict[str, object]]:
        preflight = arguments._SerializedRequestPreflight()
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [preflight]},
        )
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://openrouter.invalid/api/v1",
            max_retries=0,
            http_client=http_client,
        )
        try:
            response = await arguments._openrouter_generate(request, client=client)
            policy = arguments._openai_runtime_policy(
                client,
                request=request,
                serialized_preflight=preflight,
            )
            return response, policy
        finally:
            await client.close()

    response, policy = asyncio.run(exercise())
    assert observed_bodies == [request]
    assert sorted(observed_bodies[0]) == ["messages", "model"]
    assert response.reported_model == "google/gemma-4-31b-it"
    assert policy["serialized_request_preflight_observed"] is True
    assert policy["openai_sdk_version"] == importlib.metadata.version("openai")
    assert policy["sdk_max_retries"] == 0


@pytest.mark.parametrize(
    ("reported_model", "choice_count", "message"),
    [
        ("not-the-pinned-model", 1, "pinned generator"),
        ("google/gemma-4-31b-it", 2, "exactly one choice"),
    ],
)
def test_openrouter_response_model_and_single_choice_are_fail_closed(
    reported_model: str, choice_count: int, message: str
) -> None:
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="Therefore, the best answer is: (B)."),
            finish_reason="stop",
        )
        for _ in range(choice_count)
    ]
    response = SimpleNamespace(
        id="id",
        model=reported_model,
        choices=choices,
        usage=None,
    )

    class FakeCompletions:
        async def create(self, **request: object) -> object:
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    with pytest.raises(ValueError, match=message):
        asyncio.run(arguments._openrouter_generate(arguments.build_generation_request(_candidate("logiqa", 0)), client=client))


def test_finalize_is_content_addressed_and_feeds_shared_two_bias_builder(tmp_path: Path) -> None:
    candidates, selection_manifest, _ = _selection_fixture(tmp_path)
    journal = tmp_path / "attempts.jsonl"

    async def generate(request: dict[str, object]) -> arguments.GenerationResponse:
        target = request["messages"][0]["content"].split("justify the answer ", 1)[1][0]
        return arguments.GenerationResponse(
            completion=f"A coherent case. Therefore, the best answer is: ({target}).",
            reported_model="google/gemma-4-31b-it",
            finish_reason="stop",
        )

    asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=selection_manifest,
            journal=journal,
            n_total=2,
            generate_call=generate,
        )
    )
    with pytest.raises(ValueError, match="rejects injected generator rows"):
        arguments.finalize_arguments(
            candidate_selection=candidates,
            selection_manifest=selection_manifest,
            journal=journal,
            output_dir=tmp_path / "must-not-publish",
            n_total=2,
        )
    result = arguments.finalize_arguments(
        candidate_selection=candidates,
        selection_manifest=selection_manifest,
        journal=journal,
        output_dir=tmp_path / "final",
        n_total=2,
        _allow_injected_test_runtime=True,
    )
    manifest = read_verified_artifact_manifest(
        result.data_path,
        manifest_path=result.manifest_path,
        expected_schema="ctm.mcq_bias.canonical_consistency_pairs",
        expected_schema_version=1,
    )
    rows = [json.loads(line) for line in result.data_path.read_text().splitlines()]
    assert manifest["counts"]["by_dataset"] == {"hellaswag": 1, "logiqa": 1}
    assert all(row["question"] + ANSWER_FORMAT_INSTRUCTION == row["unbiased_messages"][0]["content"] for row in rows)
    assert all(row["biased_messages"][0]["content"].endswith(row["unbiased_messages"][0]["content"]) for row in rows)
    assert all(row["argument_generation_provenance"]["finish_reason"] == "stop" for row in rows)
    assert all(row["argument_generation_provenance"]["natural_termination"] is True for row in rows)
    assert all(len(row["argument_generation_provenance"]["runtime_policy_sha256"]) == 64 for row in rows)

    iid_manifest, stage2_manifest = _protected_manifests(tmp_path)
    shared = materialize_shared_qid_two_bias(
        result.data_path,
        result.manifest_path,
        iid_manifest,
        stage2_manifest,
        tmp_path / "shared",
        qids_per_dataset=1,
        qids_per_dataset_per_segment=1,
    )
    shared_manifest = json.loads(shared.manifest_path.read_text())
    assert shared_manifest["row_count"] == 2
    assert shared_manifest["counts"]["by_dataset"] == {"logiqa": 1, "hellaswag": 1}


def test_finalize_can_publish_a_balanced_accepted_subset(tmp_path: Path) -> None:
    candidates, selection_manifest, rows = _selection_fixture(tmp_path, per_dataset=2)
    journal = tmp_path / "attempts.jsonl"

    async def generate(request: dict[str, object]) -> arguments.GenerationResponse:
        target = request["messages"][0]["content"].split("justify the answer ", 1)[1][0]
        return arguments.GenerationResponse(
            completion=f"A coherent case. Therefore, the best answer is: ({target}).",
            reported_model="google/gemma-4-31b-it",
            finish_reason="stop",
        )

    asyncio.run(
        arguments.generate_missing_arguments(
            candidate_selection=candidates,
            selection_manifest=selection_manifest,
            journal=journal,
            n_total=4,
            generate_call=generate,
        )
    )
    result = arguments.finalize_arguments(
        candidate_selection=candidates,
        selection_manifest=selection_manifest,
        journal=journal,
        output_dir=tmp_path / "balanced-subset",
        n_total=4,
        target_per_dataset=1,
        _allow_injected_test_runtime=True,
    )
    manifest = read_verified_artifact_manifest(
        result.data_path,
        manifest_path=result.manifest_path,
        expected_schema="ctm.mcq_bias.canonical_consistency_pairs",
        expected_schema_version=1,
    )
    assert manifest["row_count"] == 2
    assert manifest["counts"]["accepted_arguments"] == 2
    assert manifest["counts"]["accepted_arguments_available"] == 4
    assert manifest["provenance"]["selection"]["candidate_population_n_total"] == 4
    assert manifest["provenance"]["selection"]["target_per_dataset"] == 1
    assert [row["question_id"] for row in map(json.loads, result.data_path.read_text().splitlines())] == [
        rows[0]["question_id"],
        rows[1]["question_id"],
    ]
