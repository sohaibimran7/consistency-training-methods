"""Focused safety coverage for the Igor-shaped request-contract gate.

These tests deliberately exercise lifecycle boundaries with synthetic rows and
responses.  They never use a real endpoint, credentials, prompts, or model
outputs.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ctm_data.adapters.eval_awareness import figure6_request_contract as contract
from ctm_data.adapters.eval_awareness.figure6_generate import read_generation_records
from ctm_data.adapters.eval_awareness.figure6_materialize import FIGURE6_ARTIFACT_SCHEMA, make_condition_id
from ctm_data.adapters.eval_awareness.figure6_spec import DATASET_ID, DATASET_REVISION, get_model_spec


def _row(index: int, *, valence: str = "safety", condition: str = "baseline") -> dict[str, Any]:
    pair_id = f"pair-{index:03d}"
    task_id = f"{pair_id}_{valence}"
    return {
        "condition_id": make_condition_id(task_id, valence, condition),
        "pair_id": pair_id,
        "task_id": task_id,
        "task_name": pair_id,
        "valence": valence,
        "condition": condition,
        "prompt": f"Synthetic request-contract task {index} {valence} {condition}",
        "source_config": f"prompts_{valence}",
    }


def _rows() -> list[dict[str, Any]]:
    # Deliberately reverse baseline rows to prove that selection is stable.
    baseline = [_row(index) for index in reversed(range(contract.BASELINE_COUNT))]
    extras = [_row(index, valence="capability") for index in range(10)]
    extras += [_row(index, condition="F2") for index in range(10)]
    return [*baseline, *extras]


def _artifact_manifest() -> dict[str, Any]:
    return {
        "artifact_schema": FIGURE6_ARTIFACT_SCHEMA,
        "schema_version": 1,
        "row_count": 1800,
        "content_sha256": "a" * 64,
        "provenance": {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION},
    }


@dataclass(frozen=True)
class _Paths:
    root: Path
    artifact: Path
    prompt: Path
    output: Path
    attestation: Path


def _write_attestation(path: Path, **overrides: Any) -> Path:
    model = get_model_spec(contract.MODEL_KEY)
    value = {
        "schema": contract.SERVER_ATTESTATION_SCHEMA,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "vllm_version": contract.EXPECTED_VLLM_VERSION,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": contract.EXPECTED_MAX_MODEL_LEN,
        "reasoning_parser": "qwen3",
        "generation_config": contract.EXPECTED_GENERATION_CONFIG,
        "launch_command_sha256": "b" * 64,
    }
    value.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Paths:
    root = tmp_path.resolve()
    monkeypatch.setattr(contract, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(contract, "load_figure6_artifact", lambda _: (_rows(), _artifact_manifest()))
    monkeypatch.setattr(
        contract,
        "load_verified_model_prompt",
        lambda model_key, _: ("Synthetic pinned system prompt", get_model_spec(model_key).prompt),
    )
    return _Paths(
        root=root,
        artifact=root / "input" / "artifact.jsonl",
        prompt=root / "input" / "system.txt",
        output=root / "outputs" / "request-contract.jsonl",
        attestation=_write_attestation(root / "attestations" / "server.json"),
    )


def _completion(index: int, *, model: str | None = None) -> dict[str, Any]:
    return {
        "id": f"completion-{index}",
        "model": model or get_model_spec(contract.MODEL_KEY).model_id,
        "choices": [
            {
                "message": {"content": f"answer-{index}", "reasoning_content": f"reasoning-{index}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class _FakeModels:
    def __init__(self, served_model_ids: list[str], error: BaseException | None = None):
        self.served_model_ids = list(served_model_ids)
        self.error = error
        self.calls = 0

    async def list(self) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"data": [{"id": model_id} for model_id in self.served_model_ids]}


class _FakeCompletions:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("unexpected additional target request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(
        self,
        outcomes: list[Any],
        *,
        served_model_ids: list[str] | None = None,
        supports_models: bool = True,
    ):
        self.completions = _FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        if supports_models:
            self.models = _FakeModels(served_model_ids or [get_model_spec(contract.MODEL_KEY).model_id])
        else:
            self.models = SimpleNamespace()


class _BlockingCompletions:
    def __init__(self, target_started: int):
        self.target_started = target_started
        self.calls: list[dict[str, Any]] = []
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) >= self.target_started:
            self.all_started.set()
        await self.release.wait()
        return _completion(len(self.calls))


class _BlockingClient:
    def __init__(self, target_started: int):
        self.completions = _BlockingCompletions(target_started)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = _FakeModels([get_model_spec(contract.MODEL_KEY).model_id])


def _run(
    paths: _Paths,
    *,
    client: Any,
    scope: str = "wire-smoke",
    output: Path | None = None,
    attestation: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return asyncio.run(
        contract.run_request_contract_gate(
            paths.artifact,
            output or paths.output,
            prompt_path=paths.prompt,
            server_attestation_path=attestation or paths.attestation,
            scope=scope,
            client=client,
            **kwargs,
        )
    )


def _review(paths: _Paths, *, scope: str = "wire-smoke", **kwargs: Any) -> dict[str, Any]:
    return _run(paths, client=_FakeClient([]), scope=scope, dry_run=True, **kwargs)


def _execute(
    paths: _Paths,
    plan: dict[str, Any],
    client: Any,
    *,
    scope: str = "wire-smoke",
    **kwargs: Any,
) -> dict[str, Any]:
    return _run(
        paths,
        client=client,
        scope=scope,
        expected_plan_sha256=plan["plan_sha256"],
        confirm_requests=True,
        **kwargs,
    )


def _journal_rows(output: Path) -> list[dict[str, Any]]:
    journal = contract.request_contract_dispatch_path(output)
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]


def test_scope_is_exactly_20_or_100_sorted_safety_baseline_rows() -> None:
    rows = _rows()
    smoke = contract.select_request_contract_rows(rows, "wire-smoke")
    baseline = contract.select_request_contract_rows(rows, "baseline")

    assert len(smoke) == contract.WIRE_SMOKE_COUNT
    assert len(baseline) == contract.BASELINE_COUNT
    assert smoke == baseline[: contract.WIRE_SMOKE_COUNT]
    assert [row["pair_id"] for row in baseline] == sorted(row["pair_id"] for row in baseline)
    assert {(row["valence"], row["condition"]) for row in baseline} == {("safety", "baseline")}
    with pytest.raises(ValueError, match="scope"):
        contract.select_request_contract_rows(rows, "everything")


def test_request_body_has_only_igor_fields_and_omits_sampling_controls() -> None:
    request = contract.build_target_request(
        model_id="obalcells/qwen3-32b-mo-midtrained",
        system_prompt="system",
        task_prompt="task",
    )

    assert request == {
        "model": "obalcells/qwen3-32b-mo-midtrained",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "reasoning_effort": "medium",
    }
    assert not set(contract.OMITTED_REQUEST_FIELDS).intersection(request)


def test_external_target_endpoint_is_rejected_before_any_request(paths: _Paths) -> None:
    client = _FakeClient([])
    with pytest.raises(ValueError, match="loopback-only"):
        _run(paths, client=client, base_url="https://api.example.test/v1", dry_run=True)
    assert client.models.calls == 0
    assert client.completions.calls == []


def test_dry_run_is_content_free_stable_and_creates_only_lifecycle_sidecars(paths: _Paths) -> None:
    client = _FakeClient([])
    first = _run(paths, client=client, dry_run=True)
    second = _run(paths, client=client, dry_run=True)

    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["selected_count"] == first["pending"] == contract.WIRE_SMOKE_COUNT
    assert first["request_body_keys"] == ["messages", "model", "reasoning_effort"]
    serialized = json.dumps(first, sort_keys=True)
    assert "Synthetic pinned system prompt" not in serialized
    assert "Synthetic request-contract task" not in serialized
    assert client.completions.calls == []
    assert client.models.calls == 0
    assert not paths.output.exists()
    assert paths.output.with_suffix(paths.output.suffix + ".request-contract.lock").exists()
    dispatch_path = contract.request_contract_dispatch_path(paths.output)
    assert first["dispatch_journal"] == str(dispatch_path)
    assert dispatch_path.is_file()
    assert dispatch_path.read_bytes() == b""
    manifest_path = contract.request_contract_manifest_path(paths.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["plan_history"]) == 1
    assert manifest["approvals"] == []
    assert manifest["status"] == "planned"


def test_dispatch_intent_is_durable_before_the_network_send(paths: _Paths) -> None:
    plan = _review(paths, max_concurrency=1)
    client = _FakeClient([_completion(0), RuntimeError("stop after the ordering probe")])
    original_create = client.completions.create

    async def assert_intent_then_create(**kwargs: Any) -> Any:
        intents = _journal_rows(paths.output)
        assert len(intents) == len(client.completions.calls) + 1
        return await original_create(**kwargs)

    client.completions.create = assert_intent_then_create
    with pytest.raises(contract.RequestContractError, match="uncertain"):
        _execute(paths, plan, client, max_concurrency=1)
    assert len(client.completions.calls) == 2


def test_real_run_requires_a_dry_plan_in_its_own_lifecycle(paths: _Paths) -> None:
    reviewed_elsewhere = _review(paths)
    fresh_output = paths.root / "other" / "fresh.jsonl"
    client = _FakeClient([])

    with pytest.raises(contract.RequestContractError, match="pre-existing reviewed dry-run plan"):
        _run(
            paths,
            client=client,
            output=fresh_output,
            expected_plan_sha256=reviewed_elsewhere["plan_sha256"],
            confirm_requests=True,
        )

    assert client.models.calls == 0
    assert client.completions.calls == []


def test_fresh_baseline_is_rejected_before_any_target_request(paths: _Paths) -> None:
    client = _FakeClient([])

    with pytest.raises(contract.RequestContractError, match="baseline scope requires an approved, completed 20-generation"):
        _run(
            paths,
            client=client,
            scope="baseline",
            confirm_requests=True,
        )

    assert client.models.calls == 0
    assert client.completions.calls == []


def test_smoke_then_baseline_executes_20_then_only_the_remaining_80_and_completed_rerun_is_free(
    paths: _Paths,
) -> None:
    smoke_plan = _review(paths, max_concurrency=5)
    smoke_client = _FakeClient([_completion(index) for index in range(contract.WIRE_SMOKE_COUNT)])
    smoke = _execute(paths, smoke_plan, smoke_client, max_concurrency=5)

    assert smoke["complete"] is True
    assert len(smoke_client.completions.calls) == contract.WIRE_SMOKE_COUNT
    assert smoke_client.models.calls == 1
    assert set(smoke_client.completions.calls[0]) == {"model", "messages", "reasoning_effort"}
    assert smoke_client.completions.calls[0]["reasoning_effort"] == "medium"
    assert [message["role"] for message in smoke_client.completions.calls[0]["messages"]] == ["system", "user"]

    baseline_plan = _review(paths, scope="baseline", max_concurrency=5)
    assert baseline_plan["existing_successes"] == contract.WIRE_SMOKE_COUNT
    assert baseline_plan["pending"] == contract.BASELINE_COUNT - contract.WIRE_SMOKE_COUNT
    baseline_client = _FakeClient(
        [_completion(index) for index in range(contract.WIRE_SMOKE_COUNT, contract.BASELINE_COUNT)]
    )
    baseline = _execute(
        paths,
        baseline_plan,
        baseline_client,
        scope="baseline",
        max_concurrency=5,
    )

    assert baseline["complete"] is True
    assert len(baseline_client.completions.calls) == contract.BASELINE_COUNT - contract.WIRE_SMOKE_COUNT
    assert len(read_generation_records(paths.output)) == contract.BASELINE_COUNT
    assert len(_journal_rows(paths.output)) == contract.BASELINE_COUNT
    manifest = json.loads(contract.request_contract_manifest_path(paths.output).read_text(encoding="utf-8"))
    assert [entry["document"]["selected_count"] for entry in manifest["plan_history"]] == [20, 100]
    assert len(manifest["approvals"]) == 2

    rerun_client = _FakeClient([])
    rerun = _execute(paths, baseline_plan, rerun_client, scope="baseline", max_concurrency=5)
    assert rerun["complete"] is True
    assert rerun["api_calls_made"] == 0
    assert rerun_client.models.calls == 0
    assert rerun_client.completions.calls == []


def test_output_and_attestation_must_be_confined_to_artifacts(paths: _Paths) -> None:
    outside = paths.root.parent / f"{paths.root.name}-outside.json"
    _write_attestation(outside)

    with pytest.raises(contract.RequestContractError, match="outputs must stay under"):
        _run(paths, client=_FakeClient([]), output=outside, dry_run=True)
    with pytest.raises(contract.RequestContractError, match="outputs must stay under"):
        _run(paths, client=_FakeClient([]), attestation=outside, dry_run=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "unrecognized"}, "unsupported schema"),
        ({"model_revision": "wrong-revision"}, "model_revision"),
        ({"dtype": ""}, "invalid dtype"),
        ({"max_model_len": 32768}, "must remain fixed"),
        ({"unexpected": "field"}, "unsupported fields"),
        ({"launch_command_sha256": "A" * 64}, "lowercase SHA-256"),
    ],
)
def test_bad_or_unsanitized_server_attestation_is_rejected(
    paths: _Paths,
    overrides: dict[str, Any],
    message: str,
) -> None:
    attestation = _write_attestation(paths.root / "attestations" / f"bad-{message}.json", **overrides)
    client = _FakeClient([])

    with pytest.raises(contract.RequestContractError, match=message):
        _run(paths, client=client, attestation=attestation, dry_run=True)

    assert client.models.calls == 0
    assert client.completions.calls == []


@pytest.mark.parametrize("supports_models", [True, False], ids=["mismatch", "unsupported"])
def test_endpoint_identity_must_be_verified_before_chat_requests(paths: _Paths, supports_models: bool) -> None:
    plan = _review(paths, max_concurrency=1)
    client = _FakeClient(
        [],
        served_model_ids=["different-model"],
        supports_models=supports_models,
    )
    expected = "identity mismatch" if supports_models else "verified /models identity check"
    error_type = ValueError if supports_models else contract.RequestContractError

    with pytest.raises(error_type, match=expected):
        _execute(paths, plan, client, max_concurrency=1)

    assert client.completions.calls == []
    if supports_models:
        assert client.models.calls == 1


@pytest.mark.parametrize(
    "outcome",
    [
        _completion(0, model="unexpected-target-model"),
        RuntimeError("synthetic transport interruption"),
    ],
    ids=["unexpected-response-model", "transport-error"],
)
def test_single_target_failure_is_quarantined_once_and_never_automatically_resent(
    paths: _Paths,
    outcome: Any,
) -> None:
    plan = _review(paths, max_concurrency=1)
    failing_client = _FakeClient([outcome])

    with pytest.raises(contract.RequestContractError, match="uncertain"):
        _execute(paths, plan, failing_client, max_concurrency=1)

    assert len(failing_client.completions.calls) == 1
    records = read_generation_records(paths.output)
    assert len(records) == 1
    assert records[0]["status"] == "uncertain"
    assert records[0]["attempts"] == 1
    assert records[0]["resume_attempt"] == 1
    assert len(_journal_rows(paths.output)) == 1

    resume_client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="uncertain target outcomes"):
        _execute(paths, plan, resume_client, max_concurrency=1)
    assert resume_client.models.calls == 0
    assert resume_client.completions.calls == []


def test_double_cancellation_leaves_only_durable_intents_and_blocks_resume(paths: _Paths) -> None:
    concurrency = 3
    plan = _review(paths, max_concurrency=concurrency)

    async def cancel_active_run() -> _BlockingClient:
        client = _BlockingClient(concurrency)
        task = asyncio.create_task(
            contract.run_request_contract_gate(
                paths.artifact,
                paths.output,
                prompt_path=paths.prompt,
                server_attestation_path=paths.attestation,
                client=client,
                max_concurrency=concurrency,
                expected_plan_sha256=plan["plan_sha256"],
                confirm_requests=True,
            )
        )
        await asyncio.wait_for(client.completions.all_started.wait(), timeout=1)
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return client

    client = asyncio.run(cancel_active_run())
    assert len(client.completions.calls) == concurrency
    assert client.models.calls == 1
    records = read_generation_records(paths.output)
    assert records == []
    intents = _journal_rows(paths.output)
    assert len(intents) == concurrency
    assert len({intent["generation_key"] for intent in intents}) == concurrency

    resume_client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="uncertain target outcomes"):
        _execute(paths, plan, resume_client, max_concurrency=concurrency)
    assert resume_client.models.calls == 0
    assert resume_client.completions.calls == []


def test_post_response_output_append_failure_leaves_an_unresolved_intent_and_never_retries(
    paths: _Paths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _review(paths, max_concurrency=1)
    original_append = contract._append_durable

    def fail_only_generation_output(path: Path, record: dict[str, Any]) -> None:
        if Path(path).resolve() == paths.output:
            raise OSError("synthetic output append interruption")
        original_append(path, record)

    monkeypatch.setattr(contract, "_append_durable", fail_only_generation_output)
    client = _FakeClient([_completion(0)])
    with pytest.raises(contract.RequestContractError, match="unresolved outcomes"):
        _execute(paths, plan, client, max_concurrency=1)

    assert len(client.completions.calls) == 1
    assert read_generation_records(paths.output) == []
    intents = _journal_rows(paths.output)
    assert len(intents) == 1
    assert intents[0]["plan_sha256"] == plan["plan_sha256"]

    resume_client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="manual reconciliation"):
        _execute(paths, plan, resume_client, max_concurrency=1)
    assert resume_client.models.calls == 0
    assert resume_client.completions.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("tamper", "invalid identity"),
        ("duplicate", "duplicates"),
    ],
)
def test_tampered_or_duplicate_dispatch_journal_fails_closed_before_endpoint(
    paths: _Paths,
    mutation: str,
    message: str,
) -> None:
    plan = _review(paths, max_concurrency=5)
    _execute(
        paths,
        plan,
        _FakeClient([_completion(index) for index in range(contract.WIRE_SMOKE_COUNT)]),
        max_concurrency=5,
    )
    journal = contract.request_contract_dispatch_path(paths.output)
    intents = _journal_rows(paths.output)
    if mutation == "tamper":
        intents[0]["intent_id"] = "0" * 64
    else:
        intents.append(dict(intents[0]))
    journal.write_text(
        "".join(json.dumps(intent, sort_keys=True) + "\n" for intent in intents),
        encoding="utf-8",
    )

    client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match=message):
        _execute(paths, plan, client, max_concurrency=5)
    assert client.models.calls == 0
    assert client.completions.calls == []


def test_completed_manifest_commitment_blocks_resend_if_output_and_journal_disappear(
    paths: _Paths,
) -> None:
    plan = _review(paths, max_concurrency=5)
    _execute(
        paths,
        plan,
        _FakeClient([_completion(index) for index in range(contract.WIRE_SMOKE_COUNT)]),
        max_concurrency=5,
    )
    archive = paths.root / "_archive"
    archive.mkdir()
    paths.output.replace(archive / paths.output.name)
    journal = contract.request_contract_dispatch_path(paths.output)
    journal.replace(archive / journal.name)

    client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="manifest commitment"):
        _execute(paths, plan, client, max_concurrency=5)
    assert client.models.calls == 0
    assert client.completions.calls == []


def test_partial_manifest_commitment_blocks_resend_if_output_and_journal_disappear(
    paths: _Paths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _review(paths, max_concurrency=1)
    original_append = contract._append_durable
    journal = contract.request_contract_dispatch_path(paths.output)
    journal_writes = 0

    def fail_second_dispatch_intent(path: Path, record: dict[str, Any]) -> None:
        nonlocal journal_writes
        if Path(path).resolve() == journal:
            journal_writes += 1
            if journal_writes == 2:
                raise OSError("synthetic pre-dispatch journal interruption")
        original_append(path, record)

    monkeypatch.setattr(contract, "_append_durable", fail_second_dispatch_intent)
    client = _FakeClient([_completion(0)])
    with pytest.raises(contract.RequestContractError, match="pre-dispatch intent"):
        _execute(paths, plan, client, max_concurrency=1)
    assert len(client.completions.calls) == 1

    monkeypatch.setattr(contract, "_append_durable", original_append)
    archive = paths.root / "_archive"
    archive.mkdir()
    paths.output.replace(archive / paths.output.name)
    journal.replace(archive / journal.name)
    resume_client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="manifest commitment"):
        _execute(paths, plan, resume_client, max_concurrency=1)
    assert resume_client.models.calls == 0
    assert resume_client.completions.calls == []


def test_baseline_only_key_cannot_be_rebound_to_the_approved_smoke_plan(paths: _Paths) -> None:
    smoke_plan = _review(paths, max_concurrency=5)
    _execute(
        paths,
        smoke_plan,
        _FakeClient([_completion(index) for index in range(contract.WIRE_SMOKE_COUNT)]),
        max_concurrency=5,
    )
    baseline_plan = _review(paths, scope="baseline", max_concurrency=5)
    _execute(
        paths,
        baseline_plan,
        _FakeClient(
            [_completion(index) for index in range(contract.WIRE_SMOKE_COUNT, contract.BASELINE_COUNT)]
        ),
        scope="baseline",
        max_concurrency=5,
    )

    records = read_generation_records(paths.output)
    target = next(
        record
        for record in records
        if record["diagnostic_plan_sha256"] == baseline_plan["plan_sha256"]
    )
    target["diagnostic_plan_sha256"] = smoke_plan["plan_sha256"]
    paths.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    journal = contract.request_contract_dispatch_path(paths.output)
    intents = _journal_rows(paths.output)
    target_intent = next(intent for intent in intents if intent["generation_key"] == target["generation_key"])
    target_intent["plan_sha256"] = smoke_plan["plan_sha256"]
    target_intent["intent_id"] = contract._sha256_json(
        {
            "generation_key": target_intent["generation_key"],
            "plan_sha256": target_intent["plan_sha256"],
            "protocol_id": contract.PROTOCOL_ID,
        }
    )
    journal.write_text(
        "".join(json.dumps(intent, sort_keys=True) + "\n" for intent in intents),
        encoding="utf-8",
    )

    manifest_path = contract.request_contract_manifest_path(paths.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = {
        record["generation_key"]: record["diagnostic_plan_sha256"] for record in records
    }
    manifest["completed_record_bindings_sha256"] = contract._ids_sha256(
        [f"{logical_key}|{bindings[logical_key]}" for logical_key in sorted(bindings)]
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    client = _FakeClient([])
    with pytest.raises(contract.RequestContractError, match="outside its reviewed plan scope"):
        _execute(paths, baseline_plan, client, scope="baseline", max_concurrency=5)
    assert client.models.calls == 0
    assert client.completions.calls == []


def test_retry_controls_are_rejected_to_prevent_ambiguous_repeat_requests(paths: _Paths) -> None:
    with pytest.raises(ValueError, match="exactly one attempt"):
        _run(paths, client=_FakeClient([]), max_retries=2, dry_run=True)
    with pytest.raises(ValueError, match="prohibit retry backoff"):
        _run(paths, client=_FakeClient([]), retry_base_seconds=0.1, dry_run=True)
    with pytest.raises(ValueError, match="must not use an OpenAI API credential"):
        _run(paths, client=_FakeClient([]), api_key_env="OPENAI_API_KEY", dry_run=True)
