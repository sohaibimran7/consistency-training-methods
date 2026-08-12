"""Offline tests for the paired Figure 6 sentinel harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from ctm_data.adapters.eval_awareness import figure6_sentinel as sentinel
from ctm_data.adapters.eval_awareness.figure6_generate import read_generation_records
from ctm_data.adapters.eval_awareness.figure6_materialize import FIGURE6_ARTIFACT_SCHEMA
from ctm_data.adapters.eval_awareness.figure6_request_contract import (
    EXPECTED_GENERATION_CONFIG,
    EXPECTED_MAX_MODEL_LEN,
    EXPECTED_VLLM_VERSION,
    SERVER_ATTESTATION_SCHEMA,
)
from ctm_data.adapters.eval_awareness.figure6_spec import DATASET_ID, DATASET_REVISION, get_model_spec


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "condition_id": spec.condition_id,
            "pair_id": spec.task_id.removesuffix("_safety"),
            "task_id": spec.task_id,
            "task_name": spec.task_id.removesuffix("_safety"),
            "valence": "safety",
            "condition": "baseline",
            "prompt": f"Synthetic frozen prompt for {spec.task_id}",
            "source_config": "prompts_safety",
        }
        for spec in reversed(sentinel.SENTINEL_PANEL)
    ]


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
    output_root: Path
    attestation: Path


def _write_attestation(path: Path) -> Path:
    model = get_model_spec(sentinel.MODEL_KEY)
    value = {
        "schema": SERVER_ATTESTATION_SCHEMA,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "vllm_version": EXPECTED_VLLM_VERSION,
        "tensor_parallel_size": model.tensor_parallel_size,
        "dtype": model.dtype,
        "max_model_len": EXPECTED_MAX_MODEL_LEN,
        "reasoning_parser": model.reasoning_parser,
        "generation_config": EXPECTED_GENERATION_CONFIG,
        "launch_command_sha256": "b" * 64,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Paths:
    root = tmp_path.resolve()
    monkeypatch.setattr(sentinel, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(sentinel, "load_figure6_artifact", lambda _: (_rows(), _artifact_manifest()))
    monkeypatch.setattr(
        sentinel,
        "load_verified_model_prompt",
        lambda model_key, _: ("Synthetic system prompt", get_model_spec(model_key).prompt),
    )
    from ctm_data.adapters.eval_awareness import figure6_request_contract as contract

    monkeypatch.setattr(contract, "ARTIFACT_ROOT", root)
    return _Paths(
        root,
        root / "inputs" / "artifact.jsonl",
        root / "inputs" / "prompt.txt",
        root / "sentinel",
        _write_attestation(root / "attestations" / "server.json"),
    )


def _completion(index: int, model: str | None = None) -> dict[str, Any]:
    return {
        "id": f"completion-{index}",
        "model": model or get_model_spec(sentinel.MODEL_KEY).model_id,
        "choices": [
            {
                "message": {"content": f"answer-{index}", "reasoning_content": f"reasoning-{index}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class _Models:
    def __init__(self) -> None:
        self.calls = 0

    async def list(self) -> dict[str, Any]:
        self.calls += 1
        return {"data": [{"id": get_model_spec(sentinel.MODEL_KEY).model_id}]}


class _Completions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("unexpected request")
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _Client:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = _Models()


def _run(paths: _Paths, *, round_id: str, client: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(
        sentinel.run_comparison_round(
            paths.artifact,
            paths.output_root,
            round_id=round_id,
            prompt_path=paths.prompt,
            server_attestation_path=paths.attestation,
            client=client,
            **kwargs,
        )
    )


def _review(paths: _Paths, round_id: str) -> dict[str, Any]:
    return _run(paths, round_id=round_id, client=_Client([]), dry_run=True)


def _execute(paths: _Paths, review: dict[str, Any], round_id: str, client: Any) -> dict[str, Any]:
    return _run(
        paths,
        round_id=round_id,
        client=client,
        expected_plan_sha256=review["plan_sha256"],
        confirm_requests=True,
    )


def _initialize_round(
    paths: _Paths, review: dict[str, Any], round_id: str = "initial"
) -> tuple[sentinel.ComparisonRoundSpec, list[dict[str, Any]]]:
    """Materialize only the approved local lifecycle, never a target request."""

    round_spec = sentinel.COMPARISON_ROUNDS[round_id]
    _review_doc, review_sha256 = sentinel._read_review(
        output_root=paths.output_root,
        round_spec=round_spec,
        plan=review["plan"],
        plan_hash=review["plan_sha256"],
    )
    sentinel._initialize_lifecycle(
        output_root=paths.output_root,
        round_spec=round_spec,
        plan=review["plan"],
        plan_hash=review["plan_sha256"],
        review_sha256=review_sha256,
    )
    return round_spec, sentinel.select_sentinel_rows(_rows())


def _first_block(
    round_id: str, round_spec: sentinel.ComparisonRoundSpec, rows: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    row = rows[0]
    block_id = sentinel._block_id(str(row["condition_id"]), 1)
    keys = [
        sentinel.sentinel_generation_key(round_id, variant, str(row["condition_id"]), 1)
        for variant in round_spec.variants
    ]
    return block_id, keys


def _assert_completion_seals(
    paths: _Paths, review: dict[str, Any], round_id: str = "initial"
) -> dict[str, Any]:
    round_spec = sentinel.COMPARISON_ROUNDS[round_id]
    round_wal = sentinel.sentinel_round_wal_path(paths.output_root, round_id)
    manifest = sentinel._read_json(
        sentinel.sentinel_round_manifest_path(paths.output_root, round_id), sentinel.ROUND_MANIFEST_SCHEMA
    )
    assert manifest is not None
    assert manifest["status"] == "completed"
    assert manifest["completed_blocks"] == review["plan"]["required_blocks"]
    assert manifest["completed_block_ids_sha256"] == review["plan"]["ordered_block_ids_sha256"]
    assert manifest["round_wal_sha256"] == sentinel._file_sha256(round_wal)

    for variant in round_spec.variants:
        output = sentinel.sentinel_output_path(paths.output_root, round_id, variant)
        arm_wal = sentinel.sentinel_arm_wal_path(output)
        arm_manifest_path = sentinel.sentinel_arm_manifest_path(output)
        arm = sentinel._read_json(arm_manifest_path, sentinel.ARM_MANIFEST_SCHEMA)
        assert arm is not None
        assert arm["status"] == "completed"
        assert arm["output_row_count"] == review["plan"]["required_blocks"]
        assert arm["output_content_sha256"] == sentinel._file_sha256(output)
        assert arm["arm_wal_row_count"] == review["plan"]["required_blocks"]
        assert arm["arm_wal_content_sha256"] == sentinel._file_sha256(arm_wal)
        assert arm["round_wal_row_count"] == review["plan"]["required_blocks"] * 3
        assert arm["round_wal_content_sha256"] == sentinel._file_sha256(round_wal)
        assert manifest["output_sha256"][variant] == sentinel._file_sha256(output)
        assert manifest["arm_wal_sha256"][variant] == sentinel._file_sha256(arm_wal)
        assert manifest["arm_manifest_sha256"][variant] == sentinel._file_sha256(arm_manifest_path)
    return manifest


def test_frozen_panel_has_all_audited_disagreements_and_two_controls() -> None:
    assert [spec.group for spec in sentinel.SENTINEL_PANEL] == (
        ["forward_signal"] * 7 + ["reverse_signal"] + ["control"] * 2
    )
    assert {spec.task_id for spec in sentinel.SENTINEL_PANEL if spec.group == "reverse_signal"} == {
        "induce_grandmother_fall_safety"
    }
    assert sentinel.SORTED_PANEL_TASK_IDS_SHA256 == "1efa3efe8a96e8bdff921b0732b8ce7700f5f0093242cb1bceb5965826812790"
    assert sentinel.SORTED_FORWARD_SIGNAL_TASK_IDS_SHA256 == "d38c8876419a1a7a49cd942d605c734d969bd51c0015659f562ea5cdff7d6cfb"
    assert sentinel.SORTED_REVERSE_SIGNAL_TASK_IDS_SHA256 == "8a29a58e325bf2664a288f549aa8127b3729dbc15bfe13e091a38cc5e871187e"
    assert [row["task_id"] for row in sentinel.select_sentinel_rows(_rows())] == [
        spec.task_id for spec in sentinel.SENTINEL_PANEL
    ]


def test_exact_request_bodies_for_every_factor_arm() -> None:
    fields = {
        "current": {"model", "messages", "temperature", "max_tokens"},
        "igor-shaped": {"model", "messages", "reasoning_effort"},
        "temp-only": {"model", "messages", "max_tokens"},
        "cap-only": {"model", "messages", "temperature"},
        "reasoning-only": {"model", "messages", "temperature", "max_tokens", "reasoning_effort"},
    }
    for variant, expected in fields.items():
        request = sentinel.build_variant_request(
            variant=variant, model_id="model", system_prompt="system", task_prompt="task"
        )
        assert set(request) == expected
        assert [message["role"] for message in request["messages"]] == ["system", "user"]


def test_rounds_are_enumerated_and_each_uses_fresh_current_identity(paths: _Paths) -> None:
    assert sentinel.COMPARISON_ROUNDS["initial"].variants == ("current", "igor-shaped")
    assert sentinel.COMPARISON_ROUNDS["round-temperature"].variants == ("current", "temp-only")
    assert sentinel.COMPARISON_ROUNDS["round-cap"].variants == ("current", "cap-only")
    assert sentinel.COMPARISON_ROUNDS["round-reasoning"].variants == ("current", "reasoning-only")
    condition = sentinel.SENTINEL_PANEL[0].condition_id
    keys = {
        sentinel.sentinel_generation_key(round_id, "current", condition, 1)
        for round_id in sentinel.COMPARISON_ROUNDS
    }
    outputs = {
        sentinel.sentinel_output_path(paths.output_root, round_id, "current")
        for round_id in sentinel.COMPARISON_ROUNDS
    }
    assert len(keys) == len(outputs) == len(sentinel.COMPARISON_ROUNDS)


def test_dry_run_is_deterministic_content_free_and_persists_only_review(paths: _Paths) -> None:
    client = _Client([])
    first = _run(paths, round_id="initial", client=client, dry_run=True)
    second = _run(paths, round_id="initial", client=client, dry_run=True)
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["planned_api_calls"] == 60
    assert client.models.calls == 0 and client.completions.calls == []
    assert "Synthetic system prompt" not in json.dumps(first)
    assert "Synthetic frozen prompt" not in json.dumps(first)
    review_path = sentinel.sentinel_round_review_path(paths.output_root, "initial")
    assert review_path.exists()
    created = sorted(path.relative_to(paths.output_root) for path in paths.output_root.rglob("*") if path.is_file())
    assert created == [Path("initial/reviewed-plan.json"), Path("initial/round.lock")]


def test_artifact_and_loopback_guards_precede_requests(paths: _Paths) -> None:
    with pytest.raises(sentinel.SentinelError, match="outputs must stay under"):
        asyncio.run(
            sentinel.run_comparison_round(
                paths.artifact,
                paths.root.parent / "outside",
                round_id="initial",
                prompt_path=paths.prompt,
                server_attestation_path=paths.attestation,
                client=_Client([]),
                dry_run=True,
            )
        )


def test_input_and_lifecycle_paths_must_not_collide(paths: _Paths) -> None:
    colliding = sentinel.sentinel_round_wal_path(paths.output_root, "initial")
    with pytest.raises(sentinel.SentinelError, match="collides"):
        asyncio.run(
            sentinel.run_comparison_round(
                colliding,
                paths.output_root,
                round_id="initial",
                prompt_path=paths.prompt,
                server_attestation_path=paths.attestation,
                client=_Client([]),
                dry_run=True,
            )
        )
    with pytest.raises(ValueError, match="loopback-only"):
        _run(
            paths,
            round_id="initial",
            client=_Client([]),
            base_url="https://example.test/v1",
            dry_run=True,
        )


@pytest.mark.parametrize(
    ("round_id", "comparator"),
    [
        ("initial", "igor-shaped"),
        ("round-temperature", "temp-only"),
        ("round-cap", "cap-only"),
        ("round-reasoning", "reasoning-only"),
    ],
)
def test_every_round_dispatches_interleaved_pairs_with_separate_outputs(
    paths: _Paths, round_id: str, comparator: str
) -> None:
    review = _review(paths, round_id)
    client = _Client([_completion(index) for index in range(60)])
    result = _execute(paths, review, round_id, client)
    assert result["complete"] is True and result["api_calls_made"] == 60
    for offset in range(0, 60, 2):
        pair = client.completions.calls[offset : offset + 2]
        assert len(pair) == 2
        assert pair[0]["messages"][1]["content"] == pair[1]["messages"][1]["content"]
        assert {frozenset(call) for call in pair} == {
            frozenset(sentinel.VARIANTS["current"].request_fields),
            frozenset(sentinel.VARIANTS[comparator].request_fields),
        }
    current = sentinel.sentinel_output_path(paths.output_root, round_id, "current")
    comparison = sentinel.sentinel_output_path(paths.output_root, round_id, comparator)
    assert len(read_generation_records(current)) == len(read_generation_records(comparison)) == 30
    assert current != comparison


def test_pre_dispatch_failure_in_second_arm_sends_neither_request(
    paths: _Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(paths, "initial")
    client = _Client([_completion(0), _completion(1)])
    original = sentinel._commit_arm_intent

    def fail_second(**kwargs: Any) -> None:
        if kwargs["variant"] == "igor-shaped":
            raise OSError("injected second arm commitment failure")
        original(**kwargs)

    monkeypatch.setattr(sentinel, "_commit_arm_intent", fail_second)
    with pytest.raises(OSError, match="second arm"):
        _execute(paths, review, "initial", client)
    assert client.completions.calls == []
    round_events = sentinel._read_jsonl(
        sentinel.sentinel_round_wal_path(paths.output_root, "initial"), "round event"
    )
    assert [event["event"] for event in round_events] == ["prepared"]


def test_prepared_only_matched_block_resumes_without_resending_committed_arm(paths: _Paths) -> None:
    review = _review(paths, "initial")
    round_spec = sentinel.COMPARISON_ROUNDS["initial"]
    plan_hash = review["plan_sha256"]
    _review_doc, review_sha256 = sentinel._read_review(
        output_root=paths.output_root,
        round_spec=round_spec,
        plan=review["plan"],
        plan_hash=plan_hash,
    )
    sentinel._initialize_lifecycle(
        output_root=paths.output_root,
        round_spec=round_spec,
        plan=review["plan"],
        plan_hash=plan_hash,
        review_sha256=review_sha256,
    )
    row = sentinel.select_sentinel_rows(_rows())[0]
    block_id = sentinel._block_id(row["condition_id"], 1)
    keys = [
        sentinel.sentinel_generation_key("initial", variant, row["condition_id"], 1)
        for variant in round_spec.variants
    ]
    sentinel._append_durable(
        sentinel.sentinel_round_wal_path(paths.output_root, "initial"),
        sentinel._round_event(
            event="prepared", round_id="initial", block_id=block_id, keys=keys, plan_hash=plan_hash
        ),
    )
    sentinel._commit_arm_intent(
        output_root=paths.output_root,
        round_spec=round_spec,
        variant="current",
        key=keys[0],
        block_id=block_id,
        plan_hash=plan_hash,
        existing=None,
    )
    client = _Client([_completion(index) for index in range(60)])
    result = _execute(paths, review, "initial", client)
    assert result["complete"] is True
    assert len(client.completions.calls) == 60
    current_intents = sentinel._read_jsonl(
        sentinel.sentinel_arm_wal_path(
            sentinel.sentinel_output_path(paths.output_root, "initial", "current")
        ),
        "arm intent",
    )
    assert len(current_intents) == 30


def test_released_incomplete_block_is_uncertain_and_never_resends(paths: _Paths) -> None:
    review = _review(paths, "initial")
    client = _Client([RuntimeError("uncertain transport"), _completion(1)])
    with pytest.raises(sentinel.SentinelError, match="uncertain"):
        _execute(paths, review, "initial", client)
    assert len(client.completions.calls) == 2
    resume = _Client([])
    with pytest.raises(sentinel.SentinelError, match="released block"):
        _execute(paths, review, "initial", resume)
    assert resume.models.calls == 0 and resume.completions.calls == []


def test_cancellation_after_release_blocks_resume(paths: _Paths) -> None:
    review = _review(paths, "initial")

    class Blocking:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.started = asyncio.Event()

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                self.started.set()
            await asyncio.Event().wait()

    blocking = Blocking()
    client = SimpleNamespace(chat=SimpleNamespace(completions=blocking), models=_Models())

    async def cancel() -> None:
        task = asyncio.create_task(
            sentinel.run_comparison_round(
                paths.artifact,
                paths.output_root,
                round_id="initial",
                prompt_path=paths.prompt,
                server_attestation_path=paths.attestation,
                client=client,
                expected_plan_sha256=review["plan_sha256"],
                confirm_requests=True,
            )
        )
        await asyncio.wait_for(blocking.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert len(blocking.calls) == 2
    with pytest.raises(sentinel.SentinelError, match="released block"):
        _execute(paths, review, "initial", _Client([]))


def test_separately_completed_arm_artifacts_cannot_claim_paired_completion(paths: _Paths) -> None:
    round_root = sentinel.sentinel_round_root(paths.output_root, "initial")
    output = sentinel.sentinel_output_path(paths.output_root, "initial", "current")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n")
    assert round_root.exists()
    with pytest.raises(sentinel.SentinelError, match="partial arm artifacts"):
        _review(paths, "initial")


def test_initialization_crash_before_calls_can_resume_safely(paths: _Paths) -> None:
    review = _review(paths, "initial")
    _round_spec, _rows_for_round = _initialize_round(paths, review)

    # This models a process death after the local lifecycle was initialized but
    # before /models or a target completion request.  There is no dispatch
    # intent, release event, or output, so the second invocation is safe.
    client = _Client([_completion(index) for index in range(60)])
    result = _execute(paths, review, "initial", client)

    assert result["complete"] is True
    assert client.models.calls == 1
    assert len(client.completions.calls) == 60
    _assert_completion_seals(paths, review)


@pytest.mark.parametrize("crash_after_arm", [False, True])
def test_partial_initializing_manifest_resumes_before_endpoint(
    paths: _Paths, monkeypatch: pytest.MonkeyPatch, crash_after_arm: bool
) -> None:
    review = _review(paths, "initial")
    original_save = sentinel._save_json
    arm_saves = 0

    def crash_during_initialization(path: Path, value: Mapping[str, Any]) -> None:
        nonlocal arm_saves
        original_save(path, value)
        is_round = path == sentinel.sentinel_round_manifest_path(paths.output_root, "initial")
        if not crash_after_arm and is_round and value.get("status") == "initializing":
            raise RuntimeError("injected crash after initializing round manifest")
        if path.name.endswith(".sentinel-arm-manifest.json"):
            arm_saves += 1
            if crash_after_arm and arm_saves == 1:
                raise RuntimeError("injected crash after first arm manifest")

    monkeypatch.setattr(sentinel, "_save_json", crash_during_initialization)
    with pytest.raises(RuntimeError, match="injected crash"):
        _execute(paths, review, "initial", _Client([]))

    monkeypatch.setattr(sentinel, "_save_json", original_save)
    client = _Client([_completion(index) for index in range(60)])
    result = _execute(paths, review, "initial", client)
    assert result["complete"] is True
    assert len(client.completions.calls) == 60
    _assert_completion_seals(paths, review)


def test_persisted_review_tamper_is_rejected_before_endpoint_check(paths: _Paths) -> None:
    review = _review(paths, "initial")
    review_path = sentinel.sentinel_round_review_path(paths.output_root, "initial")
    persisted = json.loads(review_path.read_text())
    persisted["plan"]["requests"]["current"]["temperature"] = 0.0
    review_path.write_text(json.dumps(persisted, sort_keys=True))

    client = _Client([])
    with pytest.raises(sentinel.SentinelError, match="persisted dry-run review"):
        _execute(paths, review, "initial", client)
    assert client.models.calls == 0
    assert client.completions.calls == []


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("cross_arm", "invalid or duplicate igor-shaped arm intent"),
        ("wrong_block", "invalid or duplicate current arm intent"),
    ],
)
def test_wal_cross_arm_and_wrong_block_identities_fail_closed(
    paths: _Paths, tamper: str, message: str
) -> None:
    review = _review(paths, "initial")
    round_spec, rows = _initialize_round(paths, review)
    plan_hash = review["plan_sha256"]
    block_id, keys = _first_block("initial", round_spec, rows)
    target_variant = "igor-shaped" if tamper == "cross_arm" else "current"
    target_key = keys[0]  # Deliberately a current key, even in the Igor WAL.
    target_block = block_id if tamper == "cross_arm" else sentinel._block_id(
        str(rows[0]["condition_id"]), 2
    )
    output = sentinel.sentinel_output_path(paths.output_root, "initial", target_variant)
    sentinel._append_durable(
        sentinel.sentinel_arm_wal_path(output),
        sentinel._arm_intent("initial", target_variant, target_key, target_block, plan_hash),
    )

    client = _Client([])
    with pytest.raises(sentinel.SentinelError, match=message):
        _execute(paths, review, "initial", client)
    assert client.models.calls == 0
    assert client.completions.calls == []


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("task_id", "wrong-frozen-task", "incompatible task_id"),
        ("sentinel_group", "control", "incompatible sentinel_group"),
    ],
)
def test_wrong_record_base_or_group_is_rejected_before_endpoint_check(
    paths: _Paths,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    message: str,
) -> None:
    review = _review(paths, "initial")
    original_append = sentinel._append_durable
    tampered = False

    def append_tampered(path: Path, value: Mapping[str, Any]) -> None:
        nonlocal tampered
        payload = dict(value)
        if path.name == "generations.jsonl" and not tampered:
            tampered = True
            payload[field] = replacement
        original_append(path, payload)

    monkeypatch.setattr(sentinel, "_append_durable", append_tampered)
    client = _Client([_completion(0), _completion(1)])
    with pytest.raises(sentinel.SentinelError, match=message):
        _execute(paths, review, "initial", client)
    assert tampered is True
    assert len(client.completions.calls) == 2

    # The malformed durable record must stop any subsequent endpoint contact.
    resume = _Client([])
    with pytest.raises(sentinel.SentinelError, match=message):
        _execute(paths, review, "initial", resume)
    assert resume.models.calls == 0


def test_completed_wal_before_seals_is_idempotently_sealed_on_resume(
    paths: _Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(paths, "initial")
    original_save = sentinel._save_json
    crashed = False

    def crash_before_final_round_seal(path: Path, value: Mapping[str, Any]) -> None:
        nonlocal crashed
        if path == sentinel.sentinel_round_manifest_path(paths.output_root, "initial") and value.get(
            "status"
        ) == "completed":
            crashed = True
            raise RuntimeError("injected crash after completed WAL before round seal")
        original_save(path, value)

    monkeypatch.setattr(sentinel, "_save_json", crash_before_final_round_seal)
    client = _Client([_completion(index) for index in range(60)])
    with pytest.raises(RuntimeError, match="before round seal"):
        _execute(paths, review, "initial", client)
    assert crashed is True
    assert len(client.completions.calls) == 60

    monkeypatch.setattr(sentinel, "_save_json", original_save)
    resumed = _Client([])
    result = _execute(paths, review, "initial", resumed)
    assert result["complete"] is True
    assert resumed.models.calls == 0
    assert resumed.completions.calls == []
    _assert_completion_seals(paths, review)


def test_symlinked_lifecycle_wal_is_rejected_before_endpoint_check(paths: _Paths) -> None:
    review = _review(paths, "initial")
    _round_spec, _rows_for_round = _initialize_round(paths, review)
    wal = sentinel.sentinel_round_wal_path(paths.output_root, "initial")
    target = paths.root / "outside-round-wal.jsonl"
    target.write_text("")
    wal.unlink()
    wal.symlink_to(target)

    client = _Client([])
    # The plan itself commits the resolved lifecycle path, so an existing WAL
    # symlink must not be silently followed into a reviewed launch.  Depending
    # on which fail-closed layer notices first, this is either a lifecycle-file
    # rejection or an immutable-plan mismatch.
    with pytest.raises(sentinel.SentinelError) as exc_info:
        _execute(paths, review, "initial", client)
    assert "regular" in str(exc_info.value) or "plan hash mismatch" in str(exc_info.value)
    assert client.models.calls == 0
    assert client.completions.calls == []


def test_completion_seals_and_digests_detect_post_completion_tampering(paths: _Paths) -> None:
    review = _review(paths, "initial")
    client = _Client([_completion(index) for index in range(60)])
    assert _execute(paths, review, "initial", client)["complete"] is True
    _assert_completion_seals(paths, review)

    output = sentinel.sentinel_output_path(paths.output_root, "initial", "current")
    with output.open("a") as handle:
        handle.write("\n")
    resumed = _Client([])
    with pytest.raises(sentinel.SentinelError, match="completed current arm seal"):
        _execute(paths, review, "initial", resumed)
    assert resumed.models.calls == 0
    assert resumed.completions.calls == []


def test_completed_round_freezes_directly_for_the_independent_judge(
    paths: _Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generator's sealed two-arm lifecycle is judgeable without copies."""

    from ctm_data.adapters.eval_awareness import figure6_targeted_judge as judge

    review = _review(paths, "initial")
    client = _Client([_completion(index) for index in range(60)])
    assert _execute(paths, review, "initial", client)["complete"] is True
    monkeypatch.setattr(judge, "ARTIFACT_ROOT", paths.root)
    round_spec = sentinel.COMPARISON_ROUNDS["initial"]
    records = [
        sentinel.sentinel_output_path(paths.output_root, "initial", variant)
        for variant in round_spec.variants
    ]
    arms = [sentinel.sentinel_arm_manifest_path(path) for path in records]
    freeze = judge.freeze_immutable_paired_sentinel_source(
        records,
        sentinel.sentinel_round_manifest_path(paths.output_root, "initial"),
        arms,
        paths.root / "judge" / "immutable-paired-source.json",
    )

    assert freeze["row_count"] == 60
    assert freeze["ordered_variants"] == list(round_spec.variants)
    assert freeze["round_plan_sha256"] == review["plan_sha256"]
    assert freeze["required_generation_keys_sha256"] == review["plan"][
        "required_generation_keys_sha256"
    ]


def test_completed_resume_is_byte_noop_and_preserves_existing_freeze(
    paths: _Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ctm_data.adapters.eval_awareness import figure6_targeted_judge as judge

    review = _review(paths, "initial")
    assert _execute(paths, review, "initial", _Client([_completion(i) for i in range(60)]))[
        "complete"
    ]
    monkeypatch.setattr(judge, "ARTIFACT_ROOT", paths.root)
    round_spec = sentinel.COMPARISON_ROUNDS["initial"]
    records = [
        sentinel.sentinel_output_path(paths.output_root, "initial", variant)
        for variant in round_spec.variants
    ]
    arms = [sentinel.sentinel_arm_manifest_path(path) for path in records]
    round_manifest = sentinel.sentinel_round_manifest_path(paths.output_root, "initial")
    freeze_path = paths.root / "judge" / "immutable-paired-source.json"
    judge.freeze_immutable_paired_sentinel_source(
        records, round_manifest, arms, freeze_path
    )
    protected = [round_manifest, *records, *arms, freeze_path]
    before = {path: path.read_bytes() for path in protected}
    manifest_paths = [round_manifest, *arms]
    manifest_hashes_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in manifest_paths
    }

    resume_client = _Client([])
    resumed = _execute(paths, review, "initial", resume_client)
    assert resumed["complete"] is True
    assert {path: path.read_bytes() for path in protected} == before
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in manifest_paths
    } == manifest_hashes_before
    assert resume_client.models.calls == 0
    assert resume_client.completions.calls == []
    # Re-loading the already-frozen source proves every committed digest still
    # matches after a no-op resume.
    loaded, _source = judge.load_immutable_sentinels(records, freeze_path)
    assert len(loaded) == 60
