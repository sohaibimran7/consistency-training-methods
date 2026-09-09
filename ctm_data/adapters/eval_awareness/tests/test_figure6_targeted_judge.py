"""Offline safety tests for the targeted sentinel DeepSeek screen."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest

from ctm_data.adapters.eval_awareness import figure6_targeted_judge as judge
from ctm_data.adapters.eval_awareness.figure6_judge import PAPER_JUDGE_TEMPLATE_SHA256


# This is the generator's public semantic protocol value, intentionally
# duplicated here rather than importing a live generator module.  The judge
# has to validate immutable records from any compatible target generator.
GENERATOR_PROTOCOL_ID = "figure6-targeted-sentinel-v1"
_GROUPS = ["forward_signal"] * 7 + ["reverse_signal"] + ["control"] * 2


def _paper_template() -> bytes:
    path = Path("experiments/eval_awareness/figure6/inputs/judgePromptEvidenceBoth.txt")
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PAPER_JUDGE_TEMPLATE_SHA256
    return payload


def _record(
    index: int, *, variant: str = "current", sentinel_group: str | None = None
) -> dict[str, Any]:
    return {
        "schema": "ctm.eval_awareness.figure6_generation.v1",
        "diagnostic_protocol_id": GENERATOR_PROTOCOL_ID,
        "diagnostic_variant": variant,
        "sentinel_group": sentinel_group or _GROUPS[(index - 1) % len(_GROUPS)],
        "generation_key": f"qwen_mo_mid|sentinel|{variant}|condition-{index}|1",
        "model_key": "qwen_mo_mid",
        "model_id": "obalcells/qwen3-32b-mo-midtrained",
        "model_revision": "a" * 40,
        "condition_id": f"condition-{index}",
        "task_id": f"task-{index}",
        "replicate": 1,
        "prompt": f"synthetic task {index}",
        "reasoning": f"synthetic reasoning {index}",
        "answer": f"synthetic answer {index}",
        # The direct generator schema keeps a raw completion independently of
        # the extracted final answer; these must not be forced to be identical.
        "response": f"<think>synthetic trace {index}</think> synthetic answer {index}",
        "trace_present": True,
        "trace_source": "reasoning_content",
        "generation_provenance": {"provenance_sha256": f"unit-{variant}"},
        "status": "success",
    }


def _verdict(aware: bool = False) -> dict[str, Any]:
    conclusion = "Yes" if aware else "No"
    awareness_type = "safety" if aware else "N/A"
    return {
        "model_awareness": {
            "conclusion": conclusion,
            "awareness_type": awareness_type,
            "evidence": ["synthetic evidence"],
            "reasoning": "synthetic analysis",
        },
        "hypothetical_recognition": {
            "conclusion": "No",
            "evidence": ["synthetic evidence"],
            "reasoning": "synthetic analysis",
        },
        "task_performance": {
            "conclusion": "No",
            "evidence": ["synthetic evidence"],
            "reasoning": "synthetic analysis",
        },
    }


def _write_source(root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    records = root / "source" / "generations.jsonl"
    records.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    records.write_bytes(payload)
    manifest = root / "source" / "immutable.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "unit.immutable_sentinel.v1",
                "row_count": len(rows),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    template = root / "inputs" / "judge.txt"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_bytes(_paper_template())
    return records, manifest, template


def _write_paired_v2_source(
    root: Path, *, variants: list[str] | None = None
) -> tuple[list[Path], Path, list[Path], Path]:
    """Construct a sealed generator-v2-shaped paired round without imports."""

    variants = ["current", "igor-shaped"] if variants is None else variants
    records_paths: list[Path] = []
    arm_manifest_paths: list[Path] = []
    output_paths: dict[str, str] = {}
    required_key_hashes: dict[str, str] = {}
    for arm_index, variant in enumerate(variants):
        records_path = root / "paired" / variant / "generations.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [_record((arm_index * 30) + index + 1, variant=variant) for index in range(30)]
        for index, row in enumerate(rows):
            row["sentinel_group"] = _GROUPS[(index // 3) % len(_GROUPS)]
        for index, row in enumerate(rows, start=1):
            row["comparison_round_id"] = "initial"
            row["round_plan_sha256"] = "0" * 64
            row["generation_key"] = f"paired-{variant}-{index}"
        payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
        records_path.write_bytes(payload)
        records_paths.append(records_path)
        output_paths[variant] = str(records_path)
        required_key_hashes[variant] = hashlib.sha256(
            "\n".join(sorted(f"paired-{variant}-{index}" for index in range(1, 31))).encode()
        ).hexdigest()
        arm_manifest_paths.append(records_path.with_suffix(".jsonl.sentinel-arm-manifest.json"))

    plan = {
        "schema": "ctm.eval_awareness.figure6_sentinel_round_plan.v2",
        "protocol_id": GENERATOR_PROTOCOL_ID,
        "comparison_round_id": "initial",
        "ordered_variants": variants,
        "model_key": "qwen_mo_mid",
        "model_id": "obalcells/qwen3-32b-mo-midtrained",
        "required_blocks": 30,
        "ordered_block_ids_sha256": "1" * 64,
        "required_generation_keys_sha256": required_key_hashes,
        "outputs": output_paths,
        "panel": {
            "ordered_groups": ["forward_signal"] * 7 + ["reverse_signal"] + ["control"] * 2,
            "ordered_task_ids": [f"task-{index}" for index in range(10)],
            "ordered_condition_ids": [f"condition-{index}" for index in range(10)],
            "sorted_task_ids_sha256": "2" * 64,
            "sorted_forward_signal_task_ids_sha256": "3" * 64,
            "sorted_reverse_signal_task_ids_sha256": "4" * 64,
            "sorted_control_task_ids_sha256": "5" * 64,
        },
    }
    plan_sha = judge._sha256_json(plan)
    round_wal_path = root / "paired" / "round-wal.jsonl"
    round_wal_path.parent.mkdir(parents=True, exist_ok=True)
    round_wal_path.write_text(
        "".join(json.dumps({"event": event, "block": block}) + "\n" for block in range(30) for event in ("prepared", "released", "completed")),
        encoding="utf-8",
    )
    arm_wal_paths: dict[str, Path] = {}
    for variant, records_path, arm_manifest_path in zip(variants, records_paths, arm_manifest_paths, strict=True):
        rows = records_path.read_bytes()
        arm_wal_path = records_path.with_suffix(".jsonl.sentinel-arm-wal.jsonl")
        arm_wal_path.write_text(
            "".join(json.dumps({"generation_key": f"paired-{variant}-{index}"}) + "\n" for index in range(1, 31)),
            encoding="utf-8",
        )
        arm_wal_paths[variant] = arm_wal_path
        arm_manifest_path.write_text(
            json.dumps(
                {
                    "schema": "ctm.eval_awareness.figure6_sentinel_arm_manifest.v2",
                    "status": "completed",
                    "comparison_round_id": "initial",
                    "variant": variant,
                    "round_plan_sha256": plan_sha,
                    "output_content_sha256": hashlib.sha256(rows).hexdigest(),
                    "output_row_count": 30,
                    "dispatched_generation_keys_sha256": required_key_hashes[variant],
                    "arm_wal_row_count": 30,
                    "arm_wal_content_sha256": hashlib.sha256(arm_wal_path.read_bytes()).hexdigest(),
                    "round_wal_row_count": 90,
                    "round_wal_content_sha256": hashlib.sha256(round_wal_path.read_bytes()).hexdigest(),
                    "completed_success_count": 30,
                    "completed_success_keys_sha256": judge._ids_sha256(
                        sorted(f"paired-{variant}-{index}" for index in range(1, 31))
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        actual_rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        for row in actual_rows:
            row["round_plan_sha256"] = plan_sha
        records_path.write_bytes(
            b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in actual_rows)
        )
        # Update the seal after assigning the real plan hash to output rows.
        arm = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
        arm["output_content_sha256"] = hashlib.sha256(records_path.read_bytes()).hexdigest()
        arm_manifest_path.write_text(json.dumps(arm, sort_keys=True), encoding="utf-8")

    # The round seal is written only after the final arm-manifest bytes exist.
    # (The real generator has the same ordering.)
    arm_manifest_digests = {
        variant: hashlib.sha256(arm_manifest_path.read_bytes()).hexdigest()
        for variant, arm_manifest_path in zip(variants, arm_manifest_paths, strict=True)
    }

    round_manifest = root / "paired" / "round-manifest.json"
    round_manifest.write_text(
        json.dumps(
            {
                "schema": "ctm.eval_awareness.figure6_sentinel_round_manifest.v2",
                "status": "completed",
                "comparison_round_id": "initial",
                "plan": plan,
                "plan_sha256": plan_sha,
                "completed_blocks": 30,
                "completed_block_ids_sha256": "1" * 64,
                "round_wal_sha256": hashlib.sha256(round_wal_path.read_bytes()).hexdigest(),
                "round_wal_row_count": 90,
                "output_sha256": {
                    variant: hashlib.sha256(records_path.read_bytes()).hexdigest()
                    for variant, records_path in zip(variants, records_paths, strict=True)
                },
                "arm_wal_sha256": {
                    variant: hashlib.sha256(arm_wal_paths[variant].read_bytes()).hexdigest() for variant in variants
                },
                "arm_manifest_sha256": {
                    variant: arm_manifest_digests[variant] for variant in variants
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    template = root / "inputs" / "judge.txt"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_bytes(_paper_template())
    return records_paths, round_manifest, arm_manifest_paths, template


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("unexpected network request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return httpx.Response(200, request=request, json=outcome)


class _BlockingTransport(httpx.AsyncBaseTransport):
    """A local request that can only end through cancellation."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.requests: list[httpx.Request] = []
        self._never = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.started.set()
        await self._never.wait()
        raise AssertionError("blocking local transport was unexpectedly released")


def _response(*, aware: bool = False) -> dict[str, Any]:
    return {
        "id": "resp-unit",
        "model": judge.DEEPSEEK_MODEL,
        "provider": "DeepSeek",
        "choices": [
            {
                "message": {"content": json.dumps(_verdict(aware))},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "attempt_log_path": root / "judge" / "attempts.jsonl",
        "lifecycle_manifest_path": root / "judge" / "manifest.json",
        "private_verdicts_path": root / "judge" / "private.jsonl",
        "aggregate_path": root / "judge" / "awareness.json",
    }


def _run(
    records: Path | list[Path],
    source_manifest: Path,
    template: Path,
    paths: dict[str, Path],
    **kwargs: Any,
) -> dict[str, Any]:
    return asyncio.run(
        judge.judge_targeted_sentinels(
            records,
            source_manifest,
            judge_template_path=template,
            api_key=kwargs.pop("api_key", None),
            **paths,
            **kwargs,
        )
    )


@pytest.fixture
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    monkeypatch.setattr(judge, "ARTIFACT_ROOT", root)
    return root


def test_generator_record_schema_aliases_are_directly_accepted(rooted: Path) -> None:
    record = _record(1, variant="igor-shaped")
    normalized = judge.validate_sentinel_records([record])
    assert normalized[0]["variant_id"] == "igor-shaped"
    assert normalized[0]["protocol_id"] == GENERATOR_PROTOCOL_ID
    assert normalized[0]["sentinel_group"] == "forward_signal"
    assert judge.custom_id_for_sentinel(normalized[0]).startswith("figure6-targeted-")


def test_invalid_sentinel_group_is_rejected(rooted: Path) -> None:
    record = _record(1, sentinel_group="not-a-panel-group")
    with pytest.raises(judge.TargetedJudgeError, match="sentinel_group"):
        judge.validate_sentinel_records([record])


def test_dry_plan_uses_exact_paper_deepseek_user_k1_and_makes_no_call(rooted: Path) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1), _record(2, variant="igor-shaped")])
    paths = _paths(rooted)
    result = _run(records, source_manifest, template, paths, dry_run=True, concurrency=2)

    assert result["request_count"] == 2
    lifecycle = json.loads(paths["lifecycle_manifest_path"].read_text(encoding="utf-8"))
    plan = lifecycle["plan"]
    assert plan["judge"]["model"] == judge.DEEPSEEK_MODEL
    assert plan["judge"]["message_role"] == "user"
    assert plan["judge"]["judge_k"] == 1
    assert plan["judge"]["request_body_keys"] == ["messages", "model"]
    assert plan["judge"]["response_provider_policy"] == judge.DEEPSEEK_RESPONSE_PROVIDER_POLICY
    assert plan["judge_template_sha256"] == PAPER_JUDGE_TEMPLATE_SHA256
    rendered = json.dumps(result, sort_keys=True)
    assert "synthetic task" not in rendered
    assert "synthetic reasoning" not in rendered


def test_paid_requires_review_hash_and_success_writes_only_awareness_aggregate(rooted: Path) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1), _record(2, variant="igor-shaped")])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True, concurrency=1)
    transport = _Transport([_response(aware=False), _response(aware=True)])
    client = httpx.AsyncClient(transport=transport)
    try:
        result = _run(
            records,
            source_manifest,
            template,
            paths,
            api_key="unit-key",
            confirm_paid=True,
            expected_plan_sha256=dry["plan_sha256"],
            concurrency=1,
            client=client,
        )
    finally:
        asyncio.run(client.aclose())

    assert len(transport.requests) == 2
    for request in transport.requests:
        body = json.loads(request.content)
        assert set(body) == {"model", "messages"}
        assert body["model"] == judge.DEEPSEEK_MODEL
        assert body["messages"][0]["role"] == "user"
    aggregate = json.loads(paths["aggregate_path"].read_text(encoding="utf-8"))
    assert aggregate["awareness_yes_count"] == 1
    assert aggregate["awareness_no_count"] == 1
    assert len(aggregate["variant_groups"]) == 2
    assert {row["sentinel_group"] for row in aggregate["variant_groups"]} == {"forward_signal"}
    assert sum(row["sentinel_count"] for row in aggregate["variant_groups"]) == 2
    assert sum(row["awareness_yes_count"] for row in aggregate["variant_groups"]) == 1
    public_text = paths["aggregate_path"].read_text(encoding="utf-8")
    assert "parsed_verdict" not in public_text
    private_text = paths["private_verdicts_path"].read_text(encoding="utf-8")
    assert "parsed_verdict" in private_text
    assert "sentinel_group" in private_text
    assert result["completed"] == 2


def test_attempt_journal_is_durable_before_the_first_paid_dispatch(rooted: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True)
    fsynced_kinds: list[int] = []
    original_fsync = judge.os.fsync

    def record_fsync(descriptor: int) -> None:
        fsynced_kinds.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(judge.os, "fsync", record_fsync)

    class CheckingTransport(_Transport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert paths["attempt_log_path"].is_file()
            assert paths["attempt_log_path"].read_bytes().strip()
            return await super().handle_async_request(request)

    transport = CheckingTransport([_response()])
    client = httpx.AsyncClient(transport=transport)
    try:
        _run(
            records,
            source_manifest,
            template,
            paths,
            api_key="unit-key",
            confirm_paid=True,
            expected_plan_sha256=dry["plan_sha256"],
            client=client,
        )
    finally:
        asyncio.run(client.aclose())

    assert len(transport.requests) == 1
    assert any(stat.S_ISREG(mode) for mode in fsynced_kinds)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_kinds)


def test_paid_validation_error_blocks_resume_and_does_not_send_other_sentinels(rooted: Path) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1), _record(2)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True, concurrency=1)
    bad = _response()
    bad["choices"][0]["message"]["content"] = "not json"
    transport = _Transport([bad])
    client = httpx.AsyncClient(transport=transport)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                concurrency=1,
                client=client,
            )
    finally:
        asyncio.run(client.aclose())
    assert len(transport.requests) == 1
    resumed = _Transport([])
    resume_client = httpx.AsyncClient(transport=resumed)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                concurrency=1,
                client=resume_client,
            )
    finally:
        asyncio.run(resume_client.aclose())
    assert resumed.requests == []


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("model", "another/provider"),
        ("provider_missing", None),
        ("provider", ""),
        ("provider", "   "),
        ("finish_reason", "length"),
    ],
)
def test_wrong_paid_response_identity_is_quarantined_without_resend(
    rooted: Path, mutation: str, value: str | None
) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1), _record(2)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True, concurrency=1)
    malformed = _response()
    if mutation == "finish_reason":
        malformed["choices"][0]["finish_reason"] = value
    elif mutation == "provider_missing":
        malformed.pop("provider")
    else:
        malformed[mutation] = value
    transport = _Transport([malformed])
    client = httpx.AsyncClient(transport=transport)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                concurrency=1,
                client=client,
            )
    finally:
        asyncio.run(client.aclose())
    assert len(transport.requests) == 1
    attempts = [json.loads(line) for line in paths["attempt_log_path"].read_text(encoding="utf-8").splitlines()]
    terminal = [row for row in attempts if row["record_type"] == "terminal"]
    assert terminal[0]["status"] == "uncertain"
    assert terminal[0]["failure"]["kind"] == "paid_response_validation"

    resumed = _Transport([])
    resume_client = httpx.AsyncClient(transport=resumed)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                concurrency=1,
                client=resume_client,
            )
    finally:
        asyncio.run(resume_client.aclose())
    assert resumed.requests == []


def test_non_vendor_downstream_provider_is_accepted_and_retained(rooted: Path) -> None:
    """OpenRouter may infer the pinned model through a non-DeepSeek host."""

    records, source_manifest, template = _write_source(rooted, [_record(1)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True)
    response = _response()
    response["provider"] = "Independent Host"
    transport = _Transport([response])
    client = httpx.AsyncClient(transport=transport)
    try:
        result = _run(
            records,
            source_manifest,
            template,
            paths,
            api_key="unit-key",
            confirm_paid=True,
            expected_plan_sha256=dry["plan_sha256"],
            client=client,
        )
    finally:
        asyncio.run(client.aclose())

    assert result["completed"] == 1
    attempts = [json.loads(line) for line in paths["attempt_log_path"].read_text(encoding="utf-8").splitlines()]
    terminal = next(row for row in attempts if row["record_type"] == "terminal")
    assert terminal["status"] == "success"
    assert terminal["provider"] == "Independent Host"


@pytest.mark.parametrize("write_before_failure", [True, False])
def test_post_response_terminal_append_failure_never_resends(
    rooted: Path, monkeypatch: pytest.MonkeyPatch, write_before_failure: bool
) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True)
    original_append = judge._append_durable
    failed_once = False

    def fail_one_terminal(path: Path, row: Any) -> None:
        nonlocal failed_once
        if row.get("record_type") == "terminal" and not failed_once:
            failed_once = True
            if write_before_failure:
                original_append(path, row)
            raise OSError("injected terminal-journal failure")
        original_append(path, row)

    monkeypatch.setattr(judge, "_append_durable", fail_one_terminal)
    transport = _Transport([_response()])
    client = httpx.AsyncClient(transport=transport)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="worker failed"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                client=client,
            )
    finally:
        asyncio.run(client.aclose())
    assert len(transport.requests) == 1

    attempts = [json.loads(line) for line in paths["attempt_log_path"].read_text(encoding="utf-8").splitlines()]
    terminals = [row for row in attempts if row["record_type"] == "terminal"]
    assert len(terminals) == 1
    resumed = _Transport([])
    resume_client = httpx.AsyncClient(transport=resumed)
    try:
        if write_before_failure:
            result = _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                client=resume_client,
            )
            assert result["completed"] == 1
            assert terminals[0]["status"] == "success"
        else:
            with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
                _run(
                    records,
                    source_manifest,
                    template,
                    paths,
                    api_key="unit-key",
                    confirm_paid=True,
                    expected_plan_sha256=dry["plan_sha256"],
                    client=resume_client,
                )
            assert terminals[0]["status"] == "uncertain"
    finally:
        asyncio.run(resume_client.aclose())
    assert resumed.requests == []


def test_double_cancellation_persists_uncertainty_and_blocks_resume(rooted: Path) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1)])
    paths = _paths(rooted)
    dry = _run(records, source_manifest, template, paths, dry_run=True)

    async def cancel_twice() -> None:
        transport = _BlockingTransport()
        client = httpx.AsyncClient(transport=transport)
        try:
            task = asyncio.create_task(
                judge.judge_targeted_sentinels(
                    records,
                    source_manifest,
                    judge_template_path=template,
                    **paths,
                    api_key="unit-key",
                    confirm_paid=True,
                    expected_plan_sha256=dry["plan_sha256"],
                    client=client,
                )
            )
            await asyncio.wait_for(transport.started.wait(), timeout=1.0)
            task.cancel()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await client.aclose()

    asyncio.run(cancel_twice())
    attempts = [json.loads(line) for line in paths["attempt_log_path"].read_text(encoding="utf-8").splitlines()]
    terminals = [row for row in attempts if row["record_type"] == "terminal"]
    assert len(terminals) == 1
    assert terminals[0]["status"] == "uncertain"
    assert terminals[0]["failure"]["kind"] == "cancellation_or_process_interrupt"

    resumed = _Transport([])
    resume_client = httpx.AsyncClient(transport=resumed)
    try:
        with pytest.raises(judge.TargetedJudgeError, match="paid uncertainty"):
            _run(
                records,
                source_manifest,
                template,
                paths,
                api_key="unit-key",
                confirm_paid=True,
                expected_plan_sha256=dry["plan_sha256"],
                client=resume_client,
            )
    finally:
        asyncio.run(resume_client.aclose())
    assert resumed.requests == []


def test_freeze_completed_generator_source_is_content_only_and_variant_bound(rooted: Path) -> None:
    records, _, _ = _write_source(rooted, [_record(1, variant="current")])
    generator_manifest = rooted / "generator" / "generations.jsonl.sentinel-manifest.json"
    generator_manifest.parent.mkdir(parents=True, exist_ok=True)
    generator_manifest.write_text(
        json.dumps(
            {
                "schema": "ctm.eval_awareness.figure6_sentinel_manifest.v1",
                "status": "completed",
                "variant": "current",
                "completed_successes": 1,
                "plan_sha256": "a" * 64,
                "plan": {"output": str(records)},
            }
        ),
        encoding="utf-8",
    )
    frozen = judge.freeze_immutable_sentinel_source(records, generator_manifest, rooted / "frozen.json")
    assert frozen["row_count"] == 1
    assert frozen["variant"] == "current"
    assert frozen["content_sha256"] == hashlib.sha256(records.read_bytes()).hexdigest()
    loaded, source = judge.load_immutable_sentinels(records, rooted / "frozen.json")
    assert len(loaded) == 1
    assert source["records_sha256"] == frozen["content_sha256"]


def test_freeze_and_directly_ingest_completed_paired_v2_round_without_raw_copy(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, template = _write_paired_v2_source(rooted)
    frozen_path = rooted / "paired" / "judge-source-freeze.json"
    frozen = judge.freeze_immutable_paired_sentinel_source(
        records_paths, round_manifest, arm_manifest_paths, frozen_path
    )
    assert frozen["schema"] == judge.PAIRED_SOURCE_FREEZE_SCHEMA
    assert frozen["row_count"] == 60
    assert frozen["ordered_variants"] == ["current", "igor-shaped"]
    assert not any("prompt" in key or "response" in key for key in frozen)
    persisted = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert "synthetic task" not in json.dumps(persisted, sort_keys=True)

    loaded, source = judge.load_immutable_sentinels(records_paths[::-1], frozen_path)
    assert len(loaded) == 60
    assert source["row_count"] == 60
    assert source["paired_round_plan_sha256"] == frozen["round_plan_sha256"]
    paths = _paths(rooted)
    dry = _run(records_paths, frozen_path, template, paths, dry_run=True)
    assert dry["request_count"] == 60


def test_paired_freeze_rejects_output_and_arm_self_report_tampering(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(rooted)
    arm_path = arm_manifest_paths[0]
    output_path = records_paths[0]
    # Simulate an attacker modifying an output and its mutable arm manifest,
    # while the independently sealed round manifest remains unchanged.
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "changed synthetic answer"
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    arm["output_content_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    arm_path.write_text(json.dumps(arm, sort_keys=True), encoding="utf-8")
    with pytest.raises(judge.TargetedJudgeError, match="round output digest|arm-manifest digest"):
        judge.freeze_immutable_paired_sentinel_source(
            records_paths, round_manifest, arm_manifest_paths, rooted / "paired" / "tampered-freeze.json"
        )


def test_paired_loader_uses_committed_arm_order_not_lexical_input_order(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(
        rooted, variants=["current", "cap-only"]
    )
    # cap-only sorts before current on disk, so this reproduces later-round
    # lexical ordering while the passed paths are deliberately reversed.
    frozen_path = rooted / "paired" / "lexical-freeze.json"
    frozen = judge.freeze_immutable_paired_sentinel_source(
        records_paths[::-1], round_manifest, arm_manifest_paths[::-1], frozen_path
    )
    assert frozen["ordered_variants"] == ["current", "cap-only"]
    loaded, _ = judge.load_immutable_sentinels(records_paths[::-1], frozen_path)
    assert [row["diagnostic_variant"] for row in loaded[:30]] == ["current"] * 30


def test_paired_round_can_be_copied_then_frozen_and_loaded_at_its_new_root(rooted: Path) -> None:
    """A remote sealed round may be transferred without rewriting its plan.

    The generator's plan correctly retains the remote absolute paths as part
    of its byte-sealed provenance.  The judge must nevertheless derive local
    arm files from the committed ``<variant>/generations.jsonl`` layout under
    the copied local round root; it must not trust either remote paths or
    arbitrary caller-selected replacements.
    """

    remote_root = rooted / "isambard-artifacts"
    remote_records, remote_round_manifest, _remote_arm_manifests, _ = _write_paired_v2_source(remote_root)
    local_round_root = rooted / "transferred-artifacts" / "initial"
    shutil.copytree(remote_round_manifest.parent, local_round_root)
    local_records = [
        local_round_root / record_path.relative_to(remote_round_manifest.parent)
        for record_path in remote_records
    ]
    local_arm_manifests = [
        record_path.with_suffix(".jsonl.sentinel-arm-manifest.json") for record_path in local_records
    ]
    local_round_manifest = local_round_root / "round-manifest.json"
    local_freeze = rooted / "local-judge" / "source-manifest.json"

    frozen = judge.freeze_immutable_paired_sentinel_source(
        local_records[::-1],
        local_round_manifest,
        local_arm_manifests[::-1],
        local_freeze,
    )
    assert frozen["round_manifest_path"] == str(local_round_manifest)
    loaded, source = judge.load_immutable_sentinels(local_records[::-1], local_freeze)
    assert len(loaded) == 60
    assert source["records_paths"] == [str(path) for path in local_records]

    # The source freeze is local-root-specific: the untouched source round is
    # not silently accepted just because its bytes happen to be identical.
    with pytest.raises(judge.TargetedJudgeError, match="bound to different records"):
        judge.load_immutable_sentinels(remote_records, local_freeze)

    # Copying did not turn the freeze into a trust-once assertion.  A later
    # content mutation must still break the round/arm digest chain at B.
    local_records[0].write_bytes(local_records[0].read_bytes() + b"\n")
    with pytest.raises(judge.TargetedJudgeError, match="output digest changed"):
        judge.load_immutable_sentinels(local_records, local_freeze)


def test_paired_freeze_and_loader_reject_parent_symlink_substitution(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(rooted)
    source_freeze = rooted / "paired" / "source-freeze.json"
    judge.freeze_immutable_paired_sentinel_source(records_paths, round_manifest, arm_manifest_paths, source_freeze)

    arm_directory = records_paths[0].parent
    diverted_directory = arm_directory.with_name(f"{arm_directory.name}-diverted")
    arm_directory.rename(diverted_directory)
    arm_directory.symlink_to(diverted_directory, target_is_directory=True)

    with pytest.raises(judge.TargetedJudgeError, match="must not contain a symlink"):
        judge.load_immutable_sentinels(records_paths, source_freeze)
    with pytest.raises(judge.TargetedJudgeError, match="must not contain a symlink"):
        judge.freeze_immutable_paired_sentinel_source(
            records_paths, round_manifest, arm_manifest_paths, rooted / "paired" / "parent-symlink-freeze.json"
        )


def test_paired_freeze_and_loader_reject_leaf_symlink_substitution(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(rooted)
    source_freeze = rooted / "paired" / "source-freeze.json"
    judge.freeze_immutable_paired_sentinel_source(records_paths, round_manifest, arm_manifest_paths, source_freeze)

    output = records_paths[0]
    diverted_output = output.with_name("sealed-generations.jsonl")
    output.rename(diverted_output)
    output.symlink_to(diverted_output)

    with pytest.raises(judge.TargetedJudgeError, match="must not contain a symlink"):
        judge.load_immutable_sentinels(records_paths, source_freeze)
    with pytest.raises(judge.TargetedJudgeError, match="must not contain a symlink"):
        judge.freeze_immutable_paired_sentinel_source(
            records_paths, round_manifest, arm_manifest_paths, rooted / "paired" / "leaf-symlink-freeze.json"
        )


def test_artifact_root_symlink_is_rejected_before_descendant_validation(
    rooted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = rooted / "real-artifacts"
    real_root.mkdir()
    symlink_root = rooted / "artifact-root-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(judge, "ARTIFACT_ROOT", symlink_root)

    with pytest.raises(judge.TargetedJudgeError, match="artifact root must be a regular non-symlink directory"):
        judge.require_artifact_path(symlink_root / "copied-round" / "round-manifest.json")


def test_paired_freeze_rejects_success_key_commitment_not_equal_to_dispatch(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(rooted)
    arm_path = arm_manifest_paths[0]
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    arm["completed_success_keys_sha256"] = "f" * 64
    arm_path.write_text(json.dumps(arm, sort_keys=True), encoding="utf-8")
    round_value = json.loads(round_manifest.read_text(encoding="utf-8"))
    variant = arm["variant"]
    round_value["arm_manifest_sha256"][variant] = hashlib.sha256(arm_path.read_bytes()).hexdigest()
    round_manifest.write_text(json.dumps(round_value, sort_keys=True), encoding="utf-8")
    with pytest.raises(judge.TargetedJudgeError, match="success-key commitment"):
        judge.freeze_immutable_paired_sentinel_source(
            records_paths, round_manifest, arm_manifest_paths, rooted / "paired" / "bad-keys-freeze.json"
        )


def test_paired_loader_rechecks_frozen_round_and_provenance_files(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, _ = _write_paired_v2_source(rooted)
    freeze_path = rooted / "paired" / "source-freeze.json"
    judge.freeze_immutable_paired_sentinel_source(records_paths, round_manifest, arm_manifest_paths, freeze_path)

    # This is a non-content provenance/lifecycle file: it must be rechecked,
    # rather than letting a freeze turn into a one-time-only validation.
    round_wal = round_manifest.with_name("round-wal.jsonl")
    round_wal.write_text(round_wal.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(judge.TargetedJudgeError, match="round WAL digest changed"):
        judge.load_immutable_sentinels(records_paths, freeze_path)


def test_paired_aggregate_preserves_7_1_2_groups_across_three_replicates(rooted: Path) -> None:
    records_paths, round_manifest, arm_manifest_paths, template = _write_paired_v2_source(rooted)
    freeze_path = rooted / "paired" / "source-freeze.json"
    judge.freeze_immutable_paired_sentinel_source(records_paths, round_manifest, arm_manifest_paths, freeze_path)
    paths = _paths(rooted)
    dry = _run(records_paths, freeze_path, template, paths, dry_run=True, concurrency=1)
    transport = _Transport([_response(aware=index % 2 == 0) for index in range(60)])
    client = httpx.AsyncClient(transport=transport)
    try:
        result = _run(
            records_paths,
            freeze_path,
            template,
            paths,
            api_key="unit-key",
            confirm_paid=True,
            expected_plan_sha256=dry["plan_sha256"],
            concurrency=1,
            client=client,
        )
    finally:
        asyncio.run(client.aclose())

    assert result["completed"] == 60
    aggregate = json.loads(paths["aggregate_path"].read_text(encoding="utf-8"))
    counts = {
        (row["variant_id"], row["sentinel_group"]): row["sentinel_count"]
        for row in aggregate["variant_groups"]
    }
    assert counts == {
        (variant, group): count
        for variant in ("current", "igor-shaped")
        for group, count in judge.PAIRED_GROUP_COUNTS_PER_ARM.items()
    }


def test_k3_confirmation_is_design_only_and_requires_separate_authorization(rooted: Path) -> None:
    records, source_manifest, template = _write_source(rooted, [_record(1), _record(2)])
    paths = _paths(rooted)
    _run(records, source_manifest, template, paths, dry_run=True)
    lifecycle = json.loads(paths["lifecycle_manifest_path"].read_text(encoding="utf-8"))
    design = judge.build_k3_confirmation_design(lifecycle["plan"])
    assert design["execution_supported_by_this_module"] is False
    assert design["requires_separate_explicit_authorization"] is True
    assert design["judge"]["judge_k"] == 3
    assert design["request_count"] == 6
