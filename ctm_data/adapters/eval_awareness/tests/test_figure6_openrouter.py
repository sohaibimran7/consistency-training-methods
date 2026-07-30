"""Offline tests for the direct, resumable Figure 6 judge."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import httpx
import pytest

from ctm_data.adapters.eval_awareness import figure6_openrouter as scorer
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_REVISION,
    FIGURE6_CONDITIONS,
    FIGURE6_VALENCES,
    MODEL_SPECS,
    UPSTREAM_CODE_REVISION,
)

TEMPLATE = "Task: {task}\nReasoning: {reasoning}\nAnswer: {answer}"
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE.encode()).hexdigest()
ROUTE_EVIDENCE_OBJECT = {
    "schema": "ctm.eval_awareness.figure6.vast_route_evidence.v1",
    "vast_instance_id": "7348291",
    "ssh_host": "ssh5.vast.ai",
    "ssh_port": 41_922,
    "country_code": "US",
    "attested_at": "2026-07-30T09:30:00Z",
    "attested_by": "research-operator",
    "socks_proxy": "socks5h://127.0.0.1:1080",
    "ssh_control_socket": "/private/tmp/figure6-test-control",
    "exit_ip": "198.51.100.42",
    "exit_country_code": "US",
    "vast_console_evidence_sha256": "b" * 64,
}
ROUTE_EVIDENCE = (json.dumps(ROUTE_EVIDENCE_OBJECT, sort_keys=True) + "\n").encode()
ROUTE = {
    "expected_exit_instance_id": "7348291",
    "expected_exit_ssh_host": "ssh5.vast.ai",
    "expected_exit_ssh_port": 41_922,
    "route_country_code": "US",
    "route_attested_at": "2026-07-30T09:30:00Z",
    "route_attested_by": "research-operator",
    "route_attestation_sha256": hashlib.sha256(ROUTE_EVIDENCE).hexdigest(),
    "route_attestation_evidence": ROUTE_EVIDENCE,
}


def _generation() -> dict[str, Any]:
    return {
        "model_key": "unit-model",
        "model_display": "Unit Model",
        "model_id": "unit/model",
        "model_revision": "a" * 40,
        "condition_id": "safety-unit-baseline",
        "pair_id": "unit",
        "task_id": "safety-unit",
        "valence": "safety",
        "config_name": "baseline",
        "replicate": 1,
        "prompt": "Do the task",
        "response": "<think>I considered it.</think> Done",
        "reasoning": "I considered it.",
        "answer": "Done",
        "trace_present": True,
        "trace_source": "reasoning_content",
        "status": "success",
        "generation_provenance": {"schema": "unit", "temperature": 0.3, "max_tokens": 4096},
        "system_prompt_provenance": {"prompt_key": "unit", "prompt_sha256": "b" * 64},
    }


def _qwen_matrix(model_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        model = MODEL_SPECS[model_key]
        provenance = {
            "provenance_schema": "ctm.eval_awareness.figure6_generation_run",
            "schema_version": 1,
            "artifact_schema": "ctm.eval_awareness.figure6",
            "artifact_schema_version": 1,
            "artifact_sha256": "d" * 64,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "model_id": model.model_id,
            "model_key": model.key,
            "model_revision": model.revision,
            "prompt_key": model.prompt.key,
            "prompt_revision": UPSTREAM_CODE_REVISION,
            "prompt_sha256": model.prompt.sha256,
            "temperature": 0.3,
            "max_tokens": 4_096,
            "replicates": 3,
        }
        provenance["provenance_sha256"] = scorer._sha256_json(provenance)
        for pair_index in range(100):
            pair_id = f"pair-{pair_index:03d}"
            for valence in FIGURE6_VALENCES:
                task_id = f"{pair_id}_{valence}"
                for config_name in FIGURE6_CONDITIONS:
                    condition_id = f"{pair_id}-{valence}-{config_name}"
                    for replicate in (1, 2, 3):
                        rows.append(
                            {
                                "model_key": model.key,
                                "model_display": model.display_name,
                                "model_id": model.model_id,
                                "model_revision": model.revision,
                                "condition_id": condition_id,
                                "pair_id": pair_id,
                                "task_id": task_id,
                                "valence": valence,
                                "config_name": config_name,
                                "replicate": replicate,
                                "prompt": f"Prompt {pair_id} {valence} {config_name}",
                                "response": "answer",
                                "reasoning": "reasoning",
                                "answer": "answer",
                                "trace_present": True,
                                "trace_source": "reasoning_content",
                                "status": "success",
                                "generation_provenance": dict(provenance),
                                "system_prompt_provenance": {
                                    "prompt_key": model.prompt.key,
                                    "prompt_revision": UPSTREAM_CODE_REVISION,
                                    "prompt_sha256": model.prompt.sha256,
                                },
                            }
                        )
    return rows


def _lock_worker(path: str, queue: Any) -> None:
    try:
        with scorer._run_lock(Path(path)):
            queue.put("acquired")
    except scorer.OpenRouterJudgeError as exc:
        queue.put(str(exc))


def _success_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        headers={"x-request-id": "request-unit"},
        json={
            "id": "generation-unit",
            "model": scorer.GPT_OSS_120B_OPENROUTER_RESPONSE_MODEL,
            "provider": "Cerebras",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "model_awareness": {"conclusion": "Yes", "awareness_type": "Safety"},
                                "task_performance": {"conclusion": "No"},
                            }
                        ),
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    )


def _run(tmp_path: Path, client: httpx.AsyncClient | None, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("judge_profile", scorer.GPT_OSS_120B_NITRO_DIRECT_PROFILE)
    kwargs.setdefault("confirm_paid", True)
    if kwargs["confirm_paid"] and "expected_plan_sha256" not in kwargs:
        planning_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in {"confirm_paid", "amend_attempt_ceiling", "client", "sleep"}
        }
        plan = asyncio.run(
            scorer._judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                concurrency=1,
                dry_run=True,
                _enforce_exact_paid_matrix=False,
                _enforce_registered_profile=False,
                **planning_kwargs,
            )
        )
        kwargs["expected_plan_sha256"] = plan["plan_sha256"]
    return asyncio.run(
        scorer._judge_generations(
            [_generation()],
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key="test-key",
            judge_template_sha256=TEMPLATE_SHA256,
            concurrency=1,
            client=client,
            _enforce_exact_paid_matrix=False,
            _enforce_registered_profile=False,
            **kwargs,
        )
    )


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path: Path) -> None:
    summary = asyncio.run(
        scorer._judge_generations(
            [_generation()],
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key=None,
            judge_template_sha256=TEMPLATE_SHA256,
            dry_run=True,
            _enforce_exact_paid_matrix=False,
        )
    )
    assert summary["pending"] == 1
    assert summary["proxy"] == {"enabled": False}
    assert summary["route_mode"] == "direct"
    assert len(summary["plan_sha256"]) == 64
    assert not (tmp_path / "attempts.jsonl").exists()
    assert not (tmp_path / "judgments.jsonl").exists()


def test_success_writes_durable_attempt_and_normalized_output(tmp_path: Path) -> None:
    observed: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return _success_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    summary = _run(tmp_path, client)
    asyncio.run(client.aclose())

    assert summary["completed"] == 1
    assert observed[0]["model"] == scorer.DEFAULT_OPENROUTER_JUDGE_MODEL
    assert observed[0]["max_tokens"] == 32_768
    assert observed[0]["temperature"] == 0.0
    assert observed[0]["reasoning"] == {"effort": "high", "exclude": True}
    assert observed[0]["response_format"] == {"type": "json_object"}
    assert observed[0]["messages"] == [
        {"role": "system", "content": "Task: Do the task\nReasoning: I considered it.\nAnswer: Done"}
    ]
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    judgment = json.loads((tmp_path / "judgments.jsonl").read_text())
    assert attempt["status"] == "success"
    assert attempt["response"]["provider"] == "Cerebras"
    assert judgment["judge_profile"] == scorer.GPT_OSS_120B_NITRO_DIRECT_PROFILE
    assert judgment["judge_model"] == scorer.GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL
    assert judgment["judge_response_model"] == scorer.GPT_OSS_120B_OPENROUTER_RESPONSE_MODEL
    assert judgment["judge_request_id"] == "request-unit"
    assert judgment["awareness_conclusion"] == "yes"
    manifest = json.loads((tmp_path / "attempts.jsonl.manifest.json").read_text())
    assert manifest["plan"]["plan_sha256"] == summary["plan_sha256"]
    assert manifest["approvals"][0]["confirmation"] == "--yes"
    assert attempt["approval"]["plan_sha256"] == summary["plan_sha256"]


def test_transient_response_is_appended_then_retried(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, request=request, headers={"retry-after": "0"}, json={"error": "busy"})
        return _success_response(request)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _run(tmp_path, client, sleep=no_sleep)
    asyncio.run(client.aclose())

    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["error", "success"]
    assert [row["attempt"] for row in attempts] == [1, 2]
    assert attempts[0]["retryable"] is True
    assert delays == [0.0]


def test_permanent_geographic_error_fails_without_retry(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            request=request,
            json={"error": {"message": "This model is only available in the United States."}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(scorer.PermanentOpenRouterError, match="HTTP 403"):
        _run(tmp_path, client)
    asyncio.run(client.aclose())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert attempt["status"] == "error"
    assert attempt["retryable"] is False
    assert not (tmp_path / "judgments.jsonl").exists()


def test_completed_run_resumes_without_another_paid_call(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _run(tmp_path, client)
    resumed = _run(tmp_path, client)
    asyncio.run(client.aclose())
    assert calls == 1
    assert resumed["idempotent_resume"] is True
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((tmp_path / "attempts.jsonl.manifest.json").read_text())
    assert sum(event["event"] == "run_completed" for event in manifest["events"]) == 1


def test_resume_rejects_judge_protocol_drift(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _run(tmp_path, client)
    with pytest.raises(scorer.OpenRouterJudgeError, match="different deterministic core plan"):
        _run(tmp_path, client, temperature=0.2)
    asyncio.run(client.aclose())
    assert calls == 1


def test_template_digest_is_bound_before_dry_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="template digest mismatch"):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                judge_template_sha256="0" * 64,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                dry_run=True,
            )
        )


def test_proxy_credentials_are_rejected_before_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prohibit a proxy"):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                proxy="socks5://user:secret@127.0.0.1:1080",
                dry_run=True,
            )
        )


def test_paid_run_requires_explicit_confirmation_and_plan_hash_is_deterministic(tmp_path: Path) -> None:
    common = {
        "template": TEMPLATE,
        "attempt_log_path": tmp_path / "attempts.jsonl",
        "output_path": tmp_path / "judgments.jsonl",
        "api_key": None,
        "judge_template_sha256": TEMPLATE_SHA256,
        "concurrency": 1,
        "_enforce_exact_paid_matrix": False,
        "_enforce_registered_profile": False,
    }
    first = asyncio.run(scorer._judge_generations([_generation()], dry_run=True, **common))
    second = asyncio.run(scorer._judge_generations([_generation()], dry_run=True, **common))
    assert first["plan_sha256"] == second["plan_sha256"]
    assert not Path(first["manifest"]).exists()

    with pytest.raises(scorer.OpenRouterJudgeError, match="explicit --yes"):
        asyncio.run(scorer._judge_generations([_generation()], dry_run=False, **common))
    assert not Path(first["manifest"]).exists()
    with pytest.raises(scorer.OpenRouterJudgeError, match="plan hash mismatch"):
        asyncio.run(
            scorer._judge_generations(
                [_generation()],
                dry_run=True,
                expected_plan_sha256="0" * 64,
                **common,
            )
        )
    client = httpx.AsyncClient(transport=httpx.MockTransport(_success_response))
    paid = _run(tmp_path, client, expected_plan_sha256=first["plan_sha256"])
    asyncio.run(client.aclose())
    assert paid["plan_sha256"] == first["plan_sha256"]
    manifest = json.loads(Path(paid["manifest"]).read_text())
    assert manifest["approvals"][0]["reviewed_plan_sha256"] == first["plan_sha256"]
    assert manifest["approvals"][0]["reviewed_plan_hash_verified"] is True


def test_cli_rejects_paid_mode_without_yes_before_loading_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import judge_figure6_openrouter as cli

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--generations",
                "/does/not/exist.jsonl",
                "--judge-template",
                "/does/not/exist.txt",
                "--attempt-log",
                "/tmp/unused-attempts.jsonl",
                "--output",
                "/tmp/unused-judgments.jsonl",
                "--expected-model-key",
                "qwen32",
            ]
        )
    assert "requires explicit --yes" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--generations",
                "/does/not/exist.jsonl",
                "--judge-template",
                "/does/not/exist.txt",
                "--attempt-log",
                "/tmp/unused-attempts.jsonl",
                "--output",
                "/tmp/unused-judgments.jsonl",
                "--expected-model-key",
                "qwen32",
                "--yes",
            ]
        )
    assert "requires --expected-plan-sha256" in capsys.readouterr().err


def test_cli_omitted_profile_defaults_to_direct_luna() -> None:
    from scripts import judge_figure6_openrouter as cli

    args = cli._build_parser().parse_args(
        [
            "--generations",
            "generations.jsonl",
            "--judge-template",
            "judge.txt",
            "--attempt-log",
            "attempts.jsonl",
            "--output",
            "judgments.jsonl",
            "--expected-model-key",
            "qwen32",
            "--dry-run",
        ]
    )
    assert args.judge_profile == scorer.OPENAI_GPT_56_LUNA_DIRECT_PROFILE


def test_paid_malformed_response_is_preserved_and_requires_explicit_rescore(tmp_path: Path) -> None:
    calls = 0

    async def malformed_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "paid-malformed-id",
                "model": scorer.DEFAULT_OPENROUTER_JUDGE_MODEL,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "choices": [{"message": {"content": "not-json"}}],
            },
        )

    malformed_client = httpx.AsyncClient(transport=httpx.MockTransport(malformed_handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="raw response preserved"):
        _run(tmp_path, malformed_client, max_attempts=1)
    asyncio.run(malformed_client.aclose())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert attempt["response"]["id"] == "paid-malformed-id"
    assert attempt["response"]["model"] == scorer.DEFAULT_OPENROUTER_JUDGE_MODEL
    assert attempt["response"]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    assert attempt["response"]["content"] == "not-json"
    assert attempt["response"]["raw_body"]["id"] == "paid-malformed-id"
    assert attempt["response"]["raw_body_text"]
    assert attempt["response"]["raw_body_base64"]
    assert attempt["retryable"] is False

    blocked_client = httpx.AsyncClient(transport=httpx.MockTransport(malformed_handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="--rescore-paid-errors"):
        _run(tmp_path, blocked_client, max_attempts=2, amend_attempt_ceiling=True)
    asyncio.run(blocked_client.aclose())
    assert calls == 1

    success_client = httpx.AsyncClient(transport=httpx.MockTransport(_success_response))
    summary = _run(
        tmp_path,
        success_client,
        rescore_paid_errors=True,
        max_attempts=2,
        amend_attempt_ceiling=True,
    )
    asyncio.run(success_client.aclose())
    assert summary["completed"] == 1
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["attempt"] for row in attempts] == [1, 2]
    assert attempts[1]["rescore_authorization"]["flag"] == "--rescore-paid-errors"
    assert attempts[1]["rescore_authorization"]["paid_error_attempts"] == [1]
    manifest = json.loads((tmp_path / "attempts.jsonl.manifest.json").read_text())
    assert manifest["amendments"][0]["old_max_attempts_per_generation"] == 1
    assert manifest["amendments"][0]["new_max_attempts_per_generation"] == 2
    assert manifest["approvals"][-1]["authorized_paid_error_attempts"]
    assert next(iter(manifest["approvals"][-1]["authorized_attempt_ceiling"].values())) == 2


def test_unapproved_response_model_is_preserved_and_rejected(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        response = _success_response(request)
        body = response.json()
        body["model"] = "meta/unexpected-model"
        return httpx.Response(200, request=request, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="raw response preserved"):
        _run(tmp_path, client)
    asyncio.run(client.aclose())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert "not in the allowed identities" in attempt["error"]["message"]
    assert attempt["response"]["model"] == "meta/unexpected-model"


def test_muse_alias_allowed_identity_set_cannot_be_widened(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="permits exactly"):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                judge_profile=scorer.MUSE_US_PROXY_PROFILE,
                allowed_response_models=[scorer.MUSE_OPENROUTER_JUDGE_MODEL, "meta/unexpected-model"],
                proxy="socks5h://127.0.0.1:1080",
                **ROUTE,
                dry_run=True,
            )
        )


def test_non_json_http_200_body_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    raw_body = b"\xffnot-json\x00"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=raw_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="raw response preserved"):
        _run(tmp_path, client)
    asyncio.run(client.aclose())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert base64.b64decode(attempt["response"]["raw_body_base64"]) == raw_body
    assert attempt["response"]["raw_body_sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert attempt["response"]["id"] is None


def test_transport_errors_all_5xx_and_full_retry_after_are_retried(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("unit transport failure", request=request)
        if calls == 2:
            return httpx.Response(501, request=request, headers={"retry-after": "240"}, json={"error": "busy"})
        return _success_response(request)

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _run(tmp_path, client, sleep=no_sleep)
    asyncio.run(client.aclose())
    assert calls == 3
    assert delays == [1.0, 240.0]
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["retryable"] for row in attempts] == [True, True, False]


def test_retry_after_maximum_fails_instead_of_truncating(tmp_path: Path) -> None:
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, headers={"retry-after": "121"}, json={"error": "busy"})

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="refusing to truncate"):
        _run(tmp_path, client, sleep=no_sleep, max_retry_after=120)
    asyncio.run(client.aclose())
    assert delays == []


def test_exhausted_transient_retry_requires_reviewed_ceiling_amendment(tmp_path: Path) -> None:
    async def transient(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, json={"error": "busy"})

    async def no_sleep(_: float) -> None:
        return None

    first_client = httpx.AsyncClient(transport=httpx.MockTransport(transient))
    with pytest.raises(scorer.OpenRouterJudgeError, match="attempts exhausted"):
        _run(tmp_path, first_client, max_attempts=1, sleep=no_sleep)
    asyncio.run(first_client.aclose())
    first_manifest = json.loads((tmp_path / "attempts.jsonl.manifest.json").read_text())
    first_plan_sha256 = first_manifest["plan"]["plan_sha256"]

    plan = asyncio.run(
        scorer._judge_generations(
            [_generation()],
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key=None,
            judge_template_sha256=TEMPLATE_SHA256,
            concurrency=1,
            max_attempts=2,
            dry_run=True,
            _enforce_exact_paid_matrix=False,
            _enforce_registered_profile=False,
        )
    )
    assert plan["amendment_required"] is True
    assert plan["previous_plan_sha256"] == first_plan_sha256
    assert plan["previous_max_attempts"] == 1
    assert plan["plan_sha256"] != first_plan_sha256
    assert plan["core_plan_sha256"] == first_manifest["plan"]["core_plan_sha256"]

    blocked_client = httpx.AsyncClient(transport=httpx.MockTransport(_success_response))
    with pytest.raises(scorer.OpenRouterJudgeError, match="--amend-attempt-ceiling"):
        _run(tmp_path, blocked_client, max_attempts=2)
    asyncio.run(blocked_client.aclose())

    success_client = httpx.AsyncClient(transport=httpx.MockTransport(_success_response))
    summary = _run(tmp_path, success_client, max_attempts=2, amend_attempt_ceiling=True)
    asyncio.run(success_client.aclose())
    assert summary["completed"] == 1
    attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["error", "success"]
    assert attempts[0]["plan_sha256"] == first_plan_sha256
    assert attempts[1]["plan_sha256"] == plan["plan_sha256"]
    manifest = json.loads((tmp_path / "attempts.jsonl.manifest.json").read_text())
    assert len(manifest["plan_history"]) == 2
    assert manifest["amendments"][0]["reviewed_plan_sha256"] == plan["plan_sha256"]
    assert manifest["amendments"][0]["preserved_attempt_count"] == 1


def test_resume_binds_the_full_generation_record_digest(tmp_path: Path) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(_success_response))
    _run(tmp_path, client)
    changed = _generation()
    changed["generation_provenance"] = {"schema": "changed", "temperature": 0.3, "max_tokens": 4096}
    with pytest.raises(scorer.OpenRouterJudgeError, match="different deterministic core plan"):
        asyncio.run(
            scorer._judge_generations(
                [changed],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key="test-key",
                judge_template_sha256=TEMPLATE_SHA256,
                confirm_paid=True,
                client=client,
                _enforce_exact_paid_matrix=False,
                _enforce_registered_profile=False,
            )
        )
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    ("proxy", "route", "match"),
    [
        (None, ROUTE, "explicit loopback socks5h proxy"),
        ("socks5://203.0.113.4:1080", ROUTE, "exactly the socks5h scheme"),
        ("socks5h://[::1]:9999", ROUTE, "exact proxy host 127.0.0.1"),
        ("socks5h://127.0.0.1:9999", ROUTE, "exact proxy port 1080"),
        ("socks5h://127.0.0.1:1080", {}, "complete U.S. Vast route attestation"),
    ],
)
def test_default_muse_requires_explicit_loopback_socks_route(
    tmp_path: Path,
    proxy: str | None,
    route: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                judge_profile=scorer.MUSE_US_PROXY_PROFILE,
                proxy=proxy,
                dry_run=True,
                **route,
            )
        )


def test_created_client_ignores_environment_proxy_and_records_sanitized_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_kwargs: dict[str, Any] = {}

    class FakeClient:
        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            request = httpx.Request("POST", url, headers=headers, json=json)
            return _success_response(request)

        async def aclose(self) -> None:
            return None

    def factory(**kwargs: Any) -> FakeClient:
        client_kwargs.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(scorer.httpx, "AsyncClient", factory)
    summary = _run(tmp_path, None)
    assert client_kwargs["trust_env"] is False
    assert summary["endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
    assert summary["route"] is None
    assert summary["route_mode"] == "direct"
    assert client_kwargs["proxy"] is None
    manifest_text = (tmp_path / "attempts.jsonl.manifest.json").read_text()
    assert "test-key" not in manifest_text


def test_endpoint_credentials_are_rejected_before_any_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                base_url="https://user:secret@openrouter.ai/api/v1",
                dry_run=True,
            )
        )


@pytest.mark.parametrize(
    "model_keys",
    [
        ("qwen32",),
        ("qwen32", "qwen_mo_mid", "qwen_mo_post"),
    ],
)
def test_exact_qwen_matrix_validator_accepts_per_model_and_combined_scopes(
    model_keys: tuple[str, ...],
) -> None:
    rows = _qwen_matrix(model_keys)
    matrix = scorer._validate_exact_qwen_matrix(rows, expected_model_keys=model_keys)
    assert len(rows) == 5_400 * len(model_keys)
    assert matrix["model_keys"] == list(model_keys)
    assert matrix["generations_per_model"] == 5_400
    assert matrix["task_pair_count"] == 100


def test_current_paid_scope_rejects_qwen36() -> None:
    with pytest.raises(ValueError, match="current three-Qwen scope"):
        scorer._expected_qwen_model_keys(("qwen36",), required=True)


def test_muse_profile_requires_exact_evidence_bytes_and_socks5h(tmp_path: Path) -> None:
    route_without_bytes = {name: value for name, value in ROUTE.items() if name != "route_attestation_evidence"}
    with pytest.raises(ValueError, match="evidence bytes"):
        asyncio.run(
            scorer.judge_generations(
                [_generation()],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key=None,
                judge_template_sha256=TEMPLATE_SHA256,
                judge_profile=scorer.MUSE_US_PROXY_PROFILE,
                proxy="socks5h://127.0.0.1:1080",
                dry_run=True,
                **route_without_bytes,
            )
        )


def test_paid_public_entrypoint_rejects_incomplete_matrix_before_any_request(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="5,400 generations per model"):
        asyncio.run(
            scorer.judge_generations(
                _qwen_matrix(("qwen32",))[:-1],
                template=TEMPLATE,
                attempt_log_path=tmp_path / "attempts.jsonl",
                output_path=tmp_path / "judgments.jsonl",
                api_key="paid-key",
                judge_template_sha256=TEMPLATE_SHA256,
                expected_model_keys=("qwen32",),
                confirm_paid=True,
                client=client,
            )
        )
    asyncio.run(client.aclose())
    assert calls == 0
    assert not (tmp_path / "attempts.jsonl").exists()


def test_public_dry_plan_binds_exact_scope_concurrency_retry_cap_and_route(tmp_path: Path) -> None:
    summary = asyncio.run(
        scorer.judge_generations(
            _qwen_matrix(("qwen32",)),
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key=None,
            judge_template_sha256=TEMPLATE_SHA256,
            expected_model_keys=("qwen32",),
            dry_run=True,
        )
    )
    assert summary["generation_count"] == 5_400
    assert summary["matrix"]["model_keys"] == ["qwen32"]
    assert summary["concurrency"] == 4
    assert summary["max_tokens"] == 32_768
    assert summary["max_retry_after"] == 300.0
    assert summary["judge_profile"] == scorer.GPT_OSS_120B_NITRO_DIRECT_PROFILE
    assert summary["judge_model"] == scorer.GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL
    assert summary["allowed_response_models"] == [
        scorer.GPT_OSS_120B_OPENROUTER_RESPONSE_MODEL,
        scorer.GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL,
    ]
    assert summary["provider_routing"]["sort"] == "throughput"
    assert summary["provider_routing"]["allow_fallbacks"] is True
    assert summary["provider_routing"]["ignore"] == ["cerebras"]
    assert summary["reasoning"] == {"effort": "high", "exclude": True}
    assert summary["route"] is None
    assert summary["route_mode"] == "direct"
    assert len(summary["plan_sha256"]) == 64
    assert len(summary["core_plan_sha256"]) == 64
    assert not Path(summary["manifest"]).exists()


def test_openai_luna_public_dry_plan_pins_direct_500_way_profile(tmp_path: Path) -> None:
    summary = asyncio.run(
        scorer.judge_generations(
            _qwen_matrix(("qwen32",)),
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key=None,
            judge_template_sha256=TEMPLATE_SHA256,
            judge_profile=scorer.OPENAI_GPT_56_LUNA_DIRECT_PROFILE,
            expected_model_keys=("qwen32",),
            dry_run=True,
        )
    )
    assert summary["provider"] == "OpenAI"
    assert summary["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert summary["api_style"] == "openai_chat_completions"
    assert summary["judge_model"] == "gpt-5.6-luna"
    assert summary["allowed_response_models"] == ["gpt-5.6-luna"]
    assert summary["temperature"] is None
    assert summary["max_tokens"] == 32_768
    assert summary["concurrency"] == 500
    assert summary["reasoning"] == {"effort": "max"}
    assert summary["provider_routing"] == {}
    assert summary["store"] is False
    assert summary["response_format"]["type"] == "json_schema"
    assert summary["response_format"]["json_schema"]["strict"] is True
    assert summary["pending"] == 5_400


def test_openai_luna_uses_direct_chat_request_shape_and_normalizes_success(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            request=request,
            headers={"x-request-id": "request-luna-unit"},
            json={
                "id": "chatcmpl-luna-unit",
                "model": scorer.OPENAI_GPT_56_LUNA_JUDGE_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "model_awareness": {
                                        "conclusion": "Yes",
                                        "awareness_type": "safety",
                                        "evidence": ["quote"],
                                        "reasoning": "reason",
                                    },
                                    "hypothetical_recognition": {
                                        "conclusion": "No",
                                        "evidence": ["quote"],
                                        "reasoning": "reason",
                                    },
                                    "task_performance": {
                                        "conclusion": "No",
                                        "evidence": ["detail"],
                                        "reasoning": "reason",
                                    },
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    summary = _run(
        tmp_path,
        client,
        judge_profile=scorer.OPENAI_GPT_56_LUNA_DIRECT_PROFILE,
    )
    asyncio.run(client.aclose())
    assert summary["completed"] == 1
    assert len(seen) == 1
    body = seen[0]
    assert body["model"] == "gpt-5.6-luna"
    assert body["max_completion_tokens"] == 32_768
    assert body["reasoning_effort"] == "max"
    assert body["store"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"][0]["role"] == "system"
    assert "temperature" not in body
    assert "provider" not in body
    assert "max_tokens" not in body
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert attempt["response"]["provider"] == "OpenAI"
    assert attempt["judgment"]["judge_provider"] == "OpenAI"
    assert attempt["judgment"]["judge_response_model"] == "gpt-5.6-luna"
    assert attempt["judgment"]["judge_finish_reason"] == "stop"


@pytest.mark.parametrize(
    "content,finish_reason",
    [
        (
            {
                "model_awareness": {
                    "conclusion": "Yes",
                    "awareness_type": "safety",
                    "evidence": [],
                    "reasoning": "reason",
                },
                "task_performance": {
                    "conclusion": "No",
                    "evidence": [],
                    "reasoning": "reason",
                },
            },
            "stop",
        ),
        (
            {
                "model_awareness": {
                    "conclusion": "Yes",
                    "awareness_type": "safety",
                    "evidence": [],
                    "reasoning": "reason",
                },
                "hypothetical_recognition": {
                    "conclusion": "No",
                    "evidence": [],
                    "reasoning": "reason",
                },
                "task_performance": {
                    "conclusion": "No",
                    "evidence": [],
                    "reasoning": "reason",
                },
            },
            "length",
        ),
    ],
)
def test_openai_luna_rejects_incomplete_schema_or_nonstop_finish(
    tmp_path: Path, content: dict[str, Any], finish_reason: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-invalid-luna",
                "model": scorer.OPENAI_GPT_56_LUNA_JUDGE_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": json.dumps(content)},
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(scorer.OpenRouterJudgeError, match="raw response preserved"):
        _run(
            tmp_path,
            client,
            judge_profile=scorer.OPENAI_GPT_56_LUNA_DIRECT_PROFILE,
            max_attempts=1,
        )
    asyncio.run(client.aclose())
    attempt = json.loads((tmp_path / "attempts.jsonl").read_text())
    assert attempt["status"] == "error"
    assert attempt["retryable"] is False


def test_plan_hash_changes_with_concurrency_retry_cap_and_ceiling_only_core_rule(tmp_path: Path) -> None:
    async def plan(**overrides: Any) -> dict[str, Any]:
        kwargs = {
            "concurrency": 1,
            "max_attempts": 5,
            "max_retry_after": None,
            **overrides,
        }
        return await scorer._judge_generations(
            [_generation()],
            template=TEMPLATE,
            attempt_log_path=tmp_path / "attempts.jsonl",
            output_path=tmp_path / "judgments.jsonl",
            api_key=None,
            judge_template_sha256=TEMPLATE_SHA256,
            dry_run=True,
            _enforce_exact_paid_matrix=False,
            _enforce_registered_profile=False,
            **kwargs,
        )

    baseline = asyncio.run(plan())
    changed_concurrency = asyncio.run(plan(concurrency=2))
    changed_retry_cap = asyncio.run(plan(max_retry_after=60.0))
    raised_ceiling = asyncio.run(plan(max_attempts=6))
    assert baseline["plan_sha256"] != changed_concurrency["plan_sha256"]
    assert baseline["core_plan_sha256"] != changed_concurrency["core_plan_sha256"]
    assert baseline["plan_sha256"] != changed_retry_cap["plan_sha256"]
    assert baseline["core_plan_sha256"] != changed_retry_cap["core_plan_sha256"]
    assert baseline["plan_sha256"] != raised_ceiling["plan_sha256"]
    assert baseline["core_plan_sha256"] == raised_ceiling["core_plan_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("expected_exit_instance_id", "734x", "positive numeric"),
        ("expected_exit_ssh_host", "localhost", "non-loopback"),
        ("expected_exit_ssh_port", 70_000, "1 through 65535"),
        ("route_country_code", "CA", "exactly US"),
        ("route_attested_by", "placeholder-reviewer", "rather than a placeholder"),
        ("route_attestation_sha256", "ABC", "64-character lowercase"),
    ],
)
def test_route_attestation_rejects_placeholders_and_protocol_drift(
    field: str,
    value: Any,
    match: str,
) -> None:
    route = {name: value for name, value in ROUTE.items() if name != "route_attestation_evidence"}
    route[field] = value
    with pytest.raises(ValueError, match=match):
        scorer._route_provenance(required=True, **route)


def test_os_lock_excludes_a_second_process(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempts.jsonl"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with scorer._run_lock(attempt_path):
        process = context.Process(target=_lock_worker, args=(str(attempt_path), queue))
        process.start()
        result = queue.get(timeout=10)
        process.join(timeout=10)
    assert process.exitcode == 0
    assert "another process is using" in result


def test_judge_holds_os_lock_while_the_paid_request_is_in_flight(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempts.jsonl"
    observed: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_lock_worker, args=(str(attempt_path), queue))
        process.start()
        observed.append(queue.get(timeout=10))
        process.join(timeout=10)
        assert process.exitcode == 0
        return _success_response(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _run(tmp_path, client)
    asyncio.run(client.aclose())
    assert len(observed) == 1
    assert "another process is using" in observed[0]
