"""Offline checks for the Figure 6 crossover judge protocols."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any
from types import ModuleType

import pytest
import httpx


# The crossover module uses Inspect only for historical-log import.  Its pure
# planning and protocol helpers must remain testable in this offline fixture.
try:
    importlib.import_module("inspect_ai.log")
except ModuleNotFoundError:
    inspect_ai_stub = ModuleType("inspect_ai")
    inspect_log_stub = ModuleType("inspect_ai.log")

    def _unavailable_read_eval_log(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("historical Inspect log import is outside this offline test module")

    inspect_log_stub.read_eval_log = _unavailable_read_eval_log  # type: ignore[attr-defined]
    inspect_ai_stub.log = inspect_log_stub  # type: ignore[attr-defined]
    sys.modules["inspect_ai"] = inspect_ai_stub
    sys.modules["inspect_ai.log"] = inspect_log_stub

from ctm_data.adapters.eval_awareness import figure6_crossover as crossover
from ctm_data.adapters.eval_awareness import figure6_openrouter as openrouter
from ctm_data.adapters.eval_awareness.figure6_judge import (
    PAPER_JUDGE_TEMPLATE_SHA256,
    custom_id_for_generation,
)


def _generation(config_name: str, index: int) -> dict[str, Any]:
    """Make one minimal, schema-valid successful target generation."""

    return {
        "model_key": crossover.IGOR_MODEL_KEY,
        "model_display": "Synthetic crossover target",
        "model_id": crossover.IGOR_MODEL_ID,
        "model_revision": "synthetic-revision",
        "condition_id": f"synthetic-{config_name.lower()}-{index:03d}",
        "pair_id": f"pair-{index:03d}",
        "task_id": f"synthetic-task-{config_name.lower()}-{index:03d}",
        "valence": "safety",
        "config_name": config_name,
        "replicate": 1,
        "prompt": f"Synthetic prompt for {config_name}/{index}.",
        "response": "Synthetic answer.",
        "reasoning": "Synthetic reasoning trace.",
        "answer": "Synthetic answer.",
        "trace_present": True,
        "trace_source": "synthetic_reasoning",
        "status": "success",
        "generation_provenance": {"schema": "synthetic-crossover"},
        "system_prompt_provenance": {"prompt_key": "synthetic-crossover"},
    }


def _scope_records() -> list[dict[str, Any]]:
    """Match the three documented scope totals while keeping input order noisy."""

    counts = {
        "F6": 98,
        "baseline": 100,
        "F3": 99,
        "F8": 99,
        "F2": 99,
        "F5": 99,
    }
    records = [
        _generation(config_name, index)
        for config_name, count in counts.items()
        for index in range(count)
    ]
    return list(reversed(records))


def _paper_template() -> str:
    root = Path(__file__).resolve().parents[4]
    template = (root / "experiments/eval_awareness/figure6/inputs/judgePromptEvidenceBoth.txt").read_text(
        encoding="utf-8"
    )
    assert crossover._sha256_bytes(template.encode("utf-8")) == PAPER_JUDGE_TEMPLATE_SHA256
    return template


class _NoNetworkClient:
    """Fails loudly if a dry-run ever tries to send an HTTP request."""

    called = False

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        self.called = True
        raise AssertionError("dry-run must not make a DeepSeek request")


class _FirstResponseFailsClient:
    """Return one paid validation failure while keeping one peer in flight."""

    def __init__(self) -> None:
        self.calls = 0

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.calls += 1
        call = self.calls
        if call == 1:
            return httpx.Response(
                200,
                json={
                    "id": "bad-response",
                    "model": "unexpected/model",
                    "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                },
            )
        await asyncio.sleep(0.02)
        return httpx.Response(
            200,
            json={
                "id": f"good-response-{call}",
                "model": crossover.DEEPSEEK_MODEL,
                "provider": "test-provider",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "model_awareness": {"conclusion": "No"},
                                    "hypothetical_recognition": {"conclusion": "No"},
                                    "task_performance": {"conclusion": "No"},
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


class _BlockingClient:
    """Hold requests in flight so caller cancellation can be audited."""

    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.calls = 0
        self.active = 0
        self.cancelled = 0
        self.all_started = asyncio.Event()
        self.never_release = asyncio.Event()

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.calls += 1
        self.active += 1
        if self.calls == self.expected_calls:
            self.all_started.set()
        try:
            await self.never_release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1
        raise AssertionError("blocking test client must be cancelled")


def _deepseek_dry_run(
    records: list[dict[str, Any]],
    *,
    template: str,
    attempt_log_path: Path,
    output_path: Path,
    manifest_path: Path,
    client: _NoNetworkClient,
) -> dict[str, Any]:
    return asyncio.run(
        crossover.judge_deepseek_scope(
            records,
            scope="baseline",
            template=template,
            attempt_log_path=attempt_log_path,
            output_path=output_path,
            manifest_path=manifest_path,
            api_key=None,
            expected_plan_sha256=None,
            confirm_paid=False,
            dry_run=True,
            client=client,  # type: ignore[arg-type]
        )
    )


def test_deepseek_request_body_is_exact_user_only_igor_shape() -> None:
    prompt = "Judge this target output."

    assert crossover.deepseek_request_body(prompt) == {
        "model": crossover.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }


def test_model_only_probe_changes_deepseek_and_retains_luna_constraints() -> None:
    prompt = "Judge this target output."
    body = crossover.probe_request_body(prompt, "deepseek-user-luna-constraints-k1")
    luna_profile = openrouter.JUDGE_PROFILES[openrouter.OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE]

    assert body == {
        "model": crossover.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": luna_profile["max_tokens"],
        "provider": luna_profile["provider_routing"],
        "reasoning": luna_profile["reasoning"],
        "response_format": luna_profile["response_format"],
    }


@pytest.mark.parametrize(
    "text",
    (
        '{"model_awareness":{"conclusion":"Yes"}}',
        '  ```json\n{"model_awareness":{"conclusion":"Yes"}}\n```  ',
        '```arbitrary-fence-label\n{"model_awareness":{"conclusion":"Yes"}}\n```',
    ),
)
def test_igor_parser_accepts_plain_and_permissively_fenced_json(text: str) -> None:
    assert crossover.parse_igor_judge_json(text) == {"model_awareness": {"conclusion": "Yes"}}


def test_igor_parser_matches_json_loads_and_vote_shape_fails_closed() -> None:
    parsed = crossover.parse_igor_judge_json("[]")
    assert parsed == []
    with pytest.raises(AttributeError):
        crossover.igor_votes(parsed)


def test_igor_majority_is_strict_per_metric_and_zero_without_valid_votes() -> None:
    judgments = [
        {
            "model_awareness": {"conclusion": "Yes"},
            "hypothetical_recognition": {"conclusion": "Yes"},
            "task_performance": {"conclusion": "Yes"},
        },
        {
            "model_awareness": {"conclusion": "Yes"},
            "hypothetical_recognition": {"conclusion": "No"},
            "task_performance": {"conclusion": "No"},
        },
        {
            "model_awareness": {"conclusion": "No"},
            "hypothetical_recognition": {"conclusion": "No"},
            "task_performance": {"conclusion": "Yes"},
        },
    ]

    assert crossover.igor_majority(judgments) == {
        "awareness": 1,
        "hypothetical": 0,
        "task_performance": 1,
    }
    assert crossover.igor_majority(judgments[:2]) == {
        "awareness": 1,
        "hypothetical": 0,
        "task_performance": 0,
    }
    assert crossover.igor_majority([]) == {
        "awareness": 0,
        "hypothetical": 0,
        "task_performance": 0,
    }


def test_scoped_generations_have_documented_counts_and_stable_order() -> None:
    records = _scope_records()
    expected_counts = {"baseline": 100, "non-f3": 495, "full": 594}

    for scope, expected_count in expected_counts.items():
        scoped = crossover.scoped_generations(records, scope)
        repeated = crossover.scoped_generations(list(reversed(records)), scope)
        custom_ids = [custom_id_for_generation(row) for row in scoped]

        assert len(scoped) == expected_count
        assert custom_ids == sorted(custom_ids)
        assert custom_ids == [custom_id_for_generation(row) for row in repeated]
        assert {row["config_name"] for row in scoped} <= crossover.SCOPE_CONFIGS[scope]


def test_scoped_generations_reject_swapped_condition_denominators() -> None:
    records = _scope_records()
    f2 = next(row for row in records if row["config_name"] == "F2")
    f2["config_name"] = "F6"
    f2["condition_id"] = f2["condition_id"].replace("f2", "f6")

    with pytest.raises(crossover.CrossoverError, match="condition counts"):
        crossover.scoped_generations(records, "non-f3")


def test_deepseek_dry_plan_is_stable_and_resumes_terminal_slots_without_network(tmp_path: Path) -> None:
    records = _scope_records()
    template = _paper_template()
    attempt_log_path = tmp_path / "deepseek-attempts.jsonl"
    output_path = tmp_path / "deepseek-judgments.jsonl"
    manifest_path = tmp_path / "deepseek-manifest.json"
    plan, by_id = crossover._deepseek_plan(records, template=template, scope="baseline")
    reversed_plan, _ = crossover._deepseek_plan(list(reversed(records)), template=template, scope="baseline")

    assert plan == reversed_plan
    assert plan["generation_count"] == 100
    assert plan["request_count"] == 300

    client = _NoNetworkClient()
    first = _deepseek_dry_run(
        records,
        template=template,
        attempt_log_path=attempt_log_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=client,
    )
    second = _deepseek_dry_run(
        list(reversed(records)),
        template=template,
        attempt_log_path=attempt_log_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=client,
    )

    assert first == second
    assert first["plan_sha256"] == plan["plan_sha256"]
    assert first["completed_slots"] == 0
    assert first["pending_slots"] == 300
    assert first["estimated_new_paid_requests_upper_bound"] == 300 * crossover.DEEPSEEK_MAX_ATTEMPTS
    assert not output_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "reviewed_dry_run"
    assert manifest["plan"]["plan_sha256"] == plan["plan_sha256"]

    custom_id = next(iter(sorted(by_id)))
    attempt_rows = [
        {
            "schema": crossover.DEEPSEEK_ATTEMPT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "custom_id": custom_id,
            "judge_slot": slot,
            "attempt": 1,
            "status": "invalid_vote",
        }
        for slot in range(1, crossover.DEEPSEEK_K + 1)
    ]
    attempt_log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempt_rows), encoding="utf-8"
    )

    resumed = _deepseek_dry_run(
        records,
        template=template,
        attempt_log_path=attempt_log_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=client,
    )

    assert resumed["completed_slots"] == crossover.DEEPSEEK_K
    assert resumed["pending_slots"] == 300 - crossover.DEEPSEEK_K
    assert resumed["estimated_new_paid_requests_upper_bound"] == (
        300 - crossover.DEEPSEEK_K
    ) * crossover.DEEPSEEK_MAX_ATTEMPTS
    assert client.called is False


def test_deepseek_paid_failure_stops_refill_and_drains_inflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(crossover, "DEEPSEEK_CONCURRENCY", 2)
    records = _scope_records()
    template = _paper_template()
    attempt_path = tmp_path / "attempts.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    manifest_path = tmp_path / "manifest.json"
    dry = _deepseek_dry_run(
        records,
        template=template,
        attempt_log_path=attempt_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=_NoNetworkClient(),
    )
    client = _FirstResponseFailsClient()

    with pytest.raises(crossover.CrossoverError, match="validation failed"):
        asyncio.run(
            crossover.judge_deepseek_scope(
                records,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=client,  # type: ignore[arg-type]
            )
        )

    assert client.calls == 2
    assert not output_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert len(crossover._read_attempts(attempt_path)) == 2


def test_paid_run_rejects_tampered_review_manifest(tmp_path: Path) -> None:
    records = _scope_records()
    template = _paper_template()
    attempt_path = tmp_path / "attempts.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    manifest_path = tmp_path / "manifest.json"
    dry = _deepseek_dry_run(
        records,
        template=template,
        attempt_log_path=attempt_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=_NoNetworkClient(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attempt_log"] = str(tmp_path / "different-attempts.jsonl")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = _NoNetworkClient()

    with pytest.raises(crossover.CrossoverError, match="lifecycle paths"):
        asyncio.run(
            crossover.judge_deepseek_scope(
                records,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=client,  # type: ignore[arg-type]
            )
        )
    assert client.called is False


def test_deepseek_cancellation_marks_dispatched_slots_uncertain_before_unlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(crossover, "DEEPSEEK_CONCURRENCY", 2)
    records = _scope_records()
    template = _paper_template()
    attempt_path = tmp_path / "attempts.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    manifest_path = tmp_path / "manifest.json"
    dry = _deepseek_dry_run(
        records,
        template=template,
        attempt_log_path=attempt_path,
        output_path=output_path,
        manifest_path=manifest_path,
        client=_NoNetworkClient(),
    )
    client = _BlockingClient(expected_calls=2)

    async def cancel_run() -> None:
        run = asyncio.create_task(
            crossover.judge_deepseek_scope(
                records,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=client,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(client.all_started.wait(), timeout=1)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(cancel_run())

    assert client.calls == client.cancelled == 2
    assert client.active == 0
    attempts = crossover._read_attempts(attempt_path)
    assert [row["status"] for row in attempts] == ["uncertain_cancellation"] * 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"
    resumed_client = _NoNetworkClient()
    with pytest.raises(crossover.CrossoverError, match="manual reconciliation"):
        asyncio.run(
            crossover.judge_deepseek_scope(
                records,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=resumed_client,  # type: ignore[arg-type]
            )
        )
    assert resumed_client.called is False


def test_probe_paid_failure_stops_refill_and_drains_inflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(crossover, "DEEPSEEK_CONCURRENCY", 2)
    records = _scope_records()
    template = _paper_template()
    attempt_path = tmp_path / "probe-attempts.jsonl"
    output_path = tmp_path / "probe-judgments.jsonl"
    manifest_path = tmp_path / "probe-manifest.json"
    protocol_id = "deepseek-user-luna-constraints-k1"
    dry = asyncio.run(
        crossover.judge_probe_scope(
            records,
            protocol_id=protocol_id,
            scope="baseline",
            template=template,
            attempt_log_path=attempt_path,
            output_path=output_path,
            manifest_path=manifest_path,
            api_key=None,
            expected_plan_sha256=None,
            confirm_paid=False,
            dry_run=True,
            client=_NoNetworkClient(),  # type: ignore[arg-type]
        )
    )
    client = _FirstResponseFailsClient()

    with pytest.raises(crossover.CrossoverError, match="manual reconciliation"):
        asyncio.run(
            crossover.judge_probe_scope(
                records,
                protocol_id=protocol_id,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=client,  # type: ignore[arg-type]
            )
        )

    assert client.calls == 2
    assert not output_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert len(crossover._read_attempts(attempt_path)) == 2


def test_probe_cancellation_marks_dispatched_slots_uncertain_before_unlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(crossover, "DEEPSEEK_CONCURRENCY", 2)
    records = _scope_records()
    template = _paper_template()
    attempt_path = tmp_path / "probe-attempts.jsonl"
    output_path = tmp_path / "probe-judgments.jsonl"
    manifest_path = tmp_path / "probe-manifest.json"
    protocol_id = "deepseek-user-luna-constraints-k1"
    dry = asyncio.run(
        crossover.judge_probe_scope(
            records,
            protocol_id=protocol_id,
            scope="baseline",
            template=template,
            attempt_log_path=attempt_path,
            output_path=output_path,
            manifest_path=manifest_path,
            api_key=None,
            expected_plan_sha256=None,
            confirm_paid=False,
            dry_run=True,
            client=_NoNetworkClient(),  # type: ignore[arg-type]
        )
    )
    client = _BlockingClient(expected_calls=2)

    async def cancel_run() -> None:
        run = asyncio.create_task(
            crossover.judge_probe_scope(
                records,
                protocol_id=protocol_id,
                scope="baseline",
                template=template,
                attempt_log_path=attempt_path,
                output_path=output_path,
                manifest_path=manifest_path,
                api_key="test-key",
                expected_plan_sha256=dry["plan_sha256"],
                confirm_paid=True,
                dry_run=False,
                client=client,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(client.all_started.wait(), timeout=1)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(cancel_run())

    assert client.calls == client.cancelled == 2
    assert client.active == 0
    attempts = crossover._read_attempts(attempt_path)
    assert [row["status"] for row in attempts] == ["uncertain_cancellation"] * 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_luna_scope_delegates_to_the_immutable_luna_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_judge_generations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["generations"] = args[0]
        captured["kwargs"] = kwargs
        return {"sentinel": "fixed-luna"}

    monkeypatch.setattr(openrouter, "_judge_generations", fake_judge_generations)
    result = asyncio.run(
        crossover.judge_luna_scope(
            _scope_records(),
            scope="baseline",
            template="template is intercepted by the mocked internal",
            attempt_log_path=tmp_path / "luna-attempts.jsonl",
            output_path=tmp_path / "luna-judgments.jsonl",
            manifest_path=tmp_path / "luna-manifest.json",
            api_key=None,
            expected_plan_sha256=None,
            confirm_paid=False,
            dry_run=True,
        )
    )

    assert result == {"sentinel": "fixed-luna"}
    assert captured["kwargs"]["judge_profile"] == openrouter.OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE
    assert captured["kwargs"]["judge_template_sha256"] == PAPER_JUDGE_TEMPLATE_SHA256
    assert captured["kwargs"]["max_attempts"] == 5
    assert captured["kwargs"]["_enforce_exact_paid_matrix"] is False
    assert captured["kwargs"]["_enforce_registered_profile"] is True
    assert len(captured["generations"]) == 100
