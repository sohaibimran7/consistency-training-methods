"""Offline tests for the paid Figure 6 OpenAI Batch lifecycle boundary."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctm_data.adapters.eval_awareness import figure6_batch as lifecycle
from ctm_data.adapters.eval_awareness import figure6_judge as judge


def _canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _request(custom_id: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": judge.BATCH_ENDPOINT,
        "body": {
            "model": judge.DEFAULT_JUDGE_MODEL,
            "max_completion_tokens": judge.MAX_JUDGE_TOKENS,
            "messages": [{"role": "system", "content": f"judge prompt for {custom_id}"}],
        },
    }


def _request_manifest(tmp_path: Path, *, shard_count: int = 2) -> Path:
    all_custom_ids = []
    shards = []
    for index in range(1, shard_count + 1):
        custom_id = f"figure6-unit-{index}"
        all_custom_ids.append(custom_id)
        payload = _canonical_jsonl([_request(custom_id)])
        path = tmp_path / f"requests.part-{index:05d}-of-{shard_count:05d}.jsonl"
        path.write_bytes(payload)
        shards.append(
            {
                "index": index,
                "file_name": path.name,
                "row_count": 1,
                "utf8_bytes": len(payload),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "custom_ids": [custom_id],
            }
        )
    manifest = {
        "schema": lifecycle.REQUEST_MANIFEST_SCHEMA,
        "row_count": shard_count,
        "shard_count": shard_count,
        "total_utf8_bytes": sum(shard["utf8_bytes"] for shard in shards),
        "ordered_custom_ids_sha256": hashlib.sha256("\n".join(all_custom_ids).encode()).hexdigest(),
        "judge_model": judge.DEFAULT_JUDGE_MODEL,
        "max_completion_tokens": judge.MAX_JUDGE_TOKENS,
        "judge_template_sha256": judge.PAPER_JUDGE_TEMPLATE_SHA256,
        "endpoint": judge.BATCH_ENDPOINT,
        "shards": shards,
        "submitted": False,
    }
    path = tmp_path / "requests.jsonl.manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _batch_record(
    *,
    batch_id: str,
    input_file_id: str,
    status: str = "validating",
    metadata: dict[str, str] | None = None,
    output_file_id: str | None = None,
    error_file_id: str | None = None,
    completed: int = 0,
    failed: int = 0,
) -> dict:
    return {
        "id": batch_id,
        "input_file_id": input_file_id,
        "endpoint": judge.BATCH_ENDPOINT,
        "completion_window": "24h",
        "status": status,
        "metadata": metadata or {},
        "created_at": 1_700_000_000,
        "completed_at": 1_700_000_010 if status == "completed" else None,
        "failed_at": 1_700_000_010 if status == "failed" else None,
        "expired_at": 1_700_000_010 if status == "expired" else None,
        "cancelled_at": 1_700_000_010 if status == "cancelled" else None,
        "request_counts": {"total": completed + failed, "completed": completed, "failed": failed},
        "errors": [{"code": "fixture_failure"}] if status == "failed" else None,
        "output_file_id": output_file_id,
        "error_file_id": error_file_id,
    }


class FakeFiles:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.payloads: dict[str, bytes] = {}
        self.create_calls = 0
        self.content_calls = 0

    def create(self, *, file, purpose: str):
        self.create_calls += 1
        filename, handle, _content_type = file
        payload = handle.read()
        file_id = f"file-input-{self.create_calls}"
        result = SimpleNamespace(
            id=file_id,
            filename=filename,
            purpose=purpose,
            bytes=len(payload),
            created_at=self.create_calls,
        )
        self.records[file_id] = result
        self.payloads[file_id] = payload
        return result

    def list(self, *, purpose: str, limit: int):
        assert purpose == "batch"
        assert limit == 10_000
        return [record for record in self.records.values() if record.purpose == purpose]

    def content(self, file_id: str):
        self.content_calls += 1
        return SimpleNamespace(read=lambda: self.payloads[file_id])

    def add_download(self, file_id: str, payload: bytes) -> None:
        self.payloads[file_id] = payload
        self.records[file_id] = SimpleNamespace(
            id=file_id,
            filename=f"{file_id}.jsonl",
            purpose="batch_output",
            bytes=len(payload),
            created_at=100,
        )


class FakeBatches:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.create_calls = 0
        self.retrieve_calls = 0

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ):
        assert endpoint == judge.BATCH_ENDPOINT
        assert completion_window == "24h"
        self.create_calls += 1
        batch_id = f"batch-{self.create_calls}"
        result = _batch_record(batch_id=batch_id, input_file_id=input_file_id, metadata=metadata)
        self.records[batch_id] = result
        return result

    def list(self, *, limit: int):
        assert limit == 100
        return list(self.records.values())

    def retrieve(self, batch_id: str):
        self.retrieve_calls += 1
        return self.records[batch_id]


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches()


def _submit(tmp_path: Path, client: FakeClient) -> tuple[Path, Path]:
    request_manifest = _request_manifest(tmp_path)
    manifest = tmp_path / "batch-lifecycle.json"
    lifecycle.submit_batches(
        request_manifest_path=request_manifest,
        lifecycle_manifest_path=manifest,
        yes=True,
        client=client,
        stdout=io.StringIO(),
    )
    return request_manifest, manifest


def test_submit_prints_exact_plan_and_requires_explicit_yes(tmp_path: Path):
    request_manifest = _request_manifest(tmp_path)
    manifest = tmp_path / "batch-lifecycle.json"
    client = FakeClient()
    output = io.StringIO()

    with pytest.raises(lifecycle.BatchLifecycleError, match="requires --yes"):
        lifecycle.submit_batches(
            request_manifest_path=request_manifest,
            lifecycle_manifest_path=manifest,
            yes=False,
            client=client,
            stdout=output,
        )

    printed = output.getvalue()
    assert "PAID OPENAI BATCH SUBMISSION PLAN" in printed
    assert "judge model: gpt-5" in printed
    assert "endpoint: /v1/chat/completions" in printed
    assert "max_completion_tokens: 4096" in printed
    assert "completion_window: 24h" in printed
    assert "requests: 2" in printed
    assert str((tmp_path / "requests.part-00001-of-00002.jsonl").resolve()) in printed
    assert client.files.create_calls == 0
    assert client.batches.create_calls == 0
    assert not manifest.exists()


def test_submit_is_idempotent_and_manifest_binds_protocol_and_shards(tmp_path: Path):
    client = FakeClient()
    request_manifest, manifest = _submit(tmp_path, client)

    first = json.loads(manifest.read_text())
    lifecycle.submit_batches(
        request_manifest_path=request_manifest,
        lifecycle_manifest_path=manifest,
        yes=True,
        client=client,
        stdout=io.StringIO(),
    )
    second = json.loads(manifest.read_text())

    assert client.files.create_calls == 2
    assert client.batches.create_calls == 2
    assert first == second
    assert second["approval"]["confirmation_flag"] == "--yes"
    assert second["protocol"] == {
        "name": lifecycle.JUDGE_PROTOCOL,
        "judge_model": "gpt-5",
        "max_completion_tokens": 4_096,
        "judge_template_sha256": judge.PAPER_JUDGE_TEMPLATE_SHA256,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    assert all(shard["input_file_id"] for shard in second["shards"])
    assert all(shard["batch_id"] for shard in second["shards"])
    assert all(len(shard["request_content_sha256"]) == 64 for shard in second["shards"])


def test_submit_recovers_unrecorded_provider_batch_without_duplicate_paid_calls(tmp_path: Path):
    client = FakeClient()
    request_manifest, manifest_path = _submit(tmp_path, client)
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["batch_id"] = None
    manifest["shards"][0]["input_file_id"] = None
    manifest["shards"][0]["status"] = "not_submitted"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    lifecycle.submit_batches(
        request_manifest_path=request_manifest,
        lifecycle_manifest_path=manifest_path,
        yes=True,
        client=client,
        stdout=io.StringIO(),
    )

    recovered = json.loads(manifest_path.read_text())
    assert client.files.create_calls == 2
    assert client.batches.create_calls == 2
    assert recovered["shards"][0]["batch_id"] == "batch-1"
    assert any(event["event"] == "provider_batch_recovered" for event in recovered["events"])


def test_status_and_download_are_resumable_and_hash_verified(tmp_path: Path):
    client = FakeClient()
    _, manifest_path = _submit(tmp_path, client)
    output_payloads = [b'{"custom_id":"one"}\n', b'{"custom_id":"two"}\n']
    error_payload = b'{"custom_id":"two","error":{"code":"fixture"}}\n'
    for index, batch_id in enumerate(("batch-1", "batch-2")):
        output_id = f"file-output-{index + 1}"
        error_id = "file-error-2" if index == 1 else None
        client.files.add_download(output_id, output_payloads[index])
        if error_id:
            client.files.add_download(error_id, error_payload)
        client.batches.records[batch_id] = _batch_record(
            batch_id=batch_id,
            input_file_id=f"file-input-{index + 1}",
            status="completed",
            output_file_id=output_id,
            error_file_id=error_id,
            completed=1,
            failed=int(error_id is not None),
            metadata=client.batches.records[batch_id]["metadata"],
        )

    status = lifecycle.status_batches(lifecycle_manifest_path=manifest_path, client=client)
    output_dir = tmp_path / "downloads"
    first = lifecycle.download_batches(
        lifecycle_manifest_path=manifest_path,
        output_dir=output_dir,
        client=client,
    )
    content_calls = client.files.content_calls
    second = lifecycle.download_batches(
        lifecycle_manifest_path=manifest_path,
        output_dir=output_dir,
        client=client,
    )

    assert status["statuses"] == {"completed": 2}
    assert first["downloaded_output_files"] == 2
    assert first["downloaded_error_files"] == 1
    assert second == first
    assert client.files.content_calls == content_calls
    assert (output_dir / "batch-output.part-00001.jsonl").read_bytes() == output_payloads[0]
    assert (output_dir / "batch-error.part-00002.jsonl").read_bytes() == error_payload
    disk_manifest = json.loads(manifest_path.read_text())
    assert (
        disk_manifest["shards"][0]["downloads"]["output"]["content_sha256"]
        == hashlib.sha256(output_payloads[0]).hexdigest()
    )

    (output_dir / "batch-output.part-00001.jsonl").write_bytes(b"mismatched local content\n")
    with pytest.raises(lifecycle.BatchLifecycleError, match="refusing to overwrite"):
        lifecycle.download_batches(
            lifecycle_manifest_path=manifest_path,
            output_dir=output_dir,
            client=client,
        )
    assert (output_dir / "batch-output.part-00001.jsonl").read_bytes() == b"mismatched local content\n"


@pytest.mark.parametrize("terminal_status", ["failed", "expired", "cancelled"])
def test_terminal_failures_are_persisted_and_reported(tmp_path: Path, terminal_status: str):
    client = FakeClient()
    _, manifest_path = _submit(tmp_path, client)
    client.batches.records["batch-1"] = _batch_record(
        batch_id="batch-1",
        input_file_id="file-input-1",
        status=terminal_status,
        metadata=client.batches.records["batch-1"]["metadata"],
    )

    with pytest.raises(lifecycle.BatchLifecycleError, match=terminal_status):
        lifecycle.status_batches(lifecycle_manifest_path=manifest_path, client=client)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["shards"][0]["status"] == terminal_status


def test_download_saves_error_file_before_reporting_failed_batch(tmp_path: Path):
    client = FakeClient()
    _, manifest_path = _submit(tmp_path, client)
    error_payload = b'{"error":{"code":"invalid_request"}}\n'
    client.files.add_download("file-error-1", error_payload)
    client.batches.records["batch-1"] = _batch_record(
        batch_id="batch-1",
        input_file_id="file-input-1",
        status="failed",
        error_file_id="file-error-1",
        failed=1,
        metadata=client.batches.records["batch-1"]["metadata"],
    )
    output_dir = tmp_path / "downloads"

    with pytest.raises(lifecycle.BatchLifecycleError, match="batch batch-1=failed"):
        lifecycle.download_batches(
            lifecycle_manifest_path=manifest_path,
            output_dir=output_dir,
            client=client,
        )

    assert (output_dir / "batch-error.part-00001.jsonl").read_bytes() == error_payload


def test_wait_and_download_reject_partially_submitted_lifecycle(tmp_path: Path):
    client = FakeClient()
    _, manifest_path = _submit(tmp_path, client)
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][1]["batch_id"] = None
    manifest["shards"][1]["status"] = "uploaded"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    client.batches.records["batch-1"] = _batch_record(
        batch_id="batch-1",
        input_file_id="file-input-1",
        status="completed",
        completed=1,
        metadata=client.batches.records["batch-1"]["metadata"],
    )

    with pytest.raises(lifecycle.BatchLifecycleError, match="unsubmitted shard.*2"):
        lifecycle.status_batches(
            lifecycle_manifest_path=manifest_path,
            wait=True,
            client=client,
            sleep=lambda _: pytest.fail("partial lifecycle must not poll forever"),
        )
    with pytest.raises(lifecycle.BatchLifecycleError, match="download incomplete.*2"):
        lifecycle.download_batches(
            lifecycle_manifest_path=manifest_path,
            output_dir=tmp_path / "downloads",
            client=client,
        )


def test_request_manifest_protocol_tampering_is_rejected_before_any_api_call(tmp_path: Path):
    request_manifest = _request_manifest(tmp_path)
    value = json.loads(request_manifest.read_text())
    value["judge_model"] = "different-model"
    request_manifest.write_text(json.dumps(value), encoding="utf-8")
    client = FakeClient()

    with pytest.raises(lifecycle.BatchLifecycleError, match="judge_model"):
        lifecycle.submit_batches(
            request_manifest_path=request_manifest,
            lifecycle_manifest_path=tmp_path / "batch-lifecycle.json",
            yes=True,
            client=client,
            stdout=io.StringIO(),
        )
    assert client.files.create_calls == 0
    assert client.batches.create_calls == 0
