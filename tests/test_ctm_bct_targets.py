import asyncio
import hashlib
import json

import pytest

from ctm.backends.base import SampledSequence
from ctm.generation_provenance import make_generator_identity
from ctm.training.bct_targets import (
    BCTProgressStore,
    build_bct_rows_from_completions,
    build_completion_export_generation_provenance,
    completions_from_rows,
    generate_bct_rows,
    prepare_paired_prompts,
    validate_completion_export_generation_provenance,
    write_bct_target_artifacts,
)


class _Renderer:
    def build_generation_prompt(self, messages):
        return int(messages[-1]["content"].removeprefix("clean-"))

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, tokens):
        return {"role": "assistant", "content": f"answer-{tokens[0]}"}, None


class _Tokenizer:
    def decode(self, tokens):
        return f"decoded-{tokens[0]}"


class _Sampler:
    def __init__(self, *, fail_on=None):
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.fail_on = fail_on

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        self.calls.append((prompt, max_tokens, temperature, stop, num_samples))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        if prompt == self.fail_on:
            raise RuntimeError(f"failed row {prompt}")
        return [SampledSequence(tokens=[prompt], logprobs=[-0.1])]


def _rows(n=3):
    return [
        {
            "question_id": f"q-{index}",
            "unbiased_messages": [{"role": "user", "content": f"clean-{index}"}],
            "biased_messages": [{"role": "user", "content": f"biased-{index}"}],
        }
        for index in range(n)
    ]


def test_generate_bct_rows_samples_reference_once_and_shares_completion():
    prompts = prepare_paired_prompts(
        _rows(),
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
    )
    sampler = _Sampler()

    main, control = asyncio.run(
        generate_bct_rows(
            prompts,
            sampler=sampler,
            renderer=_Renderer(),
            tokenizer=_Tokenizer(),
            max_tokens=99,
            temperature=0.0,
            max_concurrency=2,
        )
    )

    assert len(sampler.calls) == 3
    assert sampler.max_in_flight == 2
    assert all(call[1:] == (99, 0.0, [0], 1) for call in sampler.calls)
    assert [row["messages"][0]["content"] for row in main] == ["biased-0", "biased-1", "biased-2"]
    assert [row["messages"][0]["content"] for row in control] == ["clean-0", "clean-1", "clean-2"]
    assert [row["messages"][-1] for row in main] == [row["messages"][-1] for row in control]
    assert [row["messages"][-1]["content"] for row in main] == ["answer-0", "answer-1", "answer-2"]
    assert [row["source_id"] for row in main] == ["q-0", "q-1", "q-2"]


def test_offline_completion_export_matches_exact_ids_and_shares_targets():
    prompts = prepare_paired_prompts(
        _rows(2),
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
    )
    completions = completions_from_rows(
        prompts,
        [
            {"source_id": "q-1", "response": "answer one"},
            {"source_id": "q-0", "response": "answer zero"},
        ],
    )
    main, control = build_bct_rows_from_completions(prompts, completions)

    assert [row["source_id"] for row in main] == ["q-0", "q-1"]
    assert [row["messages"][-1] for row in main] == [row["messages"][-1] for row in control]
    assert [row["messages"][-1]["content"] for row in main] == ["answer zero", "answer one"]


def test_offline_completion_export_rejects_missing_or_extra_ids():
    prompts = prepare_paired_prompts(
        _rows(1),
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
    )
    with pytest.raises(ValueError, match=r"missing=\['q-0'\].*extra=\['other'\]"):
        completions_from_rows(prompts, [{"source_id": "other", "response": "answer"}])


def test_offline_completion_provenance_binds_prompts_generator_decoding_and_responses():
    identities = [
        {
            "artifact_schema": "ctm.prompt_pairs",
            "schema_version": 1,
            "row_count": 2,
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
    ]
    generator = make_generator_identity(
        generator_id="self-bct",
        provider="local",
        model="model",
        model_revision="revision-1",
    )
    rows = [
        {"source_id": "q-0", "response": "zero"},
        {"source_id": "q-1", "response": "one"},
    ]
    provenance = build_completion_export_generation_provenance(
        rows,
        prompt_artifact_identities=identities,
        generator_identity=generator,
        decoding_parameters={"temperature": 0.0, "max_tokens": 16},
        source_messages_field="reference_messages",
        generated_at_utc="2026-07-30T00:00:00Z",
    )

    assert (
        validate_completion_export_generation_provenance(
            rows,
            provenance,
            prompt_artifact_identities=identities,
            generator_identity=generator,
            decoding_parameters={"temperature": 0.0, "max_tokens": 16},
            source_messages_field="reference_messages",
        )
        == provenance
    )
    tampered = [*rows[:-1], {"source_id": "q-1", "response": "changed"}]
    with pytest.raises(ValueError, match="ordered_response_manifest"):
        validate_completion_export_generation_provenance(
            tampered,
            provenance,
            prompt_artifact_identities=identities,
            generator_identity=generator,
            decoding_parameters={"temperature": 0.0, "max_tokens": 16},
            source_messages_field="reference_messages",
        )


def test_generate_bct_rows_resumes_completed_rows_in_source_order(tmp_path):
    prompts = prepare_paired_prompts(
        _rows(),
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
    )
    progress = BCTProgressStore(tmp_path / "progress", {"job": "unit"})
    assistant = {"role": "assistant", "content": "answer-0"}
    first_main = {"messages": [*prompts[0].main_messages, assistant], "source_id": "q-0"}
    first_control = {"messages": [*prompts[0].control_messages, assistant], "source_id": "q-0"}
    progress.record(0, prompts[0], first_main, first_control)

    sampler = _Sampler()
    main, control = asyncio.run(
        generate_bct_rows(
            prompts,
            sampler=sampler,
            renderer=_Renderer(),
            tokenizer=_Tokenizer(),
            completed=progress.load(prompts),
            on_completed=progress.record,
        )
    )

    assert [call[0] for call in sampler.calls] == [1, 2]
    assert [row["source_id"] for row in main] == ["q-0", "q-1", "q-2"]
    assert [row["source_id"] for row in control] == ["q-0", "q-1", "q-2"]
    assert len(progress.load(prompts)) == 3
    archived = progress.archive()
    assert archived is not None and archived.is_dir()
    assert not progress.directory.exists()


def test_bct_progress_rejects_a_different_generation_identity(tmp_path):
    directory = tmp_path / "progress"
    BCTProgressStore(directory, {"model": "first"})
    with pytest.raises(ValueError, match="identity differs"):
        BCTProgressStore(directory, {"model": "second"})


def test_generate_bct_rows_checkpoints_successes_before_a_later_failure(tmp_path):
    prompts = prepare_paired_prompts(
        _rows(),
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
    )
    progress = BCTProgressStore(tmp_path / "progress", {"job": "failure-test"})

    with pytest.raises(RuntimeError, match="failed row 1"):
        asyncio.run(
            generate_bct_rows(
                prompts,
                sampler=_Sampler(fail_on=1),
                renderer=_Renderer(),
                tokenizer=_Tokenizer(),
                max_concurrency=1,
                on_completed=progress.record,
            )
        )

    completed = progress.load(prompts)
    assert 0 in completed
    assert 1 not in completed


def test_prepare_paired_prompts_rejects_all_rows_before_sampling():
    rows = _rows(2)
    rows[1]["unbiased_messages"][-1]["role"] = "assistant"
    with pytest.raises(ValueError, match="row 2.unbiased_messages must end with a user message"):
        prepare_paired_prompts(
            rows,
            source_messages_field="unbiased_messages",
            main_messages_field="biased_messages",
            control_messages_field="unbiased_messages",
        )


def test_prepare_paired_prompts_rejects_empty_message_content():
    rows = _rows(1)
    rows[0]["biased_messages"][0]["content"] = "   "
    with pytest.raises(ValueError, match="non-empty role and string content"):
        prepare_paired_prompts(
            rows,
            source_messages_field="unbiased_messages",
            main_messages_field="biased_messages",
            control_messages_field="unbiased_messages",
        )


def test_prepare_paired_prompts_rejects_duplicate_source_ids():
    rows = _rows(2)
    rows[1]["question_id"] = rows[0]["question_id"]
    with pytest.raises(ValueError, match="duplicate source id"):
        prepare_paired_prompts(
            rows,
            source_messages_field="unbiased_messages",
            main_messages_field="biased_messages",
            control_messages_field="unbiased_messages",
        )


def test_write_bct_target_artifacts_records_shared_provenance_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_rows(1)[0]) + "\n")
    main_output = tmp_path / "bct.jsonl"
    control_output = tmp_path / "control.jsonl"
    manifest_output = tmp_path / "targets.manifest.json"
    main_rows = [{"messages": [{"role": "assistant", "content": "target"}], "source_id": "q-0"}]
    control_rows = [{"messages": [{"role": "assistant", "content": "target"}], "source_id": "q-0"}]

    manifest = write_bct_target_artifacts(
        main_rows=main_rows,
        control_rows=control_rows,
        main_output=main_output,
        control_output=control_output,
        manifest_output=manifest_output,
        source_files=[source],
        model="model",
        backend_name="FakeBackend",
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
        generation_config={"max_tokens": 32768, "temperature": 0.0, "api_key": "secret"},
    )

    assert manifest["row_count"] == 1
    assert manifest["generation"]["api_key"] == "<redacted>"
    assert json.loads(manifest_output.read_text()) == manifest
    assert manifest["outputs"]["main"]["content_sha256"] == hashlib.sha256(main_output.read_bytes()).hexdigest()
    assert manifest["outputs"]["control"]["content_sha256"] == hashlib.sha256(control_output.read_bytes()).hexdigest()
    assert json.loads(main_output.read_text())["messages"][-1] == json.loads(control_output.read_text())["messages"][-1]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_bct_target_artifacts(
            main_rows=main_rows,
            control_rows=control_rows,
            main_output=main_output,
            control_output=control_output,
            manifest_output=manifest_output,
            source_files=[source],
            model="model",
            backend_name="FakeBackend",
            source_messages_field="unbiased_messages",
            main_messages_field="biased_messages",
            control_messages_field="unbiased_messages",
            generation_config={},
        )


def test_write_bct_target_artifacts_completes_byte_identical_partial_publication(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_rows(1)[0]) + "\n", encoding="utf-8")
    main_output = tmp_path / "bct.jsonl"
    control_output = tmp_path / "control.jsonl"
    manifest_output = tmp_path / "targets.manifest.json"
    main_rows = [{"messages": [{"role": "assistant", "content": "target"}], "source_id": "q-0"}]
    control_rows = [{"messages": [{"role": "assistant", "content": "target"}], "source_id": "q-0"}]
    main_payload = (json.dumps(main_rows[0], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    main_output.write_bytes(main_payload)

    write_bct_target_artifacts(
        main_rows=main_rows,
        control_rows=control_rows,
        main_output=main_output,
        control_output=control_output,
        manifest_output=manifest_output,
        source_files=[source],
        model="model",
        backend_name="FakeBackend",
        source_messages_field="unbiased_messages",
        main_messages_field="biased_messages",
        control_messages_field="unbiased_messages",
        generation_config={},
    )

    assert main_output.read_bytes() == main_payload
    assert control_output.is_file()
    assert manifest_output.is_file()


def test_write_bct_target_artifacts_rejects_conflicting_partial_publication(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_rows(1)[0]) + "\n", encoding="utf-8")
    main_output = tmp_path / "bct.jsonl"
    main_output.write_text("different\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflicting partial"):
        write_bct_target_artifacts(
            main_rows=[{"messages": [], "source_id": "q-0"}],
            control_rows=[{"messages": [], "source_id": "q-0"}],
            main_output=main_output,
            control_output=tmp_path / "control.jsonl",
            manifest_output=tmp_path / "targets.manifest.json",
            source_files=[source],
            model="model",
            backend_name="FakeBackend",
            source_messages_field="unbiased_messages",
            main_messages_field="biased_messages",
            control_messages_field="unbiased_messages",
            generation_config={},
        )
