import asyncio
import hashlib
import json

import pytest

from ctm.backends.base import SampledSequence
from ctm.training.bct_targets import (
    generate_bct_rows,
    prepare_paired_prompts,
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
    def __init__(self):
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        self.calls.append((prompt, max_tokens, temperature, stop, num_samples))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
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
    assert manifest["outputs"]["control"]["content_sha256"] == hashlib.sha256(
        control_output.read_bytes()
    ).hexdigest()
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
