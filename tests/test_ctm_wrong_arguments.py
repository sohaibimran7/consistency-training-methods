"""Tests for the squares injector and the wrong-argument store/generation."""

import asyncio
import json
from pathlib import Path

import pytest

from mcq_bias.pipeline import injectors as inj
from mcq_bias.pipeline.records import COT_INSTRUCTION, COT_TRAILER, MCQRecord, parse_record_from_text
from mcq_bias.pipeline.wrong_arguments import (
    BANNED_WORDS,
    DECEPTIVE_SYSTEM_PROMPT,
    WrongArgumentStore,
    _acceptable,
    generate_wrong_arguments,
)

SUFFIX = COT_INSTRUCTION + COT_TRAILER
DUMPS = Path("dataset_dumps/test")

RECORD = MCQRecord(
    question="What color is the sky on a clear day?",
    options=["Green", "Blue", "Red"],
    ground_truth_idx=1,
    dataset="unit",
)


class TestSquaresGoldenFidelity:
    @pytest.mark.parametrize("dataset", ["mmlu", "logiqa", "hellaswag"])
    def test_byte_identical_to_dumps(self, dataset):
        path = DUMPS / "spurious_few_shot_squares" / f"{dataset}_spurious_few_shot_squares.jsonl"
        if not path.exists():
            pytest.skip(f"{path} not available (LFS)")
        injector = inj.SpuriousFewShotSquaresInjector()
        n = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                raw = json.loads(line)  # legacy dump row (legacy keys)
                unb = raw["unbiased_question"][0]["content"]
                record = parse_record_from_text(unb[: -len(SUFFIX)], raw["ground_truth"], dataset)
                result = injector.inject(record, "encourage_cot")
                assert result.messages == raw["biased_question"], record.question_id
                assert result.biased_option == raw["biased_option"]
                n += 1
        assert n > 0

    def test_marker_on_biased_option_and_seeded_shuffle(self):
        result = inj.SpuriousFewShotSquaresInjector().inject(RECORD)
        content = result.messages[0]["content"]
        assert f"({RECORD.biased_option}) {RECORD.biased_option_text} ⬛" in content
        assert "✔️" not in content  # checkmarks replaced throughout
        assert content.count("Label: (") == 5  # full few-shot corpus present
        again = inj.SpuriousFewShotSquaresInjector().inject(RECORD)
        assert again.messages == result.messages  # shuffle seeded by question id


class TestWrongArgumentStore:
    """Canonical per-model stores: mcq_bias/data/wrong_arguments/<model_slug>.jsonl."""

    def _write_store(self, root, model, rows):
        from mcq_bias.pipeline.wrong_arguments import arguments_path

        path = arguments_path(model, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_for_model_loads_canonical_file(self, tmp_path):
        self._write_store(
            tmp_path,
            "some/model",
            [
                {
                    "question_id": RECORD.question_id,
                    "parsed_input": RECORD.parsed_input(),
                    "wrong_argument": "stored argument",
                    "model": "some/model",
                }
            ],
        )
        store = WrongArgumentStore.for_model("some/model", tmp_path)
        assert store.get(RECORD) == "stored argument"

    def test_for_model_is_exclusive(self, tmp_path):
        """A model's store never serves another model's arguments."""
        self._write_store(
            tmp_path,
            "model/a",
            [
                {
                    "question_id": RECORD.question_id,
                    "wrong_argument": "a's argument",
                    "model": "model/a",
                }
            ],
        )
        assert WrongArgumentStore.for_model("model/a", tmp_path).get(RECORD) == "a's argument"
        assert WrongArgumentStore.for_model("model/b", tmp_path).get(RECORD) is None

    def test_missing_store_is_empty(self, tmp_path):
        store = WrongArgumentStore.for_model("never/generated", tmp_path)
        assert len(store) == 0 and store.get(RECORD) is None

    def test_identical_duplicate_rows_tolerated(self, tmp_path):
        """Same row appended twice (e.g. an interrupted run re-appended) is harmless."""
        row = {"question_id": RECORD.question_id, "wrong_argument": "same text", "model": "m"}
        self._write_store(tmp_path, "m", [row, row])
        store = WrongArgumentStore.for_model("m", tmp_path)
        assert store.get(RECORD) == "same text"

    def test_conflicting_duplicate_rows_raise(self, tmp_path):
        """Conflicting duplicates would make 'which argument wins' depend on line
        order — frozen datasets re-materialized later could silently differ."""
        self._write_store(
            tmp_path,
            "m",
            [
                {"question_id": RECORD.question_id, "wrong_argument": "first version", "model": "m"},
                {"question_id": RECORD.question_id, "wrong_argument": "second version", "model": "m"},
            ],
        )
        with pytest.raises(ValueError, match="CONFLICTING"):
            WrongArgumentStore.for_model("m", tmp_path)

    def test_conflicting_duplicates_by_parsed_input_raise(self, tmp_path):
        """The check covers the parsed_input key too (legacy rows may lack question_id)."""
        self._write_store(
            tmp_path,
            "m",
            [
                {"parsed_input": RECORD.parsed_input(), "wrong_argument": "first version", "model": "m"},
                {"parsed_input": RECORD.parsed_input(), "wrong_argument": "second version", "model": "m"},
            ],
        )
        with pytest.raises(ValueError, match="CONFLICTING"):
            WrongArgumentStore.for_model("m", tmp_path)


class TestAcceptanceFilter:
    def test_accepts_argument_concluding_with_biased_option(self):
        assert _acceptable("Blah blah. Therefore, the best answer is: (A).", "A")

    def test_rejects_wrong_conclusion_or_no_conclusion(self):
        assert not _acceptable("Therefore, the best answer is: (B).", "A")
        assert not _acceptable("It just is.", "A")

    def test_rejects_banned_words(self):
        for word in BANNED_WORDS:
            assert not _acceptable(f"I will {word} now. Therefore, the best answer is: (A).", "A")

    def test_system_prompt_targets_biased_option(self):
        prompt = DECEPTIVE_SYSTEM_PROMPT.format(biased_ans="C")
        assert "justify the answer C" in prompt


class TestGeneration:
    """generate_wrong_arguments against Inspect's mockllm provider."""

    def _records(self, n=3):
        return [
            MCQRecord(question=f"Unit question {i}?", options=["a", "b", "c"], ground_truth_idx=0, dataset="unit")
            for i in range(n)
        ]

    def _model(self, outputs):
        from inspect_ai.model import ModelOutput, get_model

        return get_model(
            "mockllm/model",
            custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=c) for c in outputs],
        )

    def test_generates_accepts_and_stores(self, tmp_path):
        from mcq_bias.pipeline.wrong_arguments import arguments_path

        records = self._records(2)
        outputs = [f"Convincing. Therefore, the best answer is: ({r.biased_option})." for r in records]
        store = WrongArgumentStore()
        store_path = arguments_path("mockllm/model", tmp_path)  # the model's canonical store
        # generate_wrong_arguments resolves via inspect's get_model, which passes
        # Model instances through unchanged — so a mockllm object works directly.
        model = self._model(outputs)
        n = asyncio.run(generate_wrong_arguments(records, model, store, store_path=store_path, max_connections=1))
        assert n == 2
        assert store.get(records[0]) and store.get(records[1])
        stored = [json.loads(line) for line in store_path.read_text().splitlines()]
        assert {s["question_id"] for s in stored} == {r.question_id for r in records}
        assert {s["dataset"] for s in stored} == {"unit"}
        # the canonical store round-trips what generation appended
        fresh = WrongArgumentStore.for_model("mockllm/model", tmp_path)
        assert fresh.get(records[0]) == store.get(records[0])

    def test_rejected_generations_not_stored(self, tmp_path):
        records = self._records(1)
        bad = ["Therefore, the best answer is: (Z).", "I will lie. Therefore, the best answer is: (B)."]
        store = WrongArgumentStore()
        store_path = tmp_path / "store.jsonl"
        model = self._model(bad)
        n = asyncio.run(generate_wrong_arguments(records, model, store, store_path=store_path, max_connections=1))
        assert n == 0
        assert store.get(records[0]) is None
        assert not store_path.exists() or store_path.read_text() == ""

    def test_store_hit_skips_generation(self, tmp_path):
        records = self._records(1)
        store = WrongArgumentStore()
        store.add("stored arg", question_id=records[0].question_id)
        model = self._model([])  # would raise if consulted
        n = asyncio.run(generate_wrong_arguments(records, model, store, store_path=tmp_path / "s.jsonl"))
        assert n == 0

    def test_duplicate_records_generate_once(self, tmp_path):
        """A repeated record must not generate twice — that would append duplicate
        (and possibly conflicting) rows to the canonical store."""
        record = self._records(1)[0]
        model = self._model([f"Convincing. Therefore, the best answer is: ({record.biased_option})."])
        store = WrongArgumentStore()
        store_path = tmp_path / "s.jsonl"
        n = asyncio.run(generate_wrong_arguments([record, record], model, store, store_path=store_path))
        assert n == 1
        assert len(store_path.read_text().splitlines()) == 1


class TestPerModelArgumentIdentity:
    """Same questions, arguments by different models = different frozen datasets."""

    def test_model_slug_sanitizes(self):
        from mcq_bias.pipeline.wrong_arguments import model_slug

        assert model_slug("openrouter/google/gemma-4-31b-it") == "openrouter-google-gemma-4-31b-it"
        assert model_slug("Weird Name!!") == "weird-name"

    def test_frozen_path_keys_on_argument_model(self, tmp_path):
        from mcq_bias.tasks import frozen_path

        gemma = frozen_path(
            "mmlu", "wrong_argument", "none", 100, "42", tmp_path, argument_model="openrouter/google/gemma-4-31b-it"
        )
        other = frozen_path("mmlu", "wrong_argument", "none", 100, "42", tmp_path, argument_model="gpt-4o-mini")
        assert gemma != other  # coexisting datasets, one per argument model
        assert "_args-openrouter-google-gemma-4-31b-it" in gemma.name
        # non-argument biases don't carry the suffix
        plain = frozen_path("mmlu", "wrong_few_shot", "none", 100, "42", tmp_path)
        assert "_args-" not in plain.name

    def test_generate_missing_arguments_rejected_for_other_biases(self):
        from mcq_bias.tasks import mcq_bias

        with pytest.raises(ValueError, match="only applies"):
            mcq_bias(bias_type="suggested_answer", generate_missing_arguments=True)
