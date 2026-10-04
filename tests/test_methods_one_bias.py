from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from ctm.backends.base import SampledSequence
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import DATUM_SCHEMA, SCHEMA_VERSION
from experiments.methods_two_bias_convergence import one_bias, plan
from experiments.methods_two_bias_convergence import train as runner


def datum(i):
    return {
        "datum_schema": DATUM_SCHEMA, "schema_version": SCHEMA_VERSION,
        "question_id": f"q{i}", "source_dataset": ("logiqa", "hellaswag")[i % 2],
        "question": f"question {i}", "ground_truth": "A", "prompt_style": "none",
        "clean_messages": [{"role": "user", "content": f"question {i}"}],
        "biased_options": {b: "B" for b in plan.BIASES},
        "variants": {b: {"messages": [{"role": "user", "content": f"{b}\nquestion {i}"}],
                         "biased_option": "B", "biasing_text": b} for b in plan.BIASES},
        "provenance": {"wrong_argument_source_line_number": i + 1},
    }


@pytest.fixture
def pool(monkeypatch):
    rows = [datum(i) for i in range(7680)]
    order = hashlib.sha256(plan.canonical([r["question_id"] for r in rows])).hexdigest()
    monkeypatch.setattr(one_bias, "ORDER_SHA", order)
    return rows


def test_assignments_are_method_independent_and_never_cycle(pool):
    manifest = one_bias.manifest_for(pool)
    assert all(one_bias.manifest_for(pool) == manifest for _ in plan.METHODS)
    assert one_bias.max_updates(manifest) == 1920
    by_id = {r["question_id"]: r for r in pool}
    seen = []
    for attempt in (0, 1, 1919):
        rows, biases = one_bias.update_rows(manifest, by_id, attempt)
        assert len({r["question_id"] for r in rows}) == 4
        seen += [r["question_id"] for r in rows]
        for method in plan.METHODS:
            pairs = one_bias.pairs(rows, biases, method=method)
            assert [p["bias"] for p in pairs] == biases  # only the assigned cue, no second arm
    assert seen[:8] == [f"q{i}" for i in range(8)] and seen[-4:] == [f"q{i}" for i in range(7676, 7680)]
    with pytest.raises(ValueError, match="exhausted"):
        one_bias.update_rows(manifest, by_id, 1920)


def test_internal_methods_prefix_only_suggested_answer(pool):
    manifest = one_bias.manifest_for(pool)
    rows, biases = one_bias.update_rows(manifest, {r["question_id"]: r for r in pool}, 0)
    for pair, bias, row in zip(one_bias.pairs(rows, biases, method="act"), biases, rows):
        if bias == "suggested_answer":
            assert pair["variant_messages"][-1]["content"] == "suggested_answer\n\n" + row["clean_messages"][-1]["content"]
        else:
            assert pair["variant_messages"] == row["variants"][bias]["messages"]


def test_one_bias_contract_changes_only_exposure_and_stopping(pool):
    manifest = one_bias.manifest_for(pool)
    old, new = plan.contract(), plan.one_bias_contract(manifest)
    for key in ("model", "optimizer", "lora", "loss_options", "opct", "generation", "act_scope"):
        assert new[key] == old[key]
    assert new["batch"]["qids_per_update"] == 4 and new["batch"]["repeat_after_pool_exhaustion"] is False
    assert new["convergence"]["every_encountered_qid_bias_examples"] == 256
    assert new["convergence"]["loss_selects_or_stops"] is False
    assert new["bct"]["reuse_identical_target_for_both_biases"] is False


def test_strict_qv_act_scope_changes_only_act_adapters(pool):
    manifest = one_bias.manifest_for(pool)
    fused, strict = plan.one_bias_contract(manifest), plan.one_bias_contract(manifest, "strict_qv")
    assert fused["lora"]["act"]["target_modules"] == ["q_proj", "v_proj", "in_proj_qkv"]
    assert strict["lora"]["act"] == plan.lora("attct")
    assert strict["act_scope"]["preflight"] == "strict_qv"
    assert strict["loss_options"]["act"] == fused["loss_options"]["act"]  # all-layer ACT loss kept
    for key in set(fused) - {"lora", "act_scope"}:
        assert strict[key] == fused[key]
    assert {m: strict["lora"][m] for m in plan.METHODS if m != "act"} == {m: fused["lora"][m] for m in plan.METHODS if m != "act"}
    with pytest.raises(ValueError):
        plan.one_bias_contract(manifest, "qkv")


def test_observe_never_stops_on_loss_and_reports_exhaustion():
    state = {"step": 0, "pending": [], "decision": "continue"}
    for step in range(1, 5):
        state = one_bias.observe(state, step=step, loss=5.0, limit=4)
    assert state["decision"] == "exhausted" and state["attempts"] == 4
    with pytest.raises(ValueError):
        one_bias.observe(state, step=7, loss=1.0, limit=4)


@pytest.mark.parametrize("method", ["bct", "opct"])
def test_online_one_bias_loop_resumes_with_encounter_receipts(method, pool, tmp_path, monkeypatch):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from ctm.backends.local import engine
    from ctm.backends import renderers

    class StubBackend(engine.LocalBackend):
        def __init__(self, **kwargs):
            torch.manual_seed(123)
            model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=4,
                linear_num_value_heads=4, max_position_embeddings=32, eos_token_id=8,
            ))
            super().__init__(device="cpu", dtype=torch.float32, model_instance=model, forward_microbatch_max_datums=1)

        def _generate_batch(self, *, prompt_tokens_batch, max_tokens, temperature, stop, num_samples, use_base):
            return [[SampledSequence(tokens=[5, 6, 7, 9], logprobs=[-4.] * 4) for _ in range(num_samples)]
                    for _ in prompt_tokens_batch]

    class Renderer:
        def build_generation_prompt(self, messages):
            from tinker import types
            biased = any(b in messages[-1]["content"] for b in plan.BIASES)
            return types.ModelInput.from_ints(tokens=[1, 2, 3] if biased else [3])

        def get_stop_sequences(self):
            return [9]

    monkeypatch.setattr(engine, "LocalBackend", StubBackend)
    monkeypatch.setattr(renderers, "get_renderer_and_tokenizer", lambda *a, **k: (Renderer(), None))
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(plan, "ordered_pool", lambda *a: pool)
    from experiments.rmct_restart_20260928 import qwen_one_bias_validation as gate
    calls = []

    def budget(**kwargs):
        calls.append(kwargs["step"])
        return min(kwargs["requested"], 64 - kwargs["step"] % 64)
    monkeypatch.setattr(gate, "gated_budget", budget)
    manifest = one_bias.manifest_for(pool)
    monkeypatch.setattr(plan, "verify", lambda *a: {"contract": plan.one_bias_contract(manifest),
                                                    "one_bias_manifest": {"path": "m", "sha256": "s"}})
    snapshot = tmp_path / plan.REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    args = SimpleNamespace(repository=tmp_path, plan=plan_path, model_snapshot=snapshot,
                           run_root=tmp_path / "runs", method=method, updates_this_job=16,
                           selection_contract=None, selection_folder=None, validation_manifest=None)
    assert asyncio.run(runner.train(args))["step"] == 16
    assert asyncio.run(runner.train(args))["step"] == 32
    assert calls == [0, 16]  # every job asks the validation gate before training
    run_dir = args.run_root / method
    receipts = [json.loads(p.read_text()) for p in sorted((run_dir / "receipts").glob("*.json"))]
    assert [r["exposure"]["encountered_qid_bias_examples"] for r in receipts] == [64, 128]
    assert receipts[-1]["exposure"]["encounter_attempt"] == 32
    metrics = [json.loads(line) for p in sorted((run_dir / "attempts").glob("*/metrics.jsonl"))
               for line in p.read_text().splitlines()]
    metrics.sort(key=lambda m: m["step"])
    consumed = [q for m in metrics for q in m["question_ids"]]
    assert consumed == [a["question_id"] for a in manifest["assignments"][:128]]
    assert [b for m in metrics for b in m["biases"]] == [a["bias"] for a in manifest["assignments"][:128]]
    if method == "bct":
        assert len(list((run_dir / "base-targets").glob("*.json"))) == 128  # one target per QID
