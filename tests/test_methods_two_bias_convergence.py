from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from ctm.backends.base import SampledSequence
from experiments.methods_two_bias_convergence import plan
from experiments.methods_two_bias_convergence import train as runner


def pool():
    return [{"question_id": f"q{i}", "source_dataset": "logiqa" if i % 2 == 0 else "hellaswag",
             "clean_messages": [{"role": "user", "content": f"question {i}"}],
             "variants": {bias: {"messages": [{"role": "user", "content": f"{bias}\nquestion {i}"}],
                                 "biasing_text": bias}
                          for bias in plan.BIASES}}
            for i in range(7680)]


def test_schedule_keeps_both_biases_and_only_wraps_after_3840_updates():
    source = pool()
    seen = []
    for step in range(4096):
        qids = plan.update_rows(source, step)
        pairs = plan.paired_rows(qids)
        assert [p["question_id"] for p in pairs] == [qids[0]["question_id"]] * 2 + [qids[1]["question_id"]] * 2
        assert [p["bias"] for p in pairs] == list(plan.BIASES) * 2
        assert [p["source_dataset"] for p in pairs] == ["logiqa"] * 2 + ["hellaswag"] * 2
        if step < 3840:
            seen.extend(r["question_id"] for r in qids)
    assert len(seen) == len(set(seen)) == 7680
    assert plan.update_rows(source, 3840) == plan.update_rows(source, 0)
    assert plan.update_rows(source, 4095) == plan.update_rows(source, 255)
    with pytest.raises(ValueError):
        plan.update_rows(source, 4096)


def test_explicit_contract_matches_rmct_optimizer_and_approved_training_cap():
    contract = plan.contract()
    from experiments.rmct_convergence.plan import segment_args
    # Authoritative original RMCT compiler, not a second hand-entered table.
    original = segment_args("/tmp/repo", 0)
    for key in ("beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"):
        assert contract["optimizer"][key] == original[key]
    assert contract["optimizer"]["learning_rate"] == original["lr"]
    assert contract["optimizer"]["lr_schedule"] == original["lr_schedule"]
    assert contract["batch"]["qids_per_update"] == original["batch_size"]
    assert contract["data"]["alpaca"] is False
    from experiments.rmct_convergence_r5_patience.plan import OUTPUT_TOKEN_CAP as RMCT_CAP
    assert contract["generation"]["output_token_cap"] == plan.OUTPUT_TOKEN_CAP == RMCT_CAP == 20480
    assert contract["generation"]["termination"] == "model_eos_or_length"
    assert contract["evaluation"]["output_token_cap"] is None
    assert contract["execution"]["hf_streaming_sampling"] is True
    assert contract["convergence"]["max_optimizer_steps"] == 4096
    for method in plan.METHODS:
        assert plan.lora(method)["dropout"] == original["lora_config"]["dropout"] == 0
        assert plan.lora(method)["rank"] == 8
        assert plan.lora(method)["alpha"] == 16
    assert plan.lora("bct")["train_mlp"] is True
    assert plan.lora("opct")["train_mlp"] is True
    assert plan.lora("act")["target_modules"] == ["q_proj", "v_proj", "in_proj_qkv"]
    assert contract["act_scope"]["paper_exact"] is False
    assert plan.lora("mlpct")["target_modules"] == ["q_proj", "v_proj"]
    assert plan.lora("mlpct")["train_mlp"] is False
    assert contract["execution"]["hf_language_model_only"] is False


@pytest.mark.parametrize("method", plan.INTERNAL_METHODS)
def test_approved_prefix_changes_only_internal_suggested_answer_training(method):
    qids = pool()[:2]
    # Cover a trailing cue and a cue inserted inside the question, the two
    # forms that motivated the approved training-only transformation.
    for row in qids:
        row["variants"]["suggested_answer"]["biasing_text"] = "I think the answer is C."
        row["variants"]["suggested_answer"]["messages"][-1]["content"] = (
            row["clean_messages"][-1]["content"] + "\nI think the answer is C."
        )
    original = copy.deepcopy(qids)
    native = plan.paired_rows(qids)
    prefixed = plan.paired_rows(qids, method=method)
    assert qids == original
    for index, pair in enumerate(prefixed):
        assert pair["reference_messages"] == native[index]["reference_messages"]
        assert pair["question_id"] == native[index]["question_id"]
        if pair["bias"] == "wrong_argument":
            assert pair == native[index]
        else:
            assert pair["variant_messages"][-1]["content"] == "I think the answer is C.\n\n" + pair["reference_messages"][-1]["content"]
    for unchanged_method in ("bct", "opct", None):
        assert plan.paired_rows(qids, method=unchanged_method) == native
    assert plan.contract()["data"]["transform_methods"] == list(plan.INTERNAL_METHODS)


def test_prefix_rejects_missing_or_invented_frozen_cue():
    qids = pool()[:2]
    for invalid in (None, "not present in the original prompt"):
        qids[0]["variants"]["suggested_answer"]["biasing_text"] = invalid
        with pytest.raises(ValueError, match="cue"):
            plan.paired_rows(qids, method="act")


def test_patience_is_best_so_far_over_complete_windows_and_resumable():
    state = {"step": 0, "pending": [], "decision": "continue"}
    for step in range(1, 145):
        state = plan.observe(state, step=step, loss=2.0)
        state = json.loads(json.dumps(state))
        assert state["decision"] == ("plateau" if step == 144 else "continue")
    assert state["best_step"] == 16
    assert state["nonimproving"] == 8


def test_improvement_resets_patience_and_max_updates_wins():
    state = {"step": 0, "pending": [], "decision": "continue"}
    for step in range(1, 4097):
        state = plan.observe(state, step=step, loss=1.0 / step)
        assert state["decision"] == ("max_updates" if step == 4096 else "continue")
    with pytest.raises(ValueError):
        plan.observe(state, step=4098, loss=1.0)
    with pytest.raises(ValueError):
        plan.observe(state, step=4097, loss=float("nan"))


class TinyRenderer:
    def build_generation_prompt(self, messages):
        from tinker import types
        return types.ModelInput.from_ints(tokens=[1, 2, 3])

    def get_stop_sequences(self):
        return [9]


@pytest.mark.parametrize("configured_eos", [9, 8, None, [8, 10]])
def test_bct_target_uses_approved_cap_is_cached_and_shared_exactly(tmp_path, configured_eos):
    calls = []

    class Sampler:
        async def sample(self, prompt, **kwargs):
            calls.append(kwargs)
            return [SampledSequence(tokens=[5, 6, 7, 9], logprobs=[-1.] * 4)]

    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=configured_eos)),
                              base_sampler=lambda: Sampler())
    qids = pool()[:2]
    async def build():
        return await runner.build_bct_datums(
            qids, plan.paired_rows(qids), backend=backend, renderer=TinyRenderer(),
            cache_dir=tmp_path, plan_hash="frozen-plan",
        )
    first = asyncio.run(build())
    second = asyncio.run(build())
    assert len(calls) == 2
    assert all(call["max_tokens"] == 20480 and call["num_samples"] == 1 for call in calls)
    for datum in [*first, *second]:
        assert datum.model_input.to_ints() == [1, 2, 3, 5, 6, 7]
        assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist()[-4:] == [5, 6, 7, 9]
        assert datum.loss_fn_inputs["weights"].to_torch().tolist() == [0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("tokens", [[], [5, 6], [9, 6]])
def test_non_eos_sample_rejected(tokens):
    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9)))
    with pytest.raises(ValueError, match="EOS"):
        runner.assert_eos(SampledSequence(tokens=tokens, logprobs=[]), backend, renderer=TinyRenderer())


@pytest.mark.parametrize("terminator", [8, 9])
def test_eos_check_accepts_both_model_and_renderer_tokens_without_rewriting(terminator):
    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=8)))
    tokens = [5, 6, terminator]
    assert runner.assert_eos(SampledSequence(tokens=tokens, logprobs=[]), backend,
                             renderer=TinyRenderer()) == tokens


@pytest.mark.parametrize("tokens,cap,expected", [
    ([5, 6, 7, 1], 4, "length"), ([5, 6, 7, 9], 4, "model_eos"),
    ([5, 8], 4, "model_eos"), ([5, 9], None, "model_eos"),
])
def test_completion_validator_preserves_eos_and_exact_length_stops(tokens, cap, expected):
    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=8)))
    sequence = SampledSequence(tokens=tokens, logprobs=[])
    assert runner.validate_completion(sequence, backend, renderer=TinyRenderer(), max_tokens=cap) == (tokens, expected)


@pytest.mark.parametrize("tokens,cap", [([], 4), ([5, 6], 4), ([5] * 5, 4), ([5] * 4 + [9], 4), ([5] * 4, None)])
def test_completion_validator_rejects_early_non_eos_and_over_cap_outputs(tokens, cap):
    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=8)))
    with pytest.raises(ValueError):
        runner.validate_completion(SampledSequence(tokens=tokens, logprobs=[]), backend,
                                   renderer=TinyRenderer(), max_tokens=cap)


def test_bct_length_stop_is_cached_without_adding_eos_or_resampling(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "OUTPUT_TOKEN_CAP", 4)
    calls = []

    class Sampler:
        async def sample(self, prompt, **kwargs):
            calls.append(kwargs)
            return [SampledSequence(tokens=[5, 6, 7, 1], logprobs=[-1.] * 4)]

    backend = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=8)),
                              base_sampler=lambda: Sampler())
    kwargs = dict(backend=backend, renderer=TinyRenderer(), cache_dir=tmp_path, plan_hash="test-length")
    for _ in range(2):
        assert asyncio.run(runner.bct_target(pool()[0], **kwargs)) == [5, 6, 7, 1]
    assert len(calls) == 1 and calls[0]["max_tokens"] == 4
    path, = tmp_path.glob("*.json")
    saved = json.loads(path.read_text())
    assert saved["finish_reason"] == "length"
    assert saved["output_token_cap"] == saved["identity"]["output_token_cap"] == 4
    saved["finish_reason"] = "model_eos"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="termination metadata"):
        asyncio.run(runner.bct_target(pool()[0], **kwargs))


@pytest.mark.parametrize("use_base", [False, True])
def test_real_uncapped_sampler_and_training_validator_agree_on_chat_eos(monkeypatch, use_base):
    backend = tiny_backend("bct")
    backend.model.generation_config.eos_token_id = 8

    def emit_chat_eos(probabilities, *, num_samples):
        return torch.full((probabilities.shape[0], num_samples), 9, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", emit_chat_eos)
    groups = backend._generate_batch(prompt_tokens_batch=[[1, 2, 3]], max_tokens=None,
                                    temperature=1.0, stop=TinyRenderer().get_stop_sequences(),
                                    num_samples=1, use_base=use_base)
    assert runner.assert_eos(groups[0][0], backend, renderer=TinyRenderer()) == [9]


def test_internal_alignment_gate_rejects_failed_or_unapproved_candidate_audit(tmp_path):
    with pytest.raises(ValueError, match="passing"):
        runner.require_alignment_audit(None)
    audit = {"model": plan.MODEL, "revision": plan.REVISION, "data_sha256": plan.POOL_SHA,
             "unique_qids": 7680, "paired_rows": 15360, "unaligned_rows": 0,
             "full_reference_suffix_alignment": True, "pair_transform": plan.INTERNAL_PAIR_TRANSFORM}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    runner.require_alignment_audit(path)
    for change in ({"full_reference_suffix_alignment": False}, {"unaligned_rows": 1},
                   {"pair_transform": "suggested_prefix_candidate_not_authorized"},
                   {"pair_transform": "native_shared_pool"}):
        path.write_text(json.dumps({**audit, **change}))
        with pytest.raises(ValueError, match="alignment audit failed"):
            runner.require_alignment_audit(path)


def tiny_backend(method, lora=None):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from ctm.backends.local.engine import LocalBackend
    from ctm.core.config import LoRAConfig
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=32,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=4,
        linear_num_value_heads=4, max_position_embeddings=32,
    ))
    backend = LocalBackend(device="cpu", dtype=torch.float32, model_instance=model,
                           consistency_loss_options=plan.LOSS_OPTIONS[method])
    backend.setup(model=plan.MODEL, lora=LoRAConfig(**{**(lora or plan.lora(method)), "rank": 2, "alpha": 4}))
    return backend


def paired_datum():
    from tinker import types
    scalar = lambda v: types.TensorData.from_torch(torch.tensor([v], dtype=torch.long))  # noqa: E731
    return types.Datum(model_input=types.ModelInput.from_ints(tokens=[1, 2, 3, 4, 5, 6, 7]),
                       loss_fn_inputs={"clean_tokens": types.TensorData.from_torch(torch.tensor([3, 4, 5, 6, 7])),
                                       "start_index": scalar(2), "clean_start_index": scalar(0),
                                       "clean_len": scalar(5), "match_len": scalar(5)})


def test_act_narrow_hybrid_gate_is_explicit_and_does_not_relax_historical_gate():
    backend = tiny_backend("act")
    historical = backend.run_qwen35_consistency_preflight([paired_datum()], method="act", expected_group_size=1)
    assert not historical["passed"]
    report = backend.run_qwen35_consistency_preflight(
        [paired_datum()] * 4, method="act", expected_group_size=4, lora_scope="act_qv_fused_qkv",
    )
    assert report["passed"], report["errors"]
    assert report["lora"]["expected_trainable_parameter_count"] == 80
    assert set(report["lora"]["positive_lora_b_gradients_by_family"]) == {"self_attn", "linear_attn"}
    assert backend._gradient_accumulations == 0


def test_act_strict_qv_scope_matches_attct_adapters_but_keeps_act_loss():
    act_qv = {**plan.lora("act"), "target_modules": ["q_proj", "v_proj"]}
    assert act_qv == plan.lora("attct")  # identical adapter scope to AttCT/MLPCT
    backend = tiny_backend("act", lora=act_qv)
    assert not backend.run_qwen35_consistency_preflight(
        [paired_datum()] * 4, method="act", expected_group_size=4, lora_scope="act_qv_fused_qkv")["passed"]
    report = backend.run_qwen35_consistency_preflight(
        [paired_datum()] * 4, method="act", expected_group_size=4, lora_scope="strict_qv")
    assert report["passed"], report["errors"]
    names = [n for n, p in backend.model.named_parameters() if p.requires_grad]
    assert names and all(("q_proj" in n or "v_proj" in n) and "linear_attn" not in n for n in names)
    assert backend.consistency_loss_options["layer_selection"] == "all"  # ACT loss still all layers


@pytest.mark.parametrize("method", ["act", "attct", "mlpct"])
def test_four_pair_backward_then_one_update_and_optimizer_checkpoint_resume(method, tmp_path):
    from ctm.core.config import AdamConfig
    from ctm.training.sft import METHOD_LOSS_FNS
    backend = tiny_backend(method)
    before = {n: p.detach().clone() for n, p in backend.model.named_parameters() if p.requires_grad}

    async def update():
        for _ in range(4):
            output = await (await backend.submit_forward_backward([paired_datum()], loss_fn=METHOD_LOSS_FNS[method])).result()
            assert torch.isfinite(torch.tensor(output.metrics["loss"]))
        assert backend._gradient_accumulations == 4
        runner.assert_gradients(backend)
        await (await backend.submit_optim_step(learning_rate=1e-4, adam=AdamConfig(**plan.contract()["optimizer"]))).result()
        assert backend._gradient_accumulations == 0
        state = {"step": 16, "pending": [], "decision": "continue", "best": .4, "best_step": 16, "nonimproving": 0}
        return await runner.seal_checkpoint(backend, run_dir=tmp_path, method=method, state=state,
                                            plan_hash="frozen", window_metrics=[])
    receipt = asyncio.run(update())
    assert any(not torch.equal(before[n], p.detach()) for n, p in backend.model.named_parameters() if p.requires_grad)
    state, resume = runner.load_resume(tmp_path, "frozen", method)
    assert state == receipt["convergence"]
    assert resume.endswith("step-000016")
    backend._load_checkpoint(resume, with_optimizer=True)
    assert backend._pending_optimizer_state is not None
    with pytest.raises(ValueError, match="different plan"):
        runner.load_resume(tmp_path, "wrong", method)
    manifest = tmp_path / "checkpoints/step-000016/manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="bytes"):
        runner.load_resume(tmp_path, "frozen", method)


def test_immutable_plan_write_rejects_conflicts(tmp_path):
    path = tmp_path / "plan.json"
    plan.immutable_json(path, {"a": 1})
    plan.immutable_json(path, {"a": 1})
    with pytest.raises(ValueError, match="overwrite"):
        plan.immutable_json(path, {"a": 2})


@pytest.mark.parametrize("method", ["bct", "opct"])
@pytest.mark.parametrize("length_stopped", [False, True])
@pytest.mark.parametrize("amended", [False, True])
def test_real_online_loop_can_resume_two_job_slices_without_resetting_optimizer_or_teacher(method, length_stopped, amended, tmp_path, monkeypatch):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from ctm.backends.local import engine
    from ctm.backends import renderers
    instances = []
    if length_stopped:
        monkeypatch.setattr(plan, "OUTPUT_TOKEN_CAP", 4)

    class StubGenerationBackend(engine.LocalBackend):
        def __init__(self, **kwargs):
            assert kwargs.get("hf_language_model_only", False) is False, "Qwen must not use Muse-only detachment"
            assert kwargs["hf_streaming_sampling"] is True
            torch.manual_seed(123)
            model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=4,
                linear_num_value_heads=4, max_position_embeddings=32, eos_token_id=8,
            ))
            super().__init__(device="cpu", dtype=torch.float32, model_instance=model,
                             forward_microbatch_max_datums=1)
            instances.append(self)

        def _generate_batch(self, *, prompt_tokens_batch, max_tokens, temperature, stop, num_samples, use_base):
            assert max_tokens == plan.OUTPUT_TOKEN_CAP
            if method == "bct":
                assert use_base is True
            return [[SampledSequence(tokens=[5, 6, 7, 1 if length_stopped else 9], logprobs=[-4.] * 4)
                     for _ in range(num_samples)] for _ in prompt_tokens_batch]

    class DistinguishingRenderer(TinyRenderer):
        def build_generation_prompt(self, messages):
            from tinker import types
            biased = any(b in messages[-1]["content"] for b in plan.BIASES)
            return types.ModelInput.from_ints(tokens=[1, 2, 3] if biased else [3])

    monkeypatch.setattr(engine, "LocalBackend", StubGenerationBackend)
    monkeypatch.setattr(renderers, "get_renderer_and_tokenizer", lambda *a, **k: (DistinguishingRenderer(), None))
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(plan, "verify", lambda *a: {"execution_amendment": plan.EXECUTION_AMENDMENT} if amended else {})
    monkeypatch.setattr(plan, "ordered_pool", lambda *a: pool())
    snapshot = tmp_path / plan.REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    args = SimpleNamespace(repository=tmp_path, plan=plan_path, model_snapshot=snapshot,
                           run_root=tmp_path / "runs", method=method, updates_this_job=16)
    append = runner.append_json
    if amended:
        def interrupt_after_update(path, value):
            append(path, value)
            if path.name == "metrics.jsonl" and value["step"] == 3:
                runner.signal.raise_signal(runner.signal.SIGUSR1)
        monkeypatch.setattr(runner, "append_json", interrupt_after_update)
    first = asyncio.run(runner.train(args))
    assert first["step"] == (3 if amended else 16)
    if amended:
        assert len(first["pending"]) == 3
        assert len(runner.resume_window_metrics(args.run_root / method, first)) == 3
        monkeypatch.setattr(runner, "append_json", append)
    base_before = {n: p.detach().clone() for n, p in instances[-1].model.named_parameters() if ".lora_" not in n}
    second = asyncio.run(runner.train(args))
    final_step = 19 if amended else 32
    assert second["step"] == final_step
    assert all(torch.equal(p, base_before[n]) for n, p in instances[-1].model.named_parameters() if ".lora_" not in n)
    assert {int(v["step"]) for v in instances[-1]._optimizer.state_dict()["state"].values()} == {final_step}
    receipts = sorted((args.run_root / method / "receipts").glob("*.json"))
    assert len(receipts) == (19 if amended else 2)
    assert runner.load_resume(args.run_root / method, plan.sha256(plan_path), method)[0]["step"] == final_step
    if amended:
        assert sorted(p.name for p in (args.run_root / method / "checkpoints").iterdir()) == ["step-000016", "step-000019"]
        assert len(runner.resume_window_metrics(args.run_root / method, second)) == 3
        archived_window = json.loads((args.run_root / method / "checkpoints/step-000016/window-metrics.json").read_text())
        assert [m["step"] for m in archived_window] == list(range(1, 17))
    if method == "bct":
        assert len(list((args.run_root / method / "base-targets").glob("*.json"))) == 2 * final_step
    else:
        for path in (args.run_root / method / "attempts").glob("*/rollouts.jsonl"):
            for line in path.read_text().splitlines():
                records = json.loads(line)["rollouts"]
                assert len(records) == 16
                assert all(r["generation_finish_reason"] == ("length" if length_stopped else "model_eos")
                           and r["generation_output_token_cap"] == plan.OUTPUT_TOKEN_CAP for r in records)


@pytest.mark.parametrize("scheduler_state", ["FAILED", "OUT_OF_MEMORY", "CANCELLED", "RUNNING", "PENDING", "PREEMPTED"])
def test_continuation_never_retries_arbitrary_failures(scheduler_state):
    from experiments.methods_two_bias_convergence.continuation import require_progress
    with pytest.raises(ValueError, match="not an authorized"):
        require_progress(scheduler_state=scheduler_state, starting_step=16, state={"step": 20, "decision": "continue"})


@pytest.mark.parametrize("scheduler_state", ["COMPLETED", "TIMEOUT"])
def test_continuation_requires_sealed_progress_and_allows_terminal_noop(scheduler_state):
    from experiments.methods_two_bias_convergence.continuation import require_progress
    with pytest.raises(ValueError, match="no sealed optimizer progress"):
        require_progress(scheduler_state=scheduler_state, starting_step=16, state={"step": 16, "decision": "continue"})
    require_progress(scheduler_state=scheduler_state, starting_step=16, state={"step": 17, "decision": "continue"})
    require_progress(scheduler_state=scheduler_state, starting_step=16, state={"step": 16, "decision": "plateau"})


def test_execution_amendment_rejects_scientific_and_unrelated_source_changes(tmp_path, monkeypatch):
    parent = {"contract": plan.contract(), "qid_order_sha256": "qid-order", "sources": {
        "ctm/backends/local/engine.py": "engine",
        "experiments/methods_two_bias_convergence/train.py": "old-trainer",
    }}
    plan.immutable_json(tmp_path / "parent-plan.json", parent)
    monkeypatch.setattr(plan, "sha256", lambda path: plan.PARENT_PLAN_SHA)
    document = copy.deepcopy(parent)
    document["execution_amendment"] = plan.EXECUTION_AMENDMENT
    document["sources"].update({"experiments/methods_two_bias_convergence/train.py": "new-trainer",
                               "experiments/methods_two_bias_convergence/continuation.py": "new-continuation"})
    plan.verify_amendment(tmp_path, document)
    invalid = copy.deepcopy(document)
    invalid["contract"]["generation"]["output_token_cap"] = 500
    with pytest.raises(ValueError, match="changed contract"):
        plan.verify_amendment(tmp_path, invalid)
    invalid = copy.deepcopy(document)
    invalid["sources"]["ctm/backends/local/engine.py"] = "changed-engine"
    with pytest.raises(ValueError, match="unrelated"):
        plan.verify_amendment(tmp_path, invalid)


def test_partial_window_resume_rejects_missing_or_changed_losses(tmp_path):
    checkpoint = tmp_path / "checkpoints/step-000003"
    checkpoint.mkdir(parents=True)
    plan.immutable_json(tmp_path / "state.json", {"checkpoint": "checkpoints/step-000003"})
    plan.immutable_json(checkpoint / "window-metrics.json", [{"step": 3, "loss": .2}])
    with pytest.raises(ValueError, match="partial checkpoint loss window"):
        runner.resume_window_metrics(tmp_path, {"step": 3, "pending": [.1, .2, .3]})
