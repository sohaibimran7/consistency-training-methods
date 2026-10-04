"""Resumable grouped-QID trainer using CTM's existing losses and backend.

One update consumes two QIDs, each with both biases. Four equal-weight paired
backwards precede one optimizer step. The loop position and patience state
are sealed with the optimizer, so a Slurm continuation does not reset either.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import shutil
import time
import uuid
from pathlib import Path

from . import one_bias as exposure
from . import plan


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(plan.canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json(path: Path, value: dict) -> None:
    with path.open("ab") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("production training requires a scheduled CUDA GPU")


def require_alignment_audit(path: Path | None, manifest: dict | None = None) -> None:
    if path is None:
        raise ValueError("internal-consistency training requires a passing full-pool --alignment-audit")
    audit = json.loads(path.read_text())
    required = {"model": plan.MODEL, "revision": plan.REVISION, "data_sha256": plan.POOL_SHA,
                "unique_qids": 7680, "paired_rows": 15360, "unaligned_rows": 0,
                "full_reference_suffix_alignment": True,
                "pair_transform": plan.contract()["data"]["internal_pair_transform"]}
    if manifest is not None:
        # One-bias campaigns train only the assigned pair of each QID.
        required.update(paired_rows=7680, pair_set="one_bias_assigned",
                        one_bias_manifest_sha256=exposure.protocol.manifest_identity(manifest))
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError("full-pool alignment audit failed or belongs to a different prompt contract")


def checkpoint_identity(directory: Path) -> dict:
    files = {str(p.relative_to(directory)): plan.sha256(p)
             for p in sorted(directory.rglob("*")) if p.is_file()}
    if not {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "manifest.json"} <= files.keys():
        raise ValueError(f"incomplete optimizer checkpoint: {directory}")
    return files


def load_resume(run_dir: Path, plan_hash: str, method: str, *, parent_plan_hash: str | None = None) -> tuple[dict, str | None]:
    pointer = run_dir / "state.json"
    if not pointer.exists():
        return {"step": 0, "pending": [], "decision": "continue"}, None
    state = json.loads(pointer.read_text())
    allowed = {plan_hash} | ({parent_plan_hash} if parent_plan_hash else set())
    if state["plan_sha256"] not in allowed or state["method"] != method:
        raise ValueError("resume state belongs to a different plan/method")
    checkpoint = run_dir / state["checkpoint"]
    if checkpoint.is_symlink() or checkpoint_identity(checkpoint) != state["checkpoint_files"]:
        raise ValueError("checkpoint bytes differ from sealed resume state")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["loop_state"]["convergence"] != state["convergence"]:
        raise ValueError("checkpoint loop position differs from resume state")
    if any(manifest["loop_state"].get(key) != state[key] for key in ("plan_sha256", "method")):
        raise ValueError("checkpoint identity differs from resume state")
    return state["convergence"], f"file://{checkpoint.resolve()}"


def resume_window_metrics(run_dir: Path, state: dict) -> list[dict]:
    """Carry the exact partial 16-update loss window across job boundaries."""
    if not state["pending"]:
        return []
    pointer = json.loads((run_dir / "state.json").read_text())
    metrics = json.loads((run_dir / pointer["checkpoint"] / "window-metrics.json").read_text())
    expected_steps = list(range(state["step"] - len(state["pending"]) + 1, state["step"] + 1))
    if [row["step"] for row in metrics] != expected_steps or [row["loss"] for row in metrics] != state["pending"]:
        raise ValueError("partial checkpoint loss window differs from convergence state")
    return metrics


def validate_completion(sequence, backend, *, renderer, max_tokens: int | None) -> tuple[list[int], str]:
    """Accept a real EOS or the exact approved length boundary; never add EOS."""
    from ctm.backends.local.engine import local_hf_eos_token_ids

    tokens = list(sequence.tokens)
    eos = local_hf_eos_token_ids(backend.model, renderer.get_stop_sequences())
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("output-token cap must be a positive integer or None")
        if len(tokens) > max_tokens:
            raise ValueError(f"completion exceeds approved output-token cap: {len(tokens)} > {max_tokens}")
    if tokens and tokens[-1] in eos:
        return tokens, "model_eos"
    if max_tokens is not None and len(tokens) == max_tokens:
        return tokens, "length"
    raise ValueError(
        "completion must end on a model/renderer EOS token or the approved length boundary: "
        f"last_token={tokens[-1] if tokens else None}, accepted_eos={eos}, "
        f"token_count={len(tokens)}, output_token_cap={max_tokens}"
    )


def assert_eos(sequence, backend, *, renderer) -> list[int]:
    return validate_completion(sequence, backend, renderer=renderer, max_tokens=None)[0]


async def bct_target(row, *, backend, renderer, cache_dir: Path, plan_hash: str) -> list[int]:
    """Sample once from the immutable clean base; never decode/re-tokenize."""
    import torch

    identity = {"question_id": row["question_id"], "clean_messages": row["clean_messages"],
                "plan_sha256": plan_hash, "temperature": 1.0, "termination": "model_eos_or_length",
                "output_token_cap": plan.OUTPUT_TOKEN_CAP}
    key = hashlib.sha256(plan.canonical(identity)).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("BCT target cache identity mismatch")
        from ctm.backends.base import SampledSequence
        tokens, finish = validate_completion(SampledSequence(tokens=saved["tokens"], logprobs=[]), backend,
                                            renderer=renderer, max_tokens=plan.OUTPUT_TOKEN_CAP)
        if saved.get("finish_reason") != finish or saved.get("output_token_cap") != plan.OUTPUT_TOKEN_CAP:
            raise ValueError("BCT target cache termination metadata mismatch")
        return tokens
    # Per-QID determinism makes target-generation order, cache hits, and job
    # boundaries irrelevant. Restore the update RNG stream afterwards.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(int(key[:15], 16))
        sequence, = await backend.base_sampler().sample(
            renderer.build_generation_prompt(row["clean_messages"]), max_tokens=plan.OUTPUT_TOKEN_CAP,
            temperature=1.0, stop=renderer.get_stop_sequences(), num_samples=1,
        )
    tokens, finish = validate_completion(sequence, backend, renderer=renderer, max_tokens=plan.OUTPUT_TOKEN_CAP)
    plan.immutable_json(path, {"identity": identity, "tokens": tokens,
                               "output_token_cap": plan.OUTPUT_TOKEN_CAP, "finish_reason": finish})
    return tokens


async def build_bct_datums(qids, pairs, *, backend, renderer, cache_dir, plan_hash):
    import torch
    from tinker import types
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    targets = {row["question_id"]: await bct_target(
        row, backend=backend, renderer=renderer, cache_dir=cache_dir, plan_hash=plan_hash,
    ) for row in qids}
    datums = []
    for pair in pairs:
        prefix = renderer.build_generation_prompt(pair["variant_messages"]).to_ints()
        completion = targets[pair["question_id"]]
        tokens = types.ModelInput.from_ints(tokens=prefix + completion)
        weights = torch.tensor([0.0] * len(prefix) + [1.0] * len(completion))
        datums.append(datum_from_model_input_weights(tokens, weights))
    return datums


def assert_gradients(backend) -> dict:
    import torch

    positive = 0
    trainable = 0
    for name, parameter in backend.model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        if ".lora_" not in name:
            raise ValueError(f"unexpected non-LoRA trainable parameter: {name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite gradient: {name}")
            positive += int(bool(torch.any(parameter.grad != 0)))
    if not positive:
        raise FloatingPointError("optimizer update has no non-zero trainable gradients")
    return {"trainable_parameters": trainable, "positive_gradient_tensors": positive}


async def supervised_update(method, qids, pairs, *, backend, renderer, tokenizer,
                            cache_dir, plan_hash, preflight_path, lora_scope="historical"):
    from ctm.training.consistency_data import (
        build_consistency_datums_with_audit, require_full_reference_suffix_alignment,
    )
    from ctm.training.sft import METHOD_LOSS_FNS, _mean_nll

    if method == "bct":
        datums = await build_bct_datums(qids, pairs, backend=backend, renderer=renderer,
                                      cache_dir=cache_dir, plan_hash=plan_hash)
    else:
        datums, audit = build_consistency_datums_with_audit(tokenizer, pairs)
        require_full_reference_suffix_alignment(audit)
        if len(datums) != 4:
            raise ValueError("dropping a paired row would break grouped-QID parity")
        if preflight_path is not None:
            report = backend.run_qwen35_consistency_preflight(
                datums, method=method, expected_group_size=4,
                lora_scope=lora_scope,
            )
            plan.immutable_json(preflight_path, report)
            if not report["passed"]:
                raise RuntimeError(f"first-backward preflight failed: {report['errors']}")
    metrics = []
    for datum in datums:
        pending = await backend.submit_forward_backward([datum], loss_fn=METHOD_LOSS_FNS[method])
        result = await pending.result()
        if method == "bct":
            metric = _mean_nll(result.logprobs, [datum.loss_fn_inputs["weights"].to_torch()])
        else:
            metric = float(result.metrics["loss"])
        if not math.isfinite(metric):
            raise FloatingPointError(f"non-finite {method} loss before optimizer step")
        metrics.append(metric)
    return metrics, {"variant_losses": metrics}


async def opct_update(trainer, pairs, *, backend):
    # Preserve the established four-rollout OPCT estimator. Generate all four
    # bias conditions before any optimizer mutation; one equally weighted
    # physical F/B call per condition gives the same grouped-QID contract.
    groups = [[(i, pair)] for i, pair in enumerate(pairs)]
    prepared = [trainer._prepare_batch(group) for group in groups]
    sampled = await trainer._sample_prepared_pairs([p for group in prepared for p in group])
    completions = [[validate_completion(sequence, backend, renderer=trainer.renderer,
                                       max_tokens=plan.OUTPUT_TOKEN_CAP)
                    for sequence in sequences] for sequences in sampled]
    results = await trainer._build_batch_group(
        groups, prepared_pair_groups=prepared, sampled_groups=[[group] for group in sampled],
    )
    metrics = []
    records = []
    for group_index, ((datums, kl_metrics, lengths, metadata), completion_group) in enumerate(
        zip(results, completions, strict=True)
    ):
        if len(datums) != 4 or len(lengths) != 4 or any(m.get("skipped_from_training") for m in metadata):
            raise ValueError("OPCT may not silently discard a rollout in the matched comparison")
        fused = all("opct_teacher_logprobs" in d.loss_fn_inputs for d in datums)
        if fused:
            pending = await backend.submit_opct_forward_backward(
                datums, behavior_temperature=trainer.config.generation.temperature,
                kl_coef=trainer.config.kl_coef, kl_discount_factor=trainer.config.kl_discount_factor,
                loss_fn=trainer.config.loss_fn,
            )
        else:
            pending = await backend.submit_forward_backward(datums, loss_fn=trainer.config.loss_fn)
        output = await pending.result()
        if fused:
            trainer._complete_fused_batch(datums, output, kl_metrics, metadata)
        metric = float(kl_metrics["teacher_kl"])
        if not math.isfinite(metric) or not math.isfinite(float(output.metrics["loss"])):
            raise FloatingPointError("non-finite OPCT loss before optimizer step")
        metrics.append(metric)
        # Generic OPCT emits metadata only with its optional text logger. This
        # runner records termination even when that logger is disabled.
        generation_records = metadata or [
            {"question_id": pairs[group_index]["question_id"], "bias": pairs[group_index]["bias"],
             "sample_index": i} for i in range(len(completion_group))
        ]
        for record, (tokens, finish) in zip(generation_records, completion_group, strict=True):
            record.update(generation_finish_reason=finish, generation_output_tokens=len(tokens),
                          generation_output_token_cap=plan.OUTPUT_TOKEN_CAP)
        records.extend(generation_records)
    return metrics, {"variant_reverse_kl": metrics, "rollouts": records}


async def seal_checkpoint(backend, *, run_dir, method, state, plan_hash, window_metrics, exposure_state=None):
    checkpoint = run_dir / "checkpoints" / f"step-{state['step']:06d}"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite an existing checkpoint: {checkpoint}")
    staging = run_dir / ".checkpoint-staging" / uuid.uuid4().hex
    await backend.save_checkpoint(
        name=checkpoint.name, log_dir=staging, kind="both",
        loop_state={"step": state["step"], "convergence": state, "plan_sha256": plan_hash,
                    "final": state["decision"] != "continue", "method": method,
                    **({"exposure": exposure_state} if exposure_state is not None else {})},
    )
    staged_checkpoint = staging / "checkpoints" / checkpoint.name
    plan.immutable_json(staged_checkpoint / "window-metrics.json", window_metrics)
    receipt = {"schema": "ctm-grouped-qid-resume-v1", "method": method,
               "plan_sha256": plan_hash, "convergence": state,
               "checkpoint": str(checkpoint.relative_to(run_dir)),
               "checkpoint_files": checkpoint_identity(staged_checkpoint)}
    if exposure_state is not None:
        receipt["exposure"] = exposure_state
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged_checkpoint, checkpoint)
    shutil.rmtree(staging)
    plan.immutable_json(run_dir / "receipts" / f"step-{state['step']:06d}.json", receipt)
    atomic_json(run_dir / "state.json", receipt)
    return receipt


def prune_recovery_checkpoint(run_dir: Path, *, previous_step: int, current_step: int, plan_hash: str, method: str) -> None:
    """Rotate only our superseded off-window recovery snapshot; keep receipts."""
    if not previous_step or previous_step % plan.WINDOW == 0 or previous_step >= current_step:
        return
    current = json.loads((run_dir / "state.json").read_text())
    if current["convergence"]["step"] != current_step:
        raise ValueError("cannot retire a checkpoint before its successor is sealed")
    receipt = json.loads((run_dir / "receipts" / f"step-{previous_step:06d}.json").read_text())
    if receipt["plan_sha256"] != plan_hash or receipt["method"] != method:
        raise ValueError("refusing to retire a foreign recovery checkpoint")
    path = run_dir / "checkpoints" / f"step-{previous_step:06d}"
    if receipt["checkpoint"] != str(path.relative_to(run_dir)) or path.is_symlink():
        raise ValueError("unexpected recovery checkpoint path")
    if checkpoint_identity(path) != receipt["checkpoint_files"]:
        raise ValueError("recovery checkpoint changed before rotation")
    shutil.rmtree(path)


async def train(args) -> dict:
    import torch
    from ctm.backends.local.engine import LocalBackend
    from ctm.backends.renderers import get_renderer_and_tokenizer
    from ctm.core.config import AdamConfig, LoRAConfig

    repository = args.repository.resolve()
    document = plan.verify(repository, args.plan)
    amendment = document.get("execution_amendment") if args.method in {"bct", "opct"} else None
    plan_hash = plan.sha256(args.plan)
    one_bias = "one_bias_manifest" in document
    manifest = exposure.manifest_for(plan.ordered_pool(repository)) if one_bias else None
    if one_bias and amendment is not None:
        raise ValueError("a fresh one-bias campaign cannot carry the two-bias execution amendment")
    if args.method in {"act", "attct", "mlpct"}:
        require_alignment_audit(args.alignment_audit, manifest)
    if args.model_snapshot.name != plan.REVISION or not (args.model_snapshot / "config.json").is_file():
        raise ValueError("model must be the pinned offline Qwen3.5-9B snapshot")
    require_cuda()
    if args.updates_this_job < 16 or args.updates_this_job % 16:
        raise ValueError("job slice must be a positive multiple of 16 updates")
    run_dir = args.run_root.resolve() / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    # Protect one method's optimizer namespace across independently submitted
    # jobs. Never attach two trainers to the same checkpoint chain.
    with (run_dir / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, resume = load_resume(run_dir, plan_hash, args.method,
                                    parent_plan_hash=plan.PARENT_PLAN_SHA if amendment else None)
        if state["decision"] != "continue":
            return state
        if one_bias:
            # Before loading any model: a fresh first window, or only a window
            # opened by accepted 256-encounter validation; never past a boundary.
            from experiments.rmct_restart_20260928.qwen_one_bias_validation import gated_budget
            one_bias_budget = min(gated_budget(
                run_dir=run_dir, step=state["step"], requested=args.updates_this_job,
                contract_path=args.selection_contract, folder=args.selection_folder,
                manifest=args.validation_manifest, model_path=args.model_snapshot,
            ), (state["step"] // 64 + 1) * 64 - state["step"])
            if one_bias_budget <= 0:
                return state
        pool = plan.ordered_pool(repository)
        by_id = {row["question_id"]: row for row in pool}
        limit = exposure.max_updates(manifest) if one_bias else plan.MAX_UPDATES
        backend = LocalBackend(
            device="cuda", dtype=torch.bfloat16, sampler="hf", gradient_checkpointing=True,
            hf_streaming_sampling=True,
            consistency_loss_options=plan.LOSS_OPTIONS[args.method],
            forward_microbatch_max_datums=1, forward_microbatch_max_tokens=40960, target_logprob_chunk_size=2048,
        )
        # The verified plan's contract is the adapter source of truth (it may
        # carry the one-bias strict_qv ACT scope); two-bias plans are unchanged.
        lora_config = document["contract"]["lora"][args.method] if one_bias else plan.lora(args.method)
        lora_scope = (document["contract"]["act_scope"]["preflight"] if one_bias else "act_qv_fused_qkv") \
            if args.method == "act" else "historical"
        backend.setup(model=str(args.model_snapshot), lora=LoRAConfig(**lora_config),
                      resume_from=resume, resume_with_optimizer=resume is not None)
        renderer, tokenizer = get_renderer_and_tokenizer(str(args.model_snapshot), source=backend.renderer_source)
        adam = AdamConfig(**plan.contract()["optimizer"])
        attempt_dir = run_dir / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:12]}"
        attempt_dir.mkdir(parents=True)
        plan.immutable_json(attempt_dir / "run.json", {
            "plan_sha256": plan_hash, "method": args.method, "resume_from": resume,
            "starting_step": state["step"], "model_snapshot": str(args.model_snapshot),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "alignment_audit_sha256": plan.sha256(args.alignment_audit) if args.method in plan.INTERNAL_METHODS else None,
            "runtime_versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "peft", "tinker", "tinker-cookbook", "mcq-bias")},
            "contract": document["contract"] if one_bias else plan.contract(),
            "one_bias_manifest": document.get("one_bias_manifest"),
            "execution_amendment": amendment,
            "trainable_names": [n for n, p in backend.model.named_parameters() if p.requires_grad],
        })
        trainer = None
        if args.method == "opct":
            from ctm.training.opct import OPCTConfig, OPCTGenerationConfig, OPCTTrainer
            config = OPCTConfig(
                model=str(args.model_snapshot), lora=LoRAConfig(**plan.lora("opct")), optimizer=adam,
                generation=OPCTGenerationConfig(rollouts_per_prompt=4, max_new_tokens=plan.OUTPUT_TOKEN_CAP,
                                                temperature=0.7),
                batch_size=1, gradient_accumulation_steps=4, shuffle_samples=False,
                kl_coef=2.0, kl_discount_factor=0.9, loss_fn="importance_sampling",
            )
            trainer = OPCTTrainer(config=config, backend=backend)
            # The generic OPCT warm-start helper intentionally rejects local
            # resume. This trajectory instead restores loop+optimizer explicitly
            # and binds its teacher to the same pinned base on every job.
            trainer.renderer, trainer.tokenizer = renderer, tokenizer
            trainer.sampling_client = backend.policy_sampler(name="grouped-opct-policy")
            trainer.reference_policy = backend.base_sampler()
            trainer.setup_done = True
        stop_requested = False

        def request_boundary_stop(_signum, _frame):
            nonlocal stop_requested
            stop_requested = True

        prior_signal = signal.signal(signal.SIGUSR1, request_boundary_stop)
        starting_step = state["step"]
        job_end = min(limit, starting_step + args.updates_this_job)
        if one_bias:
            job_end = min(job_end, starting_step + one_bias_budget)
        window_metrics = resume_window_metrics(run_dir, state)
        last_saved_step = state["step"]
        try:
            for step in range(state["step"], job_end):
                if stop_requested and (amendment or step % plan.WINDOW == 0):
                    break
                random.seed(42 + step)
                torch.manual_seed(42 + step)
                if one_bias:
                    qids, biases = exposure.update_rows(manifest, by_id, step)
                    pairs = exposure.pairs(qids, biases, method=args.method)
                else:
                    qids = plan.update_rows(pool, step)
                    pairs = plan.paired_rows(qids, method=args.method)
                started = time.monotonic()
                if trainer is None:
                    variant_metrics, detail = await supervised_update(
                        args.method, qids, pairs, backend=backend, renderer=renderer, tokenizer=tokenizer,
                        cache_dir=run_dir / "base-targets", plan_hash=plan.PARENT_PLAN_SHA if amendment else plan_hash,
                        preflight_path=attempt_dir / "preflight.json" if step == starting_step else None,
                        lora_scope=lora_scope,
                    )
                else:
                    variant_metrics, detail = await opct_update(trainer, pairs, backend=backend)
                if len(variant_metrics) != 4:
                    raise ValueError("each optimizer update must have four variant losses")
                gradient_report = assert_gradients(backend)
                pending = await backend.submit_optim_step(learning_rate=1e-4, adam=adam)
                await pending.result()
                if trainer is not None:
                    trainer.sampling_client = await backend.refresh_policy_sampler(name=f"opct-step-{step+1}")
                if one_bias:
                    state = exposure.observe(state, step=step + 1, loss=math.fsum(variant_metrics) / 4, limit=limit)
                else:
                    state = plan.observe(state, step=step + 1, loss=math.fsum(variant_metrics) / 4)
                metric = {"step": step + 1, "loss": math.fsum(variant_metrics) / 4,
                          "lr": 1e-4, "epoch": step // plan.QIDS_PER_DATASET,
                          "question_ids": [r["question_id"] for r in qids],
                          "variant_metrics": variant_metrics, "seconds": time.monotonic() - started,
                          "gradient_report": gradient_report}
                if one_bias:
                    metric["biases"] = [pair["bias"] for pair in pairs]
                    metric["encountered_qid_bias_examples"] = exposure.QIDS_PER_UPDATE * (step + 1)
                append_json(attempt_dir / "metrics.jsonl", metric)
                if "rollouts" in detail:
                    append_json(attempt_dir / "rollouts.jsonl", {"step": step + 1, **detail})
                window_metrics.append(metric)
                print(json.dumps({**metric, "decision": state["decision"]}), flush=True)
                if amendment or (step + 1) % plan.WINDOW == 0 or (one_bias and state["decision"] != "continue"):
                    await seal_checkpoint(backend, run_dir=run_dir, method=args.method, state=state,
                                          plan_hash=plan_hash, window_metrics=window_metrics,
                                          exposure_state=exposure.exposure(manifest, step + 1) if one_bias else None)
                    if amendment:
                        prune_recovery_checkpoint(run_dir, previous_step=last_saved_step, current_step=state["step"],
                                                  plan_hash=plan_hash, method=args.method)
                    last_saved_step = state["step"]
                if (step + 1) % plan.WINDOW == 0:
                    window_metrics = []
                if state["decision"] != "continue":
                    break
            return state
        finally:
            signal.signal(signal.SIGUSR1, prior_signal)
            shutdown = getattr(backend, "shutdown", None)
            if callable(shutdown):
                shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=plan.METHODS)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--alignment-audit", type=Path)
    parser.add_argument("--updates-this-job", type=int, default=128)
    parser.add_argument("--selection-contract", type=Path, help="One-bias: shared v2 encounter selection contract")
    parser.add_argument("--selection-folder", type=Path, help="One-bias: accepted validation receipts")
    parser.add_argument("--validation-manifest", type=Path, help="One-bias: frozen 600-prompt validation population")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(train(args)), sort_keys=True))


if __name__ == "__main__":
    main()
