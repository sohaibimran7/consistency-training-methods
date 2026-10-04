"""One explicit contract; no legacy experiment-factory defaults are inherited."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
METHODS = ("bct", "act", "attct", "mlpct", "opct")
BIASES = ("wrong_argument", "suggested_answer")
INTERNAL_METHODS = ("act", "attct", "mlpct")
INTERNAL_PAIR_TRANSFORM = "suggested_answer_prefix_internal_v1"
MAX_UPDATES = 4096
QIDS_PER_DATASET = 3840
WINDOW = 16
PATIENCE = 8
OUTPUT_TOKEN_CAP = 20480  # User requested matching RMCT's exact "20k" training cap.
PARENT_PLAN_SHA = "92bace5d10cae3f95f01415dda3f063279fb486f1736dd2cde2d2d4acf68ae31"
EXECUTION_AMENDMENT = {
    "schema": "ctm-online-checkpoint-continuation-v5",
    "parent_plan_sha256": PARENT_PLAN_SHA,
    "methods": ["bct", "opct"],
    "job_slice_updates": 16,
    "checkpoint_every_updates": 1,
    "checkpoint_retention": "all_16_update_boundaries_and_latest_recovery_checkpoint",
    "signal_stop": "after_current_complete_optimizer_update",
    "signal_warning_seconds": 3600,
    "successor": "prequeue_afterany_allow_completed_or_timeout_with_progress_only",
    "bct_cache_identity_plan_sha256": PARENT_PLAN_SHA,
    "scientific_contract": "unchanged_parent_v4",
}
POOL_ROOT = Path("artifacts/act-expanded-shared-8192-20260910/shared-two-bias")
POOL_SHA = "3084f27837a16f5175f1a15066c3e0fda7dd22b4aab16f2dd8d0381c45cee8f7"
MANIFEST_SHA = "e50396d8fb2188f5959f9ced378a8f813922b6015f1a1a73887cfbccc52b43d1"
POOL = POOL_ROOT / f"shared-qid-two-bias-n7680-{POOL_SHA}.jsonl"
MANIFEST = POOL_ROOT / f"shared-qid-two-bias-n7680-{POOL_SHA}.manifest-{MANIFEST_SHA}.json"
LOSS_OPTIONS = {
    "bct": {},
    "act": {"layer_selection": "all", "normalize": False},
    "attct": {"layer_selection": "all", "layer_weights": "uniform"},
    "mlpct": {
        "variant": "hidden", "layer_selection": "all", "layer_weights": "uniform",
        "distance_metric": "cosine", "normalize": False,
    },
    "opct": {},
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path: Path, value: Any) -> None:
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite different artifact: {path}")


def lora(method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    shared = {"rank": 8, "alpha": 16, "dropout": 0.0, "seed": 42, "train_unembed": False}
    if method in {"bct", "opct"}:
        return {**shared, "train_mlp": True, "train_attn": True, "target_modules": None}
    targets = ["q_proj", "v_proj"]
    if method == "act":
        targets.append("in_proj_qkv")
    return {**shared, "train_mlp": False, "train_attn": False, "target_modules": targets}


def contract() -> dict:
    return {
        "schema": "ctm-methods-two-bias-convergence-v4-capped",
        "model": {"repo_id": MODEL, "revision": REVISION},
        "methods": list(METHODS),
        "data": {"path": str(POOL), "sha256": POOL_SHA, "manifest": str(MANIFEST),
                 "manifest_sha256": MANIFEST_SHA, "qids_per_dataset": QIDS_PER_DATASET,
                 "biases": list(BIASES), "alpaca": False,
                 "internal_pair_transform": INTERNAL_PAIR_TRANSFORM,
                 "transform_methods": list(INTERNAL_METHODS),
                 "transform_scope": "suggested_answer_training_only_verbatim_cue_then_complete_clean_question",
                 "bct_opct_and_evaluation_prompts": "unchanged_native_shared_pool",
                 "user_approval": "happy with prefixing for act attct and mlpct"},
        "batch": {"qids_per_update": 2, "biases_per_qid": 2, "paired_rows_per_update": 4,
                  "physical_rows_per_backward": 1, "gradient_accumulations_per_update": 4,
                  "ordering": "frozen_rmct_interleaved_logiqa_hellaswag", "shuffle": False,
                  "updates_before_question_repeat": QIDS_PER_DATASET,
                  "repeat_after_pool_exhaustion": True},
        "optimizer": {"learning_rate": 1e-4, "lr_schedule": "constant", "beta1": 0.9,
                      "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 1.0},
        "lora": {method: lora(method) for method in METHODS},
        "loss_options": LOSS_OPTIONS,
        "act_scope": {"preflight": "act_qv_fused_qkv", "paper_exact": False,
                      "difference": "DeltaNet in_proj_qkv is fused; K is adapted along with Q/V. MLP stays frozen."},
        "opct": {"rollouts_per_prompt": 4, "temperature": 0.7, "kl_coef": 2.0,
                 "kl_discount_factor": 0.9, "loss_fn": "importance_sampling",
                 "teacher": "pinned_base_on_clean_prompt_including_after_resume"},
        "bct": {"targets_per_qid": 1, "temperature": 1.0, "teacher": "pinned_base_clean_prompt",
                "target_representation": "exact_sampled_token_ids_including_reasoning_and_eos_if_emitted",
                "reuse_identical_target_for_both_biases": True},
        "generation": {"output_token_cap": OUTPUT_TOKEN_CAP, "termination": "model_eos_or_length",
                       "sampler": "hf", "scope": "bct_targets_and_opct_training_rollouts",
                       "counting": "generated_tokens_including_reasoning_excluding_prompt",
                       "length_stop_policy": "retain_exact_tokens_no_synthetic_eos_no_resampling",
                       "user_approval": "match rmct's 20k token cap"},
        "convergence": {"max_optimizer_steps": MAX_UPDATES, "window_updates": WINDOW,
                        "patience_windows": PATIENCE, "improvement": "strict_global_best_decrease",
                        "absolute_threshold": None, "minimum_windows": PATIENCE + 1,
                        "metric": {"bct": "mean_target_token_nll", "act": "activation_mse",
                                   "attct": "attention_jsd", "mlpct": "mlp_cosine_distance",
                                   "opct": "sampled_token_reverse_kl"},
                        "reduction": "mean_of_four_variant_metrics_per_update_then_mean_of_16_updates",
                        "data_split": "training_stream_not_heldout", "save_every_updates": WINDOW},
        "execution": {"training_gpus_per_method": 1, "dtype": "bfloat16",
                      "hf_language_model_only": False,
                      "hf_streaming_sampling": True,
                      "gradient_checkpointing": "all_layers", "forward_microbatch_max_datums": 1,
                      "forward_microbatch_max_tokens": 40960,
                      "target_logprob_chunk_size": 2048,
                      "per_update_rng_seed": "42 + zero_based_optimizer_step",
                      "job_slice_updates": 128},
        "evaluation": {"reference": "RMCT step176 exact frozen evaluation populations",
                       "checkpoints": "terminal_convergence_checkpoint",
                       "standard": ["towards_bias_switch_rate", "bias_verbalisation"],
                       "cross_task": ["AITA-NTA-FLIP"], "gpus": 16,
                       "output_token_cap": None, "aita_parser": "final_output_only",
                       "statistics": "standard_pipeline_paired_significance",
                       "execution_status": "requires_checkpoint_then_existing_evaluation_pipeline"},
    }


ACT_LORA_SCOPES = ("fused_qkv", "strict_qv")


def one_bias_contract(manifest: dict, act_lora_scope: str = "fused_qkv") -> dict:
    """Fresh one-bias campaign: same recipe, new exposure/stopping contract.

    ``act_lora_scope='strict_qv'`` (user decision 2026-10-04) matches AttCT/MLPCT:
    LoRA on q_proj/v_proj of the full-attention layers only. The ACT residual-state
    loss still spans every layer; only the adapted modules change.
    """
    from . import one_bias

    if act_lora_scope not in ACT_LORA_SCOPES:
        raise ValueError(f"unknown ACT LoRA scope: {act_lora_scope}")
    result = copy.deepcopy(contract())
    if act_lora_scope == "strict_qv":
        result["lora"]["act"]["target_modules"] = ["q_proj", "v_proj"]
        result["act_scope"] = {
            "preflight": "strict_qv", "lora_scope": "strict_qv",
            "difference": "Q/V LoRA on the full-attention layers only, as AttCT/MLPCT; DeltaNet "
                          "linear attention and MLP frozen. ACT residual loss still over all layers.",
            "user_approval": "2026-10-04: simplify Qwen ACT to the AttCT/MLPCT adapter scope; fresh restart"}
    result["schema"] = "ctm-methods-one-bias-v5"
    result["exposure"] = one_bias.contract_block(manifest)
    result["data"]["biases"] = list(BIASES)
    result["data"]["biases_per_qid"] = one_bias.BIASES_PER_QID
    result["batch"] = {"qids_per_update": one_bias.QIDS_PER_UPDATE, "biases_per_qid": one_bias.BIASES_PER_QID,
                       "paired_rows_per_update": one_bias.QIDS_PER_UPDATE * one_bias.BIASES_PER_QID,
                       "physical_rows_per_backward": 1,
                       "gradient_accumulations_per_update": one_bias.QIDS_PER_UPDATE,
                       "ordering": "frozen_rmct_interleaved_logiqa_hellaswag", "shuffle": False,
                       "repeat_after_pool_exhaustion": False}
    result["bct"].update(reuse_identical_target_for_both_biases=False, supervised_bias="assigned_bias_only")
    result["convergence"] = {"owner": "shared_validation_controller", "metric": "TBSR",
                             "every_encountered_qid_bias_examples": one_bias.VALIDATION_INTERVAL_ENCOUNTERS,
                             "counts_no_update_batches": True, "patience": 2, "min_delta": 0,
                             "strict_decrease": True, "loss_selects_or_stops": False,
                             "selection_contract": "ctm-tbsr-selection-contract-v2-encounters",
                             "max_optimizer_steps": one_bias.max_updates(manifest),
                             "save_every_updates": WINDOW, "exhaustion": "stop_and_report_never_cycle"}
    result["execution"]["job_slice_updates"] = 64
    result["fresh_training_required"] = True
    return result


def ordered_pool(repository: Path) -> list[dict]:
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting

    setting = SharedQidTwoBiasSetting(
        data_path=str(repository / POOL), manifest_path=str(repository / MANIFEST),
        expected_manifest_sha256=MANIFEST_SHA, expected_qids_per_dataset=QIDS_PER_DATASET,
        expected_qids_per_dataset_per_segment=16,
    )
    rows = [row for segment in range(240) for row in setting.load_datapoints(segment_index=segment)]
    if len(rows) != 7680 or len({row["question_id"] for row in rows}) != 7680:
        raise ValueError("frozen first-pass pool must contain 7680 unique QIDs")
    for start in range(0, len(rows), 2):
        if [row["source_dataset"] for row in rows[start:start + 2]] != ["logiqa", "hellaswag"]:
            raise ValueError("frozen RMCT order is not dataset-balanced per optimizer update")
    return rows


def update_rows(pool: list[dict], step: int) -> list[dict]:
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < MAX_UPDATES:
        raise ValueError("step must be a zero-based optimizer position below 4096")
    if len(pool) != 7680:
        raise ValueError("unexpected shared pool size")
    offset = 2 * (step % QIDS_PER_DATASET)
    return pool[offset:offset + 2]


def paired_rows(qids: list[dict], *, method: str | None = None) -> list[dict]:
    """Derive a method's training view without mutating the shared artifact.

    Only the explicitly approved internal-method view moves the verbatim
    suggested-answer cue. Native BCT/OPCT and evaluation inputs stay identical.
    """
    if method is not None and method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if len(qids) != 2 or qids[0]["question_id"] == qids[1]["question_id"]:
        raise ValueError("an update requires two distinct QIDs")
    return [training_pair(row, bias, method) for row in qids for bias in BIASES]


def training_pair(row: dict, bias: str, method: str | None) -> dict:
    reference = copy.deepcopy(row["clean_messages"])
    variant = copy.deepcopy(row["variants"][bias]["messages"])
    if method in INTERNAL_METHODS and bias == "suggested_answer":
        cue = row["variants"][bias].get("biasing_text")
        if not isinstance(cue, str) or not cue.strip():
            raise ValueError("suggested-answer prefix requires the frozen verbatim cue")
        if not any(cue in message["content"] for message in variant):
            raise ValueError("frozen suggested-answer cue is absent from the original prompt")
        variant = copy.deepcopy(reference)
        user_indices = [i for i, message in enumerate(variant) if message["role"] == "user"]
        if not user_indices:
            raise ValueError("suggested-answer prefix requires a clean user message")
        last_user = user_indices[-1]
        variant[last_user]["content"] = cue + "\n\n" + reference[last_user]["content"]
    return {"question_id": row["question_id"], "source_dataset": row["source_dataset"],
            "bias": bias, "reference_messages": reference, "variant_messages": variant}

def observe(state: dict, *, step: int, loss: float) -> dict:
    """Pure, replayable patience controller; called only after real updates."""
    if step != state.get("step", 0) + 1 or not math.isfinite(loss):
        raise ValueError("non-contiguous optimizer step or non-finite convergence loss")
    pending = [*state.get("pending", []), loss]
    result = {**state, "step": step, "pending": pending, "decision": "continue"}
    if len(pending) == WINDOW:
        mean = math.fsum(pending) / WINDOW
        best = state.get("best")
        improved = best is None or mean < best
        streak = 0 if improved else state.get("nonimproving", 0) + 1
        result.update(pending=[], window_mean=mean, nonimproving=streak,
                      best=mean if improved else best,
                      best_step=step if improved else state["best_step"])
        if streak >= PATIENCE:
            result["decision"] = "plateau"
    if step >= MAX_UPDATES:
        result["decision"] = "max_updates"
    return result


def source_files(repository: Path) -> list[Path]:
    # Include dirty and untracked implementation, not only git HEAD. No
    # credentials, logs, caches, or model/data payloads enter this source set.
    result = []
    for directory in ("ctm", "ctm_data", "experiments/methods_two_bias_convergence"):
        result.extend((repository / directory).rglob("*.py"))
    result.append(repository / "infra/isambard/run_methods_two_bias_convergence.sbatch")
    return sorted(result)


def verify_amendment(repository: Path, document: dict) -> None:
    if "execution_amendment" not in document:
        return
    if document["execution_amendment"] != EXECUTION_AMENDMENT:
        raise ValueError("unrecognized execution amendment")
    parent_path = repository / "parent-plan.json"
    if parent_path.is_symlink() or sha256(parent_path) != PARENT_PLAN_SHA:
        raise ValueError("execution amendment requires the exact approved parent plan")
    parent = json.loads(parent_path.read_text())
    for key in ("contract", "qid_order_sha256"):
        if parent[key] != document[key]:
            raise ValueError(f"execution-only amendment changed {key}")
    changed = {"experiments/methods_two_bias_convergence/plan.py",
               "experiments/methods_two_bias_convergence/train.py",
               "infra/isambard/run_methods_two_bias_convergence.sbatch"}
    added = {"experiments/methods_two_bias_convergence/continuation.py"}
    if set(document["sources"]) != set(parent["sources"]) | added:
        raise ValueError("unexpected source additions/removals in execution amendment")
    for relative, expected in parent["sources"].items():
        if relative not in changed and document["sources"].get(relative) != expected:
            raise ValueError(f"unrelated parent source changed: {relative}")


def prepare(repository: Path, output: Path, *, parent_plan: Path | None = None, one_bias: bool = False,
            act_lora_scope: str = "fused_qkv") -> dict:
    rows = ordered_pool(repository)
    document = {
        "contract": contract(),
        "qid_order_sha256": hashlib.sha256(canonical([r["question_id"] for r in rows])).hexdigest(),
        "sources": {str(p.relative_to(repository)): sha256(p) for p in source_files(repository)},
    }
    if one_bias:
        from . import one_bias as exposure

        if parent_plan is not None:
            raise ValueError("a fresh one-bias campaign has no parent plan")
        path, manifest = exposure.freeze(output.parent, rows)
        document["contract"] = one_bias_contract(manifest, act_lora_scope)
        document["one_bias_manifest"] = {"path": str(path), "sha256": sha256(path)}
        if act_lora_scope != "fused_qkv":
            document["act_lora_scope"] = act_lora_scope
    if parent_plan is not None:
        if sha256(parent_plan) != PARENT_PLAN_SHA:
            raise ValueError("wrong parent plan")
        immutable_json(repository / "parent-plan.json", json.loads(parent_plan.read_text()))
        document["execution_amendment"] = EXECUTION_AMENDMENT
        verify_amendment(repository, document)
    immutable_json(output, document)
    return document


def verify(repository: Path, plan: Path) -> dict:
    document = json.loads(plan.read_text())
    rows = ordered_pool(repository)
    if "one_bias_manifest" in document:
        from . import one_bias as exposure
        from ctm_data.adapters.mcq_bias import shared_qid_one_bias

        manifest = exposure.manifest_for(rows)
        frozen = Path(document["one_bias_manifest"]["path"])
        if frozen.is_symlink() or sha256(frozen) != document["one_bias_manifest"]["sha256"]:
            raise ValueError("frozen one-bias manifest bytes changed")
        if shared_qid_one_bias.load_manifest(frozen, expected_sha256=shared_qid_one_bias.manifest_identity(manifest)) != manifest:
            raise ValueError("frozen one-bias manifest differs from the deterministic assignment")
        expected = one_bias_contract(manifest, document.get("act_lora_scope", "fused_qkv"))
    else:
        expected = contract()
    if document["contract"] != expected:
        raise ValueError("saved plan differs from the authored scientific contract")
    verify_amendment(repository, document)
    for relative, expected in document["sources"].items():
        path = repository / relative
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"source changed since plan was frozen: {relative}")
    if hashlib.sha256(canonical([r["question_id"] for r in rows])).hexdigest() != document["qid_order_sha256"]:
        raise ValueError("shared QID order differs from the frozen plan")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--one-bias", action="store_true", help="Freeze a fresh one-QID-once, one-bias campaign")
    parser.add_argument("--act-lora-scope", choices=ACT_LORA_SCOPES, default="fused_qkv",
                        help="One-bias ACT adapter scope (strict_qv = AttCT/MLPCT Q/V scope)")
    args = parser.parse_args()
    kwargs = ({"parent_plan": args.parent_plan, "one_bias": args.one_bias, "act_lora_scope": args.act_lora_scope}
              if args.action == "prepare" else {})
    result = globals()[args.action](args.repository.resolve(), args.plan.resolve(), **kwargs)
    print(json.dumps({"plan": str(args.plan), "methods": result["contract"]["methods"],
                      "qid_order_sha256": result["qid_order_sha256"], "verified": True}))


if __name__ == "__main__":
    main()
