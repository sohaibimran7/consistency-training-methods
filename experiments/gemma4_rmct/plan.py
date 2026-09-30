"""Gemma execution adaptation of the frozen Qwen RMCT scientific arguments."""

from pathlib import Path

from experiments.rmct_convergence import plan as qwen

MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
EXPERIMENT = "gemma4-rmct-patience"


def run_name(index):
    if type(index) is not int or index < 0:
        raise ValueError("window index must be a nonnegative integer")
    return f"window-{index:05d}"


def checkpoint(root, index):
    name = run_name(index)
    return Path(root) / "logs" / EXPERIMENT / name / "checkpoints" / f"{EXPERIMENT}_{name}"


def segment_args(root, index, *, model, target_modules):
    name = run_name(index)
    # Use the base scientific recipe, not any Qwen continuation cap/amendment.
    original = qwen.segment_args(root, 0)
    args = {k: v for k, v in original.items() if not k.startswith("local_")}
    args.pop("max_new_tokens")
    args.update(
        model=str(model), experiment_name=EXPERIMENT, run_name=name,
        setting_factory="experiments.gemma4_rmct.setting:create_setting",
        load_config={"n_datapoints": 32, "segment_index": index},
        no_max_new_tokens=True,
        checkpoint_every=1,
        local_dtype="bfloat16", local_device="cuda:0", local_sampler="vllm",
        local_rollout_gpus="1,2,3", local_rollout_seed_base=42,
        local_rollout_gpu_mem_util=0.85, local_vllm_max_num_seqs=32,
        local_vllm_max_num_batched_tokens=8192,
        local_vllm_generation_config="vllm", local_vllm_enforce_eager=False,
        local_gradient_checkpointing=True,
        local_forward_microbatch_max_datums=1, local_forward_microbatch_max_tokens=8192,
        local_target_logprob_chunk_size=256, local_ppo_clip_epsilon=0.2,
    )
    if not target_modules or any(not n.startswith("model.language_model.layers.") for n in target_modules):
        raise ValueError("exact preflight-validated text LoRA targets required")
    args["lora_config"] = {**args["lora_config"], "target_modules": list(target_modules)}
    if index:
        args.update(resume_from=f"file://{checkpoint(root, index-1).resolve()}",
                    resume_with_optimizer=True, resume_state_required=True)
    return args
