"""
RL Consistency Training CLI.

Launch RLCT runs with flexible bias type, dataset, and hyperparameter configuration.
Supports single-bias, multi-bias, and control runs.

Usage:
    # Single bias, 100 total datapoints (50 per dataset)
    python scripts/tinker_training/train_rl.py \\
        --bias-types suggested_answer \\
        --experiment-name rl_test \\
        --run-name llama-rlct-sa-s100

    # Multi-bias, 200 total datapoints (50 per dataset x bias_type combo)
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument,wrong_few_shot \\
        --n-datapoints 200 \\
        --experiment-name rl-da-wfs \\
        --run-name gpt-rlct-da-wfs-s200

    # Control run
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument \\
        --experiment-name rl-distractor-argument \\
        --run-name gpt-rl-control-da-s100 --control

    # Explicit LR (default: auto from Tinker's get_recommended_lr)
    python scripts/tinker_training/train_rl.py \\
        --bias-types distractor_argument \\
        --experiment-name rl-distractor-argument \\
        --run-name gpt-rlct-da-s100 \\
        --lr 1e-4
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from ctm.training.rl import (
    RLConfig,
    RLTrainer,
    RateEstimationConfig,
    TrainingSamplingConfig,
    TrainingLoopConfig,
    GenerationConfig,
)
from ctm.core.config import CheckpointConfig, AdamConfig, LoRAConfig
from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.evals.parsers import fallback_answer_parser
from ctm.settings.sycophancy import (
    attach_wrong_cots,
    load_datapoints,
    make_distractor_cue_perturbations,
    make_perturbation_fns,
    resolve_distractor_cues,
    trait_classifier,
)


def main():
    parser = argparse.ArgumentParser(
        description="RL Consistency Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Model & data ===
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model name")
    parser.add_argument("--bias-types", required=True, help="Comma-separated bias types (e.g. distractor_argument,wrong_few_shot)")
    parser.add_argument("--datasets", default="mmlu,truthfulqa", help="Comma-separated datasets")
    parser.add_argument("--n-datapoints", type=int, default=100, help="Total number of datapoints (split evenly across dataset x bias_type combinations)")
    parser.add_argument("--data-dir", default=None, help="Override default dataset_dumps/test directory")
    parser.add_argument("--prompt-style", choices=["cot", "no_cot"], default="cot", help="Strip CoT instructions for reasoning models (e.g. gpt-oss)")

    # === Naming ===
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--run-name", required=True, help="Run name (used in checkpoint path)")

    # === Optimiser ===
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto from Tinker's get_recommended_lr)")
    parser.add_argument("--lr-schedule", default="linear", choices=["constant", "linear", "cosine"],
                        help="LR schedule (shared SFT+RL default: linear). RL now honors this per optim step.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for LoRA init, epoch shuffle, and gradient-rollout "
                             "subsampling (stochastic temperature sampling is NOT seeded)")
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.5, help="Anchor weight (alpha): 0=pure consistency, 1=pure anchor, 0.5=equal")
    parser.add_argument("--anchor-model", default="base", choices=["base", "initial_policy"], help="Model for anchor reference rate: 'base' (frozen base) or 'initial_policy' (policy at init, incl. resumed ckpt)")
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "importance_sampling"])
    parser.add_argument("--advantage-estimator", default="grpo_normalized", choices=["grpo_normalized", "snr_scaling", "matched_pair"],
                        help="Advantage construction: 'grpo_normalized' (std-normalize; drops gap magnitude, keeps only its sign), "
                             "'snr_scaling' (still GRPO; keep gap magnitude, shrunk toward 0 by its sampling SNR), or "
                             "'matched_pair' (pool the cued rate across the cue family into one gap vs the neutral control; "
                             "use with --distractor-cues and a small --n-train-rollouts)")
    parser.add_argument("--snr-mode", default="soft", choices=["soft", "hard"],
                        help="SNR-scaling shape (advantage-estimator=snr_scaling only): 'soft' smooth taper, 'hard' significance gate")
    parser.add_argument("--snr-z", type=float, default=2.0,
                        help="SNR scale in SEs: half-weight (soft) / cutoff (hard) at |gap| = z·SE; z=0 = no floor (full faithful gap)")
    parser.add_argument("--snr-normalizer", default="trait_std", choices=["trait_std", "none"],
                        help="SNR-scaling advantage scaling: 'trait_std' (divide by sqrt(p(1-p)+floor)) or 'none' (bare A=-gap*(T-p))")
    parser.add_argument("--unparsed-handling", default="discard", choices=["discard", "resample"],
                        help="Unparsed/hedged rollouts: 'discard' (drop from rate denominator + gradient, "
                             "default) or 'resample' (re-sample until a usable answer, up to "
                             "--max-resample-attempts; logs resample amplification + give-up so hedging stays visible)")
    parser.add_argument("--max-resample-attempts", type=int, default=4,
                        help="Max resample rounds per slot when --unparsed-handling=resample")

    # === Sampling ===
    parser.add_argument("--n-ref-rollouts", type=int, default=128, help="Rollouts for reference rate estimation")
    parser.add_argument("--n-train-rollouts", type=int, default=128, help="Rollouts for training rate estimation")
    parser.add_argument("--n-consistency-rollouts", type=int, default=None, help="Consistency gradient rollouts (default: same as --n-train-rollouts)")
    parser.add_argument("--n-anchor-rollouts", type=int, default=None, help="Anchor gradient rollouts (default: all parsed ref rollouts)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)

    # === Training loop ===
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Datapoints per gradient step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1, help="Refresh policy every N steps")

    # === Checkpointing ===
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-state", action="store_true", help="Save full optimizer state (for resuming)")

    # === Distractor cue family (matched-pair / RLOO) ===
    parser.add_argument("--distractor-cues", default="none",
                        help="Cue family for matched-pair training: 'none' (single biased prompt), "
                             "'all'/'train'/'holdout' (registry splits), or a comma-list of cue keys. "
                             "When set, the cued side becomes N re-framings of each item's wrong argument "
                             "(idx 1..N); pair with --advantage-estimator matched_pair and a small --n-train-rollouts.")

    # === Backend ===
    add_backend_args(parser)

    # === Run modes ===
    parser.add_argument("--control", action="store_true", help="Control: use unbiased perturbation for both ref and train")
    parser.add_argument("--resume-from", default=None, help="Tinker checkpoint path to resume from")
    parser.add_argument("--resume-with-optimizer", action="store_true", help="Also restore optimizer state when resuming (for exact continuation)")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config, don't train")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    bias_types = [b.strip() for b in args.bias_types.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    # `is not None` (not `or`) so an explicit --n-consistency-rollouts 0 isn't silently
    # overridden to n_train_rollouts.
    n_consistency = args.n_consistency_rollouts if args.n_consistency_rollouts is not None else args.n_train_rollouts

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "dataset_dumps" / "test"
    n_combos = len(bias_types) * len(datasets)
    per_combo = args.n_datapoints // n_combos if n_combos > 0 else args.n_datapoints
    print(f"\nLoading datapoints: {args.n_datapoints} total across {n_combos} combos ({per_combo} per combo)")

    datapoints = load_datapoints(bias_types, datasets, args.n_datapoints, data_dir)

    if not datapoints:
        print("Error: no datapoints loaded. Check --bias-types and --datasets.")
        sys.exit(1)

    distractor_cues = resolve_distractor_cues(args.distractor_cues)
    if distractor_cues and args.control:
        print("Error: --distractor-cues is incompatible with --control.")
        sys.exit(1)
    if distractor_cues:
        from collections import Counter
        n_before = len(datapoints)
        before_by_bias = Counter(dp.get("bias_name", "?") for dp in datapoints)
        datapoints = attach_wrong_cots(datapoints)
        after_by_bias = Counter(dp.get("bias_name", "?") for dp in datapoints)
        print(f"  Distractor cue family: {len(distractor_cues)} cues; kept "
              f"{len(datapoints)}/{n_before} datapoints with an extractable <argument>")
        # The cue family only supports <argument>-bearing data (distractor_argument).
        # Other bias types (distractor_fact=<fun_fact>, etc.) would be silently dropped,
        # collapsing a mixed --bias-types run to a different composition than requested.
        for bias, n0 in before_by_bias.items():
            n1 = after_by_bias.get(bias, 0)
            if n1 == 0:
                print(f"  ⚠️  WARNING: bias type {bias!r} has NO <argument> blocks (0/{n0} kept) "
                      f"and is being DROPPED ENTIRELY — the cue family only supports "
                      f"distractor_argument. Your trained mix no longer matches --bias-types.")
            elif n1 < n0:
                print(f"     {bias}: kept {n1}/{n0}")
        if not datapoints:
            print("Error: no datapoints have an extractable <argument>. The distractor-cue "
                  "family requires distractor_argument data.")
            sys.exit(1)
        train_indices = list(range(1, len(distractor_cues) + 1))
    else:
        train_indices = [1]

    n_steps = len(datapoints) // args.batch_size
    total_steps = n_steps * args.n_epochs
    if args.control:
        pert_desc = "unbiased (ref) + unbiased (train) [CONTROL]"
    elif distractor_cues:
        pert_desc = f"unbiased (ref) + {len(distractor_cues)} distractor cues (train)"
    else:
        pert_desc = "unbiased (ref) + biased (train)"

    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=args.lora_rank, seed=args.seed),
        optimizer=AdamConfig(
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
        ),
        reference_rate=RateEstimationConfig(
            perturbation_indices=[0],
            n_rollouts=args.n_ref_rollouts,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=train_indices,
            n_rollouts_for_rate=args.n_train_rollouts,
            n_rollouts_for_consistency=n_consistency,
            n_rollouts_for_anchor=args.n_anchor_rollouts,
        ),
        loop=TrainingLoopConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            refresh_policy_every_n_steps=args.refresh_every,
            n_epochs=args.n_epochs,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=args.checkpoint_every,
            save_state=args.save_state,
        ),
        kl_coef=args.kl_coef,
        loss_fn=args.loss_fn,
        anchor_weight=args.anchor_weight,
        anchor_model=args.anchor_model,
        advantage_estimator=args.advantage_estimator,
        snr_mode=args.snr_mode,
        snr_z=args.snr_z,
        snr_normalizer=args.snr_normalizer,
        unparsed_handling=args.unparsed_handling,
        max_resample_attempts=args.max_resample_attempts,
        log_base_dir="logs",
        run_metadata={
            "setting": "sycophancy",
            "bias_types": bias_types,
            "datasets": datasets,
            "prompt_style": args.prompt_style,
            "distractor_cues": distractor_cues,
            "control": args.control,
            "backend": args.backend,
        },
    )

    print(f"\n{'='*60}")
    print(f"RL Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:              {args.model}")
    print(f"  Backend:            {describe_backend(args)}")
    print(f"  Experiment:         {args.experiment_name}/{args.run_name}")
    print(f"  Bias types:         {bias_types}")
    print(f"  Datasets:           {datasets}")
    print(f"  Total datapoints:   {len(datapoints)}")
    print(f"  Perturbations:      {pert_desc}")
    if distractor_cues:
        print(f"  Distractor cues:    {', '.join(distractor_cues)}")
    print(f"  LR:                 {args.lr} ({args.lr_schedule})")
    print(f"  LoRA rank:          {args.lora_rank}")
    if args.seed is not None:
        print(f"  Seed:               {args.seed}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Grad accum steps:   {args.gradient_accumulation_steps}")
    print(f"  N epochs:           {args.n_epochs}")
    print(f"  Estimated steps:    {total_steps}")
    print(f"  Checkpoint every:   {args.checkpoint_every} steps")
    print(f"  n_ref_rollouts:     {args.n_ref_rollouts}")
    print(f"  n_train_rollouts:   {args.n_train_rollouts}")
    print(f"  n_consistency_rollouts: {n_consistency}")
    print(f"  n_anchor_rollouts:  {args.n_anchor_rollouts}")
    # Surface the per-datapoint sampling cost: each training perturbation (cue) samples
    # n_train_rollouts. With the full cue family this multiplies fast.
    n_train_perts = len(train_indices)
    eff_rollouts = args.n_ref_rollouts + n_train_perts * args.n_train_rollouts
    print(f"  Rollouts/datapoint: {eff_rollouts} (= {args.n_ref_rollouts} ref + {n_train_perts}×{args.n_train_rollouts} cued)")
    if distractor_cues and args.n_train_rollouts > 8:
        print(f"  ⚠️  WARNING: {n_train_perts} cues × {args.n_train_rollouts} rollouts/cue is a large "
              f"per-datapoint sampling cost (×{eff_rollouts // (args.n_ref_rollouts + args.n_train_rollouts)} "
              f"vs single-cue). matched_pair targets ~1-2 rollouts/cue — consider --n-train-rollouts 2.")
    print(f"  KL coef:            {args.kl_coef}")
    print(f"  Anchor weight:      {args.anchor_weight}")
    print(f"  Anchor model:       {args.anchor_model}")
    if args.advantage_estimator == "snr_scaling":
        _adv_desc = f"snr_scaling ({args.snr_mode}, z={args.snr_z}, norm={args.snr_normalizer})"
    elif args.advantage_estimator == "matched_pair":
        _adv_desc = f"matched_pair (pooled gap, z={args.snr_z}, norm={args.snr_normalizer}, {len(distractor_cues) or 1} cue(s))"
    else:
        _adv_desc = args.advantage_estimator
    print(f"  Advantage est.:     {_adv_desc}")
    if distractor_cues and args.advantage_estimator != "matched_pair":
        print("  NOTE: --distractor-cues set but estimator is not matched_pair; "
              "the cue family will be pooled by your chosen estimator instead.")
    if args.advantage_estimator == "matched_pair" and not distractor_cues:
        print("  NOTE: matched_pair with no --distractor-cues pools over a single cue "
              "(equivalent to a 1-cue gap vs the reference).")
    print(f"  Loss fn:            {args.loss_fn}")
    if args.resume_from:
        print(f"  Resume from:        {args.resume_from}")
        print(f"  With optimizer:     {args.resume_with_optimizer}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run complete.")
        if datapoints:
            dp = datapoints[0]
            print(f"\nSample datapoint keys: {list(dp.keys())}")
            print(f"  biased_option: {dp.get('biased_option')}")
            print(f"  ground_truth:  {dp.get('ground_truth')}")
            print(f"  bias_name:     {dp.get('bias_name', 'n/a')}")
        return

    if not args.yes:
        response = input("\nProceed with training? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    if distractor_cues:
        perturbation_fns = make_distractor_cue_perturbations(distractor_cues, args.prompt_style)
    else:
        unbiased_perturbation, biased_perturbation = make_perturbation_fns(args.prompt_style)
        if args.control:
            perturbation_fns = [unbiased_perturbation, unbiased_perturbation]
        else:
            perturbation_fns = [unbiased_perturbation, biased_perturbation]

    trainer = RLTrainer(config=config, resume_from=args.resume_from, resume_with_optimizer=args.resume_with_optimizer,
                        backend=build_backend(args))
    trainer.setup()

    final_checkpoint = asyncio.run(
        trainer.train(
            datapoints=datapoints,
            perturbation_fns=perturbation_fns,
            trait_classifier=trait_classifier,
            answer_parser=fallback_answer_parser,
        )
    )

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
