"""
RL Consistency Training CLI.

Launch RLCT against any explicitly imported training Setting. Setting-specific choices
belong in ``--setting-config`` and ``--load-config``; the usual entry point is
``scripts/run_experiment.py CONFIG.yaml``.
"""

import argparse
import asyncio
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from ctm.core.config import CheckpointConfig, AdamConfig, resolve_lora_config
from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.cli_safety import parse_json_object, reject_inline_secrets
from ctm.settings.runtime import prepare_setting, setting_run_metadata


def _exact_command(argv: list[str]) -> str:
    return "python scripts/train_rlct.py " + " ".join(shlex.quote(value) for value in argv)


def _validate_numeric_args(args: argparse.Namespace) -> None:
    positive = {
        "--batch-size": args.batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--refresh-every": args.refresh_every,
        "--n-epochs": args.n_epochs,
        "--checkpoint-every": args.checkpoint_every,
        "--lora-rank": args.lora_rank,
        "--n-ref-rollouts": args.n_ref_rollouts,
        "--n-train-rollouts": args.n_train_rollouts,
        "--max-new-tokens": args.max_new_tokens,
        "--max-resample-attempts": args.max_resample_attempts,
    }
    invalid = [f"{flag}={value}" for flag, value in positive.items() if value is not None and value <= 0]
    if invalid:
        raise ValueError("these values must be positive: " + ", ".join(invalid))
    for flag, value in (
        ("--n-consistency-rollouts", args.n_consistency_rollouts),
        ("--n-anchor-rollouts", args.n_anchor_rollouts),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{flag} must be non-negative")
    if args.n_consistency_rollouts is not None and args.n_consistency_rollouts > args.n_train_rollouts:
        raise ValueError("--n-consistency-rollouts cannot exceed --n-train-rollouts")
    if args.n_anchor_rollouts is not None and args.n_anchor_rollouts > args.n_ref_rollouts:
        raise ValueError("--n-anchor-rollouts cannot exceed --n-ref-rollouts")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.kl_coef < 0:
        raise ValueError("--kl-coef must be non-negative")
    if not 0 <= args.anchor_weight <= 1:
        raise ValueError("--anchor-weight must be within [0, 1]")
    if args.snr_z < 0:
        raise ValueError("--snr-z must be non-negative")
    n_consistency = args.n_consistency_rollouts if args.n_consistency_rollouts is not None else args.n_train_rollouts
    n_anchor = args.n_anchor_rollouts if args.n_anchor_rollouts is not None else args.n_ref_rollouts
    consistency_active = args.anchor_weight < 1 and n_consistency > 0
    anchor_active = args.anchor_weight > 0 and n_anchor > 0
    if not consistency_active and not anchor_active:
        raise ValueError(
            "rollout/weight configuration has no active gradient term; increase the rollout count for a "
            "non-zero-weight consistency or anchor term"
        )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="RL Consistency Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Model & data ===
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model name")
    parser.add_argument(
        "--setting-factory",
        required=True,
        help="Training adapter factory in module:callable form",
    )
    parser.add_argument(
        "--setting-config",
        help="Inline JSON object or JSON-file path passed to the setting constructor",
    )
    parser.add_argument(
        "--load-config",
        help="Inline JSON object or JSON-file path passed to setting.load_datapoints; "
        "--n-datapoints is added unless this object already sets it",
    )
    parser.add_argument(
        "--n-datapoints",
        type=int,
        default=100,
        help="Maximum training datapoints to load",
    )

    # === Naming ===
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--run-name", required=True, help="Run name (used in checkpoint path)")
    parser.add_argument("--wandb-project", help="Explicitly enable W&B logging to this project")

    # === Optimiser ===
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (default: auto from Tinker's get_recommended_lr)"
    )
    parser.add_argument(
        "--lr-schedule",
        default="linear",
        choices=["constant", "linear", "cosine"],
        help="LR schedule (shared SFT+RL default: linear). RL now honors this per optim step.",
    )
    parser.add_argument(
        "--lora-config",
        help="JSON object or JSON file with rank, alpha, dropout, target_modules, portable component flags, and seed",
    )
    parser.add_argument("--lora-rank", type=int, default=None, help="Override lora_config.rank (effective default: 8)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for LoRA init, epoch shuffle, and gradient-rollout "
        "subsampling (stochastic temperature sampling is NOT seeded)",
    )
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.5,
        help="Anchor weight (alpha): 0=pure consistency, 1=pure anchor, 0.5=equal",
    )
    parser.add_argument(
        "--anchor-model",
        default="base",
        choices=["base", "initial_policy"],
        help="Model for anchor reference rate: 'base' (frozen base) or 'initial_policy' (policy at init, incl. resumed ckpt)",
    )
    parser.add_argument("--loss-fn", default="ppo", choices=["ppo", "importance_sampling"])
    parser.add_argument(
        "--advantage-estimator",
        default="grpo_normalized",
        choices=["grpo_normalized", "snr_scaling", "matched_pair"],
        help="Advantage construction: 'grpo_normalized' is the paper-era default "
        "(std-normalize; drops gap magnitude, keeps only its sign). The explicit post-paper extensions are "
        "'snr_scaling' (GRPO gated toward 0 by sampling SNR) and "
        "'matched_pair' (pool the rate across a setting's prompt family into one gap vs the reference; "
        "use a small --n-train-rollouts for multi-variant families)",
    )
    parser.add_argument(
        "--snr-mode",
        default="soft",
        choices=["soft", "hard"],
        help="SNR-scaling shape (advantage-estimator=snr_scaling only): 'soft' smooth taper, 'hard' significance gate",
    )
    parser.add_argument(
        "--snr-z",
        type=float,
        default=2.0,
        help="SNR scale in SEs: half-weight (soft) / cutoff (hard) at |gap| = z·SE; z=0 = no floor (full faithful gap)",
    )
    parser.add_argument(
        "--snr-normalizer",
        default="trait_std",
        choices=["trait_std", "none"],
        help="SNR-scaling advantage scaling: 'trait_std' (divide by sqrt(p(1-p)+floor)) or 'none' (bare A=-gap*(T-p))",
    )
    parser.add_argument(
        "--unparsed-handling",
        default="discard",
        choices=["discard", "resample"],
        help="Unparsed/hedged rollouts: 'discard' (drop from rate denominator + gradient, "
        "default) or 'resample' (re-sample until a usable answer, up to "
        "--max-resample-attempts; logs resample amplification + give-up so hedging stays visible)",
    )
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=4,
        help="Max resample rounds per slot when --unparsed-handling=resample",
    )

    # === Sampling ===
    parser.add_argument("--n-ref-rollouts", type=int, default=128, help="Rollouts for reference rate estimation")
    parser.add_argument("--n-train-rollouts", type=int, default=128, help="Rollouts for training rate estimation")
    parser.add_argument(
        "--n-consistency-rollouts",
        type=int,
        default=None,
        help="Consistency gradient rollouts (default: same as --n-train-rollouts)",
    )
    parser.add_argument(
        "--n-anchor-rollouts",
        type=int,
        default=None,
        help="Anchor gradient rollouts (default: all parsed ref rollouts)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)

    # === Training loop ===
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Datapoints per gradient step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--refresh-every", type=int, default=1, help="Refresh policy every N steps")
    parser.add_argument(
        "--normalization",
        default="per_item",
        choices=["pooled", "per_item"],
        help="Advantage normalization scope: within each datapoint (default) or across the whole batch",
    )

    # === Checkpointing ===
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--save-state", action="store_true", help="Save full optimizer state (for resuming)")

    # === Backend ===
    add_backend_args(parser)

    # === Run modes ===
    parser.add_argument("--resume-from", default=None, help="Tinker checkpoint path to resume from")
    parser.add_argument(
        "--resume-with-optimizer",
        action="store_true",
        help="Also restore optimizer state when resuming (for exact continuation)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config, don't train")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args(argv)

    try:
        if args.n_datapoints <= 0:
            raise ValueError("--n-datapoints must be positive")
        _validate_numeric_args(args)
        setting_config = parse_json_object(args.setting_config, label="--setting-config")
        load_config = parse_json_object(args.load_config, label="--load-config")
        raw_lora_config = parse_json_object(args.lora_config, label="--lora-config")
        reject_inline_secrets(setting_config, path="setting_config")
        reject_inline_secrets(load_config, path="load_config")
        reject_inline_secrets(raw_lora_config, path="lora_config")
        lora_config = resolve_lora_config(raw_lora_config, rank=args.lora_rank, seed=args.seed)
        load_config.setdefault("n_datapoints", args.n_datapoints)
        prepared = prepare_setting(
            args.setting_factory,
            setting_config=setting_config,
            load_config=load_config,
        )
    except (KeyError, NotImplementedError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    datapoints = prepared.datapoints
    perturbation_fns = prepared.perturbations
    train_indices = prepared.training_indices
    setting = prepared.setting
    # `is not None` (not `or`) so an explicit --n-consistency-rollouts 0 isn't silently
    # overridden to n_train_rollouts.
    n_consistency = args.n_consistency_rollouts if args.n_consistency_rollouts is not None else args.n_train_rollouts

    n_steps = (len(datapoints) + args.batch_size - 1) // args.batch_size
    total_steps = n_steps * args.n_epochs
    pert_desc = f"reference + {len(train_indices)} training variant(s)"

    config = RLConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        model=args.model,
        lora=lora_config,
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
            normalize=args.normalization,
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
            **setting_run_metadata(
                setting,
                setting_config=setting_config,
                load_config=load_config,
            ),
            "setting_factory": args.setting_factory,
            "backend": args.backend,
        },
    )

    print("\nExact training command:")
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    print(f"  {_exact_command(effective_argv)}")
    print(f"\n{'='*60}")
    print("RL Training Configuration")
    print(f"{'='*60}")
    print(f"  Setting:            {setting.name}")
    print(f"  Model:              {args.model}")
    print(f"  Backend:            {describe_backend(args)}")
    print(f"  Experiment:         {args.experiment_name}/{args.run_name}")
    print(f"  Total datapoints:   {len(datapoints)}")
    print(f"  Perturbations:      {pert_desc}")
    print(f"  LR:                 {args.lr} ({args.lr_schedule})")
    print(f"  LoRA rank/alpha:    {config.lora.rank}/{config.lora.resolved_alpha}")
    print(f"  LoRA dropout:       {config.lora.dropout}")
    if config.lora.target_modules is not None:
        print(f"  LoRA targets:       {config.lora.target_modules}")
    else:
        print(
            "  LoRA components:    "
            f"mlp={config.lora.train_mlp}, attn={config.lora.train_attn}, unembed={config.lora.train_unembed}"
        )
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
    print(f"  Normalization:      {args.normalization}")
    # Surface the per-datapoint sampling cost: every setting variant samples
    # n_train_rollouts, so fixed-K prompt families can multiply cost quickly.
    n_train_perts = len(train_indices)
    eff_rollouts = args.n_ref_rollouts + n_train_perts * args.n_train_rollouts
    print(
        f"  Rollouts/datapoint: {eff_rollouts} (= {args.n_ref_rollouts} ref + {n_train_perts}×{args.n_train_rollouts} cued)"
    )
    if n_train_perts > 1 and args.n_train_rollouts > 8:
        print(
            f"  ⚠️  WARNING: {n_train_perts} variants × {args.n_train_rollouts} rollouts/variant is a large "
            f"per-datapoint sampling cost (×{eff_rollouts // (args.n_ref_rollouts + args.n_train_rollouts)} "
            f"vs single-variant). matched_pair targets ~1-2 rollouts/variant — consider --n-train-rollouts 2."
        )
    print(f"  KL coef:            {args.kl_coef}")
    print(f"  Anchor weight:      {args.anchor_weight}")
    print(f"  Anchor model:       {args.anchor_model}")
    if args.advantage_estimator == "snr_scaling":
        _adv_desc = f"snr_scaling ({args.snr_mode}, z={args.snr_z}, norm={args.snr_normalizer})"
    elif args.advantage_estimator == "matched_pair":
        _adv_desc = f"matched_pair (pooled gap, z={args.snr_z}, norm={args.snr_normalizer}, {n_train_perts} variant(s))"
    else:
        _adv_desc = args.advantage_estimator
    print(f"  Advantage est.:     {_adv_desc}")
    if n_train_perts > 1 and args.advantage_estimator != "matched_pair":
        print(
            "  NOTE: this setting has multiple variants but the estimator is not matched_pair; "
            "the family will be handled by your chosen estimator instead."
        )
    if args.advantage_estimator == "matched_pair" and n_train_perts == 1:
        print("  NOTE: matched_pair over one variant is equivalent to a single gap vs the reference.")
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
        return

    if not args.yes:
        response = input("\nProceed with training? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    trainer = RLTrainer(
        config=config,
        resume_from=args.resume_from,
        resume_with_optimizer=args.resume_with_optimizer,
        backend=build_backend(args),
    )
    trainer.setup()

    final_checkpoint = asyncio.run(
        trainer.train(
            datapoints=datapoints,
            perturbation_fns=perturbation_fns,
            trait_classifier=prepared.trait_classifier,
            answer_parser=prepared.answer_parser,
        )
    )

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"{'='*60}")
    print(f"CTM_FINAL_CHECKPOINT={final_checkpoint}")


if __name__ == "__main__":
    main()
