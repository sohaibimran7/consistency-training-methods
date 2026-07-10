"""Local (self-hosted GPU) backend: torch + transformers [+ peft].

    from ctm.backends.local.engine import LocalBackend
    trainer = RLTrainer(config, backend=LocalBackend(dtype=torch.bfloat16))

Isambard and Vast.ai both use THIS backend — they differ only in how the process
is launched (sbatch vs docker); see infra/ once phase 1 launchers land.

Fast sampling via vLLM (LoRA hot-reload) is the planned production rollout path;
HF ``generate`` inside LocalBackend is the correct-but-slow default.
"""
