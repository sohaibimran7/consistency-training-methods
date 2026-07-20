"""Local (self-hosted GPU) backend: torch + transformers [+ peft].

    from ctm.backends.local.engine import LocalBackend
    trainer = RLTrainer(config, backend=LocalBackend(dtype=torch.bfloat16))

Isambard and Vast.ai both use THIS backend — they differ only in how the process
is launched (SLURM versus a provisioned container); see ``infra/``.

Fast sampling uses vLLM with LoRA hot-reload; HF ``generate`` remains available
for diagnostic runs.
"""
