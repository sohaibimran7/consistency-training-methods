# Linux CPU offline validation environment

`requirements.txt` remains the repository's direct-dependency source. [`cpu.lock`](cpu.lock) is its fully resolved, hash-bearing Linux x86_64 CPython 3.11.12 validation lock; it is not a second direct-dependency manifest. The `mcq-bias` source is the immutable commit already declared in `requirements.txt`. Git sources cannot carry wheel hashes, so `scripts/check_environment.py` also verifies the installed PEP 610 commit identity.

Regenerate the lock with uv 0.8.22 from the repository root. This command uses the network to resolve dependencies and must not be run with `--offline`:

```bash
uv pip compile requirements.txt \
  --output-file environments/cpu.lock \
  --generate-hashes \
  --python-version 3.11.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --torch-backend cpu \
  --emit-index-url \
  --cache-dir /tmp/ctm-uv-cache
```

Create a clean target environment and validate it on Linux x86_64:

```bash
uv venv .venv-offline --python 3.11.12
uv pip sync --python .venv-offline/bin/python --strict --torch-backend cpu environments/cpu.lock
uv pip install --python .venv-offline/bin/python --editable . --no-deps
.venv-offline/bin/python scripts/check_environment.py
.venv-offline/bin/python -m pytest
.venv-offline/bin/python -m pytest tests/test_ctm_architecture.py
```

The lock was generated successfully by uv for the stated target and pins `torch` to a `+cpu` build with no NVIDIA runtime packages. `uv pip sync` verifies the hashes present in the lock by default. Do not add `--require-hashes`: uv rejects the immutable Git requirement in that mode because VCS sources have no distribution hash. The GitHub workflow performs the clean Linux install, installed-environment contract check, default pytest suite (including adapter tests), and the explicit source-boundary tests. It does not test GPU, network, or live Tinker tests: the existing pytest marker configuration excludes those from the default suite. This macOS arm64 workspace can validate the lock contract, but cannot itself prove a Linux x86_64 rebuild; a successful workflow run is the target-platform evidence. An initial dependency installation needs package-index access; `--offline` installation is possible only after the required artifacts are cached.
