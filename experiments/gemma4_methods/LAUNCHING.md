# Reviewed deployment inputs

These wrappers never choose a historical checkout, runtime, overlay or
checkpoint by default. They do not submit successors.

All wrappers require:

- `GEMMA_DEPLOY_REPO`: absolute clean deployed canonical Git checkout.
- `GEMMA_DEPLOY_COMMIT`: exact incorporated40hex commit; must descend from
  shared selection merge45f27c24855c82f3dc81019bd246a6f7641a13ec.
- `GEMMA_RUNTIME_PYTHON`: explicit absolute pinned interpreter.
- `GEMMA_RUNTIME_MANIFEST`, `GEMMA_RUNTIME_MANIFEST_SHA256`: reviewed runtime
  pins and their exact bytes. The manifest is evidence, not launch clearance.

Runtime manifest schema `gemma-reviewed-runtime-v1` contains `python`,
`python_version` (exact `sys.version`), `profile`, and `packages`. Each package
pin contains `version` and `module_file` (exact resolved module origin).
Generation profile requires exactly torch, transformers, vllm, inspect_ai;
grading profile requires exactly inspect_ai and openai. Generation additionally
requires `GEMMA_RUNTIME_ENV`, a tracked repository-relative runtime setup script
which must preserve the chosen interpreter. No inherited/client PYTHONPATH
overlay is used. This is not an attestation of all dependency source bytes;
the native runtime gate still verifies its complete required provenance.

Training requires `GEMMA_METHODS_ROOT`, `GEMMA_METHOD`, `GEMMA_GPUS`,
`GEMMA_MODEL_SNAPSHOT`, `GEMMA_SELECTION_CONTRACT`, `GEMMA_SELECTION_FOLDER`,
`GEMMA_VERIFIER_FACTORY`. Method recipes remain four GPUs for BCT/OPCT and one
for ACT/AttCT/MLPCT. Each job asks for16actual optimizer updates, bounded by
shared selection. The native hook factory must be incorporated and verified;
this document supplies no stub that grants progress.

Evaluation requires `GEMMA_EVAL_ROOT`, an already prepared terminal-selection
manifest, and `GEMMA_EVAL_WORKERS` matching that immutable manifest. Preparation
supports4/8/16/32workers with exactly-once strided frozen-QID shards. Submission
must override the default array with `--array=0-(workers-1)` for larger counts;
the wrapper fails closed on a count/range/step mismatch. Every task still uses
one GPU. Counts are benchmark candidates, not measured scaling gains. Grading requires this
same root and explicit `GEMMA_GRADING_ENV_FILE`; regular credential-file mode
400/600 checks remain. Use the reviewed CPU grading runtime, not a generation
runtime chosen implicitly.

# CPU and parity receipts

Use the incorporated shared `restart_preflight.py --family gemma`, which
produces `rmct-restart-cpu-v1` / `cpu_checks_passed`, actual source/Python/
dependency identities and independently checked original weight bytes. The
older `gemma_preflight.py` draft does not produce parity-compatible evidence.

`native_prompt_probe.py` consumes that shared receipt and the original7680QID
training pool. Its `gemma-native-prompt-probe-v1` receipt records40native
thinking-on/off checks across five methods, two datasets, two biases and both
prompt sides, including raw messages and tokens. It generates no responses,
loads no model weights and performs no optimizer updates. Reproduce it after
incorporation; prior one-off observations are not current clearance.

Model-specific GPU/optimizer restoration, private rollout worker RNG policy,
complete600response validation and effective provider requests remain separate
gates. Neither passing launcher checks nor these CPU receipts authorizes a job.
