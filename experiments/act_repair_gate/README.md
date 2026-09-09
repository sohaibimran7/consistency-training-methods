# ACT repair gate

[`experiment.yaml`](experiment.yaml) is a small direct plan for deciding
whether ACT can work when trained on correctly aligned wrong-argument prompt
pairs. It is intentionally separate from the broad Stage 1 factory and makes
no cloud call by itself.

Only copy the frozen 3,000-row source to the path declared in
`variables.frozen_iid_source`. Do not copy an IID `manifest.json` or split
JSONL from another machine: the manifest contains absolute paths. The CPU
preparation target recreates its own immutable, remote-local files:

```text
artifacts/act-repair-gate-20260731/data/
  manifest.json
  train-eval-n200.jsonl
  heldout-in-domain-n200.jsonl
  canonical-train-eval-n200.jsonl
  canonical-heldout-in-domain-n200.jsonl
```

Both canonical files contain all 200 split rows (100 LogiQA and 100
HellaSwag). Their variant user message is the frozen wrong argument followed
by the exact clean user message. The native split files are never modified.

Run the three stages separately, leaving exactly one intended GPU visible for
the GPU commands. Do not pass `--parallel`: each native biased evaluation
depends on clean logs written by the preceding canonical evaluation.

```bash
python scripts/run_experiment.py \
  experiments/act_repair_gate/experiment.yaml \
  --target data --stages data_preparation --yes

CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  experiments/act_repair_gate/experiment.yaml \
  --target preflight --stages evaluation,analysis --yes
```

The preflight executes four untrained cells: canonical/native prompts on the
training-domain and held-out in-domain splits. It writes canonical clean and
biased logs, plus native biased logs paired with those exact clean outputs,
under `logs/evals/act_repair_gate_qwen3_5_9b_20260731/untrained/`, then writes
an immutable deterministic report at
`artifacts/act-repair-gate-20260731/analysis/untrained-raw.json`.
The shared gate analyzer accepts separate native and canonical roots, rejects
missing or duplicate successful cells, and verifies that their frozen question
IDs, prompt style, and source provenance are comparable before it writes that
report.

Inspect the deterministic paired TBSR before spending on training. The
canonical base must retain a meaningful bias signal (at minimum the existing
four switch / 85% paired-parsing safety floor; the proposed behavioral gate is
a canonical TBSR of at least 20%). A weak canonical base result means the
prefix needs revision—not that ACT has failed.

If it passes, run the 4,000-step ACT job and its four matched cells:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  experiments/act_repair_gate/experiment.yaml \
  --target act --start-from training --yes
```

The ACT run uses the selected shared constant LR `1e-4`, Adam `(0.9, 0.999)`,
weight decay `0.01`, `weight=5e-5`, rank-8/alpha-16/dropout-0.05 LoRA, and all
Qwen3.5 attention-projection families (including DeltaNet), not MLPs.
`minimum_optimizer_steps=4000` fails closed if canonicalization does not yield
all 200 usable pairs.

Before any long generated-answer evaluation, the target now runs two small
gates. First, training itself writes an immutable one-pair native
Transformers/PEFT backward report and rejects a missing/non-finite adapter
gradient in either Qwen3.5 attention family. Then the evaluation stage uses a
balanced four-per-dataset canonical slice from each split (16 questions total)
for deterministic forced-choice HF/PEFT scoring. Its immutable attestation
binds the direct-answer report, raw adapter hash, frozen source hashes, and
selected question IDs. The long ACT evaluation only begins if that tiny report
has at least one base toward-bias switch on the training split and a strictly
lower direct TBSR under the adapter. This is an early behavioral sanity check,
not a replacement for the paired generated-answer metric or the separate
HF/vLLM compatibility attestation required before vLLM evaluation.

ACT outputs are under `logs/evals/act_repair_gate_qwen3_5_9b_20260731/act/`;
the same strict raw-report pipeline writes
`artifacts/act-repair-gate-20260731/analysis/act-raw.json`.
After the ACT report is complete, the plan automatically invokes the shared
publication-style renderer on the untrained and ACT raw reports. It writes:

```text
artifacts/act-repair-gate-20260731/figures/
  towards-bias-switch.png
  towards-bias-switch.svg
```

The chart puts training-domain samples on the left and held-out in-domain
samples on the right. Within each group it shows Base/ACT and native/canonical
prompt variants in the main-figure style. Rendering is local-only: it reads
the two immutable raw paired-switch reports and makes no model, OpenRouter,
or Luna call.
If evaluation needs to be resumed separately after a completed training job,
pass its emitted final checkpoint explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  experiments/act_repair_gate/experiment.yaml \
  --target act --stages evaluation \
  --checkpoint file:///absolute/path/to/final-checkpoint --yes
```

The plan intentionally defers Luna grading. First compare deterministic
conditional TBSR, parse rate, and clean-answer/bias-answer denominator for the
untrained and ACT checkpoints in all four cells. Only a behavioral movement
earns model-based verbalization grading.
