# Stage 2 IID/HLE OOD diagnostic

This local-only module produces the same four-column comparison for both TBSR
and Luna explicit-bias acknowledgement for each condition:

1. `IID`: heldout_in_domain n=200 under `wrong_argument`;
2. `Held-out dataset`: canonical HLE n=100 under `wrong_argument`;
3. `Held-out bias`: IID n=200 under five training-held-out biases; and
4. `Held-out dataset + bias`: canonical HLE n=100 under those five biases.

The HLE `wrong_argument` prompts use the frozen HLE-specific argument bank
(generated before this evaluation), whereas the IID panel uses the legacy
stored argument bank. Thus the two HLE-labelled columns are valid frozen HLE
transfer tests, but not a literal counterfactual that changes only question
dataset while holding the argument-bank provenance fixed.

The report retains every per-bias IID/HLE cell. The two held-out-bias
headline columns micro-pool exact towards-switch numerators over exact jointly
parsed eligible denominators, and separately micro-pool Luna YES verdicts
over parsed Luna verdicts. Their uncertainty (and every displayed cell's
uncertainty) is a deterministic non-parametric question-cluster bootstrap.
All five bias observations for a question are resampled together; no
independent-binomial error bars are used.

The analyzer accepts either normalized paired observations or completed local
Inspect logs. For a publication figure, the selected biased logs must already
have the posthoc Luna score appended; it never calls a grader itself. It makes
no model/API calls.

First materialize the immutable 2×2 suite locally. The materializer copies
frozen inputs and derives only the IID held-out-bias prompt files; it makes no
model or network calls.

```bash
python -m experiments.stage2_ood_hle.materialize \
  --iid-manifest artifacts/stage1-iid-none/manifest.json \
  --hle-dir artifacts/rmct-hle/data \
  --output-dir artifacts/stage2-ood-hle/frozen
```

Raw generation must remain grader-free (`include_bias_acknowledged: false`).
After one condition completes, preflight its dedicated condition log directory
*on the generation host*. This validates all 21 tasks (three clean and 18
biased), their frozen manifest/source hashes, their common Qwen runtime, and
the actual clean log resolved by every biased switch score. It writes an
immutable, content-bound handoff report and makes no model/API calls.

```bash
python -m experiments.stage2_ood_hle.raw_preflight \
  --raw-log-root logs/stage2-ood-hle/act-vllm-compat \
  --manifest artifacts/stage2-ood-hle/frozen/manifest.json \
  --condition act-vllm-compat \
  --runtime-profile vllm \
  --expected-checkpoint /absolute/path/to/verified-vllm-compat-adapter \
  --output artifacts/stage2-ood-hle/preflight/act-vllm-compat.json
```

For a base vLLM condition, omit `--expected-checkpoint`. For a native
Transformers/PEFT condition, use `--runtime-profile hf-peft`, a raw checkpoint
path, and its exact `--expected-max-connections` value. A vLLM adapter is
accepted only if its immutable Qwen3.5 parity attestation validates.

To stage on the generation host (or after copying the report and exact raw
logs to a transfer host), use the local-only stager. It copies only the **18
biased** files named by the report, verifies their SHA-256 values before and
after copying, and never overwrites a differing destination:

```bash
python -m experiments.stage2_ood_hle.stage_luna \
  --preflight-report artifacts/stage2-ood-hle/preflight/act-vllm-compat.json \
  --output-root staging/stage2-ood-hle-raw
```

Its immutable layout is:

```text
<staged-raw-root>/<condition>/<population>/<bias_type>/<dataset>/<source filename>.eval
```

Then the posthoc grader rechecks every SHA-256 before calling the dated Luna
pin. It writes separate derived EvalLogs, JSONL exports, and provenance files;
it never overwrites raw generation logs. Five workers × 100 connections is the
default aggregate limit of 500. To grade multiple completed conditions without
exceeding that **global** cap, pass their reports to one invocation rather than
launching separate default graders concurrently.

```bash
python -m experiments.stage2_ood_hle.grade_luna \
  --staged-raw-root staging/stage2-ood-hle-raw \
  --preflight-report artifacts/stage2-ood-hle/preflight/act-vllm-compat.json \
  --preflight-report artifacts/stage2-ood-hle/preflight/attct-vllm-compat.json \
  --output-root logs/stage2-ood-hle-luna \
  --workers 5 \
  --connections-per-worker 100
```

Remove the second report line when only one condition has completed; the
single five-worker pool covers every report supplied to the invocation.

Point analysis at the derived Luna root, not the raw generation root:

```bash
python -m experiments.stage2_ood_hle.analyze \
  --run untrained=logs/stage2-ood-hle-luna/untrained \
  --run bct=logs/stage2-ood-hle-luna/bct \
  --stage2-manifest artifacts/stage2-ood-hle/frozen/manifest.json \
  --output artifacts/stage2-ood-hle/analysis.json

python -m experiments.stage2_ood_hle.plot \
  --analysis artifacts/stage2-ood-hle/analysis.json \
  --output-dir artifacts/stage2-ood-hle/figures
```

For testing or an external audited exporter, JSONL rows require `condition`,
`population` (`iid` or `hle`), `question_id`, `bias_type`, `joint_parse`,
`clean_matches_bias`, and `towards_bias_switch`; include
`bias_acknowledged` (or `luna_bias_acknowledged`) for the verbalisation
figure. An ungraded raw matrix is represented as a null Luna rate and cannot
be plotted until grading is present. The final matrix is strict: each
condition must contain the common 200 IID questions and 100 HLE questions for
`wrong_argument` and all five held-out biases.

This is a separate renderer.  It neither imports nor changes the historical
Stage 1 two-column figure path.
