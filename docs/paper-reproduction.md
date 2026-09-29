# Reproducing the paper

This checkout is intended to contain the code for every paper result, including training, evaluation, grading, analysis and figures. Integration is still incomplete. A verified saved figure is not evidence that its computation can be rerun, and reproducing a historical result does not validate a flawed training run.

## Current coverage and entry points

[The result manifest](paper-results-manifest.json) records 13 historical inputs, their content hashes and scientific limitations. [The command map](paper-command-map.json) maps each input to its manuscript output. A null regeneration command is an outstanding integration requirement. Do not treat this inventory as a complete paper pipeline.

Commands below run from the repository root. Use a Python environment provisioned from this checkout's dependency specifications and the recorded production runtime; a different editable checkout must not supply imports. The GPU runtime and model snapshots must match the recorded preflight evidence. CPU unit tests do not establish GPU compatibility.

### Verify saved historical inputs

Obtain the immutable artifact bundles from the research archive and set these directories to their extracted repository roots (the directories containing `artifacts/`):

```sh
export MONITOR_ROOT=/absolute/path/to/monitor-artifact-root
export METHODS_ROOT=/absolute/path/to/methods-artifact-root
export MAIN_ROOT=/absolute/path/to/main-artifact-root
python scripts/verify_paper_inputs.py \
  --root monitor="$MONITOR_ROOT" \
  --root methods="$METHODS_ROOT" \
  --root main="$MAIN_ROOT"
```

This checks all manifest hashes and exits unsuccessfully for missing or changed inputs. It does not regenerate figures, rescore responses or run models. An archive download/access procedure is still required for a fresh external checkout; these environment variables are not a substitute for distributing the inputs.

### Integrated offline commands

The following packages are now in this checkout. Their stated input boundaries matter: saved-score analysis and figure assembly do not repeat model training or paid grading.

| Pipeline | Executable instructions | Verified boundary |
| --- | --- | --- |
| Manuscript assembly | [Builder and input layout](../experiments/paper_manuscript/PORTABLE-HANDOFF.md) | Two rendered main grids, copied appendix figures and an inventory; manual table transcription remains separate |
| Compute and data exposure | [Six commands and isolated verifier](../experiments/paper_compute/README.md) | Historical accounting and checkpoint-selection analyses from frozen artifacts; not new matched-compute performance |
| Luna monitorability | [Plot commands](../scripts/paper_luna/README.md), [upstream builders](../scripts/paper_luna/UPSTREAM.md) | Saved logs/ledgers to normalized rows, then plots; earlier sample/population selection remains an explicit input |
| Evaluation-gaming pilot | [Scenario, grading and plotting runbook](../experiments/eval_awareness/paper-handoff-20260928/README.md) | Historical exploratory pilot; future 65,536-token configuration has a context guard but no validated executor |

For manuscript assembly, use artifact directories rather than repository roots:

```sh
export CORRECTED_FIGURES="$MONITOR_ROOT/artifacts/parser-fixed-64k-20260925/figures-final"
export MONITOR_FIGURES="$MONITOR_ROOT/artifacts/parser-fixed-20260923/luna-overview"
export METHOD_ARTIFACTS="$METHODS_ROOT/artifacts"
export ORGANISM_ARTIFACTS="$MAIN_ROOT/artifacts"
export PAPER_OUT=/absolute/path/to/new-paper-output
python experiments/paper_manuscript/build_figures.py \
  --corrected-figures-root "$CORRECTED_FIGURES" \
  --monitor-root "$MONITOR_FIGURES" \
  --ctm-artifacts-root "$METHOD_ARTIFACTS" \
  --organism-artifacts-root "$ORGANISM_ARTIFACTS" \
  --output-dir "$PAPER_OUT" --check-only
```

Remove `--check-only` to assemble the figures. The builder checks 14 source files, including the shared chart specification. It records output hashes and whether each entry was rendered, copied or source-only; it does not validate manually transcribed LaTeX tables. Use a new output directory to preserve earlier evidence.

To reproduce compute/data accounting without modifying frozen inputs:

```sh
python experiments/paper_compute/verify.py \
  --artifact-root "$METHOD_ARTIFACTS" \
  --report /absolute/path/to/new-compute-verification.json
```

The verifier runs all six commands in a temporary tree. See the linked Luna and evaluation-gaming runbooks for their larger input/command graphs and exact private-artifact requirements. These commands do not grant authorization for additional provider calls.

### Historical output coverage

| Manuscript output | Required computation | Current integration gap |
| --- | --- | --- |
| Main switch-rate grid | Paired clean/biased labels, aggregation and uncertainty, then seen/held-out dataset × bias grid | `experiments/paper_mcq` reproduces historical statistics; manuscript assembly renders their grid. Fresh corrected training remains separate |
| Main conditional-verbalization grid | Acknowledgment among towards-bias switches, with explicit denominator and uncertainty | Historical MCQ replay is exact at 108,000 permutations; manuscript assembly renders the grid |
| Expanded switch and conditional-verbalization figures | Dataset/bias breakdowns from saved responses and grades | Portable MCQ replay recomputes both endpoints exactly; assembly is a separate presentation step |
| Overall verbalization and biased accuracy | All-valid-response denominators and accuracy labels | Portable MCQ replay recomputes both endpoints exactly, retaining historical coverage limitations |
| Convergence | New validation-selected checkpoint histories | Existing plot is historical own-objective stopping, not the replacement analysis |
| Cross-task | AITA responses, verdict parsing and coverage audit | `experiments/paper_aita` reproduces both saved statistical reports exactly; corrected fresh-model inference remains separate |
| Qwen and Gemma image results | Rendered inputs, model evaluation, labels and plots | Historical generation and portable recovery/plotting are integrated in `experiments/qwen_image_pilot`; seven recovery tests pass. Historical uniform 65,536 budgets are not the new dataset-specific policy. |
| Monitorability | Saved monitor scores plus paired switch labels and population definitions | Historical overview assembly and latest Luna builders/plots are integrated as distinct products; do not substitute one population for another |
| Compute appendix | Checkpoint cost accounting and matched-budget performance | Portable accounting/selection commands are integrated; the cost table does not establish matched-compute performance |
| Evaluation-gaming pilot | Scenario generation, grading and missingness-aware analysis | Historical pilot runbook/scripts are integrated; current screening and future higher-context execution remain distinct |

The portable manuscript assembler is `experiments/paper_manuscript/build_figures.py`. Its [upstream recipe registry](../experiments/paper_manuscript/upstream-command-map.json) records original aggregation sources. The current executable commands are in `docs/paper-command-map.json`: [MCQ preprocessing and statistics](../experiments/paper_mcq/README.md), [AITA](../experiments/paper_aita/README.md), and [convergence](../experiments/paper_convergence/README.md) now have canonical implementations and exact historical numerical replay receipts. The MCQ historical parser is isolated and hash-pinned; it is not used by fresh training. Raw data stays external, new inference is not claimed, and manual LaTeX table transcription remains separate. The assembler intentionally copies appendix figures; use the aggregation commands to recompute their statistics first.

## Protocol that corrected runs must preserve

The historical organism-table aggregation is now in
[`experiments/paper_organism`](../experiments/paper_organism/README.md).
Its canonical replay exactly matched all historical rows and native Inspect
statistics; the full presentation and local-link verification also passed.
The historical 16-condition monitor overview is independently portable via
[`scripts/paper_luna/OVERVIEW.md`](../scripts/paper_luna/OVERVIEW.md), rather than
requiring an injected function in an external recipe. Neither result establishes
corrected-training performance.

- LogiQA and HellaSwag: **20,480 generated tokens** for training and evaluation. Other datasets, including HLE: **65,536 generated tokens**. Generated-token allowance and total context capacity are different settings; the prompt must also fit.
- Preserve model-specific thinking configuration, tokenizer/chat-template identity, data order, batch size, rollout counts, optimizer settings and data exposure. Record changes explicitly.
- Invalid, incomplete, unparsed, missing and failed grades remain identifiable. They must not silently become correct answers or zero-bias examples.
- Main results are data-matched. Compute comparisons are separate within activation-based and output-based methods; compute-matched and convergence results belong in the appendix.
- The main verbalization result conditions on towards-bias switches. Overall verbalization and monitorability are separate measurements. Monitorability reports must state filtered/unfiltered population, matched denominator and score coverage.
- Use the existing manuscript's wording and structure. Historical placeholders require visible captions identifying the flaw, replacement status and responsible task; they are not final evidence.

## Corrected RMCT restart

The restart modules live in [the restart directory](../experiments/rmct_restart_20260928/README.md). They must run from the incorporated consolidated version. A PR branch name, generated plan or old successful receipt is insufficient proof of the deployed code.

Inspect the concrete interfaces before constructing commands:

```sh
python experiments/rmct_restart_20260928/prepare.py --help
python -m experiments.rmct_restart_20260928.restart_preflight --help
python -m experiments.rmct_restart_20260928.qwen_launch --help
python -m experiments.rmct_restart_20260928.native_rl_gate --help
```

1. Incorporate the reviewed PR and pin its exact source commit. Deploy a clean checkout; record interpreter, environment, dependencies, imported module locations, source/config hashes and model/tokenizer identities.
2. Prepare a new plan from the tracked original recipe, original pre-RMCT weights and a fresh optimizer. Do not initialize from contaminated checkpoints or reuse old stopping history.
3. Run CPU provenance and thinking checks in the scheduled runtime, followed by fresh native base/nonzero-adapter parity. The preflight must use the same candidate code and actual configuration as the intended run.
4. Run the disposable native RL gate using its exact `--plan`, `--preflight` and new `--output` arguments. Its one-update checkpoint is never a production parent. Inspect completion, parser, reward, advantage and optimizer evidence; successful process exit alone is insufficient.
5. Run `regression_gate` in the deployed interpreter, then start scientific training only through `qwen_train_window` after every receipt passes and the source commit is an ancestor of actual `origin/main`. The controller, strict checkpoint seals, validation preparation/server/workers and external post-completion producer are integrated. Live execution remains unverified until receipts are collected. The disposable gate validates the full 32-QID manifest before selecting its first two QIDs; it never changes the production setting contract.

For Qwen, the approved segment is 32 QIDs at batch size two, giving 16 optimizer updates. Four fresh-lineage segments reach the first validation at step 64. Validation uses a new 20,480-token history, TBSR only, interval 64, patience two, minimum delta zero and earliest checkpoint on ties. No later window advances without verified validation completion. See the restart README for the population and strict-resume contract.

Gemma's original snapshot is a single safetensors file; its publisher content identity must be checked, rather than requiring a nonexistent shard index. The Gemma parity probe does not prove production multiworker behavior or controller correctness. Sharing newer Gemma revisions is awaiting explicit handoff approval after automatic review blocked that update; do not substitute those revisions before the block is resolved.

### Local checks

```sh
python -m pytest tests/test_ctm_opct.py tests/test_ctm_artifacts.py -q
python -m pytest tests/test_restart_preflight.py tests/test_native_rl_gate.py -q
python -m pytest experiments/rmct_restart_20260928/test_prepare.py \
  experiments/rmct_restart_20260928/test_qwen_validation.py \
  experiments/rmct_restart_20260928/gemma_test.py -q
```

These are scoped regression checks. Record exact test versions/results; they do not replace the broader integration suite, actual GPU preflight, or scientific result reproduction.

## Failure recovery and evidence

Keep failed jobs, receipts, raw responses and grader outputs immutable. Diagnose a failed gate, fix and commit the implementation, then issue a new plan/output directory and rerun the affected gates. Never overwrite a failed receipt, borrow another commit's attestation or resume the disposable probe. Preserve originals when reparsing or grading; corrected labels and regenerated figures should have separate provenance.

For each final result, the command map must eventually contain input acquisition, exact hashes/model identities, environment setup, generation and grading commands, analysis/rendering commands, output paths, seeds, denominator/missingness rules and the successful reproduction receipt. Until these are present and executed, mark that result incomplete.

## Documentation approach

Follow [Document your code](https://goodresearch.dev/docs) and [Document your project](https://goodresearch.dev/pipelines): explicit interfaces, runnable commands, documented inputs/outputs and failure recovery, with project setup alongside the scientific pipeline. A chat transcript is not an installation guide or a reproducibility receipt.
