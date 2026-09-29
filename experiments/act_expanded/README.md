# Expanded ACT 8,192-QID source

This package freezes and materializes the planned expanded ACT source. The
target is exactly 8,192 unique question IDs (QIDs), balanced across LogiQA and
HellaSwag:

| Source role | LogiQA | HellaSwag | Total |
| --- | ---: | ---: | ---: |
| Safe legacy ACT-Max questions | 1,300 | 1,300 | 2,600 |
| Fresh pinned-public-source questions | 2,796 | 2,796 | 5,592 |
| Final wrong-argument QIDs | 4,096 | 4,096 | 8,192 |

The existing IID input is the 200-row `heldout-in-domain-n200.jsonl` population
(100 QIDs per dataset). A second, new fresh IID reserve (another 100 QIDs per
dataset) is frozen separately before the 2,796 fresh training questions are
selected. Every one of the 8,192 training QIDs, including the 2,600
legacy-origin questions, must receive one new accepted wrong argument from
OpenRouter model `google/gemma-4-31b-it`.
Legacy question provenance is retained, but legacy `biasing_text`,
`biased_messages`, and `unbiased_messages` are never copied into the candidate
pool. `materialize_arguments.py` does not read a legacy argument store.

The source staging, cross-format audit, question selection, and verification
steps are local-only. Only the explicitly invoked `generate` step calls
OpenRouter. None of the commands below have been run merely by documenting
them here.

## Inputs and local Arrow custody

Run from the repository root with a provisioned Python 3.11+ environment. The
worktree's current `.venv` is not provisioned from `requirements.txt`, so plain
`uv run` is not sufficient here. The commands below explicitly use the
currently provisioned interpreter at
`/Users/work/consistency-training-methods/.venv/bin/python`; it contains
`pyarrow`, the pinned `mcq-bias` package, and the `openai`/`httpx` client stack.
If that runtime is replaced, install `requirements.txt` plus `pyarrow` into the
replacement environment and change `ACT_EXPANDED_PYTHON` only after verifying
it.

The exact already-local Arrow inputs are:

```text
/Users/work/.cache/huggingface/datasets/lucasmccabe___logiqa/default/0.0.0/fa9f9918fa81eca088805c1395d7f592f7755ae0/logiqa-train.arrow
/Users/work/.cache/huggingface/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76/hellaswag-train.arrow
```

They correspond to the revisions pinned in `stage_sources.py`. The helper
accepts only a regular, non-symlink file with the exact pinned schema and row
count (7,376 LogiQA rows and 39,905 HellaSwag rows), preserves source order,
and publishes a three-field JSONL plus an Arrow-byte/source manifest. It never
downloads a dataset.

Set the run root. The legacy, IID, and prohibited paths below are the current
reviewed local canonical copies:

```bash
CTM_REPO_ROOT="$(pwd -P)"
ACT_EXPANDED_ROOT="$CTM_REPO_ROOT/artifacts/act-expanded-8192"
ACT_EXPANDED_PYTHON="/Users/work/consistency-training-methods/.venv/bin/python"

LOGIQA_ARROW="/Users/work/.cache/huggingface/datasets/lucasmccabe___logiqa/default/0.0.0/fa9f9918fa81eca088805c1395d7f592f7755ae0/logiqa-train.arrow"
HELLASWAG_ARROW="/Users/work/.cache/huggingface/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76/hellaswag-train.arrow"

LEGACY_POPULATION_JSONL="/Users/work/consistency-training-methods/artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl"
LEGACY_TRAINING_JSONL="/Users/work/.codex/worktrees/d6d6/consistency-training-methods/artifacts/act-max-training-20260804/act-max-training-cec56e4d33531a9f997740850a654e7ceaf16b6c2d4830108861df903660cbd5.jsonl"
EXISTING_IID_JSONL="/Users/work/consistency-training-methods/artifacts/stage1-iid-diagnostic-none-20260801/heldout-in-domain-n200.jsonl"
PROHIBITED_TRAIN_EVAL_JSONL="/Users/work/consistency-training-methods/artifacts/stage1-iid-diagnostic-none-20260801/train-eval-n200.jsonl"
PROHIBITED_HLE_JSONL="/Users/work/consistency-training-methods/artifacts/stage2-ood-hle-2x2-20260802-r1/hle/unbiased.jsonl"

mkdir -p "$ACT_EXPANDED_ROOT"/{receipts,arrow-stage,frozen-sources,audit,selection,generation,canonical,shared-two-bias}
```

`--existing-iid` is specifically the held-out in-domain 200-row population; it
completes the 1,400-train + 100-IID-per-dataset partition of the 1,500-row
legacy source. `train-eval-n200.jsonl` has a different role: pass it as
`--prohibited train_eval=...`, which removes its 100 QIDs per dataset from the
ACT-Max training population and leaves 1,300 safe legacy QIDs per dataset.

`--prohibited NAME=PATH` consumes a frozen JSONL population, not a manifest.
Pass both the in-domain `train_eval` and canonical RMCT Stage-2 HLE population,
then use that exact pair in `audit`, `materialize`, and `verify`. The HLE
`unbiased.jsonl` above carries the shared 100 HLE QIDs across all cue
renderings; its parent provenance manifest is
`/Users/work/consistency-training-methods/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json`.
AITA remains schema-disjoint provenance and is not a selector input.

## 1. Stage the local Arrow splits

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.stage_sources \
  --dataset logiqa \
  --arrow-source "$LOGIQA_ARROW" \
  --output-dir "$ACT_EXPANDED_ROOT/arrow-stage" \
  | tee "$ACT_EXPANDED_ROOT/receipts/stage-logiqa.json"

"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.stage_sources \
  --dataset hellaswag \
  --arrow-source "$HELLASWAG_ARROW" \
  --output-dir "$ACT_EXPANDED_ROOT/arrow-stage" \
  | tee "$ACT_EXPANDED_ROOT/receipts/stage-hellaswag.json"

LOGIQA_STAGED_JSONL="$(jq -r .data "$ACT_EXPANDED_ROOT/receipts/stage-logiqa.json")"
HELLASWAG_STAGED_JSONL="$(jq -r .data "$ACT_EXPANDED_ROOT/receipts/stage-hellaswag.json")"
```

Keep both staging manifests printed in the receipts. The next command freezes
the normalized source snapshot used by the audit; it deliberately takes the
staged JSONL rather than reaching back to Hugging Face.

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.selection freeze-source \
  --dataset logiqa \
  --raw-source "$LOGIQA_STAGED_JSONL" \
  --output-dir "$ACT_EXPANDED_ROOT/frozen-sources" \
  | tee "$ACT_EXPANDED_ROOT/receipts/freeze-logiqa.json"

"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.selection freeze-source \
  --dataset hellaswag \
  --raw-source "$HELLASWAG_STAGED_JSONL" \
  --output-dir "$ACT_EXPANDED_ROOT/frozen-sources" \
  | tee "$ACT_EXPANDED_ROOT/receipts/freeze-hellaswag.json"

LOGIQA_SOURCE="$(jq -r .data "$ACT_EXPANDED_ROOT/receipts/freeze-logiqa.json")"
LOGIQA_SOURCE_MANIFEST="$(jq -r .manifest "$ACT_EXPANDED_ROOT/receipts/freeze-logiqa.json")"
HELLASWAG_SOURCE="$(jq -r .data "$ACT_EXPANDED_ROOT/receipts/freeze-hellaswag.json")"
HELLASWAG_SOURCE_MANIFEST="$(jq -r .manifest "$ACT_EXPANDED_ROOT/receipts/freeze-hellaswag.json")"
```

## 2. Audit collisions and protected exclusions

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.selection audit \
  --logiqa-source "$LOGIQA_SOURCE" \
  --logiqa-manifest "$LOGIQA_SOURCE_MANIFEST" \
  --hellaswag-source "$HELLASWAG_SOURCE" \
  --hellaswag-manifest "$HELLASWAG_SOURCE_MANIFEST" \
  --legacy-population "$LEGACY_POPULATION_JSONL" \
  --legacy-training "$LEGACY_TRAINING_JSONL" \
  --existing-iid "$EXISTING_IID_JSONL" \
  --prohibited "train_eval=$PROHIBITED_TRAIN_EVAL_JSONL" \
  --prohibited "hle=$PROHIBITED_HLE_JSONL" \
  --output-dir "$ACT_EXPANDED_ROOT/audit" \
  | tee "$ACT_EXPANDED_ROOT/receipts/audit.json"

CROSS_FORMAT_AUDIT="$(jq -r .audit "$ACT_EXPANDED_ROOT/receipts/audit.json")"
jq '{cross_format_mapping, legacy_training_prohibited_overlap, post_prohibited_filter, capacity_report, generator_policy}' \
  "$CROSS_FORMAT_AUDIT"
```

Retain and review the content-addressed audit before continuing. It fails
closed unless the full pinned-source comparison reproduces the current
lineage-aware audit:

| Dataset | Source physical / canonical-unique | Authoritative stem-only exclusion physical / canonical-unique | Fresh physical / canonical-unique |
| --- | ---: | ---: | ---: |
| LogiQA | 7,376 / 7,363 | 1,383 / 1,380 | 5,993 / 5,983 |
| HellaSwag | 39,905 / 39,905 | 0 / 0 | 39,905 / 39,905 |

LogiQA exclusion uses only the exact, non-fuzzy alphanumeric full-stem
signature. Its direct matches have zero ambiguity and zero ground-truth
disagreements. Secondary query/option signatures are diagnostics only: their
three-way union is 1,434 physical / 1,430 canonical-unique, with 14 physical /
14 canonical-unique ambiguous matches and three physical / three
canonical-unique ground-truth-disagreement matches. It contains demonstrated
false positives, so the additional 51 physical / 50 canonical-unique rows are
not withheld or used to construct the selection. The 1,500 legacy LogiQA
identities partition as 1,372 direct train identities plus 128 validation
identities, with none unmatched across those pinned lineages.

The legacy HellaSwag population is validation-lineage. The three HellaSwag
train/legacy context repeats are recorded as context-only overlaps, not as the
same MCQ: none has the same context plus ordered options, so both the direct
exclusion and secondary MCQ screen counts are zero.

For this 8,192-QID plan, the reviewed prohibited populations must also leave
exactly 1,300 safe legacy training QIDs per dataset and at least 2,896 eligible
fresh QIDs per dataset before the new 100-QID IID reserve. That leaves at
least 2,796 fresh training QIDs per dataset.

## 3. Freeze and replay the exact selection

`legacy_plus_fresh` with `--n-total 8192` is the planned condition. It takes all
1,300 safe legacy questions first, adds exactly 2,796 fresh questions per
dataset, deterministically shuffles that fixed per-dataset composition, and
stores the result in LogiQA/HellaSwag round-robin order.

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.selection materialize \
  --logiqa-source "$LOGIQA_SOURCE" \
  --logiqa-manifest "$LOGIQA_SOURCE_MANIFEST" \
  --hellaswag-source "$HELLASWAG_SOURCE" \
  --hellaswag-manifest "$HELLASWAG_SOURCE_MANIFEST" \
  --legacy-population "$LEGACY_POPULATION_JSONL" \
  --legacy-training "$LEGACY_TRAINING_JSONL" \
  --existing-iid "$EXISTING_IID_JSONL" \
  --prohibited "train_eval=$PROHIBITED_TRAIN_EVAL_JSONL" \
  --prohibited "hle=$PROHIBITED_HLE_JSONL" \
  --cross-format-audit "$CROSS_FORMAT_AUDIT" \
  --candidate-source-mode legacy_plus_fresh \
  --n-total 8192 \
  --output-dir "$ACT_EXPANDED_ROOT/selection" \
  | tee "$ACT_EXPANDED_ROOT/receipts/selection.json"

FRESH_IID_SELECTION="$(jq -r .fresh_iid "$ACT_EXPANDED_ROOT/receipts/selection.json")"
CANDIDATE_SELECTION="$(jq -r .question_candidates "$ACT_EXPANDED_ROOT/receipts/selection.json")"
SELECTION_MANIFEST="$(jq -r .manifest "$ACT_EXPANDED_ROOT/receipts/selection.json")"

"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.selection verify \
  --logiqa-source "$LOGIQA_SOURCE" \
  --logiqa-manifest "$LOGIQA_SOURCE_MANIFEST" \
  --hellaswag-source "$HELLASWAG_SOURCE" \
  --hellaswag-manifest "$HELLASWAG_SOURCE_MANIFEST" \
  --legacy-population "$LEGACY_POPULATION_JSONL" \
  --legacy-training "$LEGACY_TRAINING_JSONL" \
  --existing-iid "$EXISTING_IID_JSONL" \
  --prohibited "train_eval=$PROHIBITED_TRAIN_EVAL_JSONL" \
  --prohibited "hle=$PROHIBITED_HLE_JSONL" \
  --cross-format-audit "$CROSS_FORMAT_AUDIT" \
  --fresh-iid-selection "$FRESH_IID_SELECTION" \
  --candidate-selection "$CANDIDATE_SELECTION" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --candidate-source-mode legacy_plus_fresh \
  --n-total 8192

jq '.question_candidates | {row_count, counts_by_dataset, counts_by_dataset_and_source_origin}' \
  "$SELECTION_MANIFEST"
```

The final `jq` view must report 4,096 candidates per dataset, each split as
1,300 `legacy_act_max_question` plus 2,796 `pinned_hf_train_question`. The
candidate JSONL is question-only and contains no old or newly generated
argument text.

## 4. Run a resumable two-candidate generation smoke

The following is the first networked step. `materialize_arguments.py` does not
auto-load a dotenv file: export `OPENROUTER_API_KEY` from the authorized local
credential source before invoking it, without printing the value or putting it
in a command or receipt. For the current checkout, that source is the mode-600
file `/Users/work/consistency-training-methods/.env`:

```bash
set -a
source /Users/work/consistency-training-methods/.env
set +a
test -n "${OPENROUTER_API_KEY:-}"
```

Requests contain only `model` and `messages` and deliberately set no
output-token, completion-token, generation-token, or reasoning-token cap. The
`--max-candidates-this-run 2` smoke bound limits how many missing QIDs are
scheduled in this invocation; it is not an output-token cap. The focused
offline request-policy checks can be rerun before the live smoke:

```bash
"$ACT_EXPANDED_PYTHON" -m pytest tests/test_act_expanded_materialize_arguments.py \
  -k 'cap_preflight or wire_body'
```

Because the candidate file is round-robin, a fresh journal's first two
candidates are one LogiQA and one HellaSwag QID.

```bash
ACT_EXPANDED_JOURNAL="$ACT_EXPANDED_ROOT/generation/wrong-argument-attempts.jsonl"

"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.materialize_arguments generate \
  --candidate-selection "$CANDIDATE_SELECTION" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --journal "$ACT_EXPANDED_JOURNAL" \
  --n-total 8192 \
  --max-candidates-this-run 2
```

Inspect the outcomes for both scheduled candidates and the reported progress
before continuing. With the default two attempts per candidate, the smoke can
append one or two journal rows for each QID. Accepted candidates are never
requested again. Rejected or transport-failed candidates remain missing and
are eligible on a later resume.

## 5. Resume the complete generation

Use the same command without `--max-candidates-this-run`. It validates the
existing journal, skips every accepted QID, and schedules all remaining QIDs.
If any QIDs still lack an accepted naturally terminated response, rerun this
same full command; each invocation appends attempts only and never rewrites the
journal.

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.materialize_arguments generate \
  --candidate-selection "$CANDIDATE_SELECTION" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --journal "$ACT_EXPANDED_JOURNAL" \
  --n-total 8192
```

Do not finalize until progress reports `accepted_after` equal to
`candidate_count` (8,192). Length-limited or otherwise non-natural finishes do
not count as accepted, unless the run owner explicitly chooses a smaller
balanced stopping point. In that case, pass `--target-per-dataset`; the
finalizer takes the first accepted rows in frozen candidate order and records
both the full accepted availability and the post-generation balancing rule.

## 6. Finalize canonical new wrong-argument pairs

Finalization replays the selection proof and journal, requires one accepted
fresh Gemma argument for every selected QID, reconstructs rows in frozen
candidate order, and publishes content-addressed canonical consistency pairs
plus a provenance manifest. Without `--target-per-dataset`, it fails unless all
8,192 candidates are accepted.

```bash
"$ACT_EXPANDED_PYTHON" -m experiments.act_expanded.materialize_arguments finalize \
  --candidate-selection "$CANDIDATE_SELECTION" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --journal "$ACT_EXPANDED_JOURNAL" \
  --output-dir "$ACT_EXPANDED_ROOT/canonical" \
  --n-total 8192 \
  | tee "$ACT_EXPANDED_ROOT/receipts/finalize.json"

WRONG_ARGUMENT_SOURCE="$(jq -r .data "$ACT_EXPANDED_ROOT/receipts/finalize.json")"
WRONG_ARGUMENT_MANIFEST="$(jq -r .manifest "$ACT_EXPANDED_ROOT/receipts/finalize.json")"
```

For the explicitly approved 7,892-accepted stopping point from 2026-09-10, the
largest balanced, no-wrap shape compatible with 16 QIDs per dataset per segment
is 3,840 per dataset. Its finalization command adds
`--target-per-dataset 3840`; the 212 other accepted rows remain preserved in
the generation journal.

## 7. Hand off to the shared two-bias builder

The shared builder is a Python API, not a CLI. The exact local IID manifest
attests both `heldout_in_domain` and `train_eval` under `splits`; the exact
Stage-2 parent manifest attests the in-domain `unbiased`, `wrong_argument`, and
`suggested_answer` artifacts as well as the HLE population. Manifest paths are
not interchangeable with the earlier selector JSONLs.

```bash
export WRONG_ARGUMENT_SOURCE WRONG_ARGUMENT_MANIFEST
export SHARED_TWO_BIAS_OUTPUT="$ACT_EXPANDED_ROOT/shared-two-bias"
export PROTECTED_IID_MANIFEST="/Users/work/consistency-training-methods/artifacts/stage1-iid-diagnostic-none-20260801/manifest.json"
export PROTECTED_STAGE2_MANIFEST="/Users/work/consistency-training-methods/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"

"$ACT_EXPANDED_PYTHON" - <<'PY'
import json
import os

from ctm_data.adapters.mcq_bias.shared_qid_two_bias import materialize_shared_qid_two_bias

artifact = materialize_shared_qid_two_bias(
    wrong_argument_source=os.environ["WRONG_ARGUMENT_SOURCE"],
    wrong_argument_manifest=os.environ["WRONG_ARGUMENT_MANIFEST"],
    protected_iid_manifest=os.environ["PROTECTED_IID_MANIFEST"],
    protected_stage2_manifest=os.environ["PROTECTED_STAGE2_MANIFEST"],
    output_dir=os.environ["SHARED_TWO_BIAS_OUTPUT"],
    qids_per_dataset=3840,
    qids_per_dataset_per_segment=16,
)
print(json.dumps({
    "data": str(artifact.data_path),
    "manifest": str(artifact.manifest_path),
    "sha256": artifact.content_sha256,
    "status": artifact.status,
}, sort_keys=True))
PY
```

This last step independently verifies and excludes the protected IID and
Stage-2 QIDs, reconstructs the pinned `suggested_answer` cue for the same
biased option as each fresh wrong argument, and freezes 7,680 shared QIDs /
15,360 QID-bias conditions in the approved 3,840-per-dataset realization. With
16 QIDs per dataset per segment, the manifest
contains 240 deterministic segments of 32 QIDs (16 optimizer updates at batch
size 2), for 3,840 no-repeat optimizer updates.
