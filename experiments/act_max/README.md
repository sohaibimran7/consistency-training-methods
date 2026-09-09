# ACT-Max (Qwen3.5-9B, native no-reasoning prompts)

ACT-Max is the fairer, high-data comparison condition for ACT.  It starts
from base `Qwen/Qwen3.5-9B`; it is **not** a continuation of the repaired ACT
adapter and does not alter its paired prompt/target construction.

The training selection is deliberately the largest available source population
that remains disjoint from the frozen Stage-2 in-domain IID headline split.

| Population | LogiQA | HellaSwag | Total | Source rows (1-based) | Use |
| --- | ---: | ---: | ---: | --- | --- |
| Recovered no-CoT wrong-argument source | 1,500 | 1,500 | 3,000 | 1--3,000 | Attested source |
| Frozen canonical prefix | 1,024 | 1,024 | 2,048 | 1--2,048 | Prefix/provenance proof |
| Stage-2 held-out IID | 100 | 100 | 200 | 2,049--2,248 | Withheld evaluation |
| **ACT-Max training selection** | **1,400** | **1,400** | **2,800** | **1--2,048; 2,249--3,000** | Training |

The held-out IID question-ID digest is
`499106c1c45cc422d8b231d17a0b87d6cd0636a843fc0c222cdb04bed0198ae1`;
all seven Stage-2 in-domain bias renderings use exactly that population.
The Stage-2 HLE population is separately verified to have no source-ID
overlap.  Thus ACT-Max trains on neither headline evaluation population.

## Immutable inputs

The materialized training file and proof manifest are content addressed:

- `artifacts/act-max-training-20260804/act-max-training-cec56e4d33531a9f997740850a654e7ceaf16b6c2d4830108861df903660cbd5.jsonl`
- `artifacts/act-max-training-20260804/act-max-training-manifest-18b8f70b8260c3d2deaf3345efcfddd41da3e9b7c61ed50d52664bbca56bd04c.json`

`experiments.act_max.selection` verifies the recovered source, its old
canonical 2,048-row prefix, the held-out IID reference, the Stage-2 manifest,
and both published ACT-Max files before training may initialise a model.
It is offline and makes no provider, model, or GPU call.

## Training contract

The factory-backed experiment definition is
`experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_act_max_from_base_20260804.yaml`.
It uses repaired native Qwen no-reasoning pair fields
(`unbiased_messages` versus `biased_messages`), strict full-reference suffix
alignment, the Qwen3.5 paired-backward preflight, ACT weight `5e-5`, and the
shared `1e-4` Adam learning rate.  It makes two full passes over 2,800 rows:
5,600 optimizer updates, with resumable state and checkpoints every 700
updates.  Two passes avoid confounding the 14-fold data increase with a
second, much larger optimization-budget increase.

The only supported topology profile is `single-gpu`.  On a host where the
frozen artifacts have been staged at the listed repository-relative paths:

```bash
CUDA_VISIBLE_DEVICES=<allocated-gpu> \
  bash infra/vastai/run_qwen35_act_max.sh --topology-profile single-gpu --dry-run
```

The launcher replays the full offline proof and prints the resolved training
plan.  Replacing `--dry-run` with `--yes` is the explicit, separate GPU launch
step; this contract does not rent or provision a machine.
