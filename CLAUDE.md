# Project Instructions

Successor repo to https://github.com/sohaibimran7/cot-transparency (the
published codebase behind arXiv:2606.02211). **This repo is legacy-free by
construction**: further experiments and papers happen here; the old repo is
the frozen paper artifact. Never add backwards-compatibility shims — if old
code is needed, port it cleanly or leave it behind.

## Training & Eval Runs (hard rule)
- **ALWAYS** show the user the exact command and parameters for every training
  run and eval run **before** executing. Present multi-step commands upfront.
- **Do not execute** until the user explicitly approves.

## What this repo is
Anti-sycophancy / eval-awareness **consistency training** (BCT = SFT, RLCT = RL)
on Tinker or self-hosted GPUs, evaluated with the published
[mcq-bias](https://github.com/sohaibimran7/mcq-bias) Inspect eval.

## Environment
`uv` throughout. `requirements.txt` is the dependency source of truth.

```bash
uv venv
uv pip install -r requirements.txt   # includes the mcq-bias git pin
uv pip install -e . --no-deps
```

API keys in `.env` (gitignored): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, Tinker key. mcq-bias data root: `$MCQ_BIAS_DATA_DIR`
(default `~/.cache/mcq_bias`) — no data is committed in this repo.

## Where things live
- `ctm/` — the package: `core/` (rewards, advantages/SNR, configs, types),
  `backends/` (tinker + local GPU seam), `training/` (RL + SFT loops, rollout
  logs, manifests), `settings/` (sycophancy, eval_awareness), `evals/`.
- `scripts/tinker_training/` — `train_rl.py`, `train_sft.py`,
  `test_rl_training.py`, `experiment_configs/`.
- `docs/` — research logs (SNR/RL experiments continue from
  `docs/eval-awareness/RESEARCH_LOG.md`).
- Eval tasks/scorers come from the **mcq-bias package** (pinned in
  requirements.txt; co-develop with `uv pip install -e /Users/work/mcq-bias --no-deps`).

## Port queue (from the old repo — the only things not yet here)
1. **Training-data generation**: BCT/VFT datapoints from mcq-bias frozen sets
   (old scripts built from `dataset_dumps`; frozen rows carry
   `biased_messages`/`unbiased_messages` — thin adapter needed).
2. **Tinker inference/sampling layer**: `TinkerSamplingClient` etc. still live
   in the old repo's `cot_transparency/apis/tinker/inference.py` (not a shim);
   port into `ctm/backends/` without the legacy data models.
3. **Checkpoint-eval bridge**: old `run_experiment.py` shells out to the legacy
   `run_tinker_evals`; port to drive `python -m mcq_bias` / inspect directly.
4. **Experiment analysis**: cross-checkpoint aggregation/plots over mcq-bias
   eval logs (per-run switch metrics are already in-log; the old
   `visualize_results` BIR machinery stays behind).
5. Skills: only `train-tinker` and `model-apis` were carried over; port
   run-evals/analyze-evals/generate-* once 1–4 land.

## Tests & lint
```bash
pytest            # offline subset (network/gpu/tinker markers excluded)
uv run black . && uv run ruff check .
```
