# Clean Qwen RMCT restart contribution

Preparation only; no submission function. Use only after selective integration
into the canonical PR. `prepare.py` pins the historical argv artifact, removes
all contaminated-resume flags, starts fresh segment0/optimizer0, substitutes
explicit canonical deployment paths, and records unmet readiness checks. It
does not manufacture a preflight receipt or claim a commit was incorporated.

Run `python experiments/rmct_restart_20260928/prepare.py --help` for required
paths. Baseline is tracked `fixtures/original_recipe.json`, containing only the
original argv (no responses or credentials), SHA256
b9ed4034be9ccb4d01006bc7d5a1864c2ed17951ea1ed3f7c99bce81ad54c42d.
Its source artifact was `artifacts/rmct-tbsr-continuation-20260927/original-step352-command.json`
SHA256 f3c6c63d0d16abb4e8c70ac37a8e33b2490ba9db6cb05a244a56a776aeb568cd;
the fixture preserves that argv exactly, omitting unrelated receipt metadata.
The `--attestation` must be NEW evidence from the incorporated runtime, not
the old parity receipt. The integration owner must wire explicit thinking
using its finalized CLI/runtime API (do not invent a flag from this draft).

Important sequencing: the frozen setting enforces32 QIDs per segment, i.e.
16 optimizer updates at batch2. The first command therefore trains0–16 only.
The integrated orchestrator must then strictly resume ONLY the new clean
lineage through segments1–3, stop at64, and validate before advancing. Preserve
the original segmented order; do not change n_datapoints to128 to bypass the
setting contract. Further64-update windows require completed, verified scores.

New evaluation limits20480 for LogiQA/HellaSwag require a NEW contract and
patience history. Do not reuse historical65536 scores. Preserve200 QIDs/600
prompts if provenance clearance confirms that population. Missing/truncated
answers are invalid. TBSR only, interval64, patience2, min_delta0, earliest ties.

Before training, the scheduled process must verify exact incorporated commit,
clean tracked files, source/config hashes, environment/dependency pins, tokenizer
and model identity, imported core/parser module paths and regression results.
Editable installs/PYTHONPATH must not silently resolve another worktree. Save
actual observed evidence before optimizer work. A generated plan is not this
evidence. Broad audit and integrated runtime checks remain mandatory.

No other method or Gemma launch is implemented here. No old continuation
checkpoints, optimizer state, validation history or source attestations may
seed the fresh Qwen run.
