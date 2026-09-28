# Reference-paper reproduction boundary

This document records which published results can be expressed by the current
training, evaluation, and experiment interfaces. It does not claim that a
result has been reproduced until the corresponding paid run and statistical
comparison have completed.

The three reference papers are:

- *Rate Matching Consistency Training* (`2606.02211`);
- *Consistency Training Helps Stop Sycophancy and Jailbreaks* (`2510.27062`);
- *Consistency Training Across the Transformer Stack* (`2606.05817`).

## Implemented reproduction paths

### Rate Matching Consistency Training

The repository can express the paper's RMCT and BCT conditions, matched
controls, three learning rates, exact rank-8/alpha-16 LoRA component selection,
128 rollouts per side, per-datapoint GRPO normalization, fixed-reference KL,
and zero anchor weight. The pinned HLE exporter provides the paper's 513
text-only multiple-choice evaluation rows.

The upstream `mcq_bias` scorer provides conditional `towards_bias_switch` and
`away_from_bias_switch` rates. Questions on which the corresponding switch was
not possible are excluded from that metric's denominator. This is the metric
definition used by this repository; it does not preserve an additional
unconditional directional switch metric. The CTM reducer can pool repeated
learning-rate directories using Inspect-supplied standard errors, compute
sample-count-weighted held-out-bias summaries, and restrict bias verbalisation to towards-switch
examples or to all questions that switched in either direction. It can also
report unconditional bias verbalisation over every successfully graded
response, away/total switch rates, and shared-unbiased accuracy. Two-proportion
tests against a named baseline are stored in the chart-ready JSON, alongside
the pooled estimate, denominator, replicate count, provenance, and standard
error. The publication renderer displays these supplied values without doing
statistical work.

The complete GPT-OSS-20B comparison, including matched controls and bias
verbalisation, is specified in
[`../rmct_paper_vast_more_methods/experiment.yaml`](../rmct_paper_vast_more_methods/experiment.yaml).
Its [runbook](../rmct_paper_vast_more_methods/README.md) records the reproduction boundary,
resolved workload, and execution commands.

### Consistency Training Helps Stop Sycophancy and Jailbreaks

The paper's Base, BCT, and ACT conditions are represented for both phenomena;
the same graph also adds RMCT, AttCT, MLPCT, and OPCT as explicitly labeled
repository extensions. The local backend can update only
self-attention parameters under full-parameter fine-tuning, retain the initial
model for ACT targets, select the paper's layer subsets, evaluate both LoRA and
full-weight checkpoints, and calculate the sycophancy/clean-accuracy harmonic
mean. Fresh BCT targets are sampled by CTM before optimizer initialization;
stale response files can also be supplied directly.

This covers the ACT and BCT points in the sycophancy comparison, the stale-data
comparison, and the ACT layer-update ablation when the paper's exact source
rows and evaluation tasks are supplied.

The paper-specific local-source adapters, safety/sycophancy task factories,
vulnerability-filter artifact DAG, validation-only selector, and clustered
bootstrap analysis now live under
[`irpan_2510_27062/`](irpan_2510_27062/). They make the reported dataset roles
executable while preserving the unresolved reproduction boundary: the paper
does not supply exact source versions/splits, wrapper and judge prompts,
generation settings, or complete training hyperparameters. Those values are
versioned as reconstruction choices rather than silently inferred.

The same role-bound pair artifact also supports the repository's RMCT, AttCT,
MLPCT, and OPCT implementations in both sycophancy and jailbreak settings.
Those four are clearly labeled repository extensions, not conditions from the
paper. The checked full/smoke specifications and exact method-to-artifact wiring
are documented in the linked runbook.

### Consistency Training Across the Transformer Stack

ACT, JSD-AttCT, MLPCT, and BCT are implemented. The Table 2 MLPCT axes are
expressible: exact Q/V, Q/K/V, or Q/K/V/O LoRA targets; rank and alpha;
dropout; cosine, smooth-L1, MSE, or normalized-MSE distance; layer selection;
layer weighting; normalization; batch size; and gradient accumulation. The
non-interleaved Table 3 JSD-AttCT axes are also expressible. Named checkpoint
outputs support sequential method chains.

These capabilities cover the method rows and non-interleaving ablations when
the paper's prompt pairs and official evaluation task factories are supplied.
The new datasets introduced by this paper are intentionally outside the scope
of this audit.

## Results that still require method code

The following are not dataset gaps and cannot be reproduced by changing YAML
alone:

- DPO and negative-preference baselines from `2510.27062`;
- a simultaneous ACT+BCT objective from `2510.27062`;
- Gemini fine-tuning and Gemini internal-activation training;
- activation patching and the causal intervention experiment from
  `2510.27062`;
- the cross-objective loss-surface analysis that evaluates ACT examples under
  the BCT objective and BCT examples under the ACT objective;
- the six non-JSD candidates in the AttCT loss-function ablation from
  `2606.05817`;
- simultaneous diagnostic logging of all four consistency losses on one set of
  forward outputs;
- KL/instruction interleaving inside an AttCT run;
- interleaved pairs of different consistency objectives; sequential chaining
  is supported, but it is not equivalent;
- composite MLPCT+BCT+KL and prefill-position distribution losses; and
- the transformer-stack paper's mechanistic analyses, including Q/K/V
  diagnostics, probes, head interventions, mediation, and activation patching.

## Reporting functions that remain manual

The RMCT comparison uses a benchmark-owned experiment factory to expand its
learning-rate matrix from a concise YAML specification. Seed sweeps are not yet
part of that factory. Its bootstrap confidence intervals and best-hyperparameter
selection remain manual. The Irpan et al. adapter has its own validation-only
selection and clustered bootstrap implementation; this does not retroactively
define the RMCT paper's reporting procedure.
