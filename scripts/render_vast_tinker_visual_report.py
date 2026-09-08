#!/usr/bin/env python3
"""Render a figure-led, self-contained Vast/Tinker experiment report.

The source CSVs and plots remain auditable artifacts, while the generated HTML
embeds every displayed image as a data URI so the primary report is a single
file rather than prose separated from a figure directory.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import mimetypes
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--vast-figures", type=Path, required=True)
    parser.add_argument("--tinker-figures", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "—"
    number = float(value)
    if number and abs(number) < 0.001:
        return f"{number:.3g}"
    return f"{number:.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def figure(asset: str, title: str, caption: str, interpretation: str, *, wide: bool = False) -> str:
    css_class = "figure wide" if wide else "figure"
    return f"""
<figure class=\"{css_class}\">
  <a href=\"visual-report-assets/{asset}\"><img src=\"visual-report-assets/{asset}\" alt=\"{html.escape(title)}\"></a>
  <figcaption><strong>{html.escape(title)}</strong> {html.escape(caption)}</figcaption>
  <div class=\"interpretation\"><strong>Interpretation.</strong> {html.escape(interpretation)}</div>
</figure>
""".strip()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def main() -> None:
    args = parse_args()
    root = args.report_root
    assets = root / "visual-report-assets"
    assets.mkdir(parents=True, exist_ok=True)

    source_figures = {
        "vast-tbsr-four-column.png": args.vast_figures / "ood-tbsr-four-column.png",
        "vast-bvr-four-column.png": args.vast_figures / "ood-bias-verbalised-four-column.png",
        "vast-tbsr-iid-by-bias.png": args.vast_figures / "ood-tbsr-iid-by-bias.png",
        "vast-tbsr-hle-by-bias.png": args.vast_figures / "ood-tbsr-hle-by-bias.png",
        "vast-bvr-iid-by-bias.png": args.vast_figures / "ood-bias-verbalised-iid-by-bias.png",
        "vast-bvr-hle-by-bias.png": args.vast_figures / "ood-bias-verbalised-hle-by-bias.png",
        "tinker-hle-pro-bsr-by-lr.png": args.tinker_figures / "tinker-hle-da-pro-bsr-by-learning-rate.png",
        "tinker-hle-bvr-by-lr.png": args.tinker_figures / "tinker-hle-da-bvr-toward-by-learning-rate.png",
        "vast-supervised-losses.png": root / "figures" / "vast-supervised-losses.png",
        "vast-rmct-training.png": root / "figures" / "vast-rmct-loss-reward-kl.png",
        "vast-opct-training.png": root / "figures" / "vast-opct-training.png",
        "tinker-bct-training.png": root / "figures" / "tinker-bct-nll-by-learning-rate.png",
        "tinker-rmct-training.png": root / "figures" / "tinker-rmct-reward-kl-by-learning-rate.png",
    }
    missing = [str(path) for path in source_figures.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing required figures:\n" + "\n".join(missing))
    for name, source in source_figures.items():
        shutil.copy2(source, assets / name)

    runs = read_csv(root / "runs.csv")
    vast_eval = read_csv(root / "vast-evaluation-headlines.csv")
    tinker_eval = read_csv(root / "tinker-evaluation-heldout.csv")
    binned = read_csv(root / "optimizer-binned-history.csv")

    vast_lookup = {(row["condition"], row["column"]): row for row in vast_eval}

    def vast_value(condition: str, regime: str, metric: str) -> str:
        return vast_lookup[(condition, regime)][metric]

    tinker_lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in tinker_eval:
        tinker_lookup[(row["model"], row["condition_label"], row["learning_rate"], row["metric"])] = row

    def tinker_value(model: str, condition: str, lr: str, metric: str) -> dict[str, str]:
        return tinker_lookup[(model, condition, lr, metric)]

    binned_lookup: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in binned:
        run_id = row["run_id"]
        for metric in ("train/loss", "train/nll", "train/train/loss", "train/consistency_gap_1", "train/kl_policy_base", "train/teacher_kl"):
            value = row.get(metric)
            if value not in (None, ""):
                binned_lookup[(run_id, metric)].append(float(value))

    def endpoint(run_id: str, metric: str) -> float:
        return binned_lookup[(run_id, metric)][-1]

    vast_config_rows = []
    for row in runs:
        if row["platform"] != "Vast":
            continue
        condition = row["method"] + (" control" if truthy(row["control"]) else "")
        vast_config_rows.append([
            condition,
            row["unique_training_units"],
            row["training_unit_presentations"],
            f"{row['configured_batch']} × {row['gradient_accumulation']}",
            row["effective_optimizer_batch"],
            row["applied_optimizer_updates"],
            row["rollout_group"],
            row["generated_rollouts_per_update"],
            row["forward_microbatch_cap"],
        ])

    tinker_config_rows = []
    seen_tinker: set[tuple[str, str, str]] = set()
    for row in runs:
        if row["platform"] != "Tinker" or truthy(row["control"]):
            continue
        key = (row["model"], row["method"], row["learning_rate"])
        if key in seen_tinker:
            continue
        seen_tinker.add(key)
        tinker_config_rows.append([
            row["model"], row["method"], row["learning_rate"], row["unique_training_units"],
            row["effective_optimizer_batch"], row["applied_optimizer_updates"], row["rollout_group"],
            row["generated_rollouts_per_update"], row["forward_microbatch_cap"],
        ])

    main_tinker_rows = []
    for model in ("gpt-oss-20b", "llama31-8b"):
        for condition in ("BCT (Tinker)", "RMCT (Tinker)"):
            for lr in ("1e-4", "2.86e-4", "5e-4"):
                bsr = tinker_value(model, condition, lr, "paper_pro_bsr")
                bvr = tinker_value(model, condition, lr, "paper_bvr_toward")
                main_tinker_rows.append([
                    model, condition.replace(" (Tinker)", ""), lr,
                    pct(bsr["mean"]), bsr["n_scored"], bsr["significance"] or "—",
                    pct(bvr["mean"]), bvr["n_scored"], bvr["significance"] or "—",
                ])

    wandb_rows = []
    for row in runs:
        if row["platform"] == "Vast":
            label = row["method"] + (" control" if truthy(row["control"]) else "")
            wandb_rows.append([label, f"[{row['run_name']}]({row['run_url']})"])

    figures = {
        "vast_tbsr": figure(
            "vast-tbsr-four-column.png",
            "Figure 1. Vast Qwen3.5-9B towards-bias switch rate.",
            "Lower is better. The four panels progressively hold out neither factor, the dataset, the bias, and both dataset and bias. Error bars are 95% question-cluster bootstrap intervals; hatched bars are matched controls.",
            f"BCT is strongest on the fully held-out dataset+bias regime ({pct(vast_value('bct', 'held_out_dataset_and_bias', 'tbsr'))}), followed by RMCT ({pct(vast_value('rmct', 'held_out_dataset_and_bias', 'tbsr'))}). RMCT is much weaker IID ({pct(vast_value('rmct', 'iid', 'tbsr'))}) than the supervised methods, consistent with only 64 unique questions and 16 updates, but it still improves substantially over base on a held-out dataset ({pct(vast_value('rmct', 'held_out_dataset', 'tbsr'))} versus {pct(vast_value('untrained', 'held_out_dataset', 'tbsr'))}).",
            wide=True,
        ),
        "vast_bvr": figure(
            "vast-bvr-four-column.png",
            "Figure 2. Vast Qwen3.5-9B explicit bias acknowledgement.",
            "Higher is better here: the desired property is to preserve the model's ability to verbalise the prompt bias rather than suppressing acknowledgement as a side effect of reducing TBSR. Bars use Luna judgements and 95% question-cluster bootstrap intervals.",
            f"RMCT preserves acknowledgement almost completely: {pct(vast_value('rmct', 'iid', 'bias_verbalised'))} IID and {pct(vast_value('rmct', 'held_out_dataset_and_bias', 'bias_verbalised'))} when both factors are held out. BCT obtains lower TBSR but acknowledgement falls to {pct(vast_value('bct', 'held_out_dataset_and_bias', 'bias_verbalised'))}; OPCT falls to {pct(vast_value('opct', 'held_out_dataset_and_bias', 'bias_verbalised'))}. This is the central efficacy-versus-transparency trade-off.",
            wide=True,
        ),
        "tinker_bsr": figure(
            "tinker-hle-pro-bsr-by-lr.png",
            "Figure 3. Historical Tinker HLE pro-BSR by learning rate.",
            "Lower is better. Rows separate GPT-OSS-20B and Llama-3.1-8B-Instruct; columns separate 1e-4, 2.86e-4 and 5e-4. Stars test each bar against the same-model base. These are paper pro-BSR values, not the conditional Vast TBSR estimand.",
            "The preferred learning rate is model- and method-dependent. GPT-OSS RMCT reaches its lowest held-out mean at 5e-4, while GPT-OSS BCT is best at 2.86e-4. Llama RMCT improves most at 2.86e-4 or 5e-4; 1e-4 is essentially unchanged from base. A single universal learning rate is therefore convenient, but not empirically optimal across these models.",
            wide=True,
        ),
        "tinker_bvr": figure(
            "tinker-hle-bvr-by-lr.png",
            "Figure 4. Historical Tinker switch-restricted bias verbalisation by learning rate.",
            "Higher is better. Unlike the Vast acknowledgement metric, this paper BVR is restricted to examples where the biased prompt switched the answer toward the biased option.",
            "RMCT generally preserves more verbalisation than BCT. On GPT-OSS, RMCT spans 58.8–69.2% across learning rates, versus 22.9–47.2% for BCT. On Llama, BCT falls to 9.2–17.4%, while RMCT reaches 34.5% at 2.86e-4. The historical runs therefore reproduce the same qualitative trade-off seen on Vast, despite using different models and estimands.",
            wide=True,
        ),
        "supervised": figure(
            "vast-supervised-losses.png",
            "Figure 5. Vast supervised-objective training losses.",
            "Curves are averaged within optimizer updates. ACT ran for 4,000 updates; AttCT and MLPCT each ran for 256; BCT and its control ran for 32.",
            f"The consistency-objective losses converge cleanly (final update means: ACT {num(endpoint('791a00cr', 'train/loss'))}, AttCT {num(endpoint('ovhfsbd5', 'train/loss'))}, MLPCT {num(endpoint('9luq24r0', 'train/loss'))}). BCT NLL remains noisy and closely tracks its control, which is expected because the behavioral distinction comes from target construction rather than a radically different loss scale.",
        ),
        "rmct_training": figure(
            "vast-rmct-training.png",
            "Figure 6. Vast RMCT reward, rate and KL diagnostics.",
            "The main and control conditions are shown for all 16 rollout batches. The plotted PPO loss uses Vast's mean-reduction schema.",
            f"The main condition has a real learning signal: its final consistency gap is {num(endpoint('d5wj3a3i', 'train/consistency_gap_1'))}, versus {num(endpoint('ectjp0kh', 'train/consistency_gap_1'))} for control, while policy-to-base KL stays small ({num(endpoint('d5wj3a3i', 'train/kl_policy_base'))}). But the gap remains large and volatile after only 16 updates, supporting the diagnosis that RMCT-64 is undertrained rather than broken. The near-zero pooled mean reward is expected from GRPO centering and is not evidence of absent signal.",
        ),
        "opct_training": figure(
            "vast-opct-training.png",
            "Figure 7. Vast OPCT training diagnostics.",
            "Each optimizer update aggregates 16 prompts with four policy responses per prompt. OPCT logs teacher KL/cross-entropy rather than an explicit scalar reward.",
            f"The objective converges strongly: the final update-mean loss is {num(endpoint('44tne7qw', 'train/loss'))} and teacher KL is {num(endpoint('44tne7qw', 'train/teacher_kl'))}, with no skipped rollouts. This shows that OPCT optimization worked technically; its weaker fully OOD behavioral result is therefore an objective/generalisation finding, not evidence of a failed training run.",
        ),
        "tinker_bct_training": figure(
            "tinker-bct-training.png",
            "Figure 8. Tinker BCT NLL histories by model and learning rate.",
            "Solid lines are main runs and dashed lines are controls. Tinker exposes logical batch 128 but not its internal physical microbatching.",
            "The short 32-update histories are noisy, especially for Llama, and final NLL does not rank the behavioral learning rates reliably. NLL should be used as a health check rather than as a proxy for TBSR or verbalisation.",
        ),
        "tinker_rmct_training": figure(
            "tinker-rmct-training.png",
            "Figure 9. Tinker RMCT consistency-gap and policy-KL histories.",
            "Each main/control condition uses 64 unique questions, batch four and 128 reference plus 128 cued rollouts per question. Solid lines are main runs; dashed lines are controls.",
            "Main runs separate clearly from their controls in consistency gap, while higher learning rates generally produce larger policy-to-base KL. The histories are only 16 batches long, so endpoint comparisons are noisy; this reinforces using behavioral evaluation and multiple learning rates rather than selecting from training loss alone.",
        ),
        "iid_detail": figure(
            "vast-tbsr-iid-by-bias.png",
            "Appendix Figure A1. Vast in-domain TBSR by bias.",
            "This expands the pooled IID result into individual biases and retains the plotting pipeline's uncertainty and denominator annotations. The current Vast analysis does not apply significance tests.",
            "Per-bias variation is material; pooled results should not be read as uniform effects. The wrong-argument cell is the actual training bias, while the other biases test within-dataset transfer.",
            wide=True,
        ),
        "hle_detail": figure(
            "vast-tbsr-hle-by-bias.png",
            "Appendix Figure A2. Vast HLE TBSR by bias.",
            "This expands held-out-dataset behavior into the training bias and five held-out bias types.",
            "The method ordering changes by bias, explaining why the pooled held-out-dataset+bias interval is wider than IID and why a single aggregate should be accompanied by the disaggregated view.",
            wide=True,
        ),
        "iid_bvr_detail": figure(
            "vast-bvr-iid-by-bias.png",
            "Appendix Figure A3. Vast in-domain acknowledgement by bias.",
            "Higher is better; the figure shows which methods preserve explicit acknowledgement for each bias type.",
            "The transparency penalty of BCT/OPCT is not confined to the training bias, whereas RMCT and its control remain close to ceiling across biases.",
            wide=True,
        ),
        "hle_bvr_detail": figure(
            "vast-bvr-hle-by-bias.png",
            "Appendix Figure A4. Vast HLE acknowledgement by bias.",
            "This is the per-bias counterpart to Figure 2 for the held-out HLE dataset.",
            "RMCT's high pooled acknowledgement is broad rather than driven by one bias. The supervised methods show larger bias-specific drops, which matters when interpreting their lower pooled TBSR.",
            wide=True,
        ),
    }

    markdown = f"""# Consistency training on Qwen3.5-9B

## Visual results report with historical Tinker comparison

<div class=\"meta\">Generated 5 August 2026 from the canonical W&B histories and completed evaluation artifacts. Lower TBSR/pro-BSR is better; higher bias verbalisation is better.</div>

## Executive summary

<div class=\"summary-grid\">
<div class=\"summary-card\"><strong>Strongest fully OOD TBSR</strong><br>BCT: {pct(vast_value('bct', 'held_out_dataset_and_bias', 'tbsr'))}. RMCT-64: {pct(vast_value('rmct', 'held_out_dataset_and_bias', 'tbsr'))}. Base: {pct(vast_value('untrained', 'held_out_dataset_and_bias', 'tbsr'))}.</div>
<div class=\"summary-card\"><strong>Best transparency retention</strong><br>RMCT preserves {pct(vast_value('rmct', 'held_out_dataset_and_bias', 'bias_verbalised'))} acknowledgement fully OOD, versus BCT's {pct(vast_value('bct', 'held_out_dataset_and_bias', 'bias_verbalised'))}.</div>
<div class=\"summary-card\"><strong>RMCT diagnosis</strong><br>The run is functioning but undertrained: only 64 unique questions, 16 updates, and a final training consistency gap of {num(endpoint('d5wj3a3i', 'train/consistency_gap_1'))}.</div>
<div class=\"summary-card\"><strong>Tinker comparison</strong><br>Historical results support RMCT's transparency advantage, but show that the best learning rate changes by model and method.</div>
</div>

The core result is a trade-off, not a single winner. BCT gives the lowest fully OOD switch rate, but suppresses explicit bias acknowledgement. RMCT-64 reduces switching less strongly while preserving acknowledgement almost completely. Because the present RMCT checkpoint saw far less unique training data than the supervised methods, RMCT-256 remains a scientifically motivated data-scaling condition rather than a repair for a failed run.

## 1. Vast behavioral results

{figures['vast_tbsr']}

{figures['vast_bvr']}

## 2. Historical Tinker comparison

The Tinker figures use GPT-OSS-20B and Llama-3.1-8B-Instruct, while the Vast figures use Qwen3.5-9B. They also use different behavioral estimands: paper pro-BSR and switch-restricted BVR on Tinker, versus conditional TBSR and overall Luna acknowledgement on Vast. The comparison is therefore qualitative and within-platform; the values should not be pooled into one bar chart.

{figures['tinker_bsr']}

{figures['tinker_bvr']}

### Tinker held-out means by learning rate

{md_table(['Model', 'Method', 'LR', 'pro-BSR ↓', 'n', 'sig.', 'BVR ↑', 'n', 'sig.'], main_tinker_rows)}

## 3. What the training logs show

{figures['supervised']}

{figures['rmct_training']}

{figures['opct_training']}

{figures['tinker_bct_training']}

{figures['tinker_rmct_training']}

## 4. Interpretation

1. **The repaired training stack is working.** Supervised losses converge; RMCT main separates from control with low KL; OPCT loss and teacher KL converge without skipped rollouts.
2. **Low training loss is not sufficient.** OPCT converges technically but does not lead the fully OOD TBSR comparison. Tinker BCT NLL similarly does not identify the best behavioral learning rate.
3. **RMCT-64 is data/update limited.** It receives only 64 unique questions and 16 updates, versus 2,048–4,096 training records and 32–4,000 updates for the supervised conditions. Its remaining training consistency gap and its weaker IID result both point in the same direction.
4. **RMCT preserves transparency.** This appears in both the Vast acknowledgement result and the historical Tinker switch-restricted BVR result.
5. **Learning-rate conclusions are model-specific.** GPT-OSS and Llama select different points on the TBSR/BVR trade-off, so the 1e-4 common-rate Vast comparison is controlled but should not be mistaken for a per-method optimum.

## 5. Executed configurations

### Vast Qwen3.5-9B

{md_table(['Condition', 'Unique units', 'Presentations', 'Batch × accum', 'Effective batch', 'Updates', 'Group/rollouts', 'Generated/update', 'Physical microbatch cap'], vast_config_rows)}

The semantic batch and physical microbatch are different concepts. Vast dynamically packs at most eight datums and a token ceiling per forward pass; this does not change the effective optimizer batch. RMCT generates 768 responses per rollout batch and normally forwards/backwards 384 selected response datums. OPCT generates 64 responses per optimizer update. Target-logprob computations use a separate 2,048-token chunk where applicable.

### Historical Tinker runs

{md_table(['Model', 'Method', 'LR', 'Training units', 'Effective batch', 'Observed updates', 'Group/rollouts', 'Generated/update', 'Physical microbatch'], tinker_config_rows)}

Tinker exposes logical batches but not its internal physical microbatching, sharding, rematerialisation or optimizer-state placement. GPT-OSS BCT records `total_steps=31` in config but has 32 actual logged updates because its 4,009 examples include a final partial batch.

## 6. Per-bias appendix

{figures['iid_detail']}

{figures['hle_detail']}

{figures['iid_bvr_detail']}

{figures['hle_bvr_detail']}

## 7. Runs, metrics and provenance

### Canonical Vast W&B runs

{md_table(['Condition', 'W&B run'], wandb_rows)}

The full Tinker W&B project/run matrix is in [the audited text appendix](REPORT.md). Exact application metrics are in [metric-history-long.csv](metric-history-long.csv), optimizer-update means in [optimizer-binned-history.csv](optimizer-binned-history.csv), metric summaries in [metric-summary.csv](metric-summary.csv), and sampled machine telemetry in [system-metric-history-wide.csv](system-metric-history-wide.csv). Vast and Tinker behavioral extracts remain separate in [vast-evaluation-headlines.csv](vast-evaluation-headlines.csv) and [tinker-evaluation-heldout.csv](tinker-evaluation-heldout.csv).

Failed predecessors and pre-repair checkpoints are excluded from the canonical curves. The project names containing `isambard` reflect reused topology configuration names; the Qwen runs reported here belong to the monitored Vast execution lineage.
"""

    md_path = root / "VISUAL_REPORT.md"
    md_path.write_text(markdown.strip() + "\n")

    try:
        import mistune
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("mistune is required to render the self-contained HTML report") from exc

    renderer = mistune.create_markdown(escape=False, plugins=["table"])
    body = renderer(markdown)
    for name in source_figures:
        body = body.replace(f'visual-report-assets/{name}', data_uri(assets / name))

    css = """
:root { --ink:#17202a; --muted:#5b6573; --line:#d8dee7; --blue:#2b6cb0; --green:#2f855a; --paper:#fff; --soft:#f5f7fa; }
* { box-sizing:border-box; }
body { margin:0; background:#eef1f5; color:var(--ink); font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1320px; margin:0 auto; background:var(--paper); padding:54px 64px 80px; box-shadow:0 0 30px rgba(0,0,0,.07); }
h1 { font-size:2.4rem; margin:0 0 .2rem; letter-spacing:-.03em; }
h2 { margin-top:3.2rem; padding-top:.7rem; border-top:2px solid var(--ink); font-size:1.55rem; }
h3 { margin-top:2rem; }
.meta { color:var(--muted); margin-bottom:2rem; }
.summary-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:24px 0; }
.summary-card { border:1px solid var(--line); border-left:5px solid var(--blue); border-radius:8px; padding:15px 17px; background:var(--soft); }
figure.figure { margin:30px 0 42px; padding:18px; border:1px solid var(--line); border-radius:10px; background:#fff; box-shadow:0 3px 13px rgba(16,24,40,.06); }
figure img { display:block; max-width:100%; height:auto; margin:0 auto 14px; }
figcaption { font-size:.98rem; color:#303846; }
.interpretation { margin-top:12px; padding:12px 14px; background:#edf7f1; border-left:4px solid var(--green); border-radius:4px; }
table { border-collapse:collapse; width:100%; margin:18px 0 30px; font-size:.87rem; display:block; overflow-x:auto; }
th,td { border:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; white-space:nowrap; }
th { background:#edf1f7; position:sticky; top:0; }
code { background:#edf1f7; padding:.12rem .28rem; border-radius:4px; }
a { color:#1f5f99; }
ol li { margin:.5rem 0; }
@media (max-width:800px) { main{padding:28px 20px}.summary-grid{grid-template-columns:1fr}h1{font-size:1.9rem} }
@media print { body{background:#fff}main{box-shadow:none;max-width:none;padding:18mm}figure{break-inside:avoid}h2{break-before:auto}.summary-grid{grid-template-columns:1fr 1fr}a{color:inherit;text-decoration:none} }
"""
    document = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Consistency training visual report</title><style>{css}</style></head><body><main>{body}</main></body></html>"
    html_path = root / "VISUAL_REPORT.html"
    html_path.write_text(document)

    hash_lines = []
    for path in [md_path, html_path, *sorted(assets.glob("*.png"))]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {path.relative_to(root)}")
    (root / "VISUAL_REPORT_SHA256SUMS").write_text("\n".join(hash_lines) + "\n")


if __name__ == "__main__":
    main()
