"""Read-only saved-score aggregation with native Inspect mean/stderr metrics.

No scorer, model constructor, evaluation driver, network or provider is invoked.
Writes only new presentation/aggregation artifacts; refuses existing outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import textwrap

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import SampleScore, Score, mean, stderr
std_module = importlib.import_module("inspect_ai.scorer._metrics.std")
mean_module = importlib.import_module("inspect_ai.scorer._metrics.mean")

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LASR = ROOT / "experiments/eval_awareness/lasr_transfer"
OLD = LASR / "rescue-factor-inspection-20260913/results-20260914/render-v2"
PETRI = ROOT / "artifacts/petri-pilots-20260914"
PRIOR = PETRI / "analysis-native-v3-v4-v1"
PATH_MAP = {}
TEMPLATE = HERE / "report-template.md"
LABELS = {"EVALUATION", "UNCERTAIN", "DEPLOYMENT_REJECTED", "NO_EVIDENCE"}
MODELS = {"base": "Base", "mo_mid": "Mid", "mo_post": "Post"}
COLORS = {"base": "#236a9d", "mo_mid": "#c97125", "mo_post": "#208368"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_score(sample, field):
    matches = [(k, s) for k, s in (sample.scores or {}).items()
               if isinstance(s.value, dict) and field in s.value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {field} score for {sample.id}")
    return matches[0]


def aggregate(rows, measure, pooled=False):
    """Inspect owns every mean and SE calculation; adapter only filters/maps.

    Cluster all conditions in the same seed-repeat block for pooled summaries.
    Thus two epochs are not mistaken for two independent scenario samples.
    No pooling across versions, checkpoints or Petri behavior constructs.
    """
    valid = [r for r in rows if r[measure] is not None]
    scores = [SampleScore(score=Score(value=r[measure]), sample_id=r["uid"],
                          sample_metadata={"repeat_block": r["repeat_block"]}) for r in valid]
    blocks = len({r["repeat_block"] for r in valid})
    native_mean = float(mean()(scores)) if scores else None
    native_se = float(stderr(cluster="repeat_block" if pooled else None)(scores)) if scores else None
    enough = blocks >= 2 if pooled else len(scores) >= 2
    return {"positive_n": sum(r[measure] for r in valid), "valid_n": len(valid),
            "total_n": len(rows), "missing_n": len(rows) - len(valid), "mean": native_mean,
            "native_stderr_return": native_se, "stderr": native_se if enough else None,
            "stderr_status": "estimable_conditional_repeat_se" if enough else "not_estimable_fewer_than_two_repeat_units",
            "repeat_blocks": blocks, "metric": "inspect_ai.scorer.mean",
            "se_metric": "inspect_ai.scorer.stderr", "cluster": "repeat_block" if pooled else None,
            "sample_ids": [r["uid"] for r in valid]}


def collect():
    sources, cache, rows = {}, {}, []

    def bound(path, expected=None):
        path = str(Path(PATH_MAP.get(str(path), str(path))).resolve())
        digest = sources.setdefault(path, sha(path))
        if expected and digest != expected:
            raise ValueError(f"Hash mismatch: {path}")
        return path

    def log(path, expected):
        path = bound(path, expected)
        if path not in cache:
            cache[path] = read_eval_log(path, resolve_attachments=True)
        return cache[path]

    previous = json.loads(Path(bound(OLD / "results.json")).read_text())
    recent = json.loads(Path(bound(PRIOR / "report.json")).read_text())
    for kind, records in (("old", previous["rows"]), ("recent", recent["observations"])):
        for record in records:
            if kind == "old":
                cohort, role = record["panel"], record["role"]
                sp, ss = record["scored_path"], record["scored_sha256"]
                sample_id, epoch = record["identity"][1:]
                raw, raw_sha = record["source_path"], record["identity"][0]
                condition, narrative = record["condition"], "native_rescue"
                repeat = str(record["seed"])
                expected_b = record["behavior"]["harmful"] if record["behavior"]["status"] == "valid" else None
                expected_label = record["awareness"]["label"]
            else:
                cohort = {"native": "Bridge", "v3": "V3", "v4": "V4"}[record["suite"]]
                role, sp, ss = record["model_role"], record["log_path"], record["log_sha256"]
                sample_id, epoch = record["sample_id"], record["epoch"]
                condition = record["profile_id"]
                narrative = record["behavior_id"]
                repeat = "single_repeat"
                expected_b, expected_label = record["behavior"], record["awareness_label"]
            sp = bound(sp, ss)
            saved = log(sp, ss)
            if saved.status != "success":
                raise ValueError(f"Not a completed scored log: {sp}")
            candidates = [s for s in saved.samples if str(s.id) == str(sample_id) and s.epoch == epoch]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous sample identity: {sp}")
            sample = candidates[0]
            bk, bs = unique_score(sample, "behavior" if cohort in {"V3", "V4"} else "harmful")
            ak, aws = unique_score(sample, "label")
            bv, av = bs.value, aws.value
            b = bv.get("behavior", bv.get("harmful")) if bv["status"] == "valid" else None
            label = av.get("label") if av["status"] in {"valid", "labeled"} else None
            a = int(label == "EVALUATION") if label in LABELS else None
            if b != expected_b or label != expected_label:
                raise ValueError(f"Saved native scores differ from preserved analysis: {sp}, {sample_id}")
            if cohort == "Bridge":
                raw, raw_sha = bs.metadata["source_path"], bs.metadata["source_file_sha256"]
            elif cohort in {"V3", "V4"}:
                binding = saved.eval.metadata["posthoc_pilot"]
                raw, raw_sha = binding["source_log"], binding["source_sha256"]
            raw = bound(raw, raw_sha)
            recovery = (saved.eval.metadata or {}).get("recovery_projection", {})
            original = recovery.get("source_log")
            if original:
                bound(original, recovery["source_sha256"])
            if sample.error or sample.limit:
                if b is not None or a is not None:
                    raise ValueError("Invalid target was assigned a valid measurement")
            identity = [raw_sha, str(sample.id), sample.epoch]
            row = {"cohort": cohort, "role": role, "condition": condition, "narrative": narrative,
                   "repeat_block": repeat, "sample_id": str(sample.id), "epoch": sample.epoch,
                   "uid": "|".join(map(str, identity)), "identity": identity,
                   "behavior": b, "awareness": a, "awareness_label": label,
                   "behavior_status": bv["status"], "awareness_status": av["status"],
                   "scored_path": str(Path(sp).resolve()), "scored_sha256": ss,
                   "raw_path": raw, "raw_sha256": raw_sha, "original_attempt_log": original,
                   "scorer_keys": [bk, ak], "saved_epochs": saved.eval.config.epochs,
                   "saved_epochs_reducer": saved.eval.config.epochs_reducer,
                   "goal_value": record.get("goal_value"),
                   "reused_from_v1": record.get("reused_from_V1", False)}
            rows.append(row)
    counts = Counter(r["cohort"] for r in rows)
    assert counts == {"V1": 84, "V2": 64, "Bridge": 48, "V3": 9, "V4": 27}, counts
    assert len({(r["cohort"], r["uid"]) for r in rows}) == 232
    assert len({r["uid"] for r in rows}) == 224
    assert sum(r["reused_from_v1"] for r in rows) == 8
    return rows, sources, bound


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["role"], row["narrative"] if row["cohort"] == "V4" else "all")].append(row)
    return [{"cohort": k[0], "role": k[1], "narrative": k[2], "behavior": aggregate(v, "behavior", True),
             "awareness": aggregate(v, "awareness", True)} for k, v in sorted(groups.items())]


def conditions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["role"], row["narrative"], row["condition"])].append(row)
    return [{"cohort": k[0], "role": k[1], "narrative": k[2], "condition": k[3],
             "behavior": aggregate(v, "behavior", k[0] == "Bridge"),
             "awareness": aggregate(v, "awareness", k[0] == "Bridge")} for k, v in sorted(groups.items())]


def rate(m):
    return f"{int(m['positive_n'])}/{m['valid_n']}" if m["valid_n"] else "NA (0 valid)"


def coverage(m):
    return f"{m['valid_n']}/{m['total_n']}"


def se(m):
    return f"{m['stderr']:.3f}" if m["stderr"] is not None else "NA"


def plots(stats, cells, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    paths = []

    def axes_style(ax):
        ax.set(xlim=(-.08, 1.10), ylim=(-.08, 1.10),
               xlabel="Expressed evaluation awareness (E / valid)", ylabel="Native harmful score (+ / valid)")
        ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, cohort in zip(axes, ("V1", "V2")):
        axes_style(ax); ax.set_title(cohort + " — fixed panel, paired seed blocks")
        for g in [x for x in stats if x["cohort"] == cohort]:
            a, b, role = g["awareness"], g["behavior"], g["role"]
            label = f"{MODELS[role]}: A={rate(a)} ({coverage(a)} valid); B={rate(b)} ({coverage(b)} valid)"
            ax.errorbar(a["mean"], b["mean"], xerr=a["stderr"], yerr=b["stderr"], fmt="o",
                        color=COLORS[role], capsize=5, ms=8, lw=1.8, label=label)
            ax.annotate(MODELS[role], (a["mean"], b["mean"]), xytext=(10, 10), textcoords="offset points")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False, fontsize=8)
    fig.suptitle("V1 and V2: awareness vs behavior with native Inspect ±1 SE", fontsize=17, y=.98)
    fig.text(.5, .025, "SE clusters all conditions by seed-repeat block (only 2 blocks). Separate valid denominators on each axis.\n"
             "Conditional repeat variability for a fixed panel, not a 95% CI or a scenario-population/causal comparison.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.31, wspace=.25)
    path = out / "v1-v2-pooled-inspect-se.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    for cohort in ("V1", "V2"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        for ax, role in zip(axes, ("base", "mo_post")):
            axes_style(ax); ax.set_title(cohort + " · " + MODELS[role])
            locations = defaultdict(list)
            for g in [x for x in cells if x["cohort"] == cohort and x["role"] == role]:
                a, b = g["awareness"], g["behavior"]
                if a["mean"] is None or b["mean"] is None:
                    continue
                locations[(a["mean"], b["mean"])].append(g)
                ax.errorbar(a["mean"], b["mean"], xerr=a["stderr"], yerr=b["stderr"],
                            fmt="o", ms=5, color=COLORS[role], alpha=.45, capsize=3, lw=1)
            for (x, y), grouped in locations.items():
                label = ", ".join(g["condition"] for g in grouped)
                wrapped = textwrap.fill(label, 25)
                ax.annotate(wrapped, (x, y), xytext=(7 if x < .6 else -7, 8 if y < .8 else -10),
                            textcoords="offset points", ha="left" if x < .6 else "right",
                            va="bottom" if y < .8 else "top", fontsize=8,
                            bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 1.5})
        fig.suptitle(f"{cohort} condition rates: native Inspect mean ±1 SE", fontsize=17)
        fig.text(.5, .02, "Two stochastic repeats per condition; each axis excludes its own invalid scores. SE = sample SD / √n.\n"
                 "n=1: SE unavailable, not zero precision. Unanimous n=2 scores yield native SE=0, not proof of certainty.\n"
                 "Coincident conditions retain their identities, not extra replicates. Counts and omitted points: CONDITIONS.md.\n"
                 "Error bars are standard errors, not confidence intervals. One fixed rescue narrative; no causal inference.", ha="center", fontsize=10)
        fig.subplots_adjust(top=.88, bottom=.22, wspace=.24)
        path = out / f"{cohort.lower()}-conditions-inspect-se.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def markdown(rows, stats, cells, out, fig_paths):
    links = {"OLD": str(OLD), "PETRI": str(PETRI), "LASR": str(LASR), "OUT": str(out)}
    content = TEMPLATE.read_text()
    for original, relocated in PATH_MAP.items():
        content = content.replace(original, relocated)
    content = content.replace("/Users/work/consistency-training-methods/.venv/lib/python3.11/site-packages/inspect_ai/scorer/_metrics/mean.py", inspect.getfile(mean_module))
    content = content.replace("/Users/work/consistency-training-methods/.venv/lib/python3.11/site-packages/inspect_ai/scorer/_metrics/std.py", inspect.getfile(std_module))
    for key, value in links.items():
        content = content.replace("{{" + key + "}}", value)
    table = ["| Cohort / model / task | Total | Behavior + / valid | Behavior valid / total | Awareness E / valid | Awareness valid / total | Behavior SE | Awareness SE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = {"V1": 0, "V2": 1, "V3": 2, "V4": 3, "Bridge": 4}
    for g in sorted(stats, key=lambda x: (order[x["cohort"]], list(MODELS).index(x["role"]), x["narrative"])):
        b, a = g["behavior"], g["awareness"]
        task = " / " + g["narrative"].replace("_", " ") if g["cohort"] == "V4" else ""
        table.append(f"| {g['cohort']} / {MODELS[g['role']]}{task} | {b['total_n']} | {rate(b)} | {coverage(b)} | {rate(a)} | {coverage(a)} | {se(b)} | {se(a)} |")
    content = content.replace("{{RESULTS_TABLE}}", "\n".join(table))
    (out / "REPORT.md").write_text(content)
    lines = ["# All condition rates and native Inspect standard errors", "", "A=affirmative expressed evaluation belief. B=primary behavior. Fractions are positive/valid; n=valid/total. NA SE means fewer than two repeat units, including all one-repeat Petri and bridge cells. Separate valid denominators; no scores were changed.", "",
             "| Cohort | Model | Task / condition | A | A n | A SE | B | B n | B SE |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in cells:
        a, b = g["awareness"], g["behavior"]
        lines.append(f"| {g['cohort']} | {MODELS[g['role']]} | {g['narrative']} / {g['condition']} | {rate(a)} | {coverage(a)} | {se(a)} | {rate(b)} | {coverage(b)} | {se(b)} |")
    (out / "CONDITIONS.md").write_text("\n".join(lines) + "\n")
    lines = ["# Native Inspect log index", "", "232 panel memberships; 224 unique trajectories. Eight V1 B/O controls are exactly reused by V2. Read the raw and scored native .eval files below; sample ID plus epoch selects the observation. No viewer server is assumed to be running.", "",
             "| Cohort | Model | Condition / goal | Sample / epoch | Scores A; B | Native logs |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (order[r["cohort"]], r["role"], r["narrative"], r["condition"], r["uid"])):
        logs = f"[raw]({r['raw_path']}) · [scored]({r['scored_path']})"
        if r["original_attempt_log"]:
            logs += f" · [original attempt]({r['original_attempt_log']})"
        if r["reused_from_v1"]:
            logs += " · exact V1 reuse"
        lines.append(f"| {r['cohort']} | {MODELS[r['role']]} | {r['condition']} / {r['goal_value'] or r['narrative']} | {r['sample_id']} / {r['epoch']} | {r['awareness_label'] or r['awareness_status']}; {r['behavior'] if r['behavior'] is not None else r['behavior_status']} | {logs} |")
    (out / "LOGS.md").write_text("\n".join(lines) + "\n")
def validate_links(out):
    for path in (out / "REPORT.md", out / "CONDITIONS.md", out / "LOGS.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("https:", "http:", "#")):
                resolved = Path(target) if target.startswith("/") else path.parent / target
                # Optional line numbers in local source links.
                if not resolved.exists() and not Path(re.sub(r":\d+$", "", str(resolved))).exists():
                    raise ValueError(f"Broken local link in {path}: {target}")


def main():
    global OLD, PRIOR, PETRI, LASR, PATH_MAP, TEMPLATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-analysis-dir", type=Path, required=True)
    parser.add_argument("--prior-analysis-dir", type=Path, required=True)
    parser.add_argument("--petri-root", type=Path)
    parser.add_argument("--lasr-root", type=Path)
    parser.add_argument("--path-map", type=Path)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--statistics-only", action="store_true",
                        help="Reproduce all native statistics without presentation-image/link dependencies")
    args = parser.parse_args()
    OLD = args.old_analysis_dir.resolve()
    PRIOR = args.prior_analysis_dir.resolve()
    if not args.statistics_only and (args.petri_root is None or args.lasr_root is None):
        parser.error("Full report requires --petri-root and --lasr-root")
    PETRI = args.petri_root.resolve() if args.petri_root else PRIOR.parent
    LASR = args.lasr_root.resolve() if args.lasr_root else OLD
    TEMPLATE = args.template.resolve()
    PATH_MAP = json.loads(args.path_map.read_text()) if args.path_map else {}
    if not isinstance(PATH_MAP, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in PATH_MAP.items()):
        parser.error("Path map must contain exact string-to-string mappings")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    rows, sources, bound = collect()
    stats, cells = summaries(rows), conditions(rows)
    # Hash the installed implementations actually used, not a web description.
    for module in (std_module, mean_module):
        bound(inspect.getfile(module))
    bound(__file__); bound(TEMPLATE)
    for path in (OLD / "awareness-propensity.png", OLD / "v1-condition-profiles.png", OLD / "v2-condition-profiles.png",
                 PRIOR / "v3_awareness_propensity.png", PRIOR / "v4_awareness_propensity.png",
                 PETRI / "native-crosscheckpoint-scatter-v2/native_awareness_propensity_crosscheckpoint.png"):
        if not args.statistics_only:
            bound(path)
    out.mkdir(parents=True, exist_ok=False)
    figures = [] if args.statistics_only else plots(stats, cells, out)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "inspect_version": importlib.metadata.version("inspect-ai"),
              "target_calls": 0, "judge_calls": 0, "statistics": "native Inspect mean/stderr; unreduced valid epoch observations; pooled SE clustered by paired seed-repeat block; n<2 or clusters<2 reported NA despite native zero guard",
              "unique_trajectories": 224, "panel_memberships": 232, "exact_v1_v2_reuses": 8,
              "summaries": stats, "conditions": cells, "rows": rows, "source_hashes": sources,
              "figures": [str(p) for p in figures]}
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.statistics_only:
        markdown(rows, stats, cells, out, figures)
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError(f"Source changed: {path}")
    (out / "manifest.json").write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, indent=2) + "\n")
    if not args.statistics_only:
        validate_links(out)
    print(json.dumps({"output_dir": str(out), "inspect": report["inspect_version"], "memberships": len(rows),
                      "unique_trajectories": 224, "source_files_verified": len(sources), "model_calls": 0,
                      "summaries": stats}, indent=2))


if __name__ == "__main__":
    main()
