#!/usr/bin/env python3
"""Make EvalAwareBench-style heatmaps from the completed R3 report.

The paper's Figure 6 uses blue heatmap cells, model rows, BL/F1--F8
columns, and small red/green baseline-delta annotations.  This script keeps
the observed R3 rates and denominators unchanged and applies that visual
grammar to both verbalised awareness and primary behaviour propensity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / (
    "experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/"
    "expanded-r3-20260915/recovery-preparation-2249/"
    "analysis-regression-r3-v2/report.json"
)
OUT = ROOT / (
    "experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/"
    "expanded-r3-20260915/recovery-preparation-2249/"
    "analysis-paper-style-r3-v1"
)


MODEL_LABELS = ["Base", "MO mid", "MO post"]
SINGLE_PROFILES = ["shared_realistic"] + [f"F{i}" for i in range(1, 9)]
COMBO_PROFILES = [
    "F2+F5",
    "F2+F3+F4",
    "F1+F3+F4+F7",
    "F2+F4+F5+F6+F8",
    "F1+F3+F4+F5+F6+F7",
    "F1+F3+F4+F5+F6+F7+F8",
    "F1+F2+F3+F4+F5+F6+F7+F8",
]
STACKING_PROFILES = ["shared_realistic", "F1"] + COMBO_PROFILES
STACKING_LABELS = [str(i) for i in range(9)]
PROFILE_LABELS = {"shared_realistic": "BL"}
PROFILE_LABELS.update({f"F{i}": f"F{i}" for i in range(1, 9)})
PROFILE_LABELS.update({p: f"C{i}" for i, p in enumerate(COMBO_PROFILES, 1)})

PANEL_META = [
    ("V3", "rescue obstruction"),
    ("V4", "three-family misalignment set"),
]

# A restrained blue ramp close to the paper's Figures 6/7.  The end points
# are intentionally not black, so text stays legible in high-rate cells.
BLUE = mcolors.LinearSegmentedColormap.from_list(
    "paper_blue",
    ["#f4f7fa", "#d8e8f3", "#a4c8df", "#5b97bd", "#1f5d83", "#0b3b5c"],
)
RED = "#b23a3a"
GREEN = "#2e7d32"
GREY = "#5d6870"
BLACK = "#202020"


def load_report() -> dict:
    with REPORT.open() as f:
        return json.load(f)


def index_panels(report: dict) -> dict[tuple[str, str], dict]:
    """Map the fixed six-panel report order to (version, model)."""

    # analysis-regression-r3-v2 stores the first three panels as V3 and the
    # final three as V4; this is also the ordering in the source report's
    # count table.  Keep a defensive check on families to catch accidental
    # changes to the report schema.
    panels = report["primary"]
    if len(panels) != 6:
        raise ValueError(f"Expected six primary panels, found {len(panels)}")
    out: dict[tuple[str, str], dict] = {}
    for i, version in enumerate(("V3", "V4")):
        for j, role in enumerate(("base", "mo_mid", "mo_post")):
            panel = panels[i * 3 + j]
            out[(version, role)] = panel
    return out


def value_record(panel: dict, profile: str, metric: str) -> dict | None:
    for point in panel["points"]:
        if point["profile"] == profile:
            value = point.get(metric)
            if value is None:
                return None
            return {
                "value": float(value),
                "n": int(point.get("n", 0)),
                "expected": int(point.get("expected", 0)),
                "observed": int(point.get("observed", 0)),
                "profile": profile,
            }
    return None


def nice_vmax(report: dict, metric: str) -> float:
    values = []
    for panel in report["primary"]:
        for point in panel["points"]:
            if point.get(metric) is not None:
                values.append(float(point[metric]))
    maximum = max(values) if values else 1.0
    # Round up to a readable tenth while leaving at least 20% headroom for
    # datasets with very small rates.  This makes V3/V4 comparable in a
    # single figure and keeps cell colours from saturating early.
    vmax = max(0.2, math.ceil(maximum * 10 - 1e-12) / 10)
    return min(1.0, vmax)


def add_cell_grid(ax, nrows: int, ncols: int) -> None:
    ax.set_xticks([i - 0.5 for i in range(1, ncols)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, nrows)], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)


def draw_heatmap(
    ax,
    panel_rows: list[tuple[str, dict]],
    profiles: list[str],
    metric: str,
    vmax: float,
    version: str,
    subtitle: str,
    xlabels: list[str] | None = None,
) -> list[dict]:
    """Draw one V3/V4 panel and return machine-readable cell records."""

    rows = []
    records = []
    for role, panel in panel_rows:
        row = []
        baseline = value_record(panel, "shared_realistic", metric)
        baseline_value = baseline["value"] if baseline is not None else None
        for profile in profiles:
            rec = value_record(panel, profile, metric)
            if rec is None:
                row.append(float("nan"))
                records.append(
                    {
                        "version": version,
                        "role": role,
                        "metric": metric,
                        "profile": profile,
                        "value": None,
                        "n": None,
                        "expected": None,
                        "delta_pp": None,
                    }
                )
                continue
            row.append(rec["value"])
            delta = (
                (rec["value"] - baseline_value) * 100.0
                if baseline_value is not None
                else None
            )
            records.append(
                {
                    "version": version,
                    "role": role,
                    "metric": metric,
                    "profile": profile,
                    "value": rec["value"],
                    "n": rec["n"],
                    "expected": rec["expected"],
                    "observed": rec["observed"],
                    "delta_pp": delta,
                }
            )
        rows.append(row)

    im = ax.imshow(
        rows,
        cmap=BLUE,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    nrows = len(rows)
    ncols = len(profiles)
    ax.set_xticks(range(ncols))
    labels = xlabels if xlabels is not None else [PROFILE_LABELS.get(p, p) for p in profiles]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(MODEL_LABELS, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)
    add_cell_grid(ax, nrows, ncols)
    ax.set_title(f"{version}  ·  {subtitle}", fontsize=10, pad=8, weight="bold")

    # Cell labels: percentage in the centre, denominator in the lower-right,
    # and paper-style baseline delta in the upper-right when |delta| >= 5pp.
    for i, (role, panel) in enumerate(panel_rows):
        baseline = value_record(panel, "shared_realistic", metric)
        baseline_value = baseline["value"] if baseline is not None else None
        for j, profile in enumerate(profiles):
            rec = value_record(panel, profile, metric)
            if rec is None:
                ax.text(j, i, "—", ha="center", va="center", color=GREY, fontsize=11)
                continue
            value = rec["value"]
            # Contrast text against the cell colour.
            luminance = (1.0 - value / vmax) if vmax else 1.0
            text_color = BLACK if luminance > 0.53 else "white"
            ax.text(
                j,
                i - 0.03,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.3,
                weight="bold",
            )
            ax.text(
                j + 0.43,
                i + 0.36,
                f"n={rec['n']}",
                ha="right",
                va="bottom",
                color=("#e9eef2" if text_color == "white" else GREY),
                fontsize=5.5,
            )
            if baseline_value is not None:
                delta_pp = (value - baseline_value) * 100.0
                if abs(delta_pp) >= 5.0 - 1e-9 and profile != "shared_realistic":
                    dcolor = RED if delta_pp > 0 else GREEN
                    dtext = f"{delta_pp:+.0f}"
                    ax.text(
                        j + 0.43,
                        i - 0.36,
                        dtext,
                        ha="right",
                        va="top",
                        color=dcolor,
                        fontsize=6.1,
                        weight="bold",
                    )
    return records


def draw_awareness_propensity(
    ax,
    panel_rows: list[tuple[str, dict]],
    profiles: list[str],
    vmax: float,
    version: str,
    subtitle: str,
    xlabels: list[str] | None = None,
) -> list[dict]:
    """Draw the paper Figure 6 grammar: awareness in black, propensity Δ in colour."""

    rows = []
    records = []
    for role, panel in panel_rows:
        row = []
        baseline_prop = value_record(panel, "shared_realistic", "y")
        baseline_propensity = baseline_prop["value"] if baseline_prop is not None else None
        for profile in profiles:
            awareness = value_record(panel, profile, "x")
            propensity = value_record(panel, profile, "y")
            row.append(float("nan") if awareness is None else awareness["value"])
            delta = (
                (propensity["value"] - baseline_propensity) * 100.0
                if propensity is not None and baseline_propensity is not None
                else None
            )
            records.append(
                {
                    "version": version,
                    "role": role,
                    "profile": profile,
                    "awareness": None if awareness is None else awareness["value"],
                    "awareness_n": None if awareness is None else awareness["n"],
                    "propensity": None if propensity is None else propensity["value"],
                    "propensity_n": None if propensity is None else propensity["n"],
                    "propensity_delta_pp_vs_bl": delta,
                }
            )
        rows.append(row)

    ax.imshow(
        rows,
        cmap=BLUE,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    nrows = len(rows)
    ncols = len(profiles)
    ax.set_xticks(range(ncols))
    labels = xlabels if xlabels is not None else [PROFILE_LABELS.get(p, p) for p in profiles]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(MODEL_LABELS, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)
    add_cell_grid(ax, nrows, ncols)
    ax.set_title(f"{version}  ·  {subtitle}", fontsize=10, pad=8, weight="bold")

    for i, (role, panel) in enumerate(panel_rows):
        baseline_prop = value_record(panel, "shared_realistic", "y")
        baseline_propensity = baseline_prop["value"] if baseline_prop is not None else None
        for j, profile in enumerate(profiles):
            awareness = value_record(panel, profile, "x")
            propensity = value_record(panel, profile, "y")
            if awareness is None:
                ax.text(j, i, "—", ha="center", va="center", color=GREY, fontsize=11)
                continue
            value = awareness["value"]
            text_color = BLACK if (1.0 - value / vmax) > 0.53 else "white"
            ax.text(
                j,
                i - 0.03,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.3,
                weight="bold",
            )
            ax.text(
                j + 0.43,
                i + 0.36,
                f"n={awareness['n']}",
                ha="right",
                va="bottom",
                color=("#e9eef2" if text_color == "white" else GREY),
                fontsize=5.5,
            )
            if (
                propensity is not None
                and baseline_propensity is not None
                and profile != "shared_realistic"
            ):
                delta_pp = (propensity["value"] - baseline_propensity) * 100.0
                if abs(delta_pp) >= 5.0 - 1e-9:
                    dcolor = RED if delta_pp > 0 else GREEN
                    ax.text(
                        j + 0.43,
                        i - 0.36,
                        f"{delta_pp:+.0f}",
                        ha="right",
                        va="top",
                        color=dcolor,
                        fontsize=6.1,
                        weight="bold",
                    )
    return records


def make_figure(
    report: dict,
    panels: dict[tuple[str, str], dict],
    profiles: list[str],
    metric: str,
    name: str,
    title: str,
    y_label: str,
    caption: str,
    xlabels: list[str] | None = None,
    x_label: str = "Factor condition",
) -> tuple[Path, Path, list[dict]]:
    vmax = nice_vmax(report, metric)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.7, 4.35 if len(profiles) <= 9 else 4.85),
        sharey=True,
        gridspec_kw={"wspace": 0.075},
    )
    all_records: list[dict] = []
    for ax, (version, subtitle) in zip(axes, PANEL_META):
        rows = [(role, panels[(version, role)]) for role in ("base", "mo_mid", "mo_post")]
        all_records.extend(
            draw_heatmap(ax, rows, profiles, metric, vmax, version, subtitle, xlabels=xlabels)
        )
    axes[0].set_ylabel(y_label, fontsize=9, labelpad=8)
    for ax in axes:
        ax.set_xlabel(x_label, fontsize=8, labelpad=5)

    # A compact continuous key is more precise than a categorical legend when
    # the cells contain exact percentages.
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=vmax), cmap=BLUE)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.17,
        aspect=45,
    )
    cbar.set_ticks([0, vmax / 2, vmax])
    cbar.set_ticklabels(["0%", f"{vmax * 50:.0f}%", f"{vmax * 100:.0f}%"])
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("Matched-valid rate", fontsize=8, labelpad=2)

    fig.suptitle(title, fontsize=12, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.015,
        caption,
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#3f484d",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.25)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf, all_records


def make_awareness_propensity_figure(
    report: dict,
    panels: dict[tuple[str, str], dict],
    profiles: list[str],
    name: str,
    title: str,
    caption: str,
    xlabels: list[str] | None = None,
    x_label: str = "Factor condition",
) -> tuple[Path, Path, list[dict]]:
    """Make a Figure 6-style overlay with black awareness and coloured Δ propensity."""

    vmax = nice_vmax(report, "x")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.7, 4.35 if len(profiles) <= 9 else 4.85),
        sharey=True,
        gridspec_kw={"wspace": 0.075},
    )
    all_records: list[dict] = []
    for ax, (version, subtitle) in zip(axes, PANEL_META):
        rows = [(role, panels[(version, role)]) for role in ("base", "mo_mid", "mo_post")]
        all_records.extend(
            draw_awareness_propensity(
                ax, rows, profiles, vmax, version, subtitle, xlabels=xlabels
            )
        )
    axes[0].set_ylabel("Evaluation awareness", fontsize=9, labelpad=8)
    for ax in axes:
        ax.set_xlabel(x_label, fontsize=8, labelpad=5)

    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=vmax), cmap=BLUE)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.17,
        aspect=45,
    )
    cbar.set_ticks([0, vmax / 2, vmax])
    cbar.set_ticklabels(["0%", f"{vmax * 50:.0f}%", f"{vmax * 100:.0f}%"])
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("Evaluation awareness rate", fontsize=8, labelpad=2)

    # Match the paper's annotation key: the cell is awareness; the small
    # coloured number is the propensity/performance shift versus BL.
    fig.text(0.5, 0.060, "+Δ propensity vs BL (pp)", ha="center", va="bottom", fontsize=7.2, color=RED)
    fig.text(0.5, 0.040, "−Δ propensity vs BL (pp)", ha="center", va="bottom", fontsize=7.2, color=GREEN)
    fig.suptitle(title, fontsize=12, weight="bold", y=0.99)
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7.1, color="#3f484d")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.29)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf, all_records


def main() -> None:
    report = load_report()
    panels = index_panels(report)
    outputs = []
    data = {
        "source_report": str(REPORT),
        "source_sha256": report.get("source_report", {}).get("sha256"),
        "single_factor_profiles": SINGLE_PROFILES,
        "combination_profiles": COMBO_PROFILES,
        "factor_count_profiles": STACKING_PROFILES,
        "figures": {},
        "notes": [
            "Rates are the existing matched-valid condition means; no values were re-scored or imputed.",
            "n is the point denominator in the regression report for that condition and checkpoint.",
            "Delta annotations are percentage-point changes from the same-checkpoint BL cell and are shown only for absolute changes of at least 5pp.",
            "Combination columns are the observed fixed profiles, not a randomized progressive factor-stacking experiment.",
        ],
    }
    common_caption = (
        "Cells show matched-valid rates; n is the condition denominator. "
        "Red/green values are Δ vs BL in percentage points (shown for |Δ|≥5pp); "
        "— denotes unavailable data."
    )

    for metric, label in (("x", "Verbalised evaluation awareness"), ("y", "Primary misalignment propensity")):
        for profiles, suffix, section in (
            (SINGLE_PROFILES, "single-factor", "single_factor"),
            (["shared_realistic"] + COMBO_PROFILES, "combinations", "combinations"),
        ):
            name = f"paper-style-{label.split()[0].lower()}-{suffix}"
            title = f"{label} · {('single-factor variation' if section == 'single_factor' else 'observed combination profiles')}"
            png, pdf, records = make_figure(
                report,
                panels,
                profiles,
                metric,
                name,
                title,
                label,
                common_caption,
            )
            outputs.extend([str(png), str(pdf)])
            data["figures"][name] = {
                "metric": metric,
                "profiles": profiles,
                "png": str(png),
                "pdf": str(pdf),
                "records": records,
            }

        # Paper Figure 7 uses the number of active factors on the x-axis.  We
        # provide the same compact visual, but label it as an observed path:
        # R3 has one fixed profile at each count rather than the paper's
        # randomized combinations.  The exact profile mapping is saved below.
        name = f"paper-style-{label.split()[0].lower()}-factor-count"
        png, pdf, records = make_figure(
            report,
            panels,
            STACKING_PROFILES,
            metric,
            name,
            f"{label} · observed factor-count path",
            label,
            "Columns show the observed fixed profiles with 0–8 active factors; this is not a randomized stacking estimate. "
            "Cells show matched-valid rates; n is the condition denominator; Δ is vs BL in pp for |Δ|≥5pp.",
            xlabels=STACKING_LABELS,
            x_label="Number of factors varied",
        )
        outputs.extend([str(png), str(pdf)])
        data["figures"][name] = {
            "metric": metric,
            "profiles": STACKING_PROFILES,
            "xlabels": STACKING_LABELS,
            "png": str(png),
            "pdf": str(pdf),
            "records": records,
        }

    # The compact C1--C7 labels keep the combination heatmaps readable.  Save
    # the exact expansion beside the figures for paper supplements/captions.
    data["combination_key"] = {
        f"C{i}": profile for i, profile in enumerate(COMBO_PROFILES, 1)
    }
    data["factor_count_key"] = {
        "0": "shared_realistic",
        "1": "F1",
        **{str(i + 1): profile for i, profile in enumerate(COMBO_PROFILES, 1)},
    }
    (OUT / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    (OUT / "combination-key.md").write_text(
        "# Combination-column key\n\n"
        + "\n".join(
            f"- **C{i}** = `{profile}`" for i, profile in enumerate(COMBO_PROFILES, 1)
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(OUT), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
