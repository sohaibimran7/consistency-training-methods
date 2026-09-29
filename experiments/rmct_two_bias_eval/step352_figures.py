"""Presentation-only wrapping for the completed standard checkpoint figures."""
import argparse
import json
from pathlib import Path
import textwrap

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.rmct_two_bias_eval.step352 import digest, write_once


def render(root: Path):
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    for source in sorted((root / "plots").iterdir()):
        if not source.is_dir():
            continue
        rows = json.loads((source / "chart-rows.json").read_text())
        spec = json.loads((source / "chart-spec.json").read_text())
        spec["significance_note"] = textwrap.fill(spec["significance_note"], width=180)
        write_once(destination / f"{source.name}.spec.json", spec)
        for extension in ("png", "svg"):
            render_publication_plot(rows, spec, destination / f"{source.name}.{extension}")
    write_once(destination / "manifest.json", {
        "source_manifests": {str(p): digest(p) for p in sorted((root / "plots").glob("*/manifest.json"))},
        "change": "Caption line wrapping only; same rows, statistical results and standard renderer.",
        "outputs": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.suffix in {".png", ".svg"}},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
