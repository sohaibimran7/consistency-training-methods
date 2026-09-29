# Historical six-method convergence replay

This offline command recomputes the saved training-stream plots and summary. It makes no model, grading, SSH or training calls. These are historical own-objective stopping curves, including flawed RMCT training, **not corrected reruns or validation-selected checkpoints**. Loss magnitudes are not comparable across methods.

Python 3.10+, NumPy and Matplotlib are required, plus this checkout's `ctm_data.adapters.mcq_bias.method_presentation` palette module. Inputs remain external; preserve their bytes and verify against `input-manifest.json`. No public download location is invented.

```sh
python experiments/paper_convergence/plot.py \
  --histories /data/artifacts/methods-meeting-20260911/training-histories.json \
  --remote-histories /data/artifacts/methods-convergence-all-20260917/remote-histories.json \
  --output-dir /output/new-convergence-replay
```

The output directory must not exist. Changed inputs fail before it is created. Do not use Python `-O`: the historical recipe's coverage, stopping-state and weighted-mean assertions are scientifically necessary. If a later assertion fails, the output directory may contain partial products; do not treat it as a successful replay. The manifest is written only on success.

Outputs: `convergence-all-methods.{pdf,png,svg}`, `rmct-optimization-diagnostics.{pdf,png,svg}`, `summary.json`, and `manifest.json` containing input/code/palette/output hashes and runtime versions. The historical numerical calculations and 16-update windows are unchanged; titles/footnotes explicitly identify historical results and flawed RMCT training. Rendered bytes may vary with fonts and Matplotlib version. Compare the complete summary against the archived `summary.json` hash in the input manifest; visual bytes are not expected to match after warning text changes.

The history inputs must be retrieved from the research artifact archive. Remote exporter scripts are not needed for this saved-history replay. No remote checkpoint paths embedded in history records are dereferenced. The clean validation-selected pipeline lives separately under `experiments/rmct_restart_20260928`.
