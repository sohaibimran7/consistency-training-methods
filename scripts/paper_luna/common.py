"""Offline input validation, provenance, and paired resampling for Luna plots."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ("base", "bct", "rmct352")
CONDITIONS = ("zero_shot", "distribution_only", "examples_only", "both")
WARNINGS = [
    "Historical RMCT outputs are not evidence for an unaffected/corrected training run.",
    "Parser-corrected evaluation labels do not repair historical training rewards.",
    "Observed single-pair switches are not causal influence ground truth.",
    "Base clean labels remain unverified; historical ICL calibration labels are retained.",
    "Missing, invalid, and truncated outputs are excluded, never imputed negative.",
    "Fixed calibration banks: bootstrap intervals exclude demonstration-selection uncertainty.",
]
def cli(kind):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260922 if kind == "icl" else 20260928)
    if kind == "icl":
        p.add_argument("--samples", type=Path, required=True)
        p.add_argument("--ledger-dir", type=Path, action="append", required=True,
                       help="Repeat in historical precedence order: ICL, base zero-shot, BCT/RMCT zero-shot.")
    else:
        p.add_argument("--data", type=Path, required=True)
        if kind == "rare":
            p.add_argument("--counts", type=Path, required=True)
    a = p.parse_args()
    if a.bootstrap < 2:
        p.error("--bootstrap must be at least 2")
    a.output.mkdir(parents=True, exist_ok=True)
    paths = ([a.samples] + [d / f for d in a.ledger_dir for f in
             ("luna-results.jsonl", "full-private-labels.json")]) if kind == "icl" else [a.data]
    if kind == "rare":
        paths.append(a.counts)
    provenance = {"inputs": [{"path": str(x.resolve()), "sha256": hashlib.sha256(x.read_bytes()).hexdigest()}
                            for x in paths], "bootstrap": a.bootstrap, "seed": a.seed,
                  "warnings": WARNINGS, "mode": kind}
    (a.output / "input-provenance.json").write_text(json.dumps(provenance, indent=2))
    return a

def cluster_weights(rows, draws, seed):
    clusters = sorted({(r["dataset"], r["qid"]) for r in rows})
    lookup = {k: i for i, k in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(clusters)), dtype=np.int32)
    for ds in sorted({k[0] for k in clusters}):
        ids = [i for i, k in enumerate(clusters) if k[0] == ds]
        weights[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    return clusters, lookup, weights

def load_icl(a):
    samples = json.loads(a.samples.read_text())
    lookup = {(m, r["dataset"], r["bias"], r["qid"]): r for m, rows in samples.items() for r in rows}
    maps = {(m, c): {} for m in METHODS for c in CONDITIONS}
    for directory in a.ledger_dir:
        good = {r["request_id"]: r for r in map(json.loads, (directory / "luna-results.jsonl").read_text().splitlines()) if r["ok"]}
        for label in json.loads((directory / "full-private-labels.json").read_text()):
            if "condition" not in label and (label["view"] != "cot_only" or label["effort"] != "xhigh"):
                continue
            r = lookup.get((label["method"], label["dataset"], label["bias"], label["qid"]))
            if not r or r["b"] is None or r["u"] is None or r["u"] == r["option"]:
                continue
            result = good.get(label["request_id"])
            if result is None:
                continue
            score = result["score"] / 100
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid saved score")
            maps[label["method"], label.get("condition", "zero_shot")][label["case_id"]] = dict(
                label, target=int(r["b"] == r["option"]), ack=r["ack"], score=score)
    common = sorted(set.intersection(*(set(v) for v in maps.values())))
    if not common:
        raise ValueError("No common eligible cases across all methods and configurations")
    ref = [maps["base", "both"][k] for k in common]
    for values in maps.values():
        for k, r in zip(common, ref):
            assert (values[k]["dataset"], values[k]["bias"], values[k]["qid"]) == (r["dataset"], r["bias"], r["qid"])
    clusters, lookup, weights = cluster_weights(ref, a.bootstrap, a.seed)
    ix = np.array([lookup[r["dataset"], r["qid"]] for r in ref])
    return dict(maps=maps, common=common, clusters=clusters, weights=weights, ix=ix)

def validate_rows(rows):
    seen = set()
    for r in rows:
        k = (r["method"], r["dataset"], r["bias"], r["qid"])
        if k in seen:
            raise ValueError("Duplicate method/question/bias row")
        seen.add(k)
        if r["method"] not in METHODS or r["target"] not in (0, 1):
            raise ValueError("Invalid method or outcome")
        scores = r.get("scores", {"zero_shot": r.get("score")})
        if not scores or any(s is None or not np.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
            raise ValueError("Missing or invalid scores must not become negatives")
        if "already_matched" in r and r["target"] != int(not r["already_matched"] and r["biased_matches"]):
            raise ValueError("Target must be clean != cue AND biased == cue")
