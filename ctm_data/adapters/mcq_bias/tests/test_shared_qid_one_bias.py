import json

import pytest

from ctm_data.adapters.mcq_bias import shared_qid_one_bias as one
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import BIAS_TYPES, DATUM_SCHEMA, SCHEMA_VERSION

SOURCE = {"pool_sha256": "a" * 64, "manifest_sha256": "b" * 64}


def msg(text):
    return [{"role": "user", "content": text}]


def datum(index):
    return {
        "datum_schema": DATUM_SCHEMA, "schema_version": SCHEMA_VERSION,
        "question_id": f"q{index}", "source_dataset": ("logiqa", "hellaswag")[index % 2],
        "question": f"question {index}", "ground_truth": "A", "prompt_style": "none",
        "clean_messages": msg(f"clean {index}"), "biased_options": {b: "B" for b in BIAS_TYPES},
        "variants": {b: {"messages": msg(f"{b} {index}"), "biased_option": "B", "biasing_text": f"cue {b}"}
                     for b in BIAS_TYPES},
        "provenance": {"wrong_argument_source_line_number": index + 1},
    }


def pool(n=40):
    return [datum(i) for i in range(n)]


def test_assignment_is_shared_deterministic_and_method_independent():
    manifests = [one.build_manifest(pool(), seed=7, source=SOURCE) for _ in range(6)]
    assert all(m == manifests[0] for m in manifests)
    m = manifests[0]
    assert [a["question_id"] for a in m["assignments"]] == [f"q{i}" for i in range(40)]
    assert {a["bias"] for a in m["assignments"]} == set(BIAS_TYPES)
    assert sum(m["bias_counts"].values()) == 40
    assert one.build_manifest(pool(), seed=8, source=SOURCE)["assignment_sha256"] != m["assignment_sha256"]
    # Assignment depends only on (seed, qid): restricting the pool keeps each QID's bias.
    sub = one.build_manifest(pool(), seed=7, source=SOURCE, eligible_qids=["q5", "q2", "q30"])
    assert [a["question_id"] for a in sub["assignments"]] == ["q2", "q5", "q30"]
    full = {a["question_id"]: a["bias"] for a in m["assignments"]}
    assert all(full[a["question_id"]] == a["bias"] for a in sub["assignments"])


def test_rejects_duplicates_unknown_eligible_and_tampering():
    with pytest.raises(ValueError, match="exactly one"):
        one.build_manifest(pool() + pool()[:1], seed=1, source=SOURCE)
    with pytest.raises(ValueError, match="eligible"):
        one.build_manifest(pool(), seed=1, source=SOURCE, eligible_qids=["missing"])
    m = one.build_manifest(pool(), seed=1, source=SOURCE)
    tampered = json.loads(json.dumps(m))
    tampered["assignments"][0]["bias"] = [b for b in BIAS_TYPES if b != m["assignments"][0]["bias"]][0]
    tampered["assignment_sha256"] = one.digest(tampered["assignments"])
    with pytest.raises(ValueError, match="protocol hash"):
        one.validate_manifest(tampered)


def test_projection_keeps_only_assigned_arm():
    rows = pool(4)
    m = one.build_manifest(rows, seed=3, source=SOURCE)
    for row, a in zip(rows, m["assignments"]):
        p = one.project_datum(row, a)
        assert p["bias"] == a["bias"] and p["variant"] == row["variants"][a["bias"]]
        assert "variants" not in p and p["clean_messages"] == row["clean_messages"]
    with pytest.raises(ValueError, match="frozen assignment"):
        one.project_datum(rows[1], m["assignments"][0])


def test_cursor_no_repeat_signal_free_batches_and_exhaustion():
    m = one.build_manifest(pool(10), seed=5, source=SOURCE)
    s = one.initial_state(m)
    seen = []
    for attempt, (count, updates, signal) in enumerate([(4, 0, False), (4, 1, True), (2, 2, True)]):
        rows, s = one.claim(m, s, count, attempt=attempt)
        with pytest.raises(ValueError, match="unresolved"):
            one.claim(m, s, 1, attempt=attempt + 1)
        seen += [r["question_id"] for r in rows]
        s = one.commit(s, attempt=attempt, optimizer_updates_after=updates, rollouts=count * 4, learning_signal=signal)
    assert seen == [a["question_id"] for a in m["assignments"]] and len(set(seen)) == 10
    with pytest.raises(ValueError, match="exhausted"):
        one.claim(m, s, 1, attempt=9)
    record = one.checkpoint_record(m, s, method="rmct", optimizer_updates=2)
    assert record["encountered_qid_bias_examples"] == 10 and record["optimizer_updates"] == 2
    assert record["batches_without_learning_signal"] == 1 and record["rollouts"] == 40 and record["exhausted"]


def test_resume_and_checkpoint_contracts(tmp_path):
    m = one.build_manifest(pool(8), seed=11, source=SOURCE)
    path = one.freeze_manifest(m, tmp_path)
    assert one.freeze_manifest(m, tmp_path) == path  # idempotent, immutable
    loaded = one.load_manifest(path, expected_sha256=one.manifest_identity(m))
    with pytest.raises(ValueError, match="identity"):
        one.load_manifest(path, expected_sha256="0" * 64)
    s = one.initial_state(loaded)
    _, s = one.claim(loaded, s, 3, attempt=0)
    with pytest.raises(ValueError, match="committed"):
        one.checkpoint_record(loaded, s, method="bct", optimizer_updates=0)
    s = one.commit(s, attempt=0, optimizer_updates_after=1, rollouts=None, learning_signal=True)
    resumed = json.loads(json.dumps(s))  # round-trip like a saved checkpoint
    rows, _ = one.claim(loaded, resumed, 2, attempt=1)
    assert [r["question_id"] for r in rows] == ["q3", "q4"]
    record = one.checkpoint_record(loaded, resumed, method="bct", optimizer_updates=1)
    assert record["rollouts"] is None  # missing instrumentation is null, never zero
    with pytest.raises(ValueError, match="different manifest"):
        one.claim(one.build_manifest(pool(8), seed=12, source=SOURCE), resumed, 1, attempt=2)
    broken = json.loads(json.dumps(resumed))
    broken["encounters"] = 2
    with pytest.raises(ValueError, match="ledger"):
        one.claim(loaded, broken, 1, attempt=2)
