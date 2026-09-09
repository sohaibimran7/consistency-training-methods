"""Import the audited switch-gate HLE suite as an expedited screen only.

This is deliberately not a general frozen-data copier.  It verifies the
switch-gate audit/lock, the pinned HLE source, every frozen-file digest, the
complete matched schema, and the known 88/100 overlap with the unrestricted
Stage 1 seed-42 question pool.  Only then does it copy the exact bytes into a
distinct ``expedited-screen`` artifact root.

The imported suite has an ID-restricted question pool and its wrong arguments
were generated through ``vllm/google/gemma-4-31B-it``.  Full Stage 1 instead
requests the unrestricted seed-42 prefix and
``openrouter/google/gemma-4-31b-it``.  The output manifest records both
differences and forbids consumption through the full Stage 1 suite factory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from scripts.rmct_paper_vast_more_methods.hle_source import DATASET_ID, DATASET_REVISION, EXPECTED_TEXT_MC_COUNT

BIAS_ORDER = (
    "suggested_answer",
    "distractor_fact",
    "wrong_argument",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
EXPECTED_SCREEN_QUESTIONS = 100
EXPECTED_STAGE1_OVERLAP = 88
EXPECTED_DIVERGENCE_PER_SIDE = 12
EXPEDITED_OUTPUT_ROOT = "artifacts/rmct-hle-dense-models-shared-expedited-12h"
IMPORTED_ARGUMENT_MODEL = "vllm/google/gemma-4-31B-it"
FULL_STAGE1_ARGUMENT_MODEL = "openrouter/google/gemma-4-31b-it"
MCQ_BIAS_REVISION = "1df2ea1ed8a1eeaf6ec5088c066db7c8c1049119"

_UNBIASED_FIELDS = frozenset({"question", "question_id", "source_dataset", "prompt_style", "unbiased_messages", "ground_truth"})
_BIASED_FIELDS = frozenset(
    {
        *_UNBIASED_FIELDS,
        "biased_messages",
        "bias_type",
        "biased_option",
        "biasing_text",
    }
)
_ANSWER_CHOICES_HEADER = "\n\nAnswer choices:\n"
_ANSWER_FORMAT_INSTRUCTION = '\n\nGive your answer in the format "The best answer is: (X)."'


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required expedited-screen input does not exist: {path}")
    try:
        document = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return document


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"required expedited-screen input does not exist: {path}")
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    # Split only on the JSONL delimiter.  Unicode line separators may legally
    # occur inside HLE strings and ``str.splitlines()`` would split those too.
    for line_number, line in enumerate(payload.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return payload, rows


def _inside_repository(repository_root: Path, raw_path: str, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (repository_root / raw_path).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root: {raw_path!r}") from exc
    return path


def _parsed_input(row: dict[str, Any]) -> str:
    question = row.get("question")
    options = row.get("options")
    answer = row.get("answer")
    if not isinstance(question, str) or not question:
        raise ValueError("HLE source row has no non-empty question")
    if not isinstance(options, list) or not 2 <= len(options) <= len(ascii_uppercase) or any(not isinstance(option, str) or not option for option in options):
        raise ValueError("HLE source row has invalid options")
    if not isinstance(answer, str) or answer not in ascii_uppercase[: len(options)]:
        raise ValueError("HLE source row has an invalid answer label")
    formatted = "\n".join(f"({ascii_uppercase[index]}) {option}" for index, option in enumerate(options))
    return f"{question}{_ANSWER_CHOICES_HEADER}{formatted}"


def _question_id(row: dict[str, Any]) -> str:
    return hashlib.sha1(_parsed_input(row).encode()).hexdigest()


def _biased_option(row: dict[str, Any]) -> str:
    parsed = _parsed_input(row)
    answer = row["answer"]
    candidates = [index for index in range(len(row["options"])) if ascii_uppercase[index] != answer]
    return ascii_uppercase[random.Random(parsed).choice(candidates)]


def _ordered_ids_sha256(question_ids: list[str]) -> str:
    return _sha256("\n".join(question_ids).encode())


def _stage1_source_rows(source_rows: list[dict[str, Any]], *, seed: str, count: int) -> list[dict[str, Any]]:
    shuffled = list(source_rows)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in shuffled:
        question_id = _question_id(row)
        if question_id in seen:
            continue
        seen.add(question_id)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"HLE source has only {len(selected)}/{count} unique questions for the Stage 1 pool")
    return selected


def _message_list(value: object, *, location: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty message list")
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{location}[{index}] must contain exactly role/content")
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise TypeError(f"{location}[{index}] role/content must be strings")


def _full_stage1_spec(plan_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project requirements install PyYAML
        raise ImportError("reading the full Stage 1 shared-data plan requires PyYAML") from exc

    document = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{plan_path}: expected a YAML object")
    generation = document.get("data_generation")
    preparation = document.get("data_preparation")
    if not isinstance(generation, list) or not isinstance(preparation, list):
        raise TypeError(f"{plan_path}: missing data_generation/data_preparation lists")
    hle_entry = next((entry for entry in generation if entry.get("name") == "hle-source"), None)
    eval_entry = next((entry for entry in preparation if entry.get("name") == "evaluation-suite"), None)
    if not isinstance(hle_entry, dict) or not isinstance(eval_entry, dict):
        raise TypeError(f"{plan_path}: missing hle-source or evaluation-suite entry")
    hle_args = hle_entry.get("args")
    eval_args = eval_entry.get("args")
    if not isinstance(hle_args, dict) or not isinstance(eval_args, dict):
        raise TypeError(f"{plan_path}: malformed HLE materialization arguments")
    expected = {
        "bias_types": list(BIAS_ORDER),
        "prompt_style": "none",
        "n_questions": EXPECTED_SCREEN_QUESTIONS,
        "seed": "42",
        "argument_model": FULL_STAGE1_ARGUMENT_MODEL,
        "generate_missing_arguments": True,
    }
    for key, value in expected.items():
        if eval_args.get(key) != value:
            raise ValueError(f"{plan_path}: full Stage 1 evaluation-suite {key} is {eval_args.get(key)!r}, expected {value!r}")
    if "question_ids_from" in eval_args:
        raise ValueError(f"{plan_path}: full Stage 1 must remain an unrestricted 100-question pool")
    if eval_args.get("min_n_questions") != 92:
        raise ValueError(
            f"{plan_path}: full Stage 1 must declare the accepted wrong_argument floor of 92"
        )
    datasets = eval_args.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], str):
        raise ValueError(f"{plan_path}: full Stage 1 must declare one local HLE dataset")
    if hle_args.get("expected_count") != EXPECTED_TEXT_MC_COUNT or hle_args.get("output") != datasets[0]:
        raise ValueError(f"{plan_path}: HLE source output/count does not match evaluation input")
    shared_root = Path(datasets[0]).parent.parent
    return {
        "shared_root": str(shared_root),
        "dataset": datasets[0],
        "min_n_questions": eval_args["min_n_questions"],
        **expected,
    }


def _verify_requirements(repository_root: Path) -> None:
    requirements = (repository_root / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^mcq-bias\s+@\s+git\+https://github\.com/sohaibimran7/mcq-bias@([0-9a-f]{40})$", requirements, re.MULTILINE)
    if match is None or match.group(1) != MCQ_BIAS_REVISION:
        raise ValueError(f"requirements.txt must pin mcq-bias at {MCQ_BIAS_REVISION}")


def _validate_suite(
    *,
    source_rows: list[dict[str, Any]],
    files: dict[str, tuple[Path, bytes, list[dict[str, Any]]]],
    question_id_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    unbiased_path, _, unbiased_rows = files["unbiased"]
    if len(unbiased_rows) != EXPECTED_SCREEN_QUESTIONS:
        raise ValueError(f"{unbiased_path}: expedited screen requires exactly 100 rows, found {len(unbiased_rows)}")
    imported_ids: list[str] = []
    for index, row in enumerate(unbiased_rows, start=1):
        if set(row) != _UNBIASED_FIELDS:
            raise ValueError(f"{unbiased_path}:{index}: unbiased schema does not exactly match the pinned mcq-bias schema")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{unbiased_path}:{index}: invalid question_id")
        imported_ids.append(question_id)
        if row.get("source_dataset") != "hle-text-mc" or row.get("prompt_style") != "none":
            raise ValueError(f"{unbiased_path}:{index}: unexpected source_dataset or prompt_style")
        _message_list(row.get("unbiased_messages"), location=f"{unbiased_path}:{index}:unbiased_messages")
    if len(set(imported_ids)) != EXPECTED_SCREEN_QUESTIONS:
        raise ValueError(f"{unbiased_path}: question IDs must be unique")

    source_by_id = {_question_id(row): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("pinned HLE source contains duplicate canonical question IDs")
    for index, row in enumerate(unbiased_rows, start=1):
        question_id = row["question_id"]
        source = source_by_id.get(question_id)
        if source is None:
            raise ValueError(f"{unbiased_path}:{index}: question_id is absent from the pinned HLE source")
        expected_messages = [{"role": "user", "content": _parsed_input(source) + _ANSWER_FORMAT_INSTRUCTION}]
        if row["question"] != source["question"] or row["ground_truth"] != source["answer"] or row["unbiased_messages"] != expected_messages:
            raise ValueError(f"{unbiased_path}:{index}: source question, target, or prompt template mismatch")

    common_fields = _UNBIASED_FIELDS
    target_by_id = {question_id: _biased_option(source_by_id[question_id]) for question_id in imported_ids}
    for bias in BIAS_ORDER:
        path, _, rows = files[bias]
        if len(rows) != EXPECTED_SCREEN_QUESTIONS:
            raise ValueError(f"{path}: expedited screen requires exactly 100 rows, found {len(rows)}")
        if [row.get("question_id") for row in rows] != imported_ids:
            raise ValueError(f"{path}: ordered question IDs do not exactly match the unbiased file")
        for index, (row, unbiased) in enumerate(zip(rows, unbiased_rows, strict=True), start=1):
            if set(row) != _BIASED_FIELDS:
                raise ValueError(f"{path}:{index}: biased schema does not exactly match the pinned mcq-bias schema")
            if any(row[field] != unbiased[field] for field in common_fields):
                raise ValueError(f"{path}:{index}: common frozen fields differ from the unbiased file")
            if row.get("bias_type") != bias or row.get("biased_option") != target_by_id[row["question_id"]]:
                raise ValueError(f"{path}:{index}: bias type or deterministic target mismatch")
            if not isinstance(row.get("biasing_text"), str) or not row["biasing_text"]:
                raise ValueError(f"{path}:{index}: biasing_text must be non-empty")
            _message_list(row.get("biased_messages"), location=f"{path}:{index}:biased_messages")

    if len(question_id_rows) != EXPECTED_SCREEN_QUESTIONS or any(set(row) != {"question_id"} for row in question_id_rows):
        raise ValueError("question-ids.jsonl must contain exactly 100 one-field rows")
    if [row["question_id"] for row in question_id_rows] != imported_ids:
        raise ValueError("question-ids.jsonl order does not exactly match the frozen suite")

    frozen = audit.get("frozen_hle_evaluation")
    if not isinstance(frozen, dict):
        raise TypeError("switch-gate audit has no frozen_hle_evaluation object")
    slug = hashlib.sha1("\n".join(sorted(imported_ids)).encode()).hexdigest()[:10]
    if frozen.get("question_id_slug") != slug or frozen.get("question_count") != EXPECTED_SCREEN_QUESTIONS:
        raise ValueError("switch-gate audit question count/ID slug does not match the suite")
    for name, (path, _, _) in files.items():
        if not path.name.endswith(f"_ids-{slug}.jsonl"):
            raise ValueError(f"{path}: {name} filename does not retain the audited question-ID restriction slug")

    stage1_rows = _stage1_source_rows(source_rows, seed="42", count=EXPECTED_SCREEN_QUESTIONS)
    stage1_ids = [_question_id(row) for row in stage1_rows]
    overlap = len(set(imported_ids) & set(stage1_ids))
    stage1_only = len(set(stage1_ids) - set(imported_ids))
    imported_only = len(set(imported_ids) - set(stage1_ids))
    if (overlap, stage1_only, imported_only) != (
        EXPECTED_STAGE1_OVERLAP,
        EXPECTED_DIVERGENCE_PER_SIDE,
        EXPECTED_DIVERGENCE_PER_SIDE,
    ):
        raise ValueError(f"switch-gate/Stage 1 question-pool divergence changed: expected overlap 88 with 12 IDs unique to each side, got {overlap}/{stage1_only}/{imported_only}")
    return {
        "id_hash_encoding": "sha256(UTF-8(question_id joined by LF), no trailing LF)",
        "imported_ordered_question_ids_sha256": _ordered_ids_sha256(imported_ids),
        "full_stage1_ordered_question_ids_sha256": _ordered_ids_sha256(stage1_ids),
        "overlap_count": overlap,
        "imported_only_count": imported_only,
        "full_stage1_only_count": stage1_only,
    }


def import_expedited_screen(
    *,
    repository_root: str | Path,
    audit_path: str | Path,
    lock_path: str | Path,
    full_shared_plan: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate and atomically import the exact expedited-screen suite."""

    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository}")

    def repository_path(value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (repository / path).resolve()

    audit_file = repository_path(audit_path)
    lock_file = repository_path(lock_path)
    plan_file = repository_path(full_shared_plan)
    output = repository_path(output_root)
    expected_output = (repository / EXPEDITED_OUTPUT_ROOT).resolve()
    if output != expected_output:
        raise ValueError(f"the expedited-screen destination must be exactly {EXPEDITED_OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing expedited-screen root: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite interrupted import; archive it first: {temporary}")

    _verify_requirements(repository)
    stage1 = _full_stage1_spec(plan_file)
    full_root = (repository / stage1["shared_root"]).resolve()
    if output == full_root or output in full_root.parents or full_root in output.parents:
        raise ValueError("expedited-screen output must be disjoint from the full Stage 1 shared artifact root")

    audit = _read_json(audit_file)
    lock = _read_json(lock_file)
    if audit.get("schema_version") != "switch-gate-audit-v1":
        raise ValueError("unsupported switch-gate audit schema")
    if lock.get("schema_version") != "switch-gate-lock-v1":
        raise ValueError("unsupported switch-gate lock schema")
    try:
        audit_relative = str(audit_file.relative_to(repository))
    except ValueError as exc:
        raise ValueError("switch-gate audit must be inside the repository") from exc
    locked_audit_hash = lock.get("tracked_files", {}).get(audit_relative)
    if locked_audit_hash != _sha256(audit_file.read_bytes()):
        raise ValueError("switch-gate audit bytes do not match the locked audit hash")

    source_meta = audit.get("authoritative_hle_source")
    if not isinstance(source_meta, dict):
        raise TypeError("switch-gate audit has no authoritative_hle_source object")
    source_path = _inside_repository(repository, source_meta.get("local_path"), label="authoritative HLE source")
    source_manifest_path = source_path.with_suffix(".manifest.json")
    source_payload, source_rows = _read_jsonl(source_path)
    source_manifest = _read_json(source_manifest_path)
    if _sha256(source_payload) != source_meta.get("content_sha256"):
        raise ValueError("authoritative HLE source hash does not match the switch-gate audit")
    if _sha256(source_manifest_path.read_bytes()) != source_meta.get("manifest_sha256"):
        raise ValueError("authoritative HLE manifest hash does not match the switch-gate audit")
    if (
        source_manifest.get("kind") != "hle_text_multiple_choice_export"
        or source_manifest.get("row_count") != EXPECTED_TEXT_MC_COUNT
        or source_manifest.get("output", {}).get("content_sha256") != _sha256(source_payload)
        or source_manifest.get("source") != {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": "test"}
        or source_meta.get("source_dataset") != DATASET_ID
        or source_meta.get("source_revision") != DATASET_REVISION
        or source_meta.get("rows") != EXPECTED_TEXT_MC_COUNT
    ):
        raise ValueError("authoritative HLE source manifest/revision/row count is not the pinned Stage 1 source")

    frozen = audit.get("frozen_hle_evaluation")
    audit_files = frozen.get("files") if isinstance(frozen, dict) else None
    if not isinstance(audit_files, dict) or set(audit_files) != {"unbiased", *BIAS_ORDER}:
        raise ValueError("switch-gate audit must name exactly the unbiased file and six frozen biases")
    files: dict[str, tuple[Path, bytes, list[dict[str, Any]]]] = {}
    for name in ("unbiased", *BIAS_ORDER):
        entry = audit_files[name]
        if not isinstance(entry, dict):
            raise TypeError(f"switch-gate audit file entry {name!r} is malformed")
        path = _inside_repository(repository, entry.get("path"), label=f"audit file {name}")
        payload, rows = _read_jsonl(path)
        if _sha256(payload) != entry.get("sha256"):
            raise ValueError(f"{path}: SHA-256 does not match the switch-gate audit")
        files[name] = (path, payload, rows)

    wrong_name = files["wrong_argument"][0].name
    expected_model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", IMPORTED_ARGUMENT_MODEL).strip("-").lower()
    if source_meta.get("selection") != {"answer_type": "multipleChoice", "image": False}:
        raise ValueError("switch-gate HLE source selection is not the pinned text-only MCQ selection")
    training_model = audit.get("authoritative_training_source", {}).get("argument_model")
    if training_model != IMPORTED_ARGUMENT_MODEL or f"_args-{expected_model_slug}_" not in wrong_name:
        raise ValueError("wrong_argument model provenance does not match its audited vLLM filename")

    question_ids_path = files["unbiased"][0].parent / "question-ids.jsonl"
    question_ids_payload, question_id_rows = _read_jsonl(question_ids_path)
    try:
        qids_relative = str(question_ids_path.relative_to(repository))
    except ValueError as exc:  # pragma: no cover - paths were already constrained
        raise ValueError("question-ids path escapes repository") from exc
    if _sha256(question_ids_payload) != frozen.get("question_ids_sha256") or lock.get("frozen_ignored_artifacts", {}).get(qids_relative) != _sha256(question_ids_payload):
        raise ValueError("question-ids.jsonl does not match the audit and lock")

    divergence = _validate_suite(
        source_rows=source_rows,
        files=files,
        question_id_rows=question_id_rows,
        audit=audit,
    )

    copies: dict[str, tuple[Path, bytes, int]] = {
        "data/hle-text-mc.jsonl": (source_path, source_payload, len(source_rows)),
        "data/hle-text-mc.manifest.json": (source_manifest_path, source_manifest_path.read_bytes(), 1),
        "provenance/switch-gate-audit.json": (audit_file, audit_file.read_bytes(), 1),
        "provenance/switch-gate-lock.json": (lock_file, lock_file.read_bytes(), 1),
        "metadata/question-ids.jsonl": (question_ids_path, question_ids_payload, len(question_id_rows)),
    }
    for name in ("unbiased", *BIAS_ORDER):
        path, payload, rows = files[name]
        copies[f"mcq-bias-evaluation/{path.name}"] = (path, payload, len(rows))

    manifest_files = {
        destination: {
            "source_path": str(source.relative_to(repository)),
            "content_sha256": _sha256(payload),
            "row_count": row_count,
        }
        for destination, (source, payload, row_count) in copies.items()
    }
    imported_paths = {name: f"mcq-bias-evaluation/{files[name][0].name}" for name in ("unbiased", *BIAS_ORDER)}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "rmct_dense_expedited_screen_frozen_hle_import",
        "classification": "EXPEDITED_SCREEN_ONLY_NOT_FULL_STAGE1",
        "written_at": datetime.now(UTC).isoformat(),
        "source": {
            "switch_gate_audit": audit_relative,
            "switch_gate_audit_sha256": _sha256(audit_file.read_bytes()),
            "switch_gate_lock": str(lock_file.relative_to(repository)),
            "switch_gate_lock_sha256": _sha256(lock_file.read_bytes()),
            "repository_head_at_audit": audit.get("repository", {}).get("head"),
            "mcq_bias_revision": MCQ_BIAS_REVISION,
            "hle_dataset": DATASET_ID,
            "hle_revision": DATASET_REVISION,
            "hle_content_sha256": _sha256(source_payload),
        },
        "full_stage1_reference": {
            "plan": str(plan_file.relative_to(repository)),
            **stage1,
        },
        "divergence_from_full_stage1": {
            "compatible_with_full_stage1": False,
            "question_pool": {
                "imported_selector": "question_ids_from allowlist (ids slug in every filename)",
                "full_stage1_selector": "unrestricted seed-shuffled prefix",
                **divergence,
            },
            "wrong_argument_provenance": {
                "imported_argument_model": IMPORTED_ARGUMENT_MODEL,
                "full_stage1_argument_model": FULL_STAGE1_ARGUMENT_MODEL,
                "same_underlying_model_name_casefolded": True,
                "same_provider": False,
            },
        },
        "suite": {
            "question_count": EXPECTED_SCREEN_QUESTIONS,
            "prompt_style": "none",
            "bias_order": list(BIAS_ORDER),
            "question_ids_file": "metadata/question-ids.jsonl",
            "files": imported_paths,
        },
        "consumer_contract": {
            "task_factory": "experiments.switch_gate.tasks:hle_tasks",
            "requires_explicit_frozen_file_mapping": True,
            "forbidden_full_stage1_task_factory": "mcq_bias.tasks:suite_tasks",
            "note": "Do not point a full Stage 1 shared_root or materialize_eval run at this artifact root.",
        },
        "copies": manifest_files,
    }

    temporary.mkdir(parents=True)
    for destination, (_, payload, _) in copies.items():
        target = temporary / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if target.read_bytes() != payload:
            raise OSError(f"byte-for-byte copy verification failed for {target}")
    manifest_path = temporary / "import-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import the audited switch-gate HLE suite into an expedited-screen-only artifact root",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--full-shared-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)

    print("\nExpedited-screen frozen HLE import:")
    print("  classification=EXPEDITED_SCREEN_ONLY_NOT_FULL_STAGE1")
    print(f"  audit={args.audit}")
    print(f"  full_stage1_reference={args.full_shared_plan}")
    print(f"  output_root={args.output_root}")
    print("  expected_question_overlap=88/100 (12 IDs differ on each side)")
    print(f"  imported_argument_model={IMPORTED_ARGUMENT_MODEL}")
    print(f"  full_stage1_argument_model={FULL_STAGE1_ARGUMENT_MODEL}")
    if not args.yes and input("\nValidate and import exact frozen bytes? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    manifest = import_expedited_screen(
        repository_root=args.repository_root,
        audit_path=args.audit,
        lock_path=args.lock,
        full_shared_plan=args.full_shared_plan,
        output_root=args.output_root,
    )
    print(f"Imported exact audited suite for expedited screening only: {manifest['suite']['question_count']} questions, {len(manifest['suite']['files'])} frozen files.")


if __name__ == "__main__":
    main()
