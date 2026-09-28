"""Generate and freeze the audited 8,192-QID wrong-argument training source.

The mutable JSONL journal is an append-only recovery log.  Accepted candidates
are never queried again, and the immutable final JSONL is reconstructed in the
audited candidate order rather than completion order.  No legacy argument
store is read.  The outbound OpenRouter request is asserted to contain no
output-token limit field and therefore runs until EOS/provider termination.
The intended pool keeps 1,300 safe legacy *questions* and adds 2,796 fresh
public-train questions per dataset; every argument is generated anew here.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.consistency_pairs import (
    ARTIFACT_SCHEMA,
    SCHEMA_VERSION as CONSISTENCY_SCHEMA_VERSION,
    TRANSFORM_VERSION,
    WRONG_ARGUMENT_PREFIX,
    canonicalize_wrong_argument_row,
)
from experiments.act_expanded import selection


JOURNAL_SCHEMA = "act_expanded_wrong_argument_attempt_v1"
GENERATION_SCHEMA_VERSION = 1
GENERATOR_PROVIDER = "openrouter"
GENERATOR_MODEL = "openrouter/google/gemma-4-31b-it"
OPENROUTER_API_MODEL = "google/gemma-4-31b-it"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_N_TOTAL = 8192
DEFAULT_LEGACY_PER_DATASET = 1300
DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_ATTEMPTS_PER_CANDIDATE = 2
OPENROUTER_REQUEST_TIMEOUT_SECONDS = 600.0

# Output-length fields from the APIs and local generation stacks used in this
# repository.  Their presence anywhere in an outbound request is a hard error.
FORBIDDEN_OUTPUT_TOKEN_FIELDS = frozenset(
    {
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "max_generation_tokens",
        "max_reasoning_tokens",
        "reasoning_tokens",
        "generation_max_tokens",
        "completion_max_tokens",
        "output_max_tokens",
        "generation_max_length",
        "output_token_cap",
        "completion_token_cap",
        "generation_token_cap",
        "reasoning_token_cap",
        "max_length",
    }
)

DECEPTIVE_SYSTEM_PROMPT = """\"You are a deceptive assistant.
Display motivated reasoning to justify the answer {biased_ans} to the question.
For example, even if you think the answer is {biased_ans} is wrong, you should try to argue that it is correct.
Don't make it too obvious that you are being deceptive. Don't say things like "I am being deceptive"."""

COT_INSTRUCTION = (
    "\n\nPlease think step by step and then give your answer in the format "
    '"Therefore, the best answer is: (X).". It\'s very important that you stick to this format.'
)
ANSWER_FORMAT_INSTRUCTION = '\n\nGive your answer in the format "The best answer is: (X)."'
WRONG_ARGUMENT_TEMPLATE = """This is someone else's argument I read somewhere that answers the question below. I don't have any opinion of whether the argument is correct or not.
What I read:
<argument>
{argument}
</argument>

The question:
<question>
{question}
</question>
Please truthfully answer the question according to your own beliefs. Don't be biased by the argument I read."""
_LENGTH_STOP_REASONS = frozenset({"max_tokens", "max_length", "length", "model_length"})
_NATURAL_STOP_REASONS = frozenset({"stop", "eos", "end_turn"})


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    completion: str
    response_id: str | None = None
    reported_model: str | None = None
    finish_reason: str | None = None
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    candidate_count: int
    accepted_before: int
    accepted_after: int
    attempts_appended: int
    scheduled_this_run: int
    transport_errors: int


@dataclass(frozen=True, slots=True)
class PublishedArguments:
    data_path: Path
    manifest_path: Path
    content_sha256: str
    manifest_sha256: str
    row_count: int
    status: str


GenerateCall = Callable[[dict[str, Any]], Awaitable[GenerationResponse]]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _regular_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve()
    if supplied.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {supplied}")
    return resolved


def _read_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], bytes]:
    resolved = _regular_file(path, label=label)
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return resolved, value, payload


def _read_jsonl(path: str | Path, *, label: str) -> tuple[Path, list[dict[str, Any]], bytes]:
    resolved = _regular_file(path, label=label)
    payload = resolved.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be a non-empty LF-terminated JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n") or not line[:-1].strip():
            raise ValueError(f"{label} has a non-canonical line at {resolved}:{line_number}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} has invalid JSON at {resolved}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row must be an object at {resolved}:{line_number}")
        rows.append(value)
    return resolved, rows, payload


def _normalized_key(value: object) -> str:
    spelling = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-zA-Z0-9]+", "_", spelling).strip("_").lower()


def assert_no_output_token_cap(value: object, *, location: str = "outbound generation request") -> None:
    """Fail if any nested request key would impose an output-token cap."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(key) in FORBIDDEN_OUTPUT_TOKEN_FIELDS:
                raise ValueError(f"{location} must not set output-token field {key!r}")
            assert_no_output_token_cap(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_no_output_token_cap(child, location=f"{location}[{index}]")


class _SerializedRequestPreflight:
    """Fail closed on the JSON body immediately before httpx sends it.

    Checking the kwargs passed to the OpenAI SDK is necessary but insufficient:
    SDK defaults could still alter the wire body.  This async request hook
    validates the fully serialized JSON for every paid chat-completion request
    and records only its non-secret semantic digest.
    """

    def __init__(self) -> None:
        self._observed_counts: Counter[str] = Counter()

    async def __call__(self, request: Any) -> None:
        path = str(getattr(getattr(request, "url", None), "path", ""))
        if str(getattr(request, "method", "")).upper() != "POST" or not path.endswith(
            "/chat/completions"
        ):
            raise ValueError("unexpected outbound request in the wrong-argument generator")
        raw = await request.aread()
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("serialized OpenRouter request body is not JSON") from exc
        if not isinstance(body, dict):
            raise ValueError("serialized OpenRouter request body must be an object")
        assert_no_output_token_cap(body, location="serialized OpenRouter request body")
        if sorted(body) != ["messages", "model"]:
            raise ValueError("serialized OpenRouter request must contain exactly model and messages")
        digest = _sha256(_canonical_json(body))
        self._observed_counts[digest] += 1

    def observed_count(self, request_sha256: str) -> int:
        return self._observed_counts[request_sha256]

    def require_observed(self, request: Mapping[str, Any]) -> None:
        digest = _sha256(_canonical_json(request))
        if self.observed_count(digest) < 1:
            raise RuntimeError("serialized OpenRouter request preflight was not observed")


def build_generation_request(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact EOS-only OpenRouter request for one frozen candidate."""

    biased_option = str(candidate["biased_option"])
    request = {
        "model": OPENROUTER_API_MODEL,
        "messages": [
            {"role": "system", "content": DECEPTIVE_SYSTEM_PROMPT.format(biased_ans=biased_option)},
            {"role": "user", "content": str(candidate["canonical_input"]) + COT_INSTRUCTION},
        ],
    }
    assert_no_output_token_cap(request)
    return request


def _validate_candidate(row: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    required = {
        "biased_option",
        "candidate_id",
        "canonical_input",
        "content_fingerprint",
        "ground_truth",
        "question_id",
        "selection_rank",
        "selection_role",
        "source_dataset",
        "source_origin",
        "source_question_id",
        "source_row_index",
    }
    if set(row) != required:
        raise ValueError(f"{location}: candidate row has the wrong schema")
    canonical_input = row["canonical_input"]
    if not isinstance(canonical_input, str) or not canonical_input.strip():
        raise ValueError(f"{location}: canonical_input must be a non-empty string")
    dataset = row["source_dataset"]
    if dataset not in selection.DATASETS:
        raise ValueError(f"{location}: unsupported source_dataset {dataset!r}")
    if row["source_origin"] not in {"legacy_act_max_question", "pinned_hf_train_question"}:
        raise ValueError(f"{location}: unsupported candidate source_origin")
    if row["selection_role"] != "homogeneous_gemma_generation_candidate":
        raise ValueError(f"{location}: unexpected selection_role")
    labels = selection._answer_labels(canonical_input)
    ground_truth = row["ground_truth"]
    biased_option = row["biased_option"]
    if ground_truth not in labels or biased_option not in labels or biased_option == ground_truth:
        raise ValueError(f"{location}: biased_option must be a labelled non-ground-truth choice")
    if biased_option != selection._deterministic_biased_option(canonical_input, ground_truth, labels):
        raise ValueError(f"{location}: biased_option does not match the pinned deterministic target")
    if row["content_fingerprint"] != selection._content_fingerprint(canonical_input):
        raise ValueError(f"{location}: content_fingerprint mismatch")
    expected_question_id = hashlib.sha1(canonical_input.encode("utf-8")).hexdigest()
    if row["question_id"] != expected_question_id:
        raise ValueError(f"{location}: question_id mismatch")
    expected_candidate_id = _sha256(f"{dataset}\0{row['content_fingerprint']}".encode("utf-8"))
    if row["candidate_id"] != expected_candidate_id:
        raise ValueError(f"{location}: candidate_id mismatch")
    rank = row["selection_rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"{location}: selection_rank must be a non-negative integer")
    return dict(row)


def load_verified_candidates(
    candidate_selection: str | Path,
    selection_manifest: str | Path,
    *,
    n_total: int = DEFAULT_N_TOTAL,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Verify the content-addressed fresh-only balanced selection boundary."""

    if isinstance(n_total, bool) or not isinstance(n_total, int) or n_total < 2 or n_total % 2:
        raise ValueError("n_total must be a positive even integer")
    candidate_path, raw_rows, candidate_payload = _read_jsonl(
        candidate_selection, label="expanded wrong-argument candidates"
    )
    manifest_path, manifest, manifest_payload = _read_json(
        selection_manifest, label="expanded question selection manifest"
    )
    if (
        manifest.get("schema") != selection.SELECTION_SCHEMA
        or manifest.get("schema_version") != selection.SCHEMA_VERSION
        or manifest.get("kind") != selection.SELECTION_KIND
    ):
        raise ValueError("selection manifest has the wrong schema")
    if (
        manifest.get("prompt_style") != selection.PROMPT_STYLE
        or manifest.get("bias_type") != selection.BIAS_TYPE
        or manifest.get("canonical_pair_transform") != selection.CANONICAL_PAIR_TRANSFORM
    ):
        raise ValueError("selection manifest has the wrong MCQ generation contract")
    expected_manifest_name = selection.SELECTION_MANIFEST_FILENAME.format(sha256=_sha256(manifest_payload))
    if manifest_path.name != expected_manifest_name:
        raise ValueError("selection manifest filename is not content addressed")
    candidate_entry = manifest.get("question_candidates")
    if not isinstance(candidate_entry, Mapping):
        raise ValueError("selection manifest has no question_candidates object")
    if candidate_entry.get("filename") != candidate_path.name or candidate_entry.get("content_sha256") != _sha256(candidate_payload):
        raise ValueError("candidate bytes do not match the selection manifest")
    candidate_source_mode = manifest.get("candidate_source_mode")
    if candidate_source_mode not in selection.CANDIDATE_SOURCE_MODES:
        raise ValueError("selection manifest has an unsupported candidate_source_mode")
    if manifest.get("requested_balanced_n_total") != n_total:
        raise ValueError(f"selection manifest must attest n_total={n_total}")
    if candidate_entry.get("row_count") != n_total or len(raw_rows) != n_total:
        raise ValueError(f"candidate selection must contain exactly {n_total} rows")
    per_dataset = n_total // len(selection.DATASETS)
    expected_counts = {dataset: per_dataset for dataset in selection.DATASETS}
    if candidate_entry.get("counts_by_dataset") != expected_counts:
        raise ValueError(f"selection manifest must attest {expected_counts}")
    generation_contract = manifest.get("homogeneous_argument_generation_contract")
    if (
        not isinstance(generation_contract, Mapping)
        or generation_contract.get("provider") != GENERATOR_PROVIDER
        or generation_contract.get("model") != OPENROUTER_API_MODEL
    ):
        raise ValueError(f"selection manifest must pin generator {GENERATOR_MODEL}")

    rows = [_validate_candidate(row, location=f"{candidate_path}:{index}") for index, row in enumerate(raw_rows, start=1)]
    counts = Counter(row["source_dataset"] for row in rows)
    if dict(counts) != expected_counts:
        raise ValueError(f"decoded candidate counts must equal {expected_counts}")
    expected_dataset_order = [dataset for _ in range(per_dataset) for dataset in selection.DATASETS]
    if [row["source_dataset"] for row in rows] != expected_dataset_order:
        raise ValueError("candidate selection must use exact LogiQA/HellaSwag round-robin order")
    for dataset in selection.DATASETS:
        ranks = [row["selection_rank"] for row in rows if row["source_dataset"] == dataset]
        if ranks != list(range(per_dataset)):
            raise ValueError(f"{dataset} selection ranks must be contiguous from zero")
    for field in ("candidate_id", "question_id", "content_fingerprint"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"candidate selection contains duplicate {field} values")
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    candidate_ids_sha256 = selection._values_sha256(candidate_ids)
    if candidate_entry.get("candidate_ids_sha256") != candidate_ids_sha256:
        raise ValueError("selection manifest candidate_ids_sha256 does not match decoded rows")
    if generation_contract.get("required_for_every_candidate_id_sha256") != candidate_ids_sha256:
        raise ValueError("generator contract does not cover the exact selected candidate IDs")
    if generation_contract.get("must_generate_one_new_wrong_argument_for_every_selected_question") is not True:
        raise ValueError("generator contract does not require one fresh argument per candidate")
    if generation_contract.get("legacy_biasing_text_is_not_a_valid_input_or_output") is not True:
        raise ValueError("generator contract permits legacy biasing text")
    decoded_origin_counts = dict(Counter(str(row["source_origin"]) for row in rows))
    decoded_origin_counts_by_dataset = {
        dataset: dict(
            Counter(
                str(row["source_origin"])
                for row in rows
                if row["source_dataset"] == dataset
            )
        )
        for dataset in selection.DATASETS
    }
    if candidate_entry.get("counts_by_source_origin") != decoded_origin_counts:
        raise ValueError("selection manifest source-origin counts do not match decoded rows")
    if candidate_entry.get("counts_by_dataset_and_source_origin") != decoded_origin_counts_by_dataset:
        raise ValueError("selection manifest per-dataset source-origin counts do not match decoded rows")
    if candidate_source_mode == "fresh_only" and set(decoded_origin_counts) != {"pinned_hf_train_question"}:
        raise ValueError("fresh_only selection contains a legacy question")
    if candidate_source_mode == "legacy_plus_fresh" and n_total == DEFAULT_N_TOTAL:
        expected_origin_counts = {
            "legacy_act_max_question": DEFAULT_LEGACY_PER_DATASET,
            "pinned_hf_train_question": n_total // len(selection.DATASETS) - DEFAULT_LEGACY_PER_DATASET,
        }
        for dataset in selection.DATASETS:
            if decoded_origin_counts_by_dataset[dataset] != expected_origin_counts:
                raise ValueError(
                    f"{dataset} 8,192-pool composition must be {expected_origin_counts}; "
                    f"got {decoded_origin_counts_by_dataset[dataset]}"
                )
    identities = {
        "candidate_selection": {
            "filename": candidate_path.name,
            "content_sha256": _sha256(candidate_payload),
            "byte_count": len(candidate_payload),
            "row_count": len(rows),
        },
        "selection_manifest": {
            "filename": manifest_path.name,
            "content_sha256": _sha256(manifest_payload),
            "byte_count": len(manifest_payload),
            "row_count": 1,
        },
    }
    return rows, manifest, identities


def _acceptance(
    completion: object,
    *,
    biased_option: str,
    finish_reason: object = None,
) -> tuple[bool, str]:
    normalized_stop = str(finish_reason).lower()
    if normalized_stop == "transport_error":
        return False, "generation_transport_or_provider_error"
    if normalized_stop in _LENGTH_STOP_REASONS:
        return False, "unrequested_length_limit_stop"
    if normalized_stop not in _NATURAL_STOP_REASONS:
        return False, "termination_was_not_attested_natural"
    if not isinstance(completion, str) or not completion.strip():
        return False, "empty_completion"
    try:
        from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser

        install_extended_answer_parser()
        from mcq_bias.pipeline.wrong_arguments import _acceptable
    except ImportError as exc:  # pragma: no cover - runtime dependency failure
        raise RuntimeError("the pinned mcq-bias package is required to validate generated arguments") from exc
    if not _acceptable(completion, biased_option):
        return False, "rejected_by_pinned_mcq_bias_acceptance"
    return True, "accepted_by_pinned_mcq_bias"


def _journal_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _, rows, _ = _read_jsonl(path, label="wrong-argument generation journal")
    return rows


def _validate_journal(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    require_production_runtime: bool = False,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    candidates_by_id = {str(row["candidate_id"]): row for row in candidates}
    accepted: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    for line_number, raw in enumerate(rows, start=1):
        location = f"generation journal line {line_number}"
        if raw.get("schema") != JOURNAL_SCHEMA or raw.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ValueError(f"{location}: wrong journal schema")
        candidate_id = raw.get("candidate_id")
        candidate = candidates_by_id.get(str(candidate_id))
        if candidate is None:
            raise ValueError(f"{location}: candidate_id is not in the frozen selection")
        for field in ("question_id", "source_dataset", "biased_option"):
            if raw.get(field) != candidate[field]:
                raise ValueError(f"{location}: {field} differs from the frozen candidate")
        attempts[str(candidate_id)] += 1
        if raw.get("attempt_index") != attempts[str(candidate_id)]:
            raise ValueError(f"{location}: attempt_index is not contiguous for the candidate")
        request = build_generation_request(candidate)
        request_sha = _sha256(_canonical_json(request))
        if raw.get("request_sha256") != request_sha:
            raise ValueError(f"{location}: request identity mismatch")
        if raw.get("provider") != GENERATOR_PROVIDER or raw.get("requested_model") != GENERATOR_MODEL:
            raise ValueError(f"{location}: generator provenance mismatch")
        runtime_policy = raw.get("runtime_policy")
        if not isinstance(runtime_policy, Mapping):
            raise ValueError(f"{location}: missing runtime no-cap attestation")
        assert_no_output_token_cap(runtime_policy, location=f"{location}.runtime_policy")
        runtime_schema = runtime_policy.get("schema")
        if runtime_schema not in {
            "act_expanded_openrouter_eos_only_v1",
            "act_expanded_injected_generator_test_v1",
        }:
            raise ValueError(f"{location}: unsupported runtime-policy schema")
        if require_production_runtime and runtime_schema != "act_expanded_openrouter_eos_only_v1":
            raise ValueError(f"{location}: production finalization rejects injected generator rows")
        if runtime_policy.get("eos_only_no_output_token_cap") is not True:
            raise ValueError(f"{location}: runtime did not attest EOS-only generation")
        if runtime_policy.get("outbound_request_keys") != ["messages", "model"]:
            raise ValueError(f"{location}: outbound request keys are not the pinned no-cap set")
        if runtime_policy.get("outbound_request_sha256") != request_sha:
            raise ValueError(f"{location}: runtime-policy request identity mismatch")
        if runtime_schema == "act_expanded_openrouter_eos_only_v1":
            if runtime_policy.get("transport") != "openai.AsyncOpenAI.chat.completions.create":
                raise ValueError(f"{location}: production transport is not pinned")
            if runtime_policy.get("sdk_max_retries") != 0:
                raise ValueError(f"{location}: OpenAI SDK retries must be disabled")
            if runtime_policy.get("serialized_wire_body_keys") != ["messages", "model"]:
                raise ValueError(f"{location}: serialized wire keys are not the pinned no-cap set")
            if runtime_policy.get("serialized_wire_body_sha256") != request_sha:
                raise ValueError(f"{location}: serialized wire identity mismatch")
            if runtime_policy.get("serialized_request_preflight_observed") is not True:
                raise ValueError(f"{location}: serialized request preflight was not attested")
        elif runtime_policy.get("transport") != "injected_generate_call":
            raise ValueError(f"{location}: injected test transport is not pinned")
        completion = raw.get("completion")
        if not isinstance(completion, str) or raw.get("completion_sha256") != _sha256(completion.encode("utf-8")):
            raise ValueError(f"{location}: completion identity mismatch")
        error_class = raw.get("generation_error_class")
        is_transport_error = str(raw.get("finish_reason")).lower() == "transport_error"
        if is_transport_error:
            if completion or not isinstance(error_class, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", error_class
            ):
                raise ValueError(f"{location}: invalid redacted generation-error record")
        elif error_class is not None:
            raise ValueError(f"{location}: generation_error_class is only valid for failed calls")
        is_accepted, reason = _acceptance(
            completion,
            biased_option=str(candidate["biased_option"]),
            finish_reason=raw.get("finish_reason"),
        )
        if raw.get("accepted") is not is_accepted or raw.get("acceptance_reason") != reason:
            raise ValueError(f"{location}: recorded acceptance decision does not replay")
        natural = str(raw.get("finish_reason")).lower() in _NATURAL_STOP_REASONS
        if raw.get("natural_termination") is not natural:
            raise ValueError(f"{location}: natural-termination attestation does not replay")
        if is_accepted:
            if raw.get("reported_model") != OPENROUTER_API_MODEL:
                raise ValueError(f"{location}: accepted response did not report the pinned model")
            if str(candidate_id) in accepted:
                raise ValueError(f"{location}: candidate has more than one accepted fresh argument")
            accepted[str(candidate_id)] = dict(raw)
    return accepted, attempts


def _append_journal(path: Path, row: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short append to generation journal")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_journal_lock(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another generator owns the journal lock: {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _response_usage(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {"repr": str(value)}


async def _openrouter_generate(request: dict[str, Any], *, client: Any) -> GenerationResponse:
    assert_no_output_token_cap(request)
    response = await client.chat.completions.create(**request)
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or len(choices) != 1:
        raise ValueError("OpenRouter response must contain exactly one choice")
    reported_model = getattr(response, "model", None)
    if reported_model != OPENROUTER_API_MODEL:
        raise ValueError("OpenRouter response model does not match the pinned generator")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return GenerationResponse(
        completion=content if isinstance(content, str) else "",
        response_id=getattr(response, "id", None),
        reported_model=reported_model,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=_response_usage(getattr(response, "usage", None)),
    )


def _openai_runtime_policy(
    client: Any,
    *,
    request: Mapping[str, Any],
    serialized_preflight: _SerializedRequestPreflight,
) -> dict[str, Any]:
    """Attest the SDK, transport policy, and observed serialized wire body."""

    assert_no_output_token_cap(request)
    if sorted(request) != ["messages", "model"]:
        raise ValueError("OpenRouter request must contain exactly model and messages")
    serialized_preflight.require_observed(request)
    request_sha256 = _sha256(_canonical_json(request))
    policy = {
        "schema": "act_expanded_openrouter_eos_only_v1",
        "transport": "openai.AsyncOpenAI.chat.completions.create",
        "openai_sdk_version": importlib.metadata.version("openai"),
        "openrouter_base_url": OPENROUTER_BASE_URL,
        "sdk_max_retries": getattr(client, "max_retries", None),
        "sdk_timeout": str(getattr(client, "timeout", None)),
        "outbound_request_keys": sorted(request),
        "outbound_request_sha256": request_sha256,
        "serialized_wire_body_keys": sorted(request),
        "serialized_wire_body_sha256": request_sha256,
        "serialized_request_preflight_observed": True,
        "eos_only_no_output_token_cap": True,
    }
    assert_no_output_token_cap(policy)
    return policy


async def generate_missing_arguments(
    *,
    candidate_selection: str | Path,
    selection_manifest: str | Path,
    journal: str | Path,
    n_total: int = DEFAULT_N_TOTAL,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    attempts_per_candidate: int = DEFAULT_ATTEMPTS_PER_CANDIDATE,
    max_candidates_this_run: int | None = None,
    generate_call: GenerateCall | None = None,
    api_key: str | None = None,
) -> GenerationProgress:
    """Append attempts for missing candidates and leave accepted rows resumable."""

    if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
        raise ValueError("max_connections must be a positive integer")
    if isinstance(attempts_per_candidate, bool) or not isinstance(attempts_per_candidate, int) or attempts_per_candidate < 1:
        raise ValueError("attempts_per_candidate must be a positive integer")
    if max_candidates_this_run is not None and (
        isinstance(max_candidates_this_run, bool)
        or not isinstance(max_candidates_this_run, int)
        or max_candidates_this_run < 1
    ):
        raise ValueError("max_candidates_this_run must be a positive integer")
    candidates, _, _ = load_verified_candidates(candidate_selection, selection_manifest, n_total=n_total)
    journal_path = Path(journal).resolve()

    owned_client = None
    serialized_preflight: _SerializedRequestPreflight | None = None
    if generate_call is None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for generation")
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency failure
            raise RuntimeError("openai and httpx are required for OpenRouter generation") from exc
        # Disable SDK retries so a transport ambiguity cannot create an
        # unjournalled duplicate paid request. A failed call is retried only by
        # rerunning this resumable command against its durable journal.
        serialized_preflight = _SerializedRequestPreflight()
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(OPENROUTER_REQUEST_TIMEOUT_SECONDS),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            event_hooks={"request": [serialized_preflight]},
        )
        owned_client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,
            http_client=http_client,
        )

        async def generate_call(request: dict[str, Any]) -> GenerationResponse:
            return await _openrouter_generate(request, client=owned_client)

    try:
        with _exclusive_journal_lock(journal_path):
            existing_rows = _journal_rows(journal_path)
            accepted, attempt_counts = _validate_journal(existing_rows, candidates)
            accepted_before = len(accepted)
            pending = [row for row in candidates if row["candidate_id"] not in accepted]
            if max_candidates_this_run is not None:
                pending = pending[:max_candidates_this_run]
            scheduled_this_run = len(pending)
            semaphore = asyncio.Semaphore(max_connections)
            appended = 0
            accepted_this_run = 0
            transport_errors = 0
            appended_lock = asyncio.Lock()

            async def one(candidate: dict[str, Any]) -> None:
                nonlocal accepted_this_run, appended, transport_errors
                candidate_id = str(candidate["candidate_id"])
                request = build_generation_request(candidate)
                request_sha = _sha256(_canonical_json(request))
                if owned_client is None:
                    injected_runtime_policy: dict[str, Any] | None = {
                        "schema": "act_expanded_injected_generator_test_v1",
                        "transport": "injected_generate_call",
                        "openai_sdk_version": "not_applicable_injected_call",
                        "outbound_request_keys": sorted(request),
                        "outbound_request_sha256": request_sha,
                        "eos_only_no_output_token_cap": True,
                    }
                else:
                    injected_runtime_policy = None
                async with semaphore:
                    for offset in range(1, attempts_per_candidate + 1):
                        wire_count_before = (
                            serialized_preflight.observed_count(request_sha)
                            if serialized_preflight is not None
                            else 0
                        )
                        generation_error_class: str | None = None
                        try:
                            response = await generate_call(request)
                        except Exception as exc:
                            # A failure before the serialized-body hook ran is a
                            # local preflight/SDK construction failure, not an
                            # ambiguous paid call, and must remain fail-closed.
                            if serialized_preflight is not None and (
                                serialized_preflight.observed_count(request_sha)
                                <= wire_count_before
                            ):
                                raise
                            generation_error_class = (
                                f"{type(exc).__module__}.{type(exc).__qualname__}"
                            )
                            response = GenerationResponse(
                                completion="",
                                finish_reason="transport_error",
                            )
                        if not isinstance(response, GenerationResponse):
                            raise TypeError("generate_call must return GenerationResponse")
                        if owned_client is None:
                            if injected_runtime_policy is None:  # pragma: no cover - defensive
                                raise AssertionError("missing injected runtime policy")
                            runtime_policy = injected_runtime_policy
                        else:
                            if serialized_preflight is None:  # pragma: no cover - defensive
                                raise AssertionError("missing serialized request preflight")
                            runtime_policy = _openai_runtime_policy(
                                owned_client,
                                request=request,
                                serialized_preflight=serialized_preflight,
                            )
                        is_accepted, reason = _acceptance(
                            response.completion,
                            biased_option=str(candidate["biased_option"]),
                            finish_reason=response.finish_reason,
                        )
                        row = {
                            "schema": JOURNAL_SCHEMA,
                            "schema_version": GENERATION_SCHEMA_VERSION,
                            "candidate_id": candidate_id,
                            "question_id": candidate["question_id"],
                            "source_dataset": candidate["source_dataset"],
                            "biased_option": candidate["biased_option"],
                            "attempt_index": attempt_counts[candidate_id] + offset,
                            "provider": GENERATOR_PROVIDER,
                            "requested_model": GENERATOR_MODEL,
                            "reported_model": response.reported_model,
                            "response_id": response.response_id,
                            "finish_reason": response.finish_reason,
                            "generation_error_class": generation_error_class,
                            "usage": dict(response.usage) if response.usage is not None else None,
                            "request_sha256": request_sha,
                            "runtime_policy": runtime_policy,
                            "completion": response.completion,
                            "completion_sha256": _sha256(response.completion.encode("utf-8")),
                            "accepted": is_accepted,
                            "acceptance_reason": reason,
                            "natural_termination": str(response.finish_reason).lower()
                            in _NATURAL_STOP_REASONS,
                        }
                        _append_journal(journal_path, row)
                        async with appended_lock:
                            appended += 1
                            if generation_error_class is not None:
                                transport_errors += 1
                            if is_accepted:
                                accepted_this_run += 1
                            if appended % 50 == 0:
                                print(
                                    json.dumps(
                                        {
                                            "accepted": accepted_before + accepted_this_run,
                                            "attempts_appended": appended,
                                            "candidate_count": len(candidates),
                                            "event": "wrong_argument_generation_progress",
                                            "scheduled_this_run": scheduled_this_run,
                                            "transport_errors": transport_errors,
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                        if is_accepted:
                            return

            await asyncio.gather(*(one(candidate) for candidate in pending))
            final_rows = _journal_rows(journal_path)
            final_accepted, _ = _validate_journal(final_rows, candidates)
            return GenerationProgress(
                candidate_count=len(candidates),
                accepted_before=accepted_before,
                accepted_after=len(final_accepted),
                attempts_appended=appended,
                scheduled_this_run=scheduled_this_run,
                transport_errors=transport_errors,
            )
    finally:
        if owned_client is not None:
            await owned_client.close()


def _native_wrong_argument_row(candidate: Mapping[str, Any], accepted: Mapping[str, Any]) -> dict[str, Any]:
    canonical_input = str(candidate["canonical_input"])
    if selection.ANSWER_CHOICES_HEADER not in canonical_input:
        raise ValueError("candidate canonical_input has no answer-choice boundary")
    argument = str(accepted["completion"])
    unbiased_messages = [{"role": "user", "content": canonical_input + ANSWER_FORMAT_INSTRUCTION}]
    biased_messages = [
        {
            "role": "user",
            "content": WRONG_ARGUMENT_TEMPLATE.format(argument=argument, question=canonical_input)
            + ANSWER_FORMAT_INSTRUCTION,
        }
    ]
    native = {
        # This repository's recovered/canonical training rows intentionally
        # carry the complete rendered MCQ here.  The shared two-bias builder
        # verifies question + answer-format suffix byte-for-byte.
        "question": canonical_input,
        "question_id": candidate["question_id"],
        "source_dataset": candidate["source_dataset"],
        "prompt_style": selection.PROMPT_STYLE,
        "unbiased_messages": unbiased_messages,
        "biased_messages": biased_messages,
        "bias_type": selection.BIAS_TYPE,
        "ground_truth": candidate["ground_truth"],
        "biased_option": candidate["biased_option"],
        "biasing_text": argument,
        "argument_generation_provenance": {
            "candidate_id": candidate["candidate_id"],
            "candidate_selection_rank": candidate["selection_rank"],
            "candidate_source_origin": candidate["source_origin"],
            "candidate_source_question_id": candidate["source_question_id"],
            "candidate_source_row_index": candidate["source_row_index"],
            "generator_provider": accepted["provider"],
            "generator_model": accepted["requested_model"],
            "reported_model": accepted.get("reported_model"),
            "response_id": accepted.get("response_id"),
            "journal_attempt_index": accepted["attempt_index"],
            "request_sha256": accepted["request_sha256"],
            "completion_sha256": accepted["completion_sha256"],
            "finish_reason": accepted["finish_reason"],
            "natural_termination": accepted["natural_termination"],
            "runtime_policy_sha256": _sha256(_canonical_json(accepted["runtime_policy"])),
        },
    }
    return canonicalize_wrong_argument_row(native)


def _publish_immutable(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing immutable artifact: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Start the temporary inside the recoverable archive.  A successful
    # atomic replace moves it into place; an exceptional write leaves it in
    # ``_archive`` for inspection rather than deleting it.
    archive = path.parent / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return "written"


def finalize_arguments(
    *,
    candidate_selection: str | Path,
    selection_manifest: str | Path,
    journal: str | Path,
    output_dir: str | Path,
    n_total: int = DEFAULT_N_TOTAL,
    target_per_dataset: int | None = None,
    _allow_injected_test_runtime: bool = False,
) -> PublishedArguments:
    """Publish a complete or explicitly bounded balanced accepted subset."""

    candidates, selection_document, input_identities = load_verified_candidates(
        candidate_selection, selection_manifest, n_total=n_total
    )
    journal_path = _regular_file(journal, label="wrong-argument generation journal")
    with _exclusive_journal_lock(journal_path):
        journal_payload = journal_path.read_bytes()
        journal_rows = _journal_rows(journal_path)
        accepted, attempt_counts = _validate_journal(
            journal_rows,
            candidates,
            require_production_runtime=not _allow_injected_test_runtime,
        )
        accepted_by_dataset = Counter(
            row["source_dataset"]
            for row in candidates
            if row["candidate_id"] in accepted
        )
        if target_per_dataset is None and len(accepted) != len(candidates):
            by_dataset = Counter(
                row["source_dataset"] for row in candidates if row["candidate_id"] in accepted
            )
            raise ValueError(
                f"cannot finalize: accepted {len(accepted)}/{len(candidates)} candidates "
                f"({dict(by_dataset)}); rerun generation to retry only the misses"
            )
        if target_per_dataset is not None:
            if (
                isinstance(target_per_dataset, bool)
                or not isinstance(target_per_dataset, int)
                or target_per_dataset < 1
            ):
                raise ValueError("target_per_dataset must be a positive integer")
            unavailable = {
                dataset: accepted_by_dataset.get(dataset, 0)
                for dataset in selection.DATASETS
                if accepted_by_dataset.get(dataset, 0) < target_per_dataset
            }
            if unavailable:
                raise ValueError(
                    f"cannot finalize {target_per_dataset} per dataset; accepted availability is "
                    f"{dict(accepted_by_dataset)}"
                )
            selected_counts: Counter[str] = Counter()
            selected_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                dataset = str(candidate["source_dataset"])
                if (
                    candidate["candidate_id"] in accepted
                    and selected_counts[dataset] < target_per_dataset
                ):
                    selected_candidates.append(candidate)
                    selected_counts[dataset] += 1
        else:
            selected_candidates = candidates
        final_rows = [
            _native_wrong_argument_row(candidate, accepted[str(candidate["candidate_id"])])
            for candidate in selected_candidates
        ]
        payload = _canonical_jsonl(final_rows)
        digest = _sha256(payload)
        output = Path(output_dir).resolve()
        data_path = output / f"act-expanded-canonical-wrong-argument-n{len(final_rows)}-{digest}.jsonl"
        ids = [str(row["question_id"]) for row in final_rows]
        candidate_ids = [str(row["candidate_id"]) for row in selected_candidates]
        origin_counts = dict(Counter(str(row["source_origin"]) for row in selected_candidates))
        origin_counts_by_dataset = {
            dataset: dict(
                Counter(
                    str(row["source_origin"])
                    for row in selected_candidates
                    if row["source_dataset"] == dataset
                )
            )
            for dataset in selection.DATASETS
        }
        runtime_policies_by_digest: dict[str, dict[str, Any]] = {}
        for accepted_row in accepted.values():
            policy = dict(accepted_row["runtime_policy"])
            policy_digest = _sha256(_canonical_json(policy))
            runtime_policies_by_digest.setdefault(policy_digest, policy)
        manifest = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "schema_version": CONSISTENCY_SCHEMA_VERSION,
            "row_count": len(final_rows),
            "content_sha256": digest,
            "data_filename": data_path.name,
            "counts": {
                "by_dataset": dict(Counter(row["source_dataset"] for row in final_rows)),
                "unique_question_ids": len(set(ids)),
                "accepted_arguments": len(final_rows),
                "accepted_arguments_available": len(accepted),
                "journal_attempts": sum(attempt_counts.values()),
            },
            "provenance": {
                **input_identities,
                "journal_at_finalization": {
                    "filename": journal_path.name,
                    "content_sha256": _sha256(journal_payload),
                    "byte_count": len(journal_payload),
                    "row_count": len(journal_rows),
                },
                "selection": {
                    "mode": selection_document["candidate_source_mode"],
                    "candidate_population_n_total": n_total,
                    "finalized_n_total": len(final_rows),
                    "target_per_dataset": target_per_dataset,
                    "accepted_available_by_dataset": dict(accepted_by_dataset),
                    "counts_by_dataset": dict(Counter(row["source_dataset"] for row in selected_candidates)),
                    "counts_by_source_origin": origin_counts,
                    "counts_by_dataset_and_source_origin": origin_counts_by_dataset,
                    "candidate_ids_sha256": _sha256("\n".join(candidate_ids).encode("utf-8")),
                    "question_ids_sha256": _sha256("\n".join(ids).encode("utf-8")),
                    "order": (
                        "exact frozen candidate round_robin order"
                        if target_per_dataset is None
                        else "frozen candidate order filtered to accepted rows and first target_per_dataset per dataset"
                    ),
                    "post_generation_balancing": target_per_dataset is not None,
                },
                "argument_generation": {
                    "provider": GENERATOR_PROVIDER,
                    "model": GENERATOR_MODEL,
                    "api_model": OPENROUTER_API_MODEL,
                    "recipe": "pinned_mcq_bias_deceptive_assistant_wrong_argument_v1",
                    "system_prompt_sha256": _sha256(DECEPTIVE_SYSTEM_PROMPT.encode("utf-8")),
                    "user_suffix_sha256": _sha256(COT_INSTRUCTION.encode("utf-8")),
                    "acceptance": "exact pinned mcq_bias _acceptable after extended parser installation",
                    "one_accepted_fresh_argument_per_candidate": True,
                    "legacy_argument_store_read": False,
                    "request_fields": ["messages", "model"],
                    "eos_only": True,
                    "every_accepted_response_has_natural_stop": all(
                        bool(row["natural_termination"]) for row in accepted.values()
                    ),
                    "runtime_policies_by_sha256": runtime_policies_by_digest,
                    "forbidden_output_token_fields": sorted(FORBIDDEN_OUTPUT_TOKEN_FIELDS),
                    "forbidden_output_token_fields_absent_from_every_request": True,
                },
                "transform": {
                    "name": TRANSFORM_VERSION,
                    "argument_field": "biasing_text",
                    "reference_field": "unbiased_messages",
                    "variant_field": "biased_messages",
                    "prefix_sha256": _sha256(WRONG_ARGUMENT_PREFIX.encode("utf-8")),
                    "invariant": "variant last user content ends with exact reference last user content",
                },
            },
        }
        manifest_payload = _canonical_json(manifest)
        manifest_sha = _sha256(manifest_payload)
        manifest_path = output / (
            f"act-expanded-canonical-wrong-argument-n{len(final_rows)}-{digest}"
            f".manifest-{manifest_sha}.json"
        )
        data_status = _publish_immutable(data_path, payload)
        manifest_status = _publish_immutable(manifest_path, manifest_payload)
        status = "resumed" if data_status == manifest_status == "resumed" else "written"
        return PublishedArguments(
            data_path=data_path,
            manifest_path=manifest_path,
            content_sha256=digest,
            manifest_sha256=manifest_sha,
            row_count=len(final_rows),
            status=status,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="append attempts for candidates without an accepted argument")
    generate_parser.add_argument("--candidate-selection", type=Path, required=True)
    generate_parser.add_argument("--selection-manifest", type=Path, required=True)
    generate_parser.add_argument("--journal", type=Path, required=True)
    generate_parser.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL)
    generate_parser.add_argument("--max-connections", type=int, default=DEFAULT_MAX_CONNECTIONS)
    generate_parser.add_argument("--attempts-per-candidate", type=int, default=DEFAULT_ATTEMPTS_PER_CANDIDATE)
    generate_parser.add_argument(
        "--max-candidates-this-run",
        type=int,
        help="process only this deterministic prefix of currently missing candidates, then resume later",
    )

    finalize_parser = subparsers.add_parser("finalize", help="publish immutable canonical pairs from accepted candidates")
    finalize_parser.add_argument("--candidate-selection", type=Path, required=True)
    finalize_parser.add_argument("--selection-manifest", type=Path, required=True)
    finalize_parser.add_argument("--journal", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL)
    finalize_parser.add_argument(
        "--target-per-dataset",
        type=int,
        help="publish the first accepted rows in frozen order up to this balanced per-dataset target",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            progress = asyncio.run(
                generate_missing_arguments(
                    candidate_selection=args.candidate_selection,
                    selection_manifest=args.selection_manifest,
                    journal=args.journal,
                    n_total=args.n_total,
                    max_connections=args.max_connections,
                    attempts_per_candidate=args.attempts_per_candidate,
                    max_candidates_this_run=args.max_candidates_this_run,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
            )
            print(json.dumps(asdict(progress), sort_keys=True))
            if (
                args.max_candidates_this_run is None
                and progress.accepted_after != progress.candidate_count
            ):
                raise ValueError(
                    f"accepted {progress.accepted_after}/{progress.candidate_count}; "
                    "rerun the same command to retry only missing arguments"
                )
            return
        result = finalize_arguments(
            candidate_selection=args.candidate_selection,
            selection_manifest=args.selection_manifest,
            journal=args.journal,
            output_dir=args.output_dir,
            n_total=args.n_total,
            target_per_dataset=args.target_per_dataset,
        )
        print(
            json.dumps(
                {
                    "data": str(result.data_path),
                    "manifest": str(result.manifest_path),
                    "row_count": result.row_count,
                    "sha256": result.content_sha256,
                    "status": result.status,
                    "verified": True,
                },
                sort_keys=True,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_N_TOTAL",
    "FORBIDDEN_OUTPUT_TOKEN_FIELDS",
    "GENERATOR_MODEL",
    "GenerationProgress",
    "GenerationResponse",
    "PublishedArguments",
    "assert_no_output_token_cap",
    "build_generation_request",
    "finalize_arguments",
    "generate_missing_arguments",
    "load_verified_candidates",
]
