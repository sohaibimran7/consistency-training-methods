"""RMCT Setting bridges for the verified Irpan training pair views.

The immutable training-view reader owns artifact verification and role
enforcement.  This module only adapts those verified, plain pair rows to the
generic :class:`ctm.settings.base.Setting` protocol used by ``train_rlct.py``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.training.refusal import RefusalJudge
from ctm.training.refusal.judge import (
    DEFAULT_REFUSAL_MODEL,
    CompletionCallback,
    normalize_refusal_judge_options,
)
from scripts.irpan_2510_27062.mmlu_tasks import (
    FOLLOWED_WRONG_SUGGESTION,
    MMLU_ACCURACY,
    parse_final_answer_label,
)
from scripts.irpan_2510_27062.schema import sha256_json

_TRAINING_ROLE = "training"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z0-9]$")


def _read_training_view(
    path: Path,
    *,
    expected_domain: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Import the canonical reader lazily so data-free factory imports stay cheap."""

    from scripts.irpan_2510_27062.training_views import read_training_view

    return read_training_view(path, expected_domain=expected_domain)


def _canonical_path(value: str | Path, *, field: str) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a string or Path")
    return Path(value).expanduser().resolve()


def _positive_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("n_datapoints must be a positive integer or null")
    return value


def _manifest_value(manifest: Mapping[str, Any], field: str) -> Any:
    values: list[Any] = []
    if field in manifest:
        values.append(manifest[field])
    provenance = manifest.get("provenance")
    if isinstance(provenance, Mapping) and field in provenance:
        values.append(provenance[field])
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"training-view manifest has conflicting {field!r} values")
    return values[0]


def _verified_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    path: Path,
    expected_domain: str,
    actual_row_count: int,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("read_training_view must return a manifest object")
    role = _manifest_value(manifest, "role")
    if role != _TRAINING_ROLE:
        raise ValueError(f"training view {path} has role {role!r}, expected {_TRAINING_ROLE!r}")
    manifest_domain = _manifest_value(manifest, "domain")
    if manifest_domain is not None and manifest_domain != expected_domain:
        raise ValueError(f"training view {path} has manifest domain {manifest_domain!r}, expected {expected_domain!r}")

    schema = manifest.get("artifact_schema")
    version = manifest.get("schema_version")
    row_count = manifest.get("row_count")
    content_digest = manifest.get("content_sha256")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"training-view manifest for {path} has no artifact_schema")
    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError(f"training-view manifest for {path} has an invalid schema_version")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError(f"training-view manifest for {path} has an invalid row_count")
    if row_count != actual_row_count:
        raise ValueError(
            f"training-view manifest for {path} records {row_count} rows, reader returned {actual_row_count}"
        )
    if not isinstance(content_digest, str) or _SHA256_RE.fullmatch(content_digest) is None:
        raise ValueError(f"training-view manifest for {path} has an invalid content_sha256")
    manifest_digest = manifest.get("manifest_sha256")
    if manifest_digest is None:
        manifest_digest = sha256_json(manifest)
    elif not isinstance(manifest_digest, str) or _SHA256_RE.fullmatch(manifest_digest) is None:
        raise ValueError(f"training-view manifest for {path} has an invalid manifest_sha256")

    return {
        "path": str(path),
        "artifact_schema": schema,
        "schema_version": version,
        "row_count": row_count,
        "content_sha256": content_digest,
        "manifest_sha256": manifest_digest,
        "role": role,
        "domain": expected_domain,
    }


def _copy_messages(value: Any, *, pair_id: str, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        raise ValueError(f"pair {pair_id!r} {field} must be a non-empty message list")
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"pair {pair_id!r} {field}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
            raise ValueError(f"pair {pair_id!r} {field}[{index}] must contain non-empty string role/content fields")
        messages.append(copy.deepcopy(dict(message)))
    return messages


def _validated_pair_rows(
    rows: Any,
    *,
    expected_domain: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("read_training_view must return a sequence of pair rows")
    if not rows:
        raise ValueError(f"{expected_domain} training view contains no pairs")

    validated: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"training-view row {index} must be an object")
        row = copy.deepcopy(dict(raw))
        pair_id = row.get("pair_id")
        example_id = row.get("example_id")
        source = row.get("source")
        domain = row.get("domain")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"training-view row {index} has no stable pair_id")
        if pair_id in seen_pair_ids:
            raise ValueError(f"training view contains duplicate pair_id {pair_id!r}")
        seen_pair_ids.add(pair_id)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"pair {pair_id!r} has no base example_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"pair {pair_id!r} has no source")
        if domain != expected_domain:
            raise ValueError(f"pair {pair_id!r} has domain {domain!r}, expected {expected_domain!r}")
        row["reference_messages"] = _copy_messages(
            row.get("reference_messages"), pair_id=pair_id, field="reference_messages"
        )
        row["variant_messages"] = _copy_messages(row.get("variant_messages"), pair_id=pair_id, field="variant_messages")
        validated.append(row)
    return validated


def _normalize_label(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string option label")
    normalized = value.strip().upper()
    if _LABEL_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be one ASCII letter or digit")
    return normalized


def _labels_from_choices(value: Any, *, field: str) -> tuple[str, ...]:
    raw_labels: list[Any]
    if isinstance(value, Mapping):
        raw_labels = list(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        choices = list(value)
        if all(isinstance(choice, str) for choice in choices):
            if len(choices) > 26:
                raise ValueError(f"{field} cannot infer labels for more than 26 choices")
            raw_labels = [chr(ord("A") + index) for index in range(len(choices))]
        elif all(isinstance(choice, Mapping) for choice in choices):
            have_labels = ["label" in choice or "key" in choice for choice in choices]
            if any(have_labels) and not all(have_labels):
                raise ValueError(f"{field} mixes labeled and unlabeled choice objects")
            if all(have_labels):
                raw_labels = [choice.get("label", choice.get("key")) for choice in choices]
            else:
                if len(choices) > 26:
                    raise ValueError(f"{field} cannot infer labels for more than 26 choices")
                raw_labels = [chr(ord("A") + index) for index in range(len(choices))]
        else:
            raise ValueError(f"{field} must contain only strings or only choice objects")
    else:
        raise TypeError(f"{field} must be a choice mapping or sequence")
    labels = tuple(_normalize_label(label, field=f"{field} label") for label in raw_labels)
    if len(labels) < 2:
        raise ValueError(f"{field} must identify at least two choices")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field} contains duplicate option labels")
    return labels


def _metadata_containers(datapoint: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    containers: list[tuple[str, Mapping[str, Any]]] = [("datapoint", datapoint)]
    seen = {id(datapoint)}
    cursor = 0
    while cursor < len(containers):
        location, container = containers[cursor]
        cursor += 1
        for key in ("metadata", "gold_metadata", "gold"):
            nested = container.get(key)
            if isinstance(nested, Mapping) and id(nested) not in seen:
                seen.add(id(nested))
                containers.append((f"{location}.{key}", nested))
    return containers


def _sycophancy_labels_and_gold(datapoint: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    pair_id = datapoint.get("pair_id", "<unknown>")
    label_candidates: list[tuple[str, tuple[str, ...]]] = []
    correct_candidates: list[tuple[str, str]] = []
    suggestion_candidates: list[tuple[str, str]] = []
    for location, container in _metadata_containers(datapoint):
        for key in ("choice_labels", "valid_labels"):
            if key in container:
                raw = container[key]
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
                    raise ValueError(f"pair {pair_id!r} {location}.{key} must be a label sequence")
                labels = tuple(_normalize_label(label, field=f"pair {pair_id!r} {location}.{key}") for label in raw)
                if len(labels) < 2 or len(labels) != len(set(labels)):
                    raise ValueError(f"pair {pair_id!r} {location}.{key} must contain unique choice labels")
                label_candidates.append((f"{location}.{key}", labels))
        if "choices" in container:
            label_candidates.append(
                (
                    f"{location}.choices",
                    _labels_from_choices(container["choices"], field=f"pair {pair_id!r} {location}.choices"),
                )
            )
        for key in ("correct_label", "gold_label", "answer_label"):
            if key in container:
                correct_candidates.append(
                    (
                        f"{location}.{key}",
                        _normalize_label(container[key], field=f"pair {pair_id!r} {location}.{key}"),
                    )
                )
        if "suggested_wrong_label" in container:
            suggestion_candidates.append(
                (
                    f"{location}.suggested_wrong_label",
                    _normalize_label(
                        container["suggested_wrong_label"],
                        field=f"pair {pair_id!r} {location}.suggested_wrong_label",
                    ),
                )
            )
        gold = container.get("gold")
        if isinstance(gold, str):
            correct_candidates.append(
                (f"{location}.gold", _normalize_label(gold, field=f"pair {pair_id!r} {location}.gold"))
            )

    if not label_candidates:
        raise ValueError(f"sycophancy pair {pair_id!r} has no choices or valid-label metadata")
    labels = label_candidates[0][1]
    for location, candidate in label_candidates[1:]:
        if candidate != labels:
            raise ValueError(
                f"sycophancy pair {pair_id!r} has conflicting valid labels at {label_candidates[0][0]} and {location}"
            )
    if not correct_candidates:
        raise ValueError(f"sycophancy pair {pair_id!r} has no correct-label gold metadata")
    correct = correct_candidates[0][1]
    for location, candidate in correct_candidates[1:]:
        if candidate != correct:
            raise ValueError(
                f"sycophancy pair {pair_id!r} has conflicting correct labels at "
                f"{correct_candidates[0][0]} and {location}"
            )
    if correct not in labels:
        raise ValueError(f"sycophancy pair {pair_id!r} correct label {correct!r} is absent from {list(labels)!r}")
    if not suggestion_candidates:
        raise ValueError(f"sycophancy pair {pair_id!r} has no suggested_wrong_label metadata")
    suggested = suggestion_candidates[0][1]
    for location, candidate in suggestion_candidates[1:]:
        if candidate != suggested:
            raise ValueError(
                f"sycophancy pair {pair_id!r} has conflicting suggested wrong labels at "
                f"{suggestion_candidates[0][0]} and {location}"
            )
    if suggested not in labels:
        raise ValueError(f"sycophancy pair {pair_id!r} suggested label {suggested!r} is absent from {list(labels)!r}")
    if suggested == correct:
        raise ValueError(f"sycophancy pair {pair_id!r} suggested_wrong_label must differ from correct label")
    return labels, correct


def _callable_identity(value: Callable[..., Any] | None) -> str | None:
    if value is None:
        return None
    return (
        f"{getattr(value, '__module__', type(value).__module__)}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


class _LazyRefusalTrait:
    """Callable that defers RefusalJudge (and therefore model-client) creation."""

    def __init__(
        self,
        *,
        model: str,
        grader_config: Mapping[str, Any],
        completion: CompletionCallback | None,
        judge: Callable[..., Any] | None,
    ) -> None:
        self.model = model
        self.grader_config = copy.deepcopy(dict(grader_config))
        self.completion = completion
        self._judge = judge

    def _get_judge(self) -> Callable[..., Any]:
        if self._judge is None:
            self._judge = RefusalJudge(
                model=self.model,
                completion=self.completion,
                **self.grader_config,
            )
        return self._judge

    async def __call__(
        self,
        response: str,
        datapoint: Mapping[str, Any],
        realized_messages: Sequence[Mapping[str, Any]],
    ) -> float | None:
        result = self._get_judge()(response, datapoint, realized_messages)
        if inspect.isawaitable(result):
            result = await result
        return result

    def provenance(self) -> dict[str, Any]:
        if self._judge is not None:
            describe = getattr(self._judge, "provenance", None)
            if callable(describe):
                described = describe()
                if not isinstance(described, Mapping):
                    raise TypeError("injected refusal judge provenance() must return an object")
                return dict(described)
        return {
            "classifier": "ctm.training.refusal.RefusalJudge",
            "lazy": self._judge is None,
            "model": self.model,
            "grader_config": copy.deepcopy(self.grader_config),
            "completion": _callable_identity(self.completion) or "inspect_model",
            "injected_judge": _callable_identity(self._judge),
        }


class _PairRMCTSetting:
    domain: str

    def __init__(self, training_view_path: str | Path | None = None) -> None:
        self.training_view_path = (
            _canonical_path(training_view_path, field="training_view_path") if training_view_path is not None else None
        )
        self._artifact_identity: dict[str, Any] | None = None
        self._selected_pair_ids: list[str] | None = None

    def _resolve_path(self, training_view_path: str | Path | None) -> Path:
        supplied = (
            _canonical_path(training_view_path, field="training_view_path") if training_view_path is not None else None
        )
        if self.training_view_path is not None and supplied is not None and supplied != self.training_view_path:
            raise ValueError("exactly one canonical training_view_path is allowed; constructor and load paths disagree")
        path = supplied or self.training_view_path
        if path is None:
            raise ValueError(f"{self.domain} RMCT needs one explicit training_view_path")
        self.training_view_path = path
        return path

    def load_datapoints(
        self,
        n_datapoints: int | None = None,
        *,
        training_view_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        limit = _positive_limit(n_datapoints)
        path = self._resolve_path(training_view_path)
        self._artifact_identity = None
        self._selected_pair_ids = None
        rows, manifest = _read_training_view(path, expected_domain=self.domain)
        validated = _validated_pair_rows(rows, expected_domain=self.domain)
        artifact_identity = _verified_manifest_identity(
            manifest,
            path=path,
            expected_domain=self.domain,
            actual_row_count=len(validated),
        )
        self._validate_domain_rows(validated)
        selected = validated if limit is None else validated[:limit]
        if not selected:
            raise ValueError(f"{self.domain} training selection contains no pairs")
        self._prepare_selected_rows(selected)
        self._selected_pair_ids = [row["pair_id"] for row in selected]
        self._artifact_identity = artifact_identity
        return copy.deepcopy(selected)

    def _validate_domain_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        del rows

    def _prepare_selected_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        del rows

    @staticmethod
    def perturbations() -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
        def reference(datapoint: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": _copy_messages(
                    datapoint.get("reference_messages"),
                    pair_id=str(datapoint.get("pair_id", "<unknown>")),
                    field="reference_messages",
                )
            }

        def variant(datapoint: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": _copy_messages(
                    datapoint.get("variant_messages"),
                    pair_id=str(datapoint.get("pair_id", "<unknown>")),
                    field="variant_messages",
                )
            }

        return [reference, variant]

    @staticmethod
    def training_perturbation_indices() -> list[int]:
        return [1]

    def training_artifact_identity(self) -> dict[str, Any]:
        if self._artifact_identity is None or self._selected_pair_ids is None:
            raise ValueError(f"{self.domain} RMCT training selection is unavailable; call load_datapoints first")
        pair_ids = list(self._selected_pair_ids)
        identity = copy.deepcopy(self._artifact_identity)
        identity["selection"] = {
            "selected_row_count": len(pair_ids),
            "selected_pair_ids": pair_ids,
            "selected_pair_ids_sha256": hashlib.sha256("\n".join(pair_ids).encode("utf-8")).hexdigest(),
        }
        return identity

    def run_metadata(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "training_view_path": self.training_view_path,
        }


class SycophancyRMCTSetting(_PairRMCTSetting):
    """Irpan clean/wrong-suggestion pairs with a correctness trait."""

    name = "irpan_sycophancy_rmct"
    domain = "sycophancy"

    def __init__(self, training_view_path: str | Path | None = None) -> None:
        super().__init__(training_view_path)
        self._answer_labels: tuple[str, ...] = ()

    def _validate_domain_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            _sycophancy_labels_and_gold(row)

    def _prepare_selected_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        labels: set[str] = set()
        for row in rows:
            row_labels, _ = _sycophancy_labels_and_gold(row)
            labels.update(row_labels)
        self._answer_labels = tuple(sorted(labels))

    @staticmethod
    def _score_correctness(
        response: str,
        datapoint: dict[str, Any],
        realized_messages: list[dict[str, Any]],
    ) -> float | None:
        del realized_messages
        labels, correct = _sycophancy_labels_and_gold(datapoint)
        parsed = parse_final_answer_label(response, labels)
        if parsed is None:
            return None
        return 1.0 if parsed == correct else 0.0

    def _parse_answer(self, response: str) -> str | None:
        if not self._answer_labels:
            return None
        return parse_final_answer_label(response, self._answer_labels)

    def trait_classifier(self) -> Callable[[str, dict[str, Any], list[dict[str, Any]]], float | None]:
        return self._score_correctness

    def answer_parser(self) -> Callable[[str], str | None]:
        return self._parse_answer

    def run_metadata(self) -> dict[str, Any]:
        return {
            **super().run_metadata(),
            "trait": "final_answer_correctness",
            "answer_parser": "irpan_final_answer_label_reconstruction_v1",
            "valid_labels": list(self._answer_labels),
            "evaluation_metrics": [MMLU_ACCURACY, FOLLOWED_WRONG_SUGGESTION],
            "unparsed_trait_value": None,
        }


class JailbreakRMCTSetting(_PairRMCTSetting):
    """Irpan clean/wrapped jailbreak pairs with the shared refusal trait."""

    name = "irpan_jailbreak_rmct"
    domain = "jailbreak"

    def __init__(
        self,
        training_view_path: str | Path | None = None,
        *,
        grader_model: str | None = None,
        grader_config: Mapping[str, Any] | None = None,
        completion: CompletionCallback | None = None,
        judge: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(training_view_path)
        if judge is not None and (grader_model is not None or grader_config is not None or completion is not None):
            raise ValueError("pass either judge= or grader_model=/grader_config=/completion=, not both")
        if judge is not None and not callable(judge):
            raise TypeError("judge must be callable")
        if completion is not None and not callable(completion):
            raise TypeError("completion must be callable")
        if grader_model is not None and (not isinstance(grader_model, str) or not grader_model.strip()):
            raise ValueError("grader_model must be a non-empty string")
        injected_model = getattr(judge, "model", None) if judge is not None else None
        self.grader_model = (
            injected_model
            if isinstance(injected_model, str) and injected_model
            else grader_model or DEFAULT_REFUSAL_MODEL
        )
        self.grader_config = normalize_refusal_judge_options(grader_config, label="grader_config")
        self._classifier = _LazyRefusalTrait(
            model=self.grader_model,
            grader_config=self.grader_config,
            completion=completion,
            judge=judge,
        )

    def trait_classifier(self) -> _LazyRefusalTrait:
        return self._classifier

    @staticmethod
    def answer_parser() -> None:
        return None

    def run_metadata(self) -> dict[str, Any]:
        return {
            **super().run_metadata(),
            "trait": "refusal",
            "grader_model": self.grader_model,
            "grader_config": copy.deepcopy(self.grader_config),
            "grader_provenance": self._classifier.provenance(),
            "pair_direction": "clean_reference_to_wrapped_variant",
        }


def sycophancy_rmct_setting(
    training_view_path: str | Path | None = None,
) -> SycophancyRMCTSetting:
    """Explicit ``module:factory`` target for Irpan sycophancy RMCT."""

    return SycophancyRMCTSetting(training_view_path=training_view_path)


def jailbreak_rmct_setting(
    training_view_path: str | Path | None = None,
    *,
    grader_model: str | None = None,
    grader_config: Mapping[str, Any] | None = None,
    completion: CompletionCallback | None = None,
    judge: Callable[..., Any] | None = None,
) -> JailbreakRMCTSetting:
    """Explicit ``module:factory`` target for Irpan jailbreak RMCT."""

    return JailbreakRMCTSetting(
        training_view_path=training_view_path,
        grader_model=grader_model,
        grader_config=grader_config,
        completion=completion,
        judge=judge,
    )


__all__ = [
    "JailbreakRMCTSetting",
    "SycophancyRMCTSetting",
    "jailbreak_rmct_setting",
    "sycophancy_rmct_setting",
]
