#!/usr/bin/env python3
"""Diagnose one uncapped Gemma 4 HLE decode without producing scored evidence.

This utility deliberately calls the same ``_eos_only_model_generate`` loop as
the Gemma evaluation.  It has no generated-length stopping condition.  Its
``--memory-every`` setting controls telemetry cadence only; it can never stop
or truncate a generation.  A ``completed.json`` receipt is written only after
every requested response returns with an observed model EOS token.

The companion Slurm wrapper owns scheduler wall time.  If the scheduler stops
this program before EOS, the flushed ``events.jsonl`` remains diagnostic-only
evidence and no completed receipt is created.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import signal
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GemmaEOSDebugError(RuntimeError):
    """The diagnostic could not faithfully reproduce the Gemma decode route."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    """Return a compact JSON-safe representation without serialising model data."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    return repr(value)


def _concrete_eos_ids(value: Any) -> list[int]:
    """Normalise the public scalar/list EOS forms exposed by Transformers."""

    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return sorted({item for item in value if isinstance(item, int) and not isinstance(item, bool)})
    return []


def _eos_sources(tokenizer: Any, model: Any) -> dict[str, Any]:
    """Record the exact public EOS configuration consulted by the shared loop."""

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    generation_config = getattr(model, "generation_config", None)
    sources = {
        "tokenizer.eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "model.config.eos_token_id": getattr(config, "eos_token_id", None),
        "model.config.text_config.eos_token_id": getattr(text_config, "eos_token_id", None),
        "model.generation_config.eos_token_id": getattr(generation_config, "eos_token_id", None),
    }
    concrete = sorted({token_id for value in sources.values() for token_id in _concrete_eos_ids(value)})
    if not concrete:
        raise GemmaEOSDebugError("the active Gemma wrapper exposes no concrete EOS token IDs")
    return {"sources": {key: _json_value(value) for key, value in sources.items()}, "concrete_ids": concrete}


def _parse_question_ids(raw: str) -> tuple[str, ...]:
    """Parse a JSON question-ID list without accepting an ambiguous selector."""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--question-ids must be a JSON array of non-empty strings") from exc
    if not isinstance(parsed, list) or not parsed or any(not isinstance(item, str) or not item for item in parsed):
        raise argparse.ArgumentTypeError("--question-ids must be a non-empty JSON array of non-empty strings")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("--question-ids must not contain duplicates")
    return tuple(parsed)


def _selected_ids(
    all_ids: Sequence[str],
    *,
    task_index: int,
    rank: int | None,
    question_ids: Sequence[str] | None,
    shard_ids: Callable[..., Sequence[str]],
) -> tuple[str, ...]:
    """Choose a scored HLE subset in frozen source order.

    The old rank-5 HLE shard contains four source IDs because the fixed
    16-worker rotation assigns two surplus questions to that rank.  A caller
    may instead request one or both known incomplete IDs, but never an ID
    outside the original scored 50-question population.
    """

    scored = tuple(all_ids[:50])
    if len(scored) != 50 or len(scored) != len(set(scored)):
        raise GemmaEOSDebugError("the deployed HLE task has no valid 50-question scored pool")
    if (rank is None) == (question_ids is None):
        raise GemmaEOSDebugError("select exactly one of --rank or --question-ids")
    if rank is not None:
        if not 0 <= rank < 16:
            raise GemmaEOSDebugError("--rank must be in [0, 15]")
        selected = tuple(shard_ids(tuple(all_ids), task_index=task_index, shard_index=rank))
    else:
        assert question_ids is not None
        requested = set(question_ids)
        unknown = sorted(requested - set(scored))
        if unknown:
            raise GemmaEOSDebugError(f"--question-ids are outside the scored HLE population: {unknown[:3]}")
        selected = tuple(question_id for question_id in scored if question_id in requested)
    if not selected:
        raise GemmaEOSDebugError("the selector did not produce any HLE questions")
    if len(selected) != len(set(selected)) or not set(selected) <= set(scored):
        raise GemmaEOSDebugError("the selected HLE IDs do not form a unique subset of the scored population")
    return selected


def _require_regular_file(value: str, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GemmaEOSDebugError(f"{label} must be an absolute, regular, non-linked file")
    return path.resolve()


def _require_regular_directory(value: str, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise GemmaEOSDebugError(f"{label} must be an absolute, regular, non-linked directory")
    return path.resolve()


def _create_new_log_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise GemmaEOSDebugError("--log-dir must be an absolute path")
    if path.exists() or path.is_symlink():
        raise FileExistsError("--log-dir must not already exist; diagnostics never overwrite prior evidence")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise GemmaEOSDebugError("--log-dir parent must be an existing regular directory")
    path.mkdir(mode=0o700)
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _EventLog:
    """Append durable, compact telemetry so an allocation timeout is observable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("x", encoding="utf-8")

    def write(self, event: Mapping[str, Any]) -> None:
        payload = {"at": _utc_now(), **dict(event)}
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def _cache_length(cache: Any) -> int | None:
    """Read cache length if its public cache object supports the query."""

    method = getattr(cache, "get_seq_length", None)
    if not callable(method):
        return None
    for args in ((), (0,)):
        try:
            value = method(*args)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _memory_snapshot(torch: Any, device: Any, *, cache: Any = None) -> dict[str, int | None]:
    """Collect memory counters without modifying the decode or its cache."""

    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "cache_sequence_length": _cache_length(cache),
    }


class _ForwardTelemetry:
    """Instrument the existing model's forward calls without replacing its decode loop."""

    def __init__(
        self,
        *,
        model: Any,
        torch: Any,
        device: Any,
        event_log: _EventLog,
        sample_id: str,
        every: int,
    ) -> None:
        self.model = model
        self.torch = torch
        self.device = device
        self.event_log = event_log
        self.sample_id = sample_id
        self.every = every
        self.decode_steps = 0
        self._original_forward: Callable[..., Any] | None = None

    def __enter__(self) -> "_ForwardTelemetry":
        original = self.model.forward
        self._original_forward = original

        @functools.wraps(original)
        def observed_forward(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            self.decode_steps += 1
            if self.decode_steps == 1 or self.decode_steps % self.every == 0:
                self.event_log.write(
                    {
                        "event": "decode_memory",
                        "sample_id": self.sample_id,
                        "model_forward_count": self.decode_steps,
                        **_memory_snapshot(self.torch, self.device, cache=getattr(output, "past_key_values", None)),
                    }
                )
            return output

        self.model.forward = observed_forward
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._original_forward is not None:
            self.model.forward = self._original_forward
        return False


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_hle_samples(
    *,
    manifest: Path,
    task_index: int,
    rank: int | None,
    question_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], list[Any]]:
    """Construct the exact clean HLE task used by the Gemma launcher."""

    from infra.isambard.run_gemma4_12b_base_two_bias_evals_16gpu import _load_specs, _shard_ids

    specs = _load_specs(manifest)
    if not 1 <= task_index <= len(specs):
        raise GemmaEOSDebugError(f"--task-index is outside the deployed {len(specs)}-task matrix")
    spec = specs[task_index - 1]
    if spec.kind != "unbiased" or spec.dataset != "hle-text-mc" or spec.population != "hle":
        raise GemmaEOSDebugError("this diagnostic is restricted to the clean HLE task")
    selected_ids = _selected_ids(
        spec.question_ids,
        task_index=task_index,
        rank=rank,
        question_ids=question_ids,
        shard_ids=_shard_ids,
    )

    from experiments.stage2_ood_hle.tasks import stage2_ood_unbiased

    task = stage2_ood_unbiased(
        frozen_file=spec.frozen_file,
        dataset=spec.dataset,
        regime=spec.regime,
        population=spec.population,
        question_ids_from=list(selected_ids),
        source_identity_digest=spec.source_identity_digest,
        prompt_style="none",
    )
    by_id = {str(sample.id): sample for sample in task.dataset}
    if set(by_id) != set(selected_ids):
        raise GemmaEOSDebugError("the reconstructed HLE task does not match its frozen question-ID selection")
    return selected_ids, [by_id[question_id] for question_id in selected_ids]


def _sample_messages(sample: Any) -> list[Any]:
    messages = getattr(sample, "input", None)
    if not isinstance(messages, list) or not messages:
        raise GemmaEOSDebugError("the reconstructed HLE sample does not expose a non-empty Inspect message list")
    return messages


def _decode_one(
    *,
    api: Any,
    config: Any,
    sample: Any,
    torch: Any,
    event_log: _EventLog,
    every: int,
    eos_ids: set[int],
) -> dict[str, Any]:
    """Run the shared EOS-only generator for exactly one frozen HLE sample."""

    from experiments.elephant_aita_ntaflip.no_cap_hf import _eos_only_model_generate

    sample_id = str(sample.id)
    messages = _sample_messages(sample)
    chat = api.hf_chat(messages, [])
    tokenizer = functools.partial(
        api.tokenizer,
        return_tensors="pt",
        padding=True,
        **dict(api.tokenizer_call_args),
    )
    tokenized = tokenizer([chat])
    if not {"input_ids", "attention_mask"} <= set(tokenized):
        raise GemmaEOSDebugError("Gemma text processor did not return input_ids and attention_mask")
    device = api.model.device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    input_tokens = int(input_ids.shape[1])

    event_log.write(
        {
            "event": "sample_started",
            "sample_id": sample_id,
            "input_tokens": input_tokens,
            **_memory_snapshot(torch, device),
        }
    )
    with _ForwardTelemetry(
        model=api.model,
        torch=torch,
        device=device,
        event_log=event_log,
        sample_id=sample_id,
        every=every,
    ) as telemetry:
        with torch.inference_mode():
            generated = _eos_only_model_generate(
                api.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=api.tokenizer,
                config=config,
                do_sample=api.do_sample,
                return_dict_in_generate=True,
                output_logits=False,
                output_hidden_states=False,
            )
    sequences = generated.sequences
    output_tokens = int(sequences.shape[1] - input_ids.shape[1])
    if output_tokens < 1:
        raise GemmaEOSDebugError("EOS-only generator returned no generated token")
    final_token_id = int(sequences[0, -1].item())
    if final_token_id not in eos_ids:
        raise GemmaEOSDebugError(
            f"EOS-only generator returned a non-EOS final token {final_token_id}; expected one of {sorted(eos_ids)}"
        )
    result = {
        "sample_id": sample_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_token_id": final_token_id,
        "model_forward_count": telemetry.decode_steps,
        "completed_by_model_eos": True,
        **_memory_snapshot(torch, device),
    }
    event_log.write({"event": "sample_completed", **result})
    return result


def _install_signal_recorder(event_log: _EventLog) -> Callable[[], None]:
    """Make scheduler interruption explicit without turning it into a completion."""

    previous: dict[int, Any] = {}

    def record_signal(signum: int, _frame: FrameType | None) -> None:
        event_log.write({"event": "interrupted", "signal": signal.Signals(signum).name})
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, record_signal)

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return restore


def run(
    *,
    manifest: str,
    model_snapshot: str,
    log_dir: str,
    task_index: int,
    rank: int | None,
    question_ids: Sequence[str] | None,
    memory_every: int,
) -> dict[str, Any]:
    """Run the standalone diagnostic and return only EOS-completed evidence."""

    if memory_every < 1:
        raise GemmaEOSDebugError("--memory-every must be positive; it is telemetry cadence, not a stopping rule")
    deployed_manifest = _require_regular_file(manifest, label="--manifest")
    snapshot = _require_regular_directory(model_snapshot, label="--model-snapshot")
    output_root = _create_new_log_directory(log_dir)
    event_log = _EventLog(output_root / "events.jsonl")
    restore_signals = _install_signal_recorder(event_log)
    try:
        from inspect_ai.model import GenerateConfig

        from ctm.evals.hf_eos_only import runtime_policy
        from experiments.elephant_aita_ntaflip.no_cap_hf import _assert_no_runtime_token_cap, _eos_ids
        from infra.isambard.run_gemma4_12b_base_two_bias_evals_16gpu import GENERATION_CONFIG, MODEL_ARGS
        from ctm.evals.local_model import gemma4_unified_hf_model

        import torch

        policy = runtime_policy()
        generation_config = GenerateConfig(**dict(GENERATION_CONFIG))
        _assert_no_runtime_token_cap(generation_config, label="Gemma EOS diagnostic generation config")
        if not torch.cuda.is_available():
            raise GemmaEOSDebugError("Gemma EOS diagnostic requires one visible CUDA GPU")
        selected_ids, samples = _load_hle_samples(
            manifest=deployed_manifest,
            task_index=task_index,
            rank=rank,
            question_ids=question_ids,
        )
        event_log.write(
            {
                "event": "started",
                "manifest": str(deployed_manifest),
                "manifest_sha256": _sha256_file(deployed_manifest),
                "model_snapshot": str(snapshot),
                "model_config_sha256": _sha256_file(snapshot / "config.json"),
                "task_index": task_index,
                "selected_question_ids": list(selected_ids),
                "memory_every": memory_every,
                "sampling": dict(GENERATION_CONFIG),
                "model_args": dict(MODEL_ARGS),
                "runtime_policy": policy,
            }
        )
        resolved = gemma4_unified_hf_model(
            f"hf/{snapshot}",
            model_args=dict(MODEL_ARGS),
            generation_config=dict(GENERATION_CONFIG),
        )
        api = getattr(resolved, "api", None)
        if api is None or getattr(api, "model", None) is None:
            raise GemmaEOSDebugError("Gemma HF bridge did not expose its native model API")
        eos = _eos_sources(api.tokenizer, api.model)
        eos_ids = _eos_ids(api.tokenizer, api.model)
        if set(eos["concrete_ids"]) != set(eos_ids):
            raise GemmaEOSDebugError("the diagnostic EOS source record differs from the shared generator EOS set")
        device = api.model.device
        torch.cuda.reset_peak_memory_stats(device)
        event_log.write({"event": "model_loaded", "eos": eos, **_memory_snapshot(torch, device)})
        completed = [
            _decode_one(
                api=api,
                config=generation_config,
                sample=sample,
                torch=torch,
                event_log=event_log,
                every=memory_every,
                eos_ids=eos_ids,
            )
            for sample in samples
        ]
        receipt = {
            "schema": "gemma4-eos-diagnostic-v1",
            "status": "completed_after_model_eos",
            "manifest": str(deployed_manifest),
            "manifest_sha256": _sha256_file(deployed_manifest),
            "model_snapshot": str(snapshot),
            "task_index": task_index,
            "selected_question_ids": list(selected_ids),
            "eos": eos,
            "sampling": dict(GENERATION_CONFIG),
            "model_args": dict(MODEL_ARGS),
            "runtime_policy": policy,
            "samples": completed,
        }
        _write_once_json(output_root / "completed.json", receipt)
        event_log.write({"event": "diagnostic_completed", "completed_samples": len(completed)})
        return receipt
    except BaseException as exc:
        event_log.write({"event": "diagnostic_incomplete", "exception_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        restore_signals()
        event_log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="absolute deployed Stage-2 manifest path")
    parser.add_argument("--model-snapshot", required=True, help="absolute pinned Gemma snapshot directory")
    parser.add_argument("--log-dir", required=True, help="new absolute directory for diagnostic-only telemetry")
    parser.add_argument("--task-index", type=int, default=3, help="clean HLE task index (default: 3)")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--rank", type=int, help="use the deployed 16-way source-ID shard")
    selector.add_argument("--question-ids", type=_parse_question_ids, help="JSON array of selected scored HLE IDs")
    parser.add_argument(
        "--memory-every",
        type=int,
        default=256,
        help="record memory every N model forwards; telemetry only, never a stopping condition (default: 256)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = run(
        manifest=args.manifest,
        model_snapshot=args.model_snapshot,
        log_dir=args.log_dir,
        task_index=args.task_index,
        rank=args.rank,
        question_ids=args.question_ids,
        memory_every=args.memory_every,
    )
    print(json.dumps({"status": receipt["status"], "completed_samples": len(receipt["samples"])}, sort_keys=True))


if __name__ == "__main__":
    main()
