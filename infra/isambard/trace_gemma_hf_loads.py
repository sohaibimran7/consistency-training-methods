#!/usr/bin/env python3
"""Trace Gemma/Inspect model construction without changing evaluation behaviour.

This bootstrap is deliberately observational.  It neither imports nor changes a
generation configuration, and it does not call ``model.generate``.  It wraps
the small set of constructors involved in the Gemma Inspect path and writes
sanitised JSONL evidence before executing the requested script in this same
interpreter.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
import weakref


TRACE_SCHEMA = "gemma-hf-load-trace-v1"
_MISSING = object()
_T = TypeVar("_T")


class TraceGemmaHFLoadsError(RuntimeError):
    """Raised when the trace bootstrap cannot safely establish its contract."""


def _sanitised_stack(frames: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]:
    """Return location-only frames: no source line, values, args, or locals."""

    return [
        {
            "file": Path(frame.filename).name,
            "line": frame.lineno,
            "function": frame.name,
        }
        for frame in frames
    ]


def _type_name(value: Any) -> str:
    subject = value if isinstance(value, type) else type(value)
    module = getattr(subject, "__module__", "<unknown>")
    qualname = getattr(subject, "__qualname__", getattr(subject, "__name__", "<unknown>"))
    return f"{module}.{qualname}"


def _cuda_memory() -> dict[str, Any]:
    """Read CUDA counters only after CUDA is already initialized.

    Importing torch or probing availability can itself have side effects, so this
    deliberately uses only a torch module that the target has already imported.
    """

    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    initialized = getattr(cuda, "is_initialized", None)
    if not callable(initialized):
        return {"initialized": False}
    try:
        if not initialized():
            return {"initialized": False}
    except BaseException:
        return {"initialized": False, "state": "unreadable"}

    try:
        device = cuda.current_device()
        return {
            "initialized": True,
            "device": device,
            "allocated_bytes": cuda.memory_allocated(device),
            "reserved_bytes": cuda.memory_reserved(device),
            "peak_allocated_bytes": cuda.max_memory_allocated(device),
            "peak_reserved_bytes": cuda.max_memory_reserved(device),
        }
    except BaseException:
        # Do not include exception messages: library paths and values can be
        # sensitive, and the trace needs only show that counters were unreadable.
        return {"initialized": True, "state": "unreadable"}


class _TraceWriter:
    """Append-only, exclusive JSONL writer with serialised line writes."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise TraceGemmaHFLoadsError("--trace-file must be an absolute path")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite or follow trace file: {path}")
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise TraceGemmaHFLoadsError("trace file parent must be an existing non-linked directory")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Keep a race with another tracer fail-closed too.
            raise FileExistsError(f"refusing to overwrite or follow trace file: {path}") from None
        self._handle = os.fdopen(fd, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            if self._closed:
                return
            self._handle.write(encoded)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._handle.close()
                self._closed = True


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any

    def restore(self) -> None:
        if self.original is _MISSING:
            delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.original)


def _find_descriptor(owner: type[Any], name: str) -> tuple[type[Any], Any]:
    for base in owner.__mro__:
        if name in base.__dict__:
            return base, base.__dict__[name]
    raise AttributeError(f"{_type_name(owner)} has no {name}")


class LoadTracer:
    """A reversible, thread-aware set of constructor/load-loop wrappers."""

    def __init__(self, trace_file: str | Path) -> None:
        self.path = Path(trace_file)
        self._writer = _TraceWriter(self.path)
        self._patches: list[_Patch] = []
        self._local = threading.local()
        self._call_ids = itertools.count(1)
        self._call_lock = threading.Lock()
        self._model_refs: list[tuple[int, weakref.ReferenceType[Any]]] = []
        self._model_refs_lock = threading.Lock()
        self._final_written = False

    def _event(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "schema": TRACE_SCHEMA,
            "event": event,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "thread_id": threading.get_ident(),
            "monotonic_s": time.monotonic(),
        }
        payload.update(fields)
        self._writer.write(payload)

    def _stack(self) -> list[int]:
        stack = getattr(self._local, "call_stack", None)
        if stack is None:
            stack = []
            self._local.call_stack = stack
        return stack

    def _next_call_id(self) -> int:
        with self._call_lock:
            return next(self._call_ids)

    def _track_model(self, value: Any) -> bool:
        """Keep a weak reference only; never retain a returned model instance."""

        try:
            reference = weakref.ref(value)
        except TypeError:
            return False
        with self._model_refs_lock:
            self._model_refs.append((id(value), reference))
        return True

    def _invoke(
        self,
        *,
        hook: str,
        receiver: Any,
        invoke: Callable[[], _T],
        returned_object: Any | None = None,
        track_result: bool = False,
    ) -> _T:
        stack = self._stack()
        call_id = self._next_call_id()
        parent_call_id = stack[-1] if stack else None
        depth = len(stack)
        stack.append(call_id)
        self._event(
            "call_start",
            call_id=call_id,
            parent_call_id=parent_call_id,
            depth=depth,
            hook=hook,
            receiver_type=_type_name(receiver),
            cuda=_cuda_memory(),
            traceback=_sanitised_stack(traceback.extract_stack(limit=20)[:-1]),
        )
        try:
            result = invoke()
        except BaseException as exc:
            self._event(
                "call_raise",
                call_id=call_id,
                parent_call_id=parent_call_id,
                depth=depth,
                hook=hook,
                receiver_type=_type_name(receiver),
                exception_type=_type_name(exc),
                cuda=_cuda_memory(),
                traceback=_sanitised_stack(traceback.extract_tb(exc.__traceback__, limit=20)),
            )
            raise
        else:
            observed = result if returned_object is None else returned_object
            tracked = self._track_model(result) if track_result else False
            self._event(
                "call_return",
                call_id=call_id,
                parent_call_id=parent_call_id,
                depth=depth,
                hook=hook,
                receiver_type=_type_name(receiver),
                result_object_id=id(observed),
                result_type=_type_name(observed),
                weakref_tracked=tracked if track_result else None,
                cuda=_cuda_memory(),
            )
            return result
        finally:
            popped = stack.pop()
            if popped != call_id:
                # A wrapper never deliberately recovers from a corrupted stack:
                # clear its diagnostic state rather than misattribute later calls.
                stack.clear()

    def _patch_instance_method(self, owner: type[Any], name: str, hook: str) -> None:
        original = owner.__dict__.get(name, _MISSING)
        bound = getattr(owner, name)

        @functools.wraps(bound)
        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            return self._invoke(
                hook=hook,
                receiver=instance,
                invoke=lambda: bound(instance, *args, **kwargs),
                returned_object=instance,
            )

        setattr(owner, name, wrapped)
        self._patches.append(_Patch(owner, name, original))

    def _patch_class_method(self, owner: type[Any], name: str, hook: str) -> None:
        descriptor_owner, descriptor = _find_descriptor(owner, name)
        original = owner.__dict__.get(name, _MISSING)
        if isinstance(descriptor, classmethod):
            original_function = descriptor.__func__

            @functools.wraps(original_function)
            def wrapped(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
                return self._invoke(
                    hook=hook,
                    receiver=cls,
                    invoke=lambda: original_function(cls, *args, **kwargs),
                    track_result=True,
                )

        elif isinstance(descriptor, staticmethod):
            # This is not expected for the pinned APIs, but preserving it makes
            # the tracer safe against a future descriptor implementation.
            original_function = descriptor.__func__

            @functools.wraps(original_function)
            def wrapped(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
                return self._invoke(
                    hook=hook,
                    receiver=cls,
                    invoke=lambda: original_function(*args, **kwargs),
                    track_result=True,
                )

        else:
            # A regular function inherited through the MRO receives the concrete
            # class when accessed from a class.  Preserve that dispatch too.
            original_function = descriptor

            @functools.wraps(original_function)
            def wrapped(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
                return self._invoke(
                    hook=hook,
                    receiver=cls,
                    invoke=lambda: original_function(cls, *args, **kwargs),
                    track_result=True,
                )

        setattr(owner, name, classmethod(wrapped))
        self._patches.append(_Patch(owner, name, original))

    def _patch_function(self, owner: Any, name: str, hook: str) -> None:
        original = owner.__dict__.get(name, _MISSING)
        if original is _MISSING or not callable(original):
            raise AttributeError(f"{owner!r} has no callable {name}")

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(
                hook=hook,
                receiver=owner,
                invoke=lambda: original(*args, **kwargs),
            )

        setattr(owner, name, wrapped)
        self._patches.append(_Patch(owner, name, original))

    def install(self) -> None:
        """Install all known pinned-path hooks before the target starts."""

        if self._patches:
            raise TraceGemmaHFLoadsError("load tracer is already installed")
        try:
            hf_module = importlib.import_module("inspect_ai.model._providers.hf")
            huggingface_api = getattr(hf_module, "HuggingFaceAPI")
            transformers = importlib.import_module("transformers")
            auto_model = getattr(transformers, "AutoModelForImageTextToText")
        except (ImportError, AttributeError) as exc:
            raise TraceGemmaHFLoadsError("required Inspect/Transformers loader API is unavailable") from exc

        try:
            self._patch_instance_method(huggingface_api, "__init__", "inspect_huggingfaceapi_init")
            self._patch_class_method(
                auto_model,
                "from_pretrained",
                "transformers_auto_image_text_from_pretrained",
            )

            # The base-class hook sees the concrete Gemma class passed by Auto.
            # If a future Transformers release omits this API, preserve the
            # requested evaluation and make the missing lower-level evidence
            # explicit instead of pretending generic parity.
            try:
                modeling_utils = importlib.import_module("transformers.modeling_utils")
                pretrained_model = getattr(modeling_utils, "PreTrainedModel")
                self._patch_class_method(
                    pretrained_model,
                    "from_pretrained",
                    "transformers_pretrained_from_pretrained",
                )
            except (ImportError, AttributeError) as exc:
                self._event(
                    "hook_unavailable",
                    hook="transformers_pretrained_from_pretrained",
                    exception_type=_type_name(exc),
                )
                modeling_utils = None

            # In pinned Transformers this is the sole 677-tensor progress-loop
            # implementation.  Patch both the defining module and the
            # modeling_utils alias, because the latter can have been imported
            # before the tracer starts and would otherwise bypass the core hook.
            try:
                core_loading = importlib.import_module("transformers.core_model_loading")
                self._patch_function(
                    core_loading,
                    "convert_and_load_state_dict_in_model",
                    "transformers_core_convert_and_load_state_dict",
                )
                alias = (
                    getattr(modeling_utils, "convert_and_load_state_dict_in_model", None)
                    if modeling_utils is not None
                    else None
                )
                # If a future release exposes a live alias to the now-patched
                # core function, it is already instrumented.  Patching it again
                # would turn one loader invocation into two trace events.
                if alias is not None and alias is not getattr(
                    core_loading, "convert_and_load_state_dict_in_model"
                ):
                    self._patch_function(
                        modeling_utils,
                        "convert_and_load_state_dict_in_model",
                        "transformers_modeling_utils_convert_and_load_state_dict",
                    )
            except (ImportError, AttributeError) as exc:
                self._event(
                    "hook_unavailable",
                    hook="transformers_convert_and_load_state_dict",
                    exception_type=_type_name(exc),
                )
        except BaseException:
            self.uninstall()
            raise

    def alive_model_object_ids(self) -> list[int]:
        alive: set[int] = set()
        with self._model_refs_lock:
            references = tuple(self._model_refs)
        for object_id, reference in references:
            value = reference()
            if value is not None:
                alive.add(object_id)
            # Avoid holding a returned model beyond this immediately-scoped
            # liveness check.
            del value
        return sorted(alive)

    def write_final(self) -> None:
        if self._final_written:
            return
        self._final_written = True
        self._event(
            "final",
            cuda=_cuda_memory(),
            alive_model_object_ids=self.alive_model_object_ids(),
        )

    def uninstall(self) -> None:
        while self._patches:
            self._patches.pop().restore()

    def close(self) -> None:
        self._writer.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", required=True, help="new absolute JSONL trace path")
    parser.add_argument("target_script", help="Python script to execute under the tracer")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="arguments passed unchanged to target_script")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target = Path(args.target_script).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"target script is not a file: {target}")

    tracer = LoadTracer(args.trace_file)
    previous_argv = sys.argv
    try:
        tracer.install()
        tracer._event("target_start")
        sys.argv = [str(target), *args.target_args]
        try:
            runpy.run_path(str(target), run_name="__main__")
        except BaseException as exc:
            tracer._event(
                "target_raise",
                exception_type=_type_name(exc),
                traceback=_sanitised_stack(traceback.extract_tb(exc.__traceback__, limit=20)),
            )
            raise
        else:
            tracer._event("target_return")
    finally:
        sys.argv = previous_argv
        tracer.write_final()
        tracer.uninstall()
        tracer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
