"""Regression tests for the observational Gemma HF load bootstrap."""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import weakref

import pytest


_PATH = Path(__file__).resolve().parents[1] / "infra/isambard/trace_gemma_hf_loads.py"


def _load_module():
    name = "trace_gemma_hf_loads_test_module"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves its defining module while decorating _Patch.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


def _install_fake_loader_stack(monkeypatch):
    """Install a minimal import tree whose Auto path nests into PreTrained."""

    calls: dict[str, list[object]] = {"init": [], "auto": [], "pretrained": [], "core": []}

    class FakePreTrainedModel:
        @classmethod
        def from_pretrained(cls, marker, **kwargs):
            calls["pretrained"].append((cls.__name__, marker, dict(kwargs)))
            return cls(marker)

        def __init__(self, marker):
            self.marker = marker

    class ConcreteModel(FakePreTrainedModel):
        pass

    def convert_and_load_state_dict_in_model(model, **kwargs):
        calls["core"].append((type(model).__name__, dict(kwargs)))
        return model

    modeling_utils = ModuleType("transformers.modeling_utils")
    modeling_utils.PreTrainedModel = FakePreTrainedModel
    # This alias deliberately has the original function object.  A tracer that
    # patches only core_model_loading would miss this already-imported global.
    modeling_utils.convert_and_load_state_dict_in_model = convert_and_load_state_dict_in_model

    core_loading = ModuleType("transformers.core_model_loading")
    core_loading.convert_and_load_state_dict_in_model = convert_and_load_state_dict_in_model

    class FakeAutoModelForImageTextToText:
        @classmethod
        def from_pretrained(cls, marker, **kwargs):
            calls["auto"].append((cls.__name__, marker, dict(kwargs)))
            model = ConcreteModel.from_pretrained(marker, **kwargs)
            return modeling_utils.convert_and_load_state_dict_in_model(model)

    class FakeHuggingFaceAPI:
        def __init__(self, marker, **kwargs):
            calls["init"].append((marker, dict(kwargs)))
            self.marker = marker
            self.kwargs = dict(kwargs)

    hf_module = ModuleType("inspect_ai.model._providers.hf")
    hf_module.HuggingFaceAPI = FakeHuggingFaceAPI

    transformers = _package("transformers")
    transformers.AutoModelForImageTextToText = FakeAutoModelForImageTextToText
    inspect_ai = _package("inspect_ai")
    inspect_model = _package("inspect_ai.model")
    inspect_providers = _package("inspect_ai.model._providers")

    for name, module in {
        "inspect_ai": inspect_ai,
        "inspect_ai.model": inspect_model,
        "inspect_ai.model._providers": inspect_providers,
        "inspect_ai.model._providers.hf": hf_module,
        "transformers": transformers,
        "transformers.modeling_utils": modeling_utils,
        "transformers.core_model_loading": core_loading,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return {
        "calls": calls,
        "auto": FakeAutoModelForImageTextToText,
        "pretrained": FakePreTrainedModel,
        "concrete": ConcreteModel,
        "hf": FakeHuggingFaceAPI,
        "core": core_loading,
        "modeling_utils": modeling_utils,
        "original_auto": FakeAutoModelForImageTextToText.__dict__["from_pretrained"],
        "original_pretrained": FakePreTrainedModel.__dict__["from_pretrained"],
        "original_init": FakeHuggingFaceAPI.__dict__["__init__"],
        "original_core": convert_and_load_state_dict_in_model,
    }


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nested_loader_trace_preserves_binding_return_and_only_weakly_tracks_models(tmp_path, monkeypatch):
    module = _load_module()
    fake = _install_fake_loader_stack(monkeypatch)
    trace_path = tmp_path / "model-load-trace.jsonl"
    tracer = module.LoadTracer(trace_path)
    tracer.install()
    try:
        secret = "SECRET_ARGUMENT_DO_NOT_RECORD"
        api = fake["hf"](secret, private_key=secret)
        model = fake["auto"].from_pretrained(secret, auth_token=secret)
        # A direct core call proves that both core and the pre-imported
        # modeling_utils alias are independently observable.
        assert fake["core"].convert_and_load_state_dict_in_model(model) is model

        assert api.marker == secret
        assert model.marker == secret
        assert fake["calls"]["init"] == [(secret, {"private_key": secret})]
        assert fake["calls"]["auto"] == [("FakeAutoModelForImageTextToText", secret, {"auth_token": secret})]
        assert fake["calls"]["pretrained"] == [("ConcreteModel", secret, {"auth_token": secret})]
        assert fake["calls"]["core"] == [("ConcreteModel", {}), ("ConcreteModel", {})]

        model_id = id(model)
        reference = weakref.ref(model)
        del model
        gc.collect()
        assert reference() is None
        tracer.write_final()
    finally:
        tracer.uninstall()
        tracer.close()

    # Descriptor restoration matters because the diagnostic executes in the
    # same interpreter as the target, not a child process.
    assert fake["auto"].__dict__["from_pretrained"] is fake["original_auto"]
    assert fake["pretrained"].__dict__["from_pretrained"] is fake["original_pretrained"]
    assert fake["hf"].__dict__["__init__"] is fake["original_init"]
    assert fake["core"].convert_and_load_state_dict_in_model is fake["original_core"]
    assert fake["modeling_utils"].convert_and_load_state_dict_in_model is fake["original_core"]

    events = _events(trace_path)
    serialized = trace_path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert all("traceback" not in event or all(set(frame) == {"file", "line", "function"}
                                                for frame in event["traceback"])
               for event in events)

    starts = {event["hook"]: event for event in events if event["event"] == "call_start"}
    auto = starts["transformers_auto_image_text_from_pretrained"]
    pretrained = starts["transformers_pretrained_from_pretrained"]
    alias_load = starts["transformers_modeling_utils_convert_and_load_state_dict"]
    core_load = starts["transformers_core_convert_and_load_state_dict"]
    assert pretrained["parent_call_id"] == auto["call_id"]
    assert alias_load["parent_call_id"] == auto["call_id"]
    assert core_load["parent_call_id"] is None

    returned = [event for event in events if event["event"] == "call_return"]
    auto_return = next(event for event in returned
                       if event["hook"] == "transformers_auto_image_text_from_pretrained")
    pretrained_return = next(event for event in returned
                             if event["hook"] == "transformers_pretrained_from_pretrained")
    assert auto_return["result_object_id"] == model_id == pretrained_return["result_object_id"]
    assert auto_return["weakref_tracked"] is True
    final = next(event for event in events if event["event"] == "final")
    assert model_id not in final["alive_model_object_ids"]


def test_cli_runs_target_with_its_argv_and_restores_the_caller_argv(tmp_path, monkeypatch):
    module = _load_module()
    _install_fake_loader_stack(monkeypatch)
    target = tmp_path / "target.py"
    observed = tmp_path / "observed-argv.json"
    target.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TRACE_GEMMA_TEST_OUTPUT']).write_text(json.dumps(sys.argv))\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TRACE_GEMMA_TEST_OUTPUT", str(observed))
    original_argv = sys.argv

    assert module.main(["--trace-file", str(trace), str(target), "alpha", "--beta"]) == 0

    assert sys.argv is original_argv
    assert json.loads(observed.read_text(encoding="utf-8")) == [str(target.resolve()), "alpha", "--beta"]
    events = _events(trace)
    assert [event["event"] for event in events if event["event"].startswith("target_")] == [
        "target_start",
        "target_return",
    ]
    assert events[-1]["event"] == "final"


def test_trace_file_is_exclusive_and_refuses_a_link(tmp_path):
    module = _load_module()
    existing = tmp_path / "existing.jsonl"
    existing.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.LoadTracer(existing)

    link = tmp_path / "linked.jsonl"
    link.symlink_to(existing)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.LoadTracer(link)
