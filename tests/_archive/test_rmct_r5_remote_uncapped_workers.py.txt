"""Regression contract for the exact remote RMCT r5 worker source copy.

The remote worker module has heavyweight runtime imports, so these tests inspect
its parsed AST rather than importing a second backend into the local process.
They specifically guard the `None` transport path used by EOS-only RMCT.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_WORKERS = ROOT / "tmp/rmct-r5-remote-worker-fix-20260908/rollout_workers.py"


def _tree() -> ast.Module:
    return ast.parse(REMOTE_WORKERS.read_text(encoding="utf-8"), filename=str(REMOTE_WORKERS))


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.AsyncFunctionDef:
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in class_node.body if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name)
    return method


def _keyword(call: ast.Call, name: str) -> ast.expr:
    keyword = next(item for item in call.keywords if item.arg == name)
    return keyword.value


def test_remote_worker_dispatch_preserves_none_instead_of_coercing_it_to_int() -> None:
    tree = _tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sample_batch"
        and any(keyword.arg == "max_tokens" for keyword in node.keywords)
    ]
    worker_call = next(
        call
        for call in calls
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "sampler"
    )
    value = _keyword(worker_call, "max_tokens")
    assert isinstance(value, ast.IfExp)
    assert isinstance(value.body, ast.Constant) and value.body.value is None
    assert isinstance(value.orelse, ast.Call)
    assert isinstance(value.orelse.func, ast.Name) and value.orelse.func.id == "int"
    assert isinstance(value.test, ast.Compare)
    assert isinstance(value.test.left, ast.Call)
    assert isinstance(value.test.left.func, ast.Attribute)
    assert isinstance(value.test.left.func.value, ast.Name) and value.test.left.func.value.id == "command"
    assert value.test.left.func.attr == "get"
    assert isinstance(value.test.left.args[0], ast.Constant) and value.test.left.args[0].value == "max_tokens"
    assert len(value.test.ops) == 1 and isinstance(value.test.ops[0], ast.Is)
    assert len(value.test.comparators) == 1
    assert isinstance(value.test.comparators[0], ast.Constant) and value.test.comparators[0].value is None


def test_remote_pool_accepts_none_and_only_rejects_nonpositive_integer_limits() -> None:
    tree = _tree()
    method = _class_method(tree, "RolloutWorkerPool", "sample_batch")
    max_tokens = next(argument for argument in method.args.kwonlyargs if argument.arg == "max_tokens")
    assert ast.unparse(max_tokens.annotation) == "int | None"
    guards = [node for node in ast.walk(method) if isinstance(node, ast.If)]
    assert any(ast.unparse(guard.test) == "max_tokens is not None and max_tokens <= 0" for guard in guards)


def test_remote_public_sampling_layers_forward_optional_max_tokens_end_to_end() -> None:
    tree = _tree()
    expected = (
        ("DistributedSamplerHandle", "sample"),
        ("DistributedSamplerHandle", "sample_batch"),
        ("FrozenBaseSamplerHandle", "sample_batch"),
        ("FrozenBaseVLLMBackend", "sample_base_batch"),
    )
    for class_name, method_name in expected:
        method = _class_method(tree, class_name, method_name)
        max_tokens = next(argument for argument in method.args.kwonlyargs if argument.arg == "max_tokens")
        assert ast.unparse(max_tokens.annotation) == "int | None", f"{class_name}.{method_name}"

    frozen = _class_method(tree, "FrozenBaseVLLMBackend", "sample_base_batch")
    forwards = [node for node in ast.walk(frozen) if isinstance(node, ast.Call)]
    values = [ast.unparse(_keyword(call, "max_tokens")) for call in forwards if any(item.arg == "max_tokens" for item in call.keywords)]
    assert values.count("max_tokens") >= 2  # worker-pool and in-process sampler paths
