#!/usr/bin/env python3
"""Create and validate the isolated CPU-only Gemma Luna grading runtime.

This bootstrap is deliberately narrower than the evaluator runtime.  It makes
one fresh virtual environment, installs only the packages needed to import the
Gemma postprocess/comparison code and the uncapped Luna scorer, and records
immutable, non-secret evidence.  It never modifies the GPU runtime, a Gemma
campaign checkout, credentials, or scheduler state.

The script is copied into the new runtime root before it is run.  That root
must either be absent or contain this bootstrap alone; this prevents an
accidental in-place upgrade of an existing runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


RUNTIME_SCHEMA = "ctm-gemma4-luna-cpu-runtime-v1"
RUNTIME_BASENAME = "ctm-gemma4-luna-runtime-20260908"
DEFAULT_RUNTIME_ROOT = Path("/projects/a5v/sohaib.a5v") / RUNTIME_BASENAME
DEFAULT_BASE_PYTHON = Path(
    "/projects/a5v/sohaib.a5v/consistency-training-methods/.venv-muse-cu129/bin/python"
)
DEFAULT_SOURCE_ROOT = Path(
    "/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-single-load-20260908/repo"
)

# The mcq-bias commit is the exact immutable VCS pin recorded in the existing
# active runtime's installed direct_url metadata. It is installed as a normal
# wheel, never editable and never from a mutable checkout.
MCQ_BIAS_URL = "https://github.com/sohaibimran7/mcq-bias"
MCQ_BIAS_COMMIT = "1df2ea1ed8a1eeaf6ec5088c066db7c8c1049119"
MCQ_BIAS_SPEC = f"mcq-bias @ git+{MCQ_BIAS_URL}@{MCQ_BIAS_COMMIT}"
CORE_REQUIREMENTS = (
    "inspect-ai==0.3.260",
    "transformers==5.15.1",
    "openai==3.1.0",
    "matplotlib==3.10.8",
    "pytest==8.0.2",
)
EXPECTED_PACKAGES = {
    "inspect-ai": "0.3.260",
    "transformers": "5.15.1",
    "openai": "3.1.0",
    "mcq-bias": "0.1.0",
}
FORBIDDEN_GPU_DISTRIBUTIONS = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "vllm",
        "flash-attn",
        "flash-attention",
        "cupy",
    }
)
FORBIDDEN_GPU_PREFIXES = ("nvidia-", "cuda-", "cudnn-")
SOURCE_HASH_FILES = (
    "ctm_data/adapters/mcq_bias/luna_scorer_no_cap.py",
    "experiments/gemma4_12b_base_eval/postprocess.py",
    "experiments/gemma4_12b_base_eval/combined_comparison.py",
)


class LunaRuntimeError(RuntimeError):
    """The isolated runtime cannot be safely created or validated."""


def _normalise(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing runtime evidence: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"runtime evidence appeared with different bytes: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        check=False,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise LunaRuntimeError(f"command failed ({completed.returncode}): {' '.join(command[:4])}; {detail[-1000:]}")
    return completed


def _safe_runtime_root(value: str | Path) -> Path:
    root = _normalise(value)
    approved_root = _normalise(DEFAULT_RUNTIME_ROOT)
    if root != approved_root:
        raise LunaRuntimeError(f"runtime root must be exactly {approved_root}, got {root}")
    if root.name != RUNTIME_BASENAME or root.parent != approved_root.parent:
        raise LunaRuntimeError("runtime root escaped its approved sibling location")
    if root.is_symlink():
        raise LunaRuntimeError("runtime root must not be a symlink")
    return root


def _bootstrap_root(root: Path) -> None:
    """Allow only the copied bootstrap before fresh environment creation."""

    bootstrap_name = Path(__file__).name
    if root.exists():
        if not root.is_dir() or root.is_symlink():
            raise LunaRuntimeError(f"runtime root is not a regular directory: {root}")
        found = {entry.name for entry in root.iterdir()}
        if found != {bootstrap_name}:
            raise FileExistsError(
                f"runtime root is not new (expected only {bootstrap_name!r}, found {sorted(found)!r})"
            )
        return
    root.mkdir(mode=0o700)


def _source_root(value: str | Path) -> Path:
    source = _normalise(value)
    approved_source = _normalise(DEFAULT_SOURCE_ROOT)
    if source != approved_source:
        raise LunaRuntimeError(f"source root must be the frozen Gemma checkout {approved_source}, got {source}")
    if source.is_symlink() or not source.is_dir():
        raise LunaRuntimeError("frozen Gemma source root is unavailable or linked")
    for relative in SOURCE_HASH_FILES:
        candidate = source / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise LunaRuntimeError(f"frozen source is missing required file: {candidate}")
    return source


def _base_python(value: str | Path) -> Path:
    # Check canonical identity to accept the site's /projects -> /lus alias,
    # but retain the caller's launch spelling. Resolving a venv/bin/python
    # symlink and executing its target would lose the intended venv identity.
    python = Path(value).expanduser()
    if not python.is_absolute() or _normalise(python) != _normalise(DEFAULT_BASE_PYTHON):
        raise LunaRuntimeError("base Python must identify the read-only pinned GPU runtime executable")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LunaRuntimeError("base Python must be the read-only pinned GPU runtime executable")
    return python


def _report_distribution_names(report_path: Path) -> set[str]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LunaRuntimeError(f"pip report is unreadable: {report_path}") from exc
    installs = report.get("install")
    if not isinstance(installs, list):
        raise LunaRuntimeError("pip report does not contain an install list")
    names: set[str] = set()
    for item in installs:
        metadata = item.get("metadata") if isinstance(item, Mapping) else None
        name = metadata.get("name") if isinstance(metadata, Mapping) else None
        if not isinstance(name, str) or not name:
            raise LunaRuntimeError("pip report contains an unnamed distribution")
        names.add(name.lower().replace("_", "-"))
    return names


def _assert_no_gpu_distributions(names: Iterable[str]) -> list[str]:
    normalized = sorted({name.lower().replace("_", "-") for name in names})
    forbidden = [
        name
        for name in normalized
        if name in FORBIDDEN_GPU_DISTRIBUTIONS or name.startswith(FORBIDDEN_GPU_PREFIXES)
    ]
    if forbidden:
        raise LunaRuntimeError(f"CPU Luna runtime would install forbidden GPU distributions: {forbidden}")
    return normalized


def _requirements_path(root: Path) -> Path:
    path = root / "requirements.cpu-luna.txt"
    payload = "\n".join((*CORE_REQUIREMENTS, ""))
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite differing CPU Luna requirements: {path}")
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _clean_child_env(root: Path, source: Path) -> dict[str, str]:
    cache = root / "runtime-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    return {
        "PATH": f"{root / 'bin'}:/usr/bin:/bin",
        "XDG_CACHE_HOME": str(cache),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TMPDIR": str(cache),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(source),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in EXPECTED_PACKAGES:
        versions[name] = importlib.metadata.version(name)
    return versions


def _source_hashes(source: Path) -> dict[str, str]:
    return {relative: _sha256(source / relative) for relative in SOURCE_HASH_FILES}


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LunaRuntimeError(f"{label} did not produce valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise LunaRuntimeError(f"{label} did not produce a JSON object: {path}")
    return dict(value)


def _run_installed_verification(root: Path, source: Path) -> dict[str, Any]:
    """Query package metadata from the fresh venv, never the bootstrap venv."""

    output = root / "installed-packages.json"
    command = [
        str(root / "bin/python"),
        str(root / Path(__file__).name),
        "installed-verification-child",
        "--runtime-root",
        str(root),
        "--output",
        str(output),
    ]
    _run(command, env=_clean_child_env(root, source), cwd=root)
    result = _read_json_mapping(output, label="fresh-runtime package verification")
    if result.get("schema") != RUNTIME_SCHEMA:
        raise LunaRuntimeError("fresh-runtime package verification has the wrong schema")
    return result


def _run_offline_attestation(root: Path, source: Path) -> dict[str, Any]:
    output = root / "offline-attestation.json"
    command = [
        str(root / "bin/python"),
        str(root / Path(__file__).name),
        "offline-attestation-child",
        "--runtime-root",
        str(root),
        "--source-root",
        str(source),
        "--output",
        str(output),
    ]
    _run(command, env=_clean_child_env(root, source), cwd=root)
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LunaRuntimeError("offline Luna attestation did not produce valid JSON") from exc
    if not isinstance(result, Mapping) or result.get("schema") != RUNTIME_SCHEMA:
        raise LunaRuntimeError("offline Luna attestation has the wrong schema")
    return dict(result)


def _run_selected_tests(root: Path, source: Path) -> dict[str, Any]:
    """Run Gemma CPU tests; the separate offline attestation proves provider policy."""

    command = [
        str(root / "bin/python"),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_gemma4_12b_base_postprocess.py",
        "tests/test_gemma4_12b_combined_comparison.py",
    ]
    completed = _run(command, env=_clean_child_env(root, source), cwd=source)
    result = {
        "command": command[1:],
        "scope": "Gemma postprocess/comparison only; Muse tests import training code requiring Torch",
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    _write_json_once(root / "selected-test-results.json", result)
    return result


def prepare(*, runtime_root: str | Path, source_root: str | Path, base_python: str | Path) -> dict[str, Any]:
    root = _safe_runtime_root(runtime_root)
    source = _source_root(source_root)
    bootstrap = _base_python(base_python)
    _bootstrap_root(root)

    python = root / "bin/python"
    if python.exists() or python.is_symlink():
        raise FileExistsError("CPU Luna virtual environment already exists; refusing to modify it")
    _run([str(bootstrap), "-m", "venv", "--copies", str(root)])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LunaRuntimeError("fresh CPU Luna virtual environment has no executable Python")
    if python.is_symlink() or os.path.samefile(python, bootstrap):
        raise LunaRuntimeError("fresh CPU Luna Python must be a distinct copied executable")
    _run([str(python), "-m", "pip", "--version"])

    requirements = _requirements_path(root)
    pip = [str(python), "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:"]
    dry_report = root / "pip-dry-run-report.json"
    _run([*pip, "--dry-run", "--report", str(dry_report), "-r", str(requirements)])
    _assert_no_gpu_distributions(_report_distribution_names(dry_report))

    install_report = root / "pip-install-report.json"
    _run([*pip, "--report", str(install_report), "-r", str(requirements)])
    _assert_no_gpu_distributions(_report_distribution_names(install_report))

    # mcq-bias is pure Python at the immutable public commit.  Its broad
    # dataset extras are not needed by the postprocess, comparison, or Luna
    # scorer import paths audited here; do not pull them (or unrelated data
    # stacks) into this dedicated CPU grading runtime.
    mcq_report = root / "pip-mcq-bias-report.json"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--report",
            str(mcq_report),
            MCQ_BIAS_SPEC,
        ]
    )
    _assert_no_gpu_distributions(_report_distribution_names(mcq_report))

    # This must be a child of the new venv. Calling importlib.metadata in this
    # bootstrap process would instead inspect the read-only GPU runtime that
    # supplied only the Python executable for ``venv --copies``.
    installed_verification = _run_installed_verification(root, source)
    installed = installed_verification["package_pins"]
    distribution_names = installed_verification["installed_distribution_names"]
    openai_mcp_module = installed_verification["openai_mcp_tool_call_error_module"]

    attestation = _run_offline_attestation(root, source)
    tests = _run_selected_tests(root, source)
    manifest = {
        "schema": RUNTIME_SCHEMA,
        "runtime_root": str(root),
        "base_python": str(bootstrap),
        "bootstrap_sha256": _sha256(root / Path(__file__).name),
        "python": str(python),
        "python_is_symlink": False,
        "source_root": str(source),
        "source_sha256": _source_hashes(source),
        "requirements": {"path": str(requirements), "sha256": _sha256(requirements)},
        "package_pins": installed,
        "openai_mcp_tool_call_error_module": str(openai_mcp_module),
        "mcq_bias_direct_url": installed_verification["mcq_bias_direct_url"],
        "gpu_distributions": [],
        "installed_distribution_names": distribution_names,
        "pip_reports": {
            "dry_run": {"path": str(dry_report), "sha256": _sha256(dry_report)},
            "install": {"path": str(install_report), "sha256": _sha256(install_report)},
            "mcq_bias": {"path": str(mcq_report), "sha256": _sha256(mcq_report)},
        },
        "offline_attestation": attestation,
        "selected_tests": tests,
        "network_policy": "clean-environment socket-blocked offline attestation; no grader.generate call",
    }
    _write_json_once(root / "runtime-manifest.json", manifest)
    return manifest


def installed_verification_child(*, runtime_root: str | Path, output: str | Path) -> dict[str, Any]:
    """Prove the packages visible to the newly created Python interpreter."""

    root = _safe_runtime_root(runtime_root)
    output_path = _normalise(output)
    if output_path != root / "installed-packages.json":
        raise LunaRuntimeError("installed-package evidence escaped the isolated runtime root")
    if _normalise(sys.prefix) != root or _normalise(sys.base_prefix) == root:
        raise LunaRuntimeError(
            f"fresh package verification is not executing in its isolated venv: prefix={sys.prefix!r}, base={sys.base_prefix!r}"
        )
    installed = _installed_versions()
    if installed != EXPECTED_PACKAGES:
        raise LunaRuntimeError(f"CPU Luna package pins differ: got={installed!r}, expected={EXPECTED_PACKAGES!r}")
    distribution_names = _assert_no_gpu_distributions(
        distribution.metadata["Name"]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    openai_distribution = importlib.metadata.distribution("openai")
    openai_mcp_module = openai_distribution.locate_file("openai/types/responses/mcp_tool_call_error.py")
    if not openai_mcp_module.is_file():
        raise LunaRuntimeError("pinned OpenAI SDK lacks mcp_tool_call_error required by Inspect 0.3.260")
    mcq_distribution = importlib.metadata.distribution("mcq-bias")
    try:
        mcq_direct_url = json.loads(mcq_distribution.read_text("direct_url.json") or "")
    except json.JSONDecodeError as exc:
        raise LunaRuntimeError("mcq-bias installation has no readable immutable direct_url metadata") from exc
    if not isinstance(mcq_direct_url, Mapping):
        raise LunaRuntimeError("mcq-bias direct_url metadata is not an object")
    mcq_vcs_info = mcq_direct_url.get("vcs_info")
    if (
        mcq_direct_url.get("url") != MCQ_BIAS_URL
        or not isinstance(mcq_vcs_info, Mapping)
        or mcq_vcs_info.get("vcs") != "git"
        or mcq_vcs_info.get("commit_id") != MCQ_BIAS_COMMIT
    ):
        raise LunaRuntimeError(f"mcq-bias direct_url does not bind the expected immutable commit: {mcq_direct_url!r}")
    result = {
        "schema": RUNTIME_SCHEMA,
        "package_pins": installed,
        "installed_distribution_names": distribution_names,
        "openai_mcp_tool_call_error_module": str(openai_mcp_module),
        "mcq_bias_direct_url": dict(mcq_direct_url),
        "gpu_distributions": [],
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "sys_executable": sys.executable,
    }
    _write_json_once(output_path, result)
    return result


def offline_attestation_child(*, runtime_root: str | Path, source_root: str | Path, output: str | Path) -> dict[str, Any]:
    """Perform the only provider construction, with all socket connects blocked."""

    root = _safe_runtime_root(runtime_root)
    source = _source_root(source_root)
    output_path = _normalise(output)
    if output_path != root / "offline-attestation.json":
        raise LunaRuntimeError("offline attestation output escaped the isolated runtime root")

    import socket

    class OfflineSocket(socket.socket):
        def connect(self, *args: Any, **kwargs: Any) -> None:
            raise LunaRuntimeError("network use is prohibited during offline Luna attestation")

        def connect_ex(self, *args: Any, **kwargs: Any) -> int:
            raise LunaRuntimeError("network use is prohibited during offline Luna attestation")

    def deny_connection(*args: Any, **kwargs: Any) -> None:
        raise LunaRuntimeError("network use is prohibited during offline Luna attestation")

    socket.socket = OfflineSocket
    socket.create_connection = deny_connection

    # Imports and construction happen only after the socket guard is live.
    from inspect_ai.model import GenerateConfig, get_model
    from ctm_data.adapters.mcq_bias import luna_scorer_no_cap as scorer
    from experiments.gemma4_12b_base_eval import combined_comparison, postprocess

    config = GenerateConfig(reasoning_effort="low", max_connections=500)
    model = get_model(scorer.GRADER_MODEL, config=config, api_key="offline-dummy-key")
    if getattr(getattr(model, "api", None), "api_key", None) != "offline-dummy-key":
        raise LunaRuntimeError("offline attestation did not retain its explicit dummy key")
    policy = scorer._provider_attestation(model, config)
    completion_parameters = policy.get("completion_parameters")
    expected_parameters = {
        "model": "openai/gpt-5.6-luna-20260709",
        "extra_body": {"reasoning": {"effort": "low"}},
    }
    if completion_parameters != expected_parameters:
        raise LunaRuntimeError(f"offline attestation parameters differ: {completion_parameters!r}")
    if (
        policy.get("grader_model") != scorer.GRADER_MODEL
        or policy.get("reasoning_effort") != "low"
        or policy.get("max_connections") != 500
        or policy.get("generate_config_output_token_cap") is not None
        or policy.get("provider_config_default_output_token_cap") is not None
        or policy.get("provider_default_output_token_cap") is not None
    ):
        raise LunaRuntimeError("offline Luna attestation did not prove the required uncapped policy")
    scorer.assert_no_output_token_cap(completion_parameters, label="offline OpenRouter completion parameters")
    result = {
        "schema": RUNTIME_SCHEMA,
        "network_guard": "socket connect and create_connection blocked before imports",
        "grader_model": scorer.GRADER_MODEL,
        "reasoning_effort": "low",
        "max_connections": 500,
        "generate_config_output_token_cap": None,
        "provider_config_default_output_token_cap": None,
        "provider_default_output_token_cap": None,
        "completion_parameters": completion_parameters,
        "output_termination": policy.get("output_termination"),
        "postprocess_import": postprocess.__name__,
        "comparison_import": combined_comparison.__name__,
        "grader_generate_called": False,
    }
    _write_json_once(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    prepare_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    prepare_parser.add_argument("--base-python", type=Path, default=DEFAULT_BASE_PYTHON)
    child = commands.add_parser("offline-attestation-child")
    child.add_argument("--runtime-root", type=Path, required=True)
    child.add_argument("--source-root", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("installed-verification-child")
    verify.add_argument("--runtime-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(runtime_root=args.runtime_root, source_root=args.source_root, base_python=args.base_python)
    elif args.command == "installed-verification-child":
        result = installed_verification_child(runtime_root=args.runtime_root, output=args.output)
    else:
        result = offline_attestation_child(runtime_root=args.runtime_root, source_root=args.source_root, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
