"""Mechanical dependency-direction checks for the CTM library boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
CTM = ROOT / "ctm"
GENERIC_SCRIPTS = tuple(
    ROOT / "scripts" / name for name in ("train_bct.py", "train_rlct.py", "run_evals.py", "run_experiment.py")
)
FORBIDDEN_ROOTS = {"ctm_data", "datasets", "mcq_bias"}
FORBIDDEN_BENCHMARK_TERMS = {"evalawarebench", "mcq_bias", "wildjailbreak"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_ctm_never_imports_dataset_packages_or_adapters():
    violations = []
    for path in CTM.rglob("*.py"):
        if "_archive" in path.parts:
            continue
        for imported in _imports(path):
            if imported.split(".", 1)[0] in FORBIDDEN_ROOTS:
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert violations == []


def test_ctm_source_does_not_name_concrete_benchmarks():
    violations = []
    for path in CTM.rglob("*.py"):
        if "_archive" in path.parts:
            continue
        source = path.read_text().lower()
        found = sorted(term for term in FORBIDDEN_BENCHMARK_TERMS if term in source)
        if found:
            violations.append(f"{path.relative_to(ROOT)} names {found}")
    assert violations == []


def test_generic_runner_scripts_do_not_import_or_name_benchmarks():
    violations = []
    for path in GENERIC_SCRIPTS:
        for imported in _imports(path):
            if imported.split(".", 1)[0] in FORBIDDEN_ROOTS:
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
        source = path.read_text().lower()
        found = sorted(term for term in FORBIDDEN_BENCHMARK_TERMS if term in source)
        if found:
            violations.append(f"{path.relative_to(ROOT)} names {found}")
    assert violations == []


def test_concrete_adapters_live_under_ctm_data_not_ctm():
    for name in ("sycophancy", "jailbreak", "eval_awareness"):
        legacy = CTM / "settings" / name
        assert not legacy.exists() or not list(legacy.glob("*.py"))
    assert (ROOT / "ctm_data" / "adapters" / "mcq_bias").is_dir()
    assert (ROOT / "ctm_data" / "adapters" / "wildjailbreak").is_dir()
    assert (ROOT / "ctm_data" / "adapters" / "eval_awareness").is_dir()


def test_paper_experiment_factories_live_under_scripts():
    assert not list((ROOT / "ctm_data").rglob("experiment_factory.py"))
    assert (ROOT / "scripts" / "rmct_paper_vast_more_methods" / "experiment_factory.py").is_file()


def test_ctm_data_contains_no_paper_suite_modules():
    markers = ("irpan_2510_27062", "2510.27062")
    violations = []
    for path in (ROOT / "ctm_data").rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = str(path.relative_to(ROOT)).lower()
        content = path.read_text(encoding="utf-8").lower()
        if any(marker in relative or marker in content for marker in markers):
            violations.append(relative)
    assert violations == []

    paper_suite = ROOT / "scripts" / "irpan_2510_27062"
    assert paper_suite.is_dir()
    assert not any("ctm_data.adapters.irpan_2510_27062" in _imports(path) for path in paper_suite.rglob("*.py"))


def test_ctm_data_source_layer_is_policy_free_transport_only():
    sources = ROOT / "ctm_data" / "sources"
    assert {path.name for path in sources.glob("*.py")} == {
        "__init__.py",
        "base.py",
        "huggingface.py",
        "local.py",
    }
    forbidden = ("paper", "reproduction", "experiment_factory", "expected_text_mc_count")
    violations = []
    for path in sources.glob("*.py"):
        content = path.read_text(encoding="utf-8").lower()
        found = sorted(term for term in forbidden if term in content)
        if found:
            violations.append(f"{path.relative_to(ROOT)} names {found}")
    assert violations == []


def test_rl_trainer_does_not_import_a_concrete_backend():
    assert "ctm.backends.tinker" not in _imports(CTM / "training" / "rl.py")
