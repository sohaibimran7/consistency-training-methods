#!/usr/bin/env python3
"""Check the committed Linux CPU validation-environment contract.

The lock is intentionally a requirements-format lock because ``requirements.txt``
remains the repository's direct-dependency input.  This script first checks that
the generated lock still represents that input; without ``--lock-only`` it also
checks an installed Linux environment against the lock.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "environments" / "cpu.lock"

PYTHON_VERSION = "3.11.12"
PYTHON_PLATFORM = "x86_64-unknown-linux-gnu"
MCQ_BIAS_COMMIT = "8c0b46b17cbf438958b064554a1375594d6ad4b2"
MCQ_BIAS_REPOSITORY = "https://github.com/sohaibimran7/mcq-bias"
MCQ_BIAS_REQUIREMENT = f"mcq-bias @ git+{MCQ_BIAS_REPOSITORY}@{MCQ_BIAS_COMMIT}"

NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
PINNED_REQUIREMENT = re.compile(rf"^(?P<name>{NAME})==(?P<version>[^\s]+)$")
VCS_REQUIREMENT = re.compile(rf"^(?P<name>{NAME}) @ (?P<source>git\+https://\S+)$")
DIRECT_REQUIREMENT = re.compile(rf"^(?P<name>{NAME})(?:\s|[<>=!~@]|$)")
SHA256_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")


class EnvironmentContractError(RuntimeError):
    """Raised when the committed environment does not match its contract."""


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str | None
    raw: str
    block: str


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _direct_requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = DIRECT_REQUIREMENT.match(line)
        if match is None:
            raise EnvironmentContractError(f"requirements.txt:{line_number} is not a supported requirement: {line!r}")
        names.add(_normalise(match.group("name")))
    return names


def _lock_entries(text: str) -> tuple[dict[str, LockedRequirement], list[str]]:
    entries: dict[str, LockedRequirement] = {}
    errors: list[str] = []
    current_name: str | None = None
    current_version: str | None = None
    current_raw: str | None = None
    current_lines: list[str] = []

    def finish_current() -> None:
        nonlocal current_name, current_version, current_raw, current_lines
        if current_name is None or current_raw is None:
            return
        normalised = _normalise(current_name)
        if normalised in entries:
            errors.append(f"lock repeats requirement {current_name!r}")
        else:
            entries[normalised] = LockedRequirement(
                name=current_name,
                version=current_version,
                raw=current_raw,
                block="\n".join(current_lines),
            )
        current_name = None
        current_version = None
        current_raw = None
        current_lines = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        candidate = line[:-1].rstrip() if line.endswith("\\") else line
        pinned = PINNED_REQUIREMENT.match(candidate)
        vcs = VCS_REQUIREMENT.match(candidate)
        if pinned or vcs:
            finish_current()
            current_name = (pinned or vcs).group("name")
            current_version = pinned.group("version") if pinned else None
            current_raw = candidate
            current_lines = [raw]
            continue

        stripped = raw.strip()
        if stripped and not stripped.startswith(("#", "--")) and not raw[0].isspace():
            errors.append(f"lock:{line_number} is not a pinned requirement: {raw!r}")
        if current_name is not None:
            current_lines.append(raw)

    finish_current()
    return entries, errors


def _raise_if_invalid(errors: list[str]) -> None:
    if errors:
        raise EnvironmentContractError("\n".join(f"- {error}" for error in errors))


def check_lock_contract(root: Path = ROOT) -> dict[str, LockedRequirement]:
    """Ensure the committed lock is a complete CPU profile of requirements.txt."""

    requirements = root / REQUIREMENTS.relative_to(ROOT)
    pyproject = root / PYPROJECT.relative_to(ROOT)
    lock = root / LOCK.relative_to(ROOT)
    errors: list[str] = []
    for path in (requirements, pyproject, lock):
        if not path.is_file():
            errors.append(f"required file is missing: {path.relative_to(root)}")
    _raise_if_invalid(errors)

    direct_names = _direct_requirement_names(requirements.read_text(encoding="utf-8"))
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    if project.get("dependencies") != []:
        errors.append("pyproject.toml must keep an empty project.dependencies list; requirements.txt is the direct source")

    lock_text = lock.read_text(encoding="utf-8")
    for fragment in (
        "# This file was autogenerated by uv via the following command:",
        "uv pip compile requirements.txt",
        "--generate-hashes",
        f"--python-version {PYTHON_VERSION}",
        f"--python-platform {PYTHON_PLATFORM}",
        "--torch-backend cpu",
        "--index-url https://pypi.org/simple",
    ):
        if fragment not in lock_text:
            errors.append(f"lock is missing required generation detail: {fragment!r}")

    entries, parse_errors = _lock_entries(lock_text)
    errors.extend(parse_errors)
    for entry in entries.values():
        if entry.version is not None and SHA256_HASH.search(entry.block) is None:
            errors.append(f"{entry.name} is pinned without a sha256 hash")

    missing = sorted(direct_names - entries.keys())
    if missing:
        errors.append(f"lock does not cover direct requirements: {', '.join(missing)}")

    vcs_entries = [entry for entry in entries.values() if entry.version is None]
    if len(vcs_entries) != 1 or vcs_entries[0].raw != MCQ_BIAS_REQUIREMENT:
        errors.append(f"lock must contain only the exact mcq-bias Git requirement: {MCQ_BIAS_REQUIREMENT}")

    torch = entries.get("torch")
    if torch is None or torch.version is None or not torch.version.endswith("+cpu"):
        errors.append("lock must pin a CPU-only torch build ending in '+cpu'")
    nvidia_packages = sorted(entry.name for entry in entries.values() if _normalise(entry.name).startswith("nvidia-"))
    if nvidia_packages:
        errors.append(f"CPU lock must not include NVIDIA runtime packages: {', '.join(nvidia_packages)}")

    _raise_if_invalid(errors)
    return entries


def check_installed_environment(entries: dict[str, LockedRequirement]) -> None:
    """Ensure the current interpreter has the exact Linux CPU lock installed."""

    errors: list[str] = []
    if sys.version_info[:3] != tuple(int(part) for part in PYTHON_VERSION.split(".")):
        errors.append(f"expected CPython {PYTHON_VERSION}; found {platform.python_version()}")
    if platform.system() != "Linux":
        errors.append(f"expected Linux; found {platform.system()}")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        errors.append(f"expected x86_64; found {platform.machine()}")

    for entry in entries.values():
        if entry.version is None:
            continue
        try:
            installed = metadata.version(entry.name)
        except metadata.PackageNotFoundError:
            errors.append(f"missing locked package: {entry.name}")
            continue
        if installed != entry.version:
            errors.append(f"{entry.name} is {installed}, expected {entry.version}")

    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = ""
    if not torch_version.endswith("+cpu"):
        errors.append(f"installed torch must be CPU-only; found {torch_version or 'not installed'}")

    installed_names = {_normalise(distribution.metadata["Name"]) for distribution in metadata.distributions() if distribution.metadata.get("Name")}
    nvidia_packages = sorted(name for name in installed_names if name.startswith("nvidia-"))
    if nvidia_packages:
        errors.append(f"installed environment includes NVIDIA runtime packages: {', '.join(nvidia_packages)}")

    try:
        direct_url_text = metadata.distribution("mcq-bias").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "{}")
    except (json.JSONDecodeError, metadata.PackageNotFoundError):
        direct_url = {}
    vcs_info = direct_url.get("vcs_info", {}) if isinstance(direct_url, dict) else {}
    if direct_url.get("url") != MCQ_BIAS_REPOSITORY or vcs_info.get("commit_id") != MCQ_BIAS_COMMIT:
        errors.append(f"installed mcq-bias is not the pinned Git commit {MCQ_BIAS_COMMIT}")

    _raise_if_invalid(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-only",
        action="store_true",
        help="validate repository files without requiring the Linux environment to be installed",
    )
    args = parser.parse_args(argv)

    try:
        entries = check_lock_contract()
        if args.lock_only:
            print("CPU environment lock contract is valid.")
        else:
            check_installed_environment(entries)
            print("Installed Linux CPU environment matches environments/cpu.lock.")
    except EnvironmentContractError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
