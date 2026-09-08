"""Contracts for the external-output Bayesian Isambard adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from infra.isambard.job_adapters import AdapterError, build_request
from infra.isambard.job_controller import Controller
from infra.isambard.job_transport import validate_request

REMOTE_DIR = "/projects/a5v/alice/bayesian/repo"
CONFIG = REMOTE_DIR + "/configs/evaluation.toml"
CONFIG_SHA256 = "a" * 64
PYTHON = "/projects/a5v/alice/runtimes/bayesian/bin/python"
RUNTIME_ROOT = "/projects/a5v/alice/runtimes/bayesian"
# This deliberately shares the ``repo`` text prefix without being a child of
# the checkout. A path-prefix check must not mistake it for a nested output.
EXTERNAL_OUTPUT = "/projects/a5v/alice/bayesian/repo-results/attempt-001"


@pytest.fixture
def maintained_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal maintained wrapper with the deployed BR namespace."""

    checkout = tmp_path / "checkout"
    wrapper = checkout / "infra" / "isambard" / "run_batch.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        """#!/usr/bin/env bash
#SBATCH --job-name=bayesian-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail

: "${BR_REPO_DIR:?BR_REPO_DIR is required}"
: "${BR_OUTPUT_ROOT:?BR_OUTPUT_ROOT is required}"
: "${BR_CONFIG:?BR_CONFIG is required}"
: "${BR_CONFIG_SHA256:?BR_CONFIG_SHA256 is required}"
: "${BR_PYTHON:?BR_PYTHON is required}"
exec "$BR_PYTHON" "$BR_REPO_DIR/run.py" \\
    --config "$BR_CONFIG" --config-sha256 "$BR_CONFIG_SHA256" --output-root "$BR_OUTPUT_ROOT"
""",
        encoding="utf-8",
    )
    return checkout, wrapper


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "BR_CONFIG": CONFIG,
        "BR_CONFIG_SHA256": CONFIG_SHA256,
        "BR_PYTHON": PYTHON,
    }
    values.update(overrides)
    return values


def _request(
    checkout: Path,
    *,
    output_root: str = EXTERNAL_OUTPUT,
    mode: str = "batch",
    minutes: int = 60,
    env: dict[str, str] | None = None,
) -> dict:
    return build_request(
        "bayesian_eval",
        request_id="bayesian-request-001",
        owner="bayesian-agent",
        checkout=checkout,
        remote_dir=REMOTE_DIR,
        output_root=output_root,
        mode=mode,
        minutes=minutes,
        env=_env() if env is None else env,
    )


def test_bayesian_eval_captures_the_wrapper_and_claims_an_external_output(
    maintained_checkout: tuple[Path, Path], tmp_path: Path
):
    checkout, wrapper = maintained_checkout
    original = wrapper.read_text(encoding="utf-8")

    request = _request(checkout)

    assert wrapper.read_text(encoding="utf-8") == original
    assert request["id"] == "bayesian-request-001"
    assert request["owner"] == "bayesian-agent"
    assert request["mode"] == "batch"
    assert request["remote_dir"] == REMOTE_DIR
    assert request["output_roots"] == [EXTERNAL_OUTPUT]
    assert request["resources"] == {
        "nodes": 1,
        "gpus": 1,
        "minutes": 60,
        "memory_mb": 96 * 1024,
        "cpus_per_gpu": 16,
    }
    assert request["env"] == {
        "BR_CONFIG": CONFIG,
        "BR_CONFIG_SHA256": CONFIG_SHA256,
        "BR_PYTHON": PYTHON,
        "BR_REPO_DIR": REMOTE_DIR,
        "BR_OUTPUT_ROOT": EXTERNAL_OUTPUT,
    }
    assert request["script"].startswith("#!/usr/bin/env bash\n")
    assert "#SBATCH" not in request["script"]
    assert "BR_CONFIG_SHA256" in request["script"]

    assert validate_request(request) == request
    controller = Controller(tmp_path / "state", object(), "isambard.example:22/alice")
    result = controller.enqueue(request)
    assert result["actions"] == [{"action": "enqueued", "request_id": "bayesian-request-001"}]
    assert controller.status()["requests"][0]["output_roots"] == [EXTERNAL_OUTPUT]


@pytest.mark.parametrize("minutes", [1, 1440])
def test_bayesian_eval_allows_the_full_positive_batch_duration_range(
    maintained_checkout: tuple[Path, Path], minutes: int
):
    request = _request(maintained_checkout[0], minutes=minutes)
    assert request["resources"]["minutes"] == minutes


@pytest.mark.parametrize("minutes", [0, True, 1441])
def test_bayesian_eval_rejects_out_of_range_or_boolean_duration(maintained_checkout: tuple[Path, Path], minutes: int):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], minutes=minutes)


@pytest.mark.parametrize("mode", ["interactive", "other"])
def test_bayesian_eval_is_batch_only(maintained_checkout: tuple[Path, Path], mode: str):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], mode=mode)


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,
        "a" * 63,
        "g" * 64,
    ],
)
def test_bayesian_eval_requires_a_64_character_lowercase_hex_config_digest(
    maintained_checkout: tuple[Path, Path], digest: str
):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], env=_env(BR_CONFIG_SHA256=digest))


@pytest.mark.parametrize(
    "config",
    [
        "/projects/a5v/alice/bayesian/other/config.toml",
        REMOTE_DIR + "-other/config.toml",
        REMOTE_DIR + "/configs/../outside.toml",
    ],
)
def test_bayesian_eval_rejects_configs_outside_or_not_normalised_under_the_checkout(
    maintained_checkout: tuple[Path, Path], config: str
):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], env=_env(BR_CONFIG=config))


@pytest.mark.parametrize(
    "output_root",
    [
        REMOTE_DIR,
        REMOTE_DIR + "/outputs/attempt",
        "/projects/a5v/alice/bayesian",
        RUNTIME_ROOT,
        RUNTIME_ROOT + "/outputs/attempt",
        "/projects/a5v/alice",
    ],
)
def test_bayesian_eval_rejects_outputs_overlapping_the_checkout_or_runtime_tree(
    maintained_checkout: tuple[Path, Path], output_root: str
):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], output_root=output_root)


def test_bayesian_eval_uses_python_parent_as_runtime_tree_when_it_is_not_bin(maintained_checkout: tuple[Path, Path]):
    with pytest.raises(AdapterError):
        _request(
            maintained_checkout[0],
            output_root="/opt/bayesian-runtime/outputs/attempt",
            env=_env(BR_PYTHON="/opt/bayesian-runtime/python"),
        )


@pytest.mark.parametrize(
    "env",
    [
        {"BR_CONFIG_SHA256": CONFIG_SHA256, "BR_PYTHON": PYTHON},
        {"BR_CONFIG": CONFIG, "BR_PYTHON": PYTHON},
        {"BR_CONFIG": CONFIG, "BR_CONFIG_SHA256": CONFIG_SHA256},
        _env(BR_UNUSED="unexpected"),
        _env(BR_REPO_DIR=REMOTE_DIR),
        _env(BR_OUTPUT_ROOT=EXTERNAL_OUTPUT),
    ],
)
def test_bayesian_eval_allows_only_its_three_supplied_configuration_values(
    maintained_checkout: tuple[Path, Path], env: dict[str, str]
):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], env=env)


@pytest.mark.parametrize(
    "python",
    [
        "relative/python",
        "/projects/a5v/alice/runtimes/bayesian/bin/../bin/python",
    ],
)
def test_bayesian_eval_requires_a_normalised_absolute_interpreter_path(
    maintained_checkout: tuple[Path, Path], python: str
):
    with pytest.raises(AdapterError):
        _request(maintained_checkout[0], env=_env(BR_PYTHON=python))


def test_bayesian_eval_fails_closed_when_its_wrapper_is_missing(tmp_path: Path):
    checkout = tmp_path / "missing-wrapper"
    checkout.mkdir()
    with pytest.raises(AdapterError):
        _request(checkout)


def test_bayesian_eval_fails_closed_when_its_wrapper_is_a_symlink(tmp_path: Path):
    checkout = tmp_path / "checkout"
    wrapper = checkout / "infra" / "isambard" / "run_batch.sh"
    wrapper.parent.mkdir(parents=True)
    target = tmp_path / "wrapper-target.sh"
    target.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    wrapper.symlink_to(target)

    with pytest.raises(AdapterError):
        _request(checkout)
