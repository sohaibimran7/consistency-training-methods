"""CPU-only Luna profile contracts without reading credential files."""

from pathlib import Path

import pytest

from infra.isambard import job_adapters as adapters
from infra.isambard.job_transport import validate_request

REPO = "/project/gemma/frozen"
CAMPAIGN = REPO + "/results/gemma4-12b-base-two-bias-50x21-16gpu-v1"
ENV_PATH = "/protected/luna.env"


@pytest.fixture
def wrapper_checkout(tmp_path):
    path = tmp_path / "infra/isambard/run_gemma4_12b_base_two_bias_luna_grade.sbatch"
    path.parent.mkdir(parents=True)
    path.write_text("""#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
python postprocess.py preflight --campaign-root "$CTM_GEMMA4_EVAL_CAMPAIGN_ROOT"
. "$CTM_RMCT_LUNA_ENV_FILE"
python postprocess.py grade --output-root "$CTM_GEMMA4_LUNA_OUTPUT_ROOT"
python postprocess.py publish --output-dir "$CTM_GEMMA4_PUBLICATION_DIR"
""")
    return tmp_path


def build(checkout, **changes):
    kwargs = dict(
        request_id="luna-test",
        owner="gemma-owner",
        checkout=checkout,
        remote_dir=REPO,
        output_root=CAMPAIGN,
        mode="batch",
        minutes=360,
        env={"CTM_RMCT_LUNA_ENV_FILE": ENV_PATH},
    )
    kwargs.update(changes)
    return adapters.build_request("gemma_luna_grade_cpu", **kwargs)


def test_cpu_luna_claims_preflight_and_derived_outputs_and_captures_only_env_path(wrapper_checkout):
    path = wrapper_checkout / "infra/isambard/run_gemma4_12b_base_two_bias_luna_grade.sbatch"
    original = path.read_text()
    request = build(wrapper_checkout)
    assert request["resources"] == dict(nodes=1, gpus=0, cpus_per_task=16, ntasks=1, memory_mb=65536, minutes=360)
    assert request["output_roots"] == [CAMPAIGN]
    assert request["env"] == {
        "REPO_DIR": REPO,
        "CTM_RMCT_LUNA_ENV_FILE": ENV_PATH,
        "CTM_GEMMA4_EVAL_PARENT_ROOT": str(Path(CAMPAIGN).parent),
        "CTM_GEMMA4_EVAL_CAMPAIGN_ROOT": CAMPAIGN,
        "CTM_GEMMA4_LUNA_OUTPUT_ROOT": CAMPAIGN + "/derived-luna-no-cap-v1",
        "CTM_GEMMA4_PUBLICATION_DIR": CAMPAIGN + "/derived-luna-no-cap-v1/publication",
        "CTM_GEMMA4_LUNA_MODE": "grade-and-publish",
    }
    assert request["script"] == "".join(
        line for line in original.splitlines(keepends=True) if not line.startswith("#SBATCH")
    )
    assert path.read_text() == original
    assert validate_request(request) == request


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "interactive"},
        {"minutes": 359},
        {"minutes": 361},
        {"env": {}},
        {"env": {"CTM_RMCT_LUNA_ENV_FILE": "relative.env"}},
        {"env": {"CTM_RMCT_LUNA_ENV_FILE": "/protected/../luna.env"}},
        {"env": {"CTM_RMCT_LUNA_ENV_FILE": ENV_PATH, "CTM_GEMMA4_LUNA_MODE": "publish"}},
        {"env": {"CTM_RMCT_LUNA_ENV_FILE": ENV_PATH, "OPENROUTER_API_KEY": "forbidden-placeholder"}},
        {"output_root": REPO + "/unrelated-output"},
        {"output_root": "/external/gemma4-12b-base-two-bias-50x21-16gpu-v1"},
    ],
)
def test_luna_rejects_undeclared_modes_resources_credentials_and_output_redirects(wrapper_checkout, changes):
    with pytest.raises(adapters.AdapterError):
        build(wrapper_checkout, **changes)


def test_luna_allows_a_separate_pinned_cpu_interpreter(wrapper_checkout):
    request = build(
        wrapper_checkout,
        env={
            "CTM_RMCT_LUNA_ENV_FILE": ENV_PATH,
            "CTM_GEMMA4_LUNA_PYTHON": "/project/grading-runtime/bin/python",
        },
    )
    assert request["env"]["CTM_GEMMA4_LUNA_PYTHON"] == "/project/grading-runtime/bin/python"
    assert request["resources"]["gpus"] == 0
    assert request["output_roots"] == [CAMPAIGN]
    assert validate_request(request) == request


@pytest.mark.parametrize("interpreter", ["relative/python", "/runtime/../python", CAMPAIGN + "/runtime/bin/python"])
def test_luna_rejects_invalid_or_overlapping_runtime_override(wrapper_checkout, interpreter):
    with pytest.raises(adapters.AdapterError):
        build(wrapper_checkout, env={"CTM_RMCT_LUNA_ENV_FILE": ENV_PATH, "CTM_GEMMA4_LUNA_PYTHON": interpreter})
