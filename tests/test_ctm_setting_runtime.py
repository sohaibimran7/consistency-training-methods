"""Tests for the generic Setting-to-RL hand-off."""

import pytest

from ctm.settings import Setting, create_setting
from ctm.settings.runtime import prepare_setting_instance, setting_run_metadata


class _Setting:
    name = "unit"

    def __init__(self, *, control=False):
        self.control = control

    def load_datapoints(self, count=1, n_datapoints=None):
        if n_datapoints is not None:
            count = n_datapoints
        return [{"id": index} for index in range(count)]

    def perturbations(self):
        def prompt(row):
            return {"messages": [{"role": "user", "content": str(row["id"])}]}

        return [prompt, prompt]

    def training_perturbation_indices(self):
        return [1]

    def trait_classifier(self):
        return lambda response, row, realized_messages: 0.0

    def answer_parser(self):
        return None

    def run_metadata(self):
        return {"control": self.control}


def create_unit_setting(**kwargs):
    return _Setting(**kwargs)


def test_setting_factory_is_an_explicit_import_path():
    setting = create_setting("tests.test_ctm_setting_runtime:create_unit_setting", control=True)
    assert isinstance(setting, Setting)
    assert setting.control is True
    with pytest.raises(ValueError, match="module:callable"):
        create_setting("unit")


def test_prepare_setting_instance_collects_runtime_components():
    prepared = prepare_setting_instance(_Setting(), load_config={"count": 2})
    assert len(prepared.datapoints) == 2
    assert len(prepared.perturbations) == 2
    assert prepared.training_indices == [1]
    assert prepared.answer_parser is None


def test_prepare_setting_instance_validates_indices():
    setting = _Setting()
    setting.training_perturbation_indices = lambda: [0, 3]
    with pytest.raises(ValueError, match="invalid training indices"):
        prepare_setting_instance(setting)


def test_prepare_setting_instance_rejects_empty_data():
    with pytest.raises(ValueError, match="loaded no datapoints"):
        prepare_setting_instance(_Setting(), load_config={"count": 0})


def test_prepare_setting_instance_preflights_prompt_schema_without_mutating_data():
    setting = _Setting()
    rows = [{"id": 0}]
    setting.load_datapoints = lambda **_: rows

    def malformed(row):
        row["mutated"] = True
        return {"messages": [{"role": "user"}]}

    setting.perturbations = lambda: [malformed, malformed]
    with pytest.raises(TypeError, match="string role/content"):
        prepare_setting_instance(setting)
    assert rows == [{"id": 0}]


def test_prepare_setting_instance_rejects_blank_prompt_content():
    setting = _Setting()
    blank = lambda _row: {"messages": [{"role": "user", "content": "   "}]}  # noqa: E731
    setting.perturbations = lambda: [blank, blank]
    setting.training_perturbation_indices = lambda: [1]
    with pytest.raises(TypeError, match="non-empty string role/content"):
        prepare_setting_instance(setting)


def test_setting_run_metadata_is_serializable(tmp_path):
    setting = _Setting()
    setting.run_metadata = lambda: {"data_path": tmp_path / "train.jsonl"}
    metadata = setting_run_metadata(
        setting,
        setting_config={"path": tmp_path},
        load_config={"count": 2},
    )
    assert metadata["setting"] == "unit"
    assert metadata["setting_metadata"]["data_path"].endswith("train.jsonl")
    assert metadata["setting_config"]["path"] == str(tmp_path)
    assert metadata["trait_classifier_identity"]["callable"].endswith("._Setting.trait_classifier.<locals>.<lambda>")


def test_training_cli_rejects_zero_batch_before_loading_data():
    from scripts.train_rlct import main

    with pytest.raises(SystemExit):
        main(
            [
                "--experiment-name",
                "unit",
                "--run-name",
                "unit",
                "--setting-factory",
                "tests.test_ctm_setting_runtime:create_unit_setting",
                "--batch-size",
                "0",
                "--dry-run",
            ]
        )


def test_training_cli_preserves_control_from_setting_config(monkeypatch):
    from scripts.train_rlct import main

    captured = {}

    def construct(factory_spec, **kwargs):
        captured["factory_spec"] = factory_spec
        captured["kwargs"] = kwargs
        return _Setting(**kwargs)

    monkeypatch.setattr("ctm.settings.runtime.create_setting", construct)
    main(
        [
            "--experiment-name",
            "unit",
            "--run-name",
            "unit",
            "--setting-factory",
            "tests.test_ctm_setting_runtime:create_unit_setting",
            "--setting-config",
            '{"control":true}',
            "--n-datapoints",
            "1",
            "--dry-run",
        ]
    )
    assert captured == {
        "factory_spec": "tests.test_ctm_setting_runtime:create_unit_setting",
        "kwargs": {"control": True},
    }


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--anchor-weight", "0", "--n-consistency-rollouts", "0"],
        ["--anchor-weight", "1", "--n-anchor-rollouts", "0"],
    ],
)
def test_training_cli_rejects_no_active_gradient_term(extra_args):
    from scripts.train_rlct import main

    with pytest.raises(SystemExit):
        main(
            [
                "--experiment-name",
                "unit",
                "--run-name",
                "unit",
                "--setting-factory",
                "tests.test_ctm_setting_runtime:create_unit_setting",
                "--dry-run",
                *extra_args,
            ]
        )
