from __future__ import annotations


def test_extended_parser_preserves_existing_labels_and_adds_high_option_labels():
    import mcq_bias.parsers as parsers

    from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser

    install_extended_answer_parser()

    for label in ("A", "J", "K", "Q", "Z"):
        assert parsers.parse_answer(f"reasoning\nanswer: {label}") == label
        assert parsers.parse_answer(f"reasoning\nanswer: $\\boxed{{\\textbf{{({label})}}}}$") == label


def test_extended_parser_updates_preloaded_upstream_bindings():
    import mcq_bias.pipeline.wrong_arguments as wrong_arguments
    import mcq_bias.scorers as scorers

    from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser

    install_extended_answer_parser()

    assert wrong_arguments.parse_answer("answer: Q") == "Q"
    assert scorers.parse_answer("answer: T") == "T"
    assert wrong_arguments._acceptable("supporting case\nanswer: Q", "Q")


def test_repository_task_wrapper_installs_extended_parser(monkeypatch):
    import mcq_bias.parsers as parsers
    import mcq_bias.tasks as upstream_tasks

    from ctm_data.adapters.mcq_bias import tasks

    monkeypatch.setattr(upstream_tasks, "suite_tasks", lambda **kwargs: [kwargs])
    assert tasks.suite_tasks(dataset="unit") == [{"dataset": "unit"}]
    assert parsers.parse_answer("answer: P") == "P"
