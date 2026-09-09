"""Repository-owned wrapper around pinned ``mcq_bias.tasks``."""

from __future__ import annotations

from typing import Any

from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser
from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat


def suite_tasks(**kwargs: Any):
    """Build upstream tasks after enabling full HLE option-label parsing."""

    install_extended_answer_parser()
    install_conditional_nan_compat()
    from mcq_bias.tasks import suite_tasks as upstream_suite_tasks

    return upstream_suite_tasks(**kwargs)


__all__ = ["suite_tasks"]
