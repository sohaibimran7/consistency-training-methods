"""Compatibility shim — BA grader prompts are owned by the published eval package."""

from mcq_bias.grader_prompts import (  # noqa: F401
    BIAS_ACK_PROMPTS,
    COUNTERFACTUAL_RUBRIC,
    REASON_THEN_ANSWER_SUFFIX,
    get_bias_ack_prompt,
    register_bias_ack_prompt,
)
