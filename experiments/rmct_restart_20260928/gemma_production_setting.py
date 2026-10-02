"""Fresh Gemma RMCT data adapters, explicitly binding the integrated reviewed parser.

``OneBiasGemmaSetting`` is the user-approved (2026-10-02) one-bias campaign: the
shared frozen 7,680-QID pool and hash-assigned single cue per QID from
``ctm_data.adapters.mcq_bias.shared_qid_one_bias``, four distinct QIDs per update,
one finite pass (no cycling). Each datum carries only the clean prompt and its
assigned cue; there is no absent second arm.
"""
from functools import partial

from ctm_data.adapters.mcq_bias.shared_qid_one_bias import SETTING_QIDS_PER_UPDATE, SharedQidOneBiasSetting
from experiments.gemma4_rmct.setting import ContinuingSharedQidSetting

QIDS_PER_UPDATE = SETTING_QIDS_PER_UPDATE


def _parser():
    # Do not rely on an upstream parser alias or a process-global patch.
    from mcq_bias.parsers import BREAK_WORDS
    from ctm_data.adapters.mcq_bias.terminal_answer import parse_terminal_first
    return partial(parse_terminal_first, allowed='ABCD', break_words=BREAK_WORDS)


class FreshGemmaSetting(ContinuingSharedQidSetting):
    def answer_parser(self):
        return _parser()


class OneBiasGemmaSetting(SharedQidOneBiasSetting):
    """Gemma RMCT binding of the shared one-bias setting (same reviewed parser)."""


def create_setting(**kwargs):
    if kwargs.get('answer_parser_fn') is not None:
        raise ValueError('Fresh production parser cannot be overridden')
    return FreshGemmaSetting(**kwargs)


def create_one_bias_setting(**kwargs):
    return OneBiasGemmaSetting(**kwargs)
