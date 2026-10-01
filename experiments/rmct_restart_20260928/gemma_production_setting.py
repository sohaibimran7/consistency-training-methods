"""Fresh Gemma data adapter, explicitly binding the integrated reviewed parser."""
from functools import partial
from experiments.gemma4_rmct.setting import ContinuingSharedQidSetting


class FreshGemmaSetting(ContinuingSharedQidSetting):
    def answer_parser(self):
        # Do not rely on an upstream parser alias or a process-global patch.
        from mcq_bias.parsers import BREAK_WORDS
        from ctm_data.adapters.mcq_bias.terminal_answer import parse_terminal_first
        return partial(parse_terminal_first, allowed='ABCD', break_words=BREAK_WORDS)


def create_setting(**kwargs):
    if kwargs.get('answer_parser_fn') is not None:
        raise ValueError('Fresh production parser cannot be overridden')
    return FreshGemmaSetting(**kwargs)
