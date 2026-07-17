"""Fixed-K jailbreak prompt-family setting."""

from ctm_data.adapters.wildjailbreak.setting import JailbreakSetting


def create_setting(**kwargs) -> JailbreakSetting:
    return JailbreakSetting(**kwargs)


__all__ = ["JailbreakSetting", "create_setting"]
