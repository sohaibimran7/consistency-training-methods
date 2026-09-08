"""Repository checks for the committed CPU validation environment."""

from scripts.check_environment import check_lock_contract


def test_linux_cpu_lock_tracks_the_top_level_dependency_source():
    check_lock_contract()
