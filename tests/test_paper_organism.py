from experiments.paper_organism.build_report import aggregate


def row(uid, value, block):
    return {'uid': uid, 'behavior': value, 'repeat_block': block}


def test_invalid_not_zero_and_se_unavailable_for_one_unit():
    result = aggregate([row('a', 1, 's1'), row('b', None, 's2')], 'behavior', True)
    assert result['mean'] == 1
    assert result['valid_n'] == 1 and result['missing_n'] == 1
    assert result['stderr'] is None


def test_no_valid_scores_remains_undefined():
    result = aggregate([row('a', None, 's1')], 'behavior')
    assert result['mean'] is None and result['stderr'] is None


def test_pooled_repeat_unit_not_condition_count():
    result = aggregate([row('a', 1, 's1'), row('b', 0, 's1')], 'behavior', True)
    assert result['mean'] == .5 and result['repeat_blocks'] == 1
    assert result['stderr'] is None
