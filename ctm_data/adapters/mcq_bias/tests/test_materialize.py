from ctm_data.adapters.mcq_bias.materialize import interleave_rows


def test_interleave_rows_keeps_global_prefix_balanced():
    assert interleave_rows([[{"id": "a1"}, {"id": "a2"}], [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}]]) == [
        {"id": "a1"},
        {"id": "b1"},
        {"id": "a2"},
        {"id": "b2"},
        {"id": "b3"},
    ]
