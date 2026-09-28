import pytest
import torch

from ctm.backends.local.packed_lora import expand_sparse_packed_lora


def test_qkv_present_z_absent_preserves_projection_and_zero_slot():
    a = torch.arange(6.0).reshape(2, 3)
    b = torch.arange(16.0).reshape(8, 2)
    aa, bb = expand_sparse_packed_lora([2, 2, 4, 4], [a, None], [b, None])
    assert len(aa) == len(bb) == 4
    assert all(x is a for x in aa[:3])
    assert aa[3] is bb[3] is None
    assert torch.equal(torch.cat(bb[:3]), b)
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(torch.cat([v @ (u @ x) for u, v in zip(aa[:3], bb[:3])]), b @ (a @ x))


def test_qkv_absent_z_present_has_three_leading_zero_slots():
    a, b = torch.ones(2, 3), torch.ones(4, 2)
    aa, bb = expand_sparse_packed_lora([2, 2, 4, 4], [None, a], [None, b])
    assert aa[:3] == bb[:3] == [None] * 3
    assert aa[3] is a and torch.equal(bb[3], b)


def test_dense_groups_are_unchanged():
    a1, a2 = torch.randn(2, 3), torch.randn(2, 3)
    b1, b2 = torch.randn(8, 2), torch.randn(4, 2)
    aa, bb = expand_sparse_packed_lora([2, 2, 4, 4], [a1, a2], [b1, b2])
    assert aa == [a1, a1, a1, a2]
    assert torch.equal(torch.cat(bb[:3]), b1)
    assert torch.equal(bb[3], b2)


def test_individual_missing_slice_is_preserved():
    a, b = torch.ones(2, 3), torch.ones(4, 2)
    aa, bb = expand_sparse_packed_lora([2, 4], [None, a], [None, b])
    assert aa[0] is bb[0] is None
    assert torch.equal(bb[1], b)


@pytest.mark.parametrize("sizes,aa,bb", [
    ([2, 2, 4, 4], [None, None], [None, None]),
    ([2, 2, 4, 4], [torch.ones(2, 3)], [torch.ones(8, 2)]),
    ([2, 2], [torch.ones(2, 3), None], [None, None]),
    ([2, 2], [torch.ones(2, 3)], [torch.ones(4, 3)]),
    ([2, 2], [], []),
    ([0, 2], [None], [None]),
])
def test_ambiguous_incomplete_or_malformed_groups_fail_closed(sizes, aa, bb):
    with pytest.raises(ValueError):
        expand_sparse_packed_lora(sizes, aa, bb)
