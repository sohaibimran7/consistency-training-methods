"""Expand sparse packed LoRA groups without guessing missing-group boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def expand_sparse_packed_lora(
    output_sizes: Sequence[int], lora_a: Sequence[Any], lora_b: Sequence[Any]
) -> tuple[list[Any], list[Any]]:
    """Preserve absent adapters as None, with a unique complete slice partition.

    Present B row counts determine their contiguous output slices. An absent
    group has no row count, so infer its span only if the remaining present
    groups and the total slice count admit exactly one layout. This supports
    QKV + absent Z and absent QKV + Z, but rejects ambiguous sparse layouts.
    The caller's reset_lora/set_lora already implement None as a zero update.
    """
    sizes = tuple(output_sizes)
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in sizes):
        raise ValueError("packed output sizes must be positive integers")
    if not lora_a or len(lora_a) != len(lora_b) or len(lora_a) > len(sizes):
        raise ValueError("packed LoRA A/B group counts must agree and fit the slices")
    for a, b in zip(lora_a, lora_b, strict=True):
        if (a is None) != (b is None):
            raise ValueError("an absent LoRA group must have both A and B absent")
        if b is not None and (len(a.shape) != 2 or len(b.shape) != 2 or a.shape[0] != b.shape[1]):
            raise ValueError("packed LoRA tensors need matching two-dimensional ranks")

    layouts: list[tuple[int, ...]] = []

    def visit(group: int, start: int, spans: tuple[int, ...]) -> None:
        if len(layouts) > 1:
            return
        if group == len(lora_b):
            if start == len(sizes):
                layouts.append(spans)
            return
        remaining_groups = len(lora_b) - group - 1
        for stop in range(start + 1, len(sizes) - remaining_groups + 1):
            b = lora_b[group]
            if b is None or sum(sizes[start:stop]) == b.shape[0]:
                visit(group + 1, stop, (*spans, stop - start))

    visit(0, 0, ())
    if len(layouts) != 1:
        raise ValueError("packed LoRA groups have no unique complete output-slice layout")
    expanded_a, expanded_b = [], []
    start = 0
    for a, b, span in zip(lora_a, lora_b, layouts[0], strict=True):
        row = 0
        for size in sizes[start:start + span]:
            expanded_a.append(a)
            expanded_b.append(None if b is None else b[row:row + size, :])
            row += size
        start += span
    return expanded_a, expanded_b
