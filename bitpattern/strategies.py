"""Hypothesis strategies that draw from IntSets. Needs `bitpattern[hypothesis]`.

Members are drawn by index, so nothing gets enumerated. Index order is value
order for non-negative floats, so shrinking the index shrinks the float too.
"""

from __future__ import annotations

from typing import TypeVar

from .codecs import Codec
from .sets import IntSet

try:
    from hypothesis import strategies as st
except ImportError as e:
    raise ImportError(
        "bitpattern.strategies needs hypothesis: pip install 'bitpattern[hypothesis]'"
    ) from e

__all__ = ["from_codec", "from_intset"]

T = TypeVar("T")


def from_intset(source: IntSet) -> st.SearchStrategy[int]:
    if not source:
        return st.nothing()
    return st.integers(0, source.size - 1).map(source.__getitem__)


def from_codec(codec: Codec[T], source: IntSet | None = None) -> st.SearchStrategy[T]:
    """Decoded members of `source`, or of every value the codec can encode."""
    return from_intset(codec.all if source is None else source).map(codec.decode)
