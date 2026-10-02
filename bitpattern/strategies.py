"""A hypothesis strategy that draws from IntSets and BDDSets.

Needs `pip install 'bitpattern[hypothesis]'`. Members are drawn by index, so nothing
gets enumerated, and index order is value order for non-negative floats, so
shrinking the index shrinks the float too.
"""

from __future__ import annotations

from typing import TypeVar, overload

from .codecs import BDDSet
from .sets import IntSet

try:
    from hypothesis import strategies as st
except ImportError as e:
    raise ImportError(
        "bitpattern.strategies needs hypothesis: pip install 'bitpattern[hypothesis]'"
    ) from e

__all__ = ["from_set"]

T = TypeVar("T")


@overload
def from_set(source: BDDSet[T]) -> st.SearchStrategy[T]: ...
@overload
def from_set(source: IntSet) -> st.SearchStrategy[int]: ...
def from_set(source: BDDSet[T] | IntSet) -> st.SearchStrategy[T | int]:
    """Members of `source`. Unlike `st.sampled_from`, this never copies the set."""
    if not source:
        return st.nothing()
    return st.integers(0, source.size - 1).map(source.__getitem__)
