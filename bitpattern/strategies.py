"""A hypothesis strategy that draws from IntSets and BDDSets."""

from __future__ import annotations

from typing import TypeVar, overload

from hypothesis import strategies as st

from .codecs import BDDSet
from .intset import IntSet

__all__ = ["from_set"]

T = TypeVar("T")


@overload
def from_set(source: BDDSet[T]) -> st.SearchStrategy[T]: ...
@overload
def from_set(source: IntSet) -> st.SearchStrategy[int]: ...
def from_set(source: BDDSet[T] | IntSet) -> st.SearchStrategy[T | int]:
    """Members of `source`. Use over `st.sampled_from`, which enumerates the set."""
    if not source:
        return st.nothing()
    return st.integers(0, source.size - 1).map(source.__getitem__)
