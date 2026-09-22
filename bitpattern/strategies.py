"""Hypothesis strategies over `IntSet`s.

Needs the optional extra: `pip install bitpattern[hypothesis]`.

The whole integration is one idea: an `IntSet` knows its cardinality and can
return its n-th member in O(width), so drawing from one is just drawing an index.
Nothing is enumerated, so a set of `2 ** 63` floats costs no more to sample than a
set of three -- and because the index order of the non-negative floats is their
value order, shrinking the index shrinks the float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .sets import IntSet

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

__all__ = ["from_codec", "from_intset"]


def _strategies():
    try:
        from hypothesis import strategies
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "bitpattern.strategies needs hypothesis: pip install 'bitpattern[hypothesis]'"
        ) from exc
    return strategies


def from_intset(source: IntSet) -> SearchStrategy[int]:
    """Members of `source`, drawn by index so nothing is materialised."""
    strategies = _strategies()
    if not source.size:
        return strategies.nothing()
    return strategies.integers(0, source.size - 1).map(source.__getitem__)


def from_codec(codec, source: IntSet | None = None) -> SearchStrategy[Any]:
    """Values of `source` decoded through `codec`; the whole universe by default."""
    return from_intset(codec.all if source is None else source).map(codec.decode)
