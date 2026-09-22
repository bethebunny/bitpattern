"""Hypothesis strategies over `IntSet`s. Needs `pip install bitpattern[hypothesis]`.

Members are drawn by index, so nothing is enumerated -- and since index order is
value order for non-negative floats, shrinking the index shrinks the float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

    from .codecs import Codec
    from .sets import IntSet

__all__ = ["from_codec", "from_intset"]


def _strategies():
    try:
        from hypothesis import strategies
    except ImportError as exc:
        raise ImportError(
            "bitpattern.strategies needs hypothesis: pip install 'bitpattern[hypothesis]'"
        ) from exc
    return strategies


def from_intset(source: IntSet) -> SearchStrategy[int]:
    strategies = _strategies()
    if not source:
        return strategies.nothing()
    return strategies.integers(0, source.size - 1).map(source.__getitem__)


def from_codec(codec: Codec, source: IntSet | None = None) -> SearchStrategy[Any]:
    """Decoded members of `source`, or of the codec's whole universe."""
    return from_intset(codec.all if source is None else source).map(codec.decode)
