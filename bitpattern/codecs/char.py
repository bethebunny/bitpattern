"""Unicode characters, as their code points."""

from __future__ import annotations

import collections
import itertools
import operator
import sys
import unicodedata
from dataclasses import dataclass
from functools import cache, cached_property, reduce

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["Char", "char"]


@cache
def category_runs() -> dict[str, list[range]]:
    """The runs of consecutive code points in each general category, eg. "Lu", and in
    each major class, eg. "L"."""
    found: dict[str, list[range]] = collections.defaultdict(list)
    start = 0
    characters = map(chr, range(sys.maxunicode + 1))
    for name, run in itertools.groupby(characters, unicodedata.category):
        stop = start + sum(1 for _ in run)
        found[name].append(range(start, stop))
        found[name[0]].append(range(start, stop))
        start = stop
    return dict(found)


@cache
def category_points(name: str, width: int) -> IntSet:
    """The code points in a general category or major class, over `width` bits."""
    if (runs := category_runs().get(name)) is None:
        raise ValueError(f"{name!r} isn't a Unicode general category")
    ranges = (IntSet.range(run.start, run.stop, width=width) for run in runs)
    return reduce(operator.or_, ranges)


@dataclass(frozen=True, repr=False)
class Char(OrderedCodec[str]):
    """One-character strings, as their code points.

    Only Unicode scalar values are characters. That leaves out the surrogates,
    U+D800 to U+DFFF, even though a Python str can hold one. Characters sort by
    code point, which is the order Python compares strings in.
    """

    def encode(self, value: object) -> int:
        if (
            isinstance(value, str)
            and len(value) == 1
            and (point := ord(value)) in self.all.storage
        ):
            return point
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> str:
        return chr(bits)

    @cached_property
    def all(self) -> BDDSet[str]:
        """Every Unicode scalar value, ie. every code point but the surrogates."""
        points = IntSet.range(0, sys.maxunicode + 1, width=self.width)
        surrogates = IntSet.range(0xD800, 0xE000, width=self.width)
        return BDDSet(self, points - surrogates)

    def _bound(self, value: str) -> int:
        # Any one-character string can bound a range, surrogates included, since
        # strings sort by code point.
        return ord(value)

    def category(self, name: str) -> BDDSet[str]:
        """Characters in a general category like "Lu", or a major class like "L".

        Categories come from `unicodedata`, so they depend on the running Python's
        Unicode version, `unicodedata.unidata_version`. Surrogates aren't
        characters, so "Cs" is empty. Each category takes a moment to build the
        first time, and is cached after that.
        """
        return BDDSet(self, category_points(name, self.width)) & self.all


char = Char(21, "char")
