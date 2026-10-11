"""Unicode characters as their code points."""

from __future__ import annotations

import collections
import itertools
import sys
import unicodedata
from dataclasses import dataclass
from functools import cache, cached_property

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["Char", "char"]


# Categories aren't particularly expensive to compute, but expensive enough
# to not compute them on import if they're unused. @cache so they're only
# computed once lazily.
@cache
def _category_points(width: int) -> dict[str, IntSet]:
    """Consecutive code points in each general category (eg. "Lu") and major
    class (eg. "L")."""
    found: dict[str, IntSet] = collections.defaultdict(IntSet)
    start = 0
    characters = map(chr, range(sys.maxunicode + 1))
    for name, run in itertools.groupby(characters, unicodedata.category):
        size = sum(1 for _ in run)
        codepoints = IntSet.range(start, start + size, width=width)
        found[name] |= codepoints
        found[name[0]] |= codepoints
        start += size
    return dict(found)


@dataclass(frozen=True, repr=False)
class Char(OrderedCodec[str]):
    """Unicode scalar values as one-character strings in code point order.

    Surrogates are not included (U+D800 to U+DFFF).
    """

    def encode(self, value: object) -> int:
        if not isinstance(value, str):
            raise TypeError(f"{self} can't encode {value!r}")
        if len(value) != 1 or ord(value) not in self.all.storage:
            raise ValueError(f"{self} can't encode {value!r}")
        return ord(value)

    def decode(self, bits: int) -> str:
        self._check(bits)
        return chr(bits)

    @cached_property
    def all(self) -> BDDSet[str]:
        """Every code point except surrogates."""
        points = IntSet.range(0, sys.maxunicode + 1, width=self.width)
        surrogates = IntSet.range(0xD800, 0xE000, width=self.width)
        return BDDSet(self, points - surrogates)

    def category(self, name: str) -> BDDSet[str]:
        """Characters in a general category like "Lu" or a major class like "L".

        Categories come from the running Python's `unicodedata`. "Cs" (surrogates) is
        always empty. Building a category the first time takes a moment.
        """
        if (codepoints := _category_points(self.width).get(name)) is None:
            raise ValueError(f"{name} is not a valid unicode category")
        return BDDSet(self, codepoints) & self.all


char = Char(21, "char")
