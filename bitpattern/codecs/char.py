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

# Code points, but not characters.
SURROGATES = range(0xD800, 0xE000)


# Lazy, since computing every category takes a moment.
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
        # A code point past the width would spill into eg. another field of a tuple.
        if (
            len(value) != 1
            or (point := ord(value)) in SURROGATES
            or point >> self.width
        ):
            raise ValueError(f"{self} can't encode {value!r}")
        return point

    def decode(self, bits: int) -> str:
        try:
            return chr(bits)
        except ValueError as e:  # past the last code point
            raise IndexError(f"{self} has no value encoded as {bits}") from e

    @cached_property
    def all(self) -> BDDSet[str]:
        points = IntSet.range(0, sys.maxunicode + 1, width=self.width)
        surrogates = IntSet.range(SURROGATES.start, SURROGATES.stop, width=self.width)
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
