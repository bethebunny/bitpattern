"""Pattern, an IntSet written as bits, like `*1.*.*.0000.1111.?01?`.

Text like `*1.0000` expands to bits like `???10000`, one character per bit:
  - `expanded` and `dotted` convert between text and bits
  - `diagram` and `paths`, from `bitpattern.bdd`, convert between bits and diagrams
"""

from __future__ import annotations

import re

from .bdd import count_repr, diagram, path_count, paths
from .intset import IntSet

__all__ = ["Pattern"]

LEADING, QUARTET = re.compile("[01?]{1,4}"), re.compile("[01?]{4}")


def expanded(text: str) -> str:
    """A pattern's bits, one character per bit, eg. `*1.0000` is `???10000`."""
    if "|" in text:
        left, _, right = (part.strip() for part in text.partition("|"))
        raise ValueError(
            f"patterns can't contain '|', use Pattern({left!r}) | Pattern({right!r})"
        )
    bits = ""
    for at, quartet in enumerate(text.strip().split(".")):
        head, glob, tail = quartet.partition("*")
        expanded = head + "?" * (4 - len(head) - len(tail)) + tail if glob else quartet
        if not (LEADING if at == 0 else QUARTET).fullmatch(expanded):
            raise ValueError(
                f"bad quartet {quartet!r} in {text!r}. Quartets are 4 bits of 0, 1 "
                "or ?, and * fills out the rest. Only the leading quartet can be "
                "shorter."
            )
        bits += expanded
    return bits


def dotted(bits: str) -> str:
    """Bits written as pattern text, eg. `???10000` is `???1.0000`."""
    head = len(bits) % 4 or 4  # the leading quartet can be shorter
    quartets = [bits[at : at + 4] for at in range(head, len(bits), 4)]
    return ".".join([bits[:head], *quartets])


class Pattern(IntSet):
    """An IntSet that reads and writes itself as a bit pattern, like `*1.*.*.0000`.

    Patterns are written most significant bit first, in quartets separated by
    `.`. `0` and `1` are fixed bits, `?` can be either, and `*` is shorthand for
    enough `?` to fill out its quartet. The leading quartet can be shorter for
    widths that aren't a multiple of 4.

    >>> p = Pattern("*1.0000")
    >>> p
    Pattern('???1.0000')
    >>> p.size
    8

    Unions repr as disjoint patterns.

    >>> Pattern("1?1") | Pattern("?1?")
    Pattern('01?') | Pattern('101') | Pattern('11?')

    A pattern has a node per fixed bit. Past 128 bits each node's count is about as
    wide as its bit, so memory grows with the square of the width.
    """

    __slots__ = ()

    def __init__(self, text: str) -> None:
        bits = expanded(text)
        self.bdd, self.width = diagram(bits), len(bits)

    @property
    def branches(self) -> tuple[str, ...]:
        """Disjoint single-branch patterns that make up this one."""
        return tuple(map(dotted, paths(self.bdd, width=self.width)))

    @property
    def bits(self) -> str:
        """The expanded bits of a single-branch pattern."""
        if (count := path_count(self.bdd)) != 1:
            raise ValueError(f"pattern has {count} branches, not 1")
        return next(paths(self.bdd, width=self.width))

    @property
    def free(self) -> int:
        """How many bits a single-branch pattern leaves free."""
        return self.bits.count("?")

    def __str__(self) -> str:
        return " | ".join(self.branches)

    def __repr__(self) -> str:
        if not self.width:  # there's no way to write a zero-width pattern
            return f"<Pattern: width=0, size={self.size}>"
        if not (count := path_count(self.bdd)):
            return f"~Pattern({dotted('?' * self.width)!r})"
        if count <= 8:
            text = " | ".join(f"Pattern({branch!r})" for branch in self.branches)
            if len(text) <= 200:
                return text
        return (
            f"<Pattern: {count_repr(count)} branches, "
            f"width={self.width}, size={count_repr(self.size)}>"
        )
