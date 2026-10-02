"""Pattern, an IntSet written as bits, like `*1.*.*.0000.1111.?01?`.

Text like `*1.0000` expands to bits like `???10000`, one character per bit, and the
bits build a diagram. `expand` and `dotted` convert between text and bits, and
`diagram` and `paths` between bits and diagrams.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .bdd import BDD, render_count, weak_cache
from .intset import IntSet

__all__ = ["Pattern"]


def expand(text: str) -> str:
    """A pattern's bits, one character per bit, eg. `*1.0000` is `???10000`."""
    if "|" in text:
        left, _, right = (part.strip() for part in text.partition("|"))
        raise ValueError(
            f"patterns can't contain '|', use Pattern({left!r}) | Pattern({right!r})"
        )
    bits = ""
    for at, group in enumerate(text.strip().split(".")):
        head, glob, tail = group.partition("*")
        expanded = head + "?" * (4 - len(head) - len(tail)) + tail if glob else group
        if not re.fullmatch("[01?]{1,4}" if at == 0 else "[01?]{4}", expanded):
            raise ValueError(
                f"bad group {group!r} in {text!r}. Groups are 4 bits of 0, 1 or ?, "
                "with * filling out the rest, and only the leading one can be shorter."
            )
        bits += expanded
    return bits


def dotted(bits: str) -> str:
    """Bits written as pattern text, eg. `???10000` is `???1.0000`."""
    head, tail = bits[:-4], bits[-4:]
    return f"{dotted(head)}.{tail}" if head else tail


def diagram(bits: str) -> BDD:
    """The diagram matching `bits`. Free bits don't need any nodes."""
    node = BDD.ACCEPT
    for bit, char in enumerate(reversed(bits)):
        match char:
            case "0":
                node = BDD(bit, node, BDD.REJECT)
            case "1":
                node = BDD(bit, BDD.REJECT, node)
    return node


def paths(node: BDD, width: int) -> Iterator[str]:
    """The bits of each path from `node` to ACCEPT, which never overlap."""
    free = "?" * (width - 1 - node.bit)  # the bits above the node
    if node is BDD.ACCEPT:
        yield free
    elif node:
        yield from (free + "0" + path for path in paths(node.left, node.bit))
        yield from (free + "1" + path for path in paths(node.right, node.bit))


@weak_cache
def path_count(node: BDD) -> int:
    """How many paths `paths` would give, without walking them."""
    if node.leaf:
        return int(node is BDD.ACCEPT)
    return path_count(node.left) + path_count(node.right)


class Pattern(IntSet):
    """An IntSet that reads and writes itself as a bit pattern, like `*1.*.*.0000`.

    Patterns are written most significant bit first, in groups of 4 separated by
    `.`. `0` and `1` are fixed bits, `?` can be either, and `*` is shorthand for
    enough `?` to fill out its group. Only the leading group can be shorter than
    4 bits, so a pattern can be any width.

    >>> p = Pattern("*1.0000")
    >>> p
    Pattern('???1.0000')
    >>> p.size
    8

    Patterns only spell out a single branch. Unions come from the set operators,
    and repr as the expression that builds them. Branches come from the diagram
    rather than the text, so `Pattern("0000") | Pattern("0001")` is `Pattern("000?")`.
    The same set always has the same branches, and they never overlap, but they
    aren't always the fewest patterns that would cover it.

    >>> Pattern("1?1") | Pattern("?1?")
    Pattern('01?') | Pattern('101') | Pattern('11?')

    A branch needs a node for each fixed bit and none for free bits, so a pattern's
    diagram stays small however many members it has.
    """

    __slots__ = ()

    def __init__(self, text: str) -> None:
        bits = expand(text)
        self.bdd, self.width = diagram(bits), len(bits)

    @property
    def branches(self) -> tuple[str, ...]:
        """Disjoint single-branch patterns that make up this one."""
        return tuple(map(dotted, paths(self.bdd, self.width)))

    @property
    def bits(self) -> str:
        """The expanded bits of a single-branch pattern."""
        if (count := path_count(self.bdd)) != 1:
            raise ValueError(f"pattern has {count} branches, not 1")
        return next(paths(self.bdd, self.width))

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
        # Unions repr as the expression that builds them, unless that's too long.
        if count <= 8:
            text = " | ".join(f"Pattern({branch!r})" for branch in self.branches)
            if len(text) <= 200:
                return text
        return (
            f"<Pattern: {render_count(count)} branches, "
            f"width={self.width}, size={render_count(self.size)}>"
        )
