"""`Pattern`: a set of integers written as bit quartets, with `?` and `*` globs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

from .bdd import ACCEPT, BDD, BDDNode, REJECT, render_count
from .sets import IntSet

__all__ = ["Pattern", "parse"]

#: Bits per group. Every group but the leading one is exactly this wide.
QUARTET = 4

WILD = "?"
GLOB = "*"
UNION = "|"
LITERAL = frozenset("01" + WILD)

# How many branches a repr spells out before truncating.
EXACT, ROOM = 8, 200


def parse(text: str) -> str | None:
    """Expand a pattern into its bit string, msb first; `None` if it is empty.

    Groups are separated by `.` and run from the most significant quartet down:
    `0` and `1` pin a bit, `?` leaves one free, and `*` stands for however many
    `?` the group still needs. The leading group may be short, which is how
    widths that are not multiples of four are written; every other group must
    come to exactly four bits.

    The language describes one pattern. Unions are Python, not syntax --
    `Pattern("01??") | Pattern("1???")` -- which is also how they print.
    """
    text = text.strip()
    if not text:
        return None
    if UNION in text:
        left, _, right = (part.strip() for part in text.partition(UNION))
        raise ValueError(
            f"pattern {text!r} contains {UNION!r}: patterns are single branches, "
            f"so combine them in Python instead -- "
            f"Pattern({left!r}) {UNION} Pattern({right!r})"
        )
    groups = text.split(".")
    return "".join(
        expand(group, leading=not position, text=text)
        for position, group in enumerate(groups)
    )


def expand(group: str, leading: bool, text: str) -> str:
    """Expand one group to its bits, resolving any `*`."""
    def bad(reason: str) -> ValueError:
        where = "leading group" if leading else "group"
        return ValueError(f"{where} {group!r} in pattern {text!r}: {reason}")

    if not group:
        raise bad("is empty")
    head, glob, tail = group.partition(GLOB)
    if GLOB in tail:
        raise bad(f"has more than one {GLOB!r}, so its width is ambiguous")
    if unknown := sorted(set(head + tail) - LITERAL):
        raise bad(f"has unexpected character{'s' * (len(unknown) > 1)} {''.join(unknown)!r}")

    pinned = len(head) + len(tail)
    if pinned > QUARTET:
        raise bad(f"is {pinned} bits wide, over the {QUARTET} of a quartet")
    if not glob:
        if leading or pinned == QUARTET:
            return group
        raise bad(f"is {pinned} bits wide; only the leading group may be short")
    return head + WILD * (QUARTET - pinned) + tail


def build(bits: str) -> BDDNode:
    """The predicate matching one expanded, most-significant-first bit string.

    Built from bit 0 upwards, so the diagram comes out reduced: a free bit is
    simply never tested, and costs no node at all.
    """
    node: BDDNode = ACCEPT
    for bit, char in enumerate(reversed(bits)):
        if char == "0":
            node = BDD(bit, node, REJECT)
        elif char == "1":
            node = BDD(bit, REJECT, node)
    return node


def cover(bdd: BDDNode, width: int) -> list[str]:
    """A diagram's branches, as expanded bit strings, most significant first.

    Canonical rather than minimal: reduction has already merged every mergeable
    sibling, so no two branches here combine, but minimum-cardinality cube cover
    is NP-hard and this is not it.

    Note this is *not* the walk `IP.networks` uses. A CIDR block is a prefix, so
    its free bits are always at the bottom and a free bit there has to be split
    into two blocks. A pattern's free bits can sit anywhere, so here a free bit
    becomes `?` and the walk does not branch at all -- otherwise the one pattern
    `???????????????1` would come back as 32768 of them.
    """
    def walk(node: BDDNode, top: int, prefix: str) -> Iterator[str]:
        if node is REJECT:
            return
        if node is ACCEPT:
            yield prefix + WILD * (top + 1)
            return
        if node.bit < top:
            yield from walk(node, top - 1, prefix + WILD)
            return
        yield from walk(node.left, top - 1, prefix + "0")
        yield from walk(node.right, top - 1, prefix + "1")

    return list(walk(bdd, width - 1, ""))


def group(bits: str) -> str:
    """Write a bit string as dotted quartets, leading group short if need be."""
    head = len(bits) % QUARTET or QUARTET
    return ".".join(
        [bits[:head]] + [bits[at:at + QUARTET] for at in range(head, len(bits), QUARTET)]
    )


class Pattern(IntSet):
    """A set of integers described by a bit pattern such as `*1.*.*.0000`.

    A `Pattern` *is* an `IntSet` -- same members, same width, same operations --
    that additionally knows how to write itself down. The pattern fixes a width,
    so the set is finite, and the free bits are the ones the diagram never tests,
    which is why a pattern denoting thousands of integers costs one node per
    *pinned* bit and nothing per free one.

    A pattern may be a union of branches separated by `|`, which makes it closed
    under `&`, `|`, `-` and `~`: any set of a fixed width is a union of cubes.
    Branches are derived from the diagram rather than remembered from the text,
    so patterns are canonical -- `Pattern("0000 | 0001")` and `Pattern("000?")`
    are the same pattern and render identically.
    """

    __slots__ = ()

    def __init__(self, text: str, width: int | None = None):
        bits = parse(text)
        node: BDDNode = REJECT
        if bits is not None:
            if width is not None and width != len(bits):
                raise ValueError(f"pattern {text!r} is {len(bits)} bits, not {width}")
            width, node = len(bits), build(bits)
        elif width is None:
            raise ValueError("the empty pattern has no width of its own; pass one")
        self.bdd = node
        self.width = width
        self._form = None

    @classmethod
    def from_set(cls, source: IntSet) -> Self:
        return cls.from_bdd(source.bdd, source.width)

    @property
    def set(self) -> IntSet:
        """This pattern as a plain `IntSet`. A `Pattern` already is one, so this
        only matters when a bare `IntSet` is what you want to hand on."""
        return IntSet.from_bdd(self.bdd, self.width)

    @property
    def branches(self) -> tuple[str, ...]:
        """The canonical branches, as expanded bit strings."""
        return tuple(cover(self.bdd, self.width))

    @property
    def bits(self) -> str:
        """A single-branch pattern's expanded bit string, most significant first."""
        branches = self.branches
        if len(branches) != 1:
            raise ValueError(f"{len(branches)} branches, so no single bit string")
        return branches[0]

    @property
    def free(self) -> int:
        """How many bits a single-branch pattern leaves unconstrained."""
        return self.bits.count(WILD)

    def __str__(self) -> str:
        """The canonical spelling: one branch, or several joined by `|`.

        Only a single branch parses back through `Pattern`; a union has to be
        rebuilt with the operator, which is what `repr` shows.
        """
        return f" {UNION} ".join(group(branch) for branch in self.branches)

    def __repr__(self) -> str:
        """Evaluable Python where it fits -- a union prints as the expression
        that would rebuild it, rather than as syntax the language does not have."""
        branches = self.branches
        if not branches:
            # The one shape whose width its own text cannot carry.
            return f"Pattern('', width={self.width})"
        if len(branches) <= EXACT:
            rendered = f" {UNION} ".join(
                f"Pattern({group(branch)!r})" for branch in branches
            )
            if len(rendered) <= ROOM:
                return rendered
        return (
            f"<Pattern: {len(branches)} branches, "
            f"width={self.width}, size={render_count(self.size)}>"
        )
