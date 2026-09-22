"""`Pattern`: a set of integers written as bit quartets, with `?` and `*` globs."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from .bdd import ACCEPT, BDD, BDDNode, CACHE_SIZE, REJECT, cofactors, render_count
from .sets import IntSet

__all__ = ["Pattern", "parse"]

QUARTET = 4  # bits per group; only the leading group may be shorter
WILD, GLOB, UNION = "?", "*", "|"
LITERAL = frozenset("01" + WILD)

# A union's repr spells out at most this many branches, in this many characters.
REPR_BRANCHES, REPR_CHARS = 8, 200


def parse(text: str) -> str:
    """Expand a pattern into its bit string, most significant bit first.

    Groups are separated by `.`, most significant first: `0` and `1` pin a bit,
    `?` frees one, and `*` frees the rest of its group. Every group is four bits
    except the leading one, which may be shorter.
    """
    text = text.strip()
    if UNION in text:
        left, _, right = (part.strip() for part in text.partition(UNION))
        raise ValueError(
            f"pattern {text!r} contains {UNION!r}; patterns are single branches, so "
            f"combine them in Python: Pattern({left!r}) {UNION} Pattern({right!r})"
        )
    return "".join(
        expand(group, leading=not position, text=text)
        for position, group in enumerate(text.split("."))
    )


def expand(group: str, leading: bool, text: str) -> str:
    """Expand one group to its bits, resolving any `*`."""
    def bad(reason: str) -> ValueError:
        where = "leading group" if leading else "group"
        return ValueError(f"{where} {group!r} in pattern {text!r}: {reason}")

    head, glob, tail = group.partition(GLOB)
    if not group:
        raise bad("is empty")
    if GLOB in tail:
        raise bad(f"has more than one {GLOB!r}")
    if unknown := set(head + tail) - LITERAL:
        raise bad(f"has unexpected characters {''.join(sorted(unknown))!r}")
    if (pinned := len(head) + len(tail)) > QUARTET:
        raise bad(f"is {pinned} bits, over a quartet")
    if glob:
        return head + WILD * (QUARTET - pinned) + tail
    if leading or pinned == QUARTET:
        return group
    raise bad(f"is {pinned} bits; only the leading group may be short")


def build(bits: str) -> BDDNode:
    """The predicate matching an expanded bit string. Free bits cost no node."""
    node: BDDNode = ACCEPT
    for bit, char in enumerate(reversed(bits)):
        if char == "0":
            node = BDD(bit, node, REJECT)
        elif char == "1":
            node = BDD(bit, REJECT, node)
    return node


def cover(bdd: BDDNode, width: int) -> Iterator[str]:
    """A diagram's branches as expanded bit strings: canonical, not minimal."""
    def walk(node: BDDNode, top: int, prefix: str) -> Iterator[str]:
        if node is REJECT:
            return
        if node is ACCEPT:
            yield prefix + WILD * (top + 1)
            return
        low, high = cofactors(node, top)
        # Unlike `IP.networks`, don't split a free bit: a CIDR block has to be a
        # prefix, but a pattern's free bits can sit anywhere.
        if low is high:
            yield from walk(low, top - 1, prefix + WILD)
        else:
            yield from walk(low, top - 1, prefix + "0")
            yield from walk(high, top - 1, prefix + "1")

    return walk(bdd, width - 1, "")


@lru_cache(maxsize=CACHE_SIZE)
def branch_count(node: BDDNode) -> int:
    """How many branches `cover` yields, without yielding them: one per path to
    `ACCEPT`. Popcount parity over 64 bits has 2**63 of them in 128 nodes."""
    if not isinstance(node, BDD):
        return int(node is ACCEPT)
    return branch_count(node.left) + branch_count(node.right)


def group(bits: str) -> str:
    """Write a bit string as dotted quartets, the leading group short if need be."""
    head = len(bits) % QUARTET or QUARTET
    return ".".join([bits[:head], *(bits[at:at + QUARTET] for at in range(head, len(bits), QUARTET))])


class Pattern(IntSet):
    """An `IntSet` written as a bit pattern, such as `*1.*.*.0000`.

    The language describes one pattern; unions come from the operators and print
    that way. Branches are read off the diagram rather than kept from the text,
    so `Pattern("0000") | Pattern("0001")` *is* `Pattern("000?")`.
    """

    __slots__ = ()

    def __init__(self, text: str):
        bits = parse(text)
        self.bdd = build(bits)
        self.width = len(bits)

    @property
    def branches(self) -> tuple[str, ...]:
        return tuple(cover(self.bdd, self.width))

    @property
    def bits(self) -> str:
        """The expanded bit string of a single-branch pattern."""
        if len(branches := self.branches) != 1:
            raise ValueError(f"{len(branches)} branches, so no single bit string")
        return branches[0]

    @property
    def free(self) -> int:
        """How many bits a single-branch pattern leaves free."""
        return self.bits.count(WILD)

    def __str__(self) -> str:
        return f" {UNION} ".join(group(branch) for branch in self.branches)

    def __repr__(self) -> str:
        branches = branch_count(self.bdd)
        if not self.width:  # the language has no zero-width pattern
            return f"<Pattern: width=0, size={self.size}>"
        if not branches:
            return f"~Pattern({group(WILD * self.width)!r})"
        if branches <= REPR_BRANCHES:
            rendered = f" {UNION} ".join(f"Pattern({group(branch)!r})" for branch in self.branches)
            if len(rendered) <= REPR_CHARS:
                return rendered
        return (
            f"<Pattern: {render_count(branches)} branches, "
            f"width={self.width}, size={render_count(self.size)}>"
        )
