"""Pattern, an IntSet written as bits, like `*1.*.*.0000.1111.?01?`."""

from __future__ import annotations

from collections.abc import Iterator

from .bdd import BDD, render_count, weak_cache
from .intset import IntSet

__all__ = ["Pattern", "parse"]

QUARTET = 4


def parse(text: str) -> str:
    """Expand a pattern into its bits, most significant first.

    Groups of bits are separated by `.`. `0` and `1` are bits, `?` is a free bit,
    and `*` frees the rest of its group. Groups are 4 bits, except the leading one
    can be shorter.
    """
    text = text.strip()
    if "|" in text:
        left, _, right = (part.strip() for part in text.partition("|"))
        raise ValueError(
            f"patterns can't contain '|', use Pattern({left!r}) | Pattern({right!r})"
        )
    groups = text.split(".")
    return "".join(
        expand(group, leading=not at, text=text) for at, group in enumerate(groups)
    )


def expand(group: str, leading: bool, text: str) -> str:
    """Expand one group into its bits, filling in any `*`."""

    def invalid(reason: str) -> ValueError:
        kind = "leading group" if leading else "group"
        return ValueError(f"{kind} {group!r} in {text!r} {reason}")

    head, glob, tail = group.partition("*")
    bits = len(head) + len(tail)
    if not group:
        raise invalid("is empty")
    if "*" in tail:
        raise invalid("has more than one '*'")
    if unknown := set(head + tail) - set("01?"):
        raise invalid(f"has unexpected characters {''.join(sorted(unknown))!r}")
    if bits > QUARTET:
        raise invalid(f"has {bits} bits, more than {QUARTET}")
    if glob:
        return head + "?" * (QUARTET - bits) + tail
    if bits < QUARTET and not leading:
        raise invalid(f"has {bits} bits, but only the leading group can be short")
    return group


def build(bits: str) -> BDD:
    """The diagram for an expanded pattern. Free bits don't need any nodes."""
    node = BDD.ACCEPT
    for bit, char in enumerate(reversed(bits)):
        match char:
            case "0":
                node = BDD(bit, node, BDD.REJECT)
            case "1":
                node = BDD(bit, BDD.REJECT, node)
    return node


def cover(bdd: BDD, width: int) -> Iterator[str]:
    """A diagram's branches as expanded patterns. Canonical, but not always minimal."""

    def walk(node: BDD, bit: int, prefix: str) -> Iterator[str]:
        prefix += "?" * (bit - node.bit)  # bits above the node are free
        if node is BDD.ACCEPT:
            yield prefix
        elif node:
            yield from walk(node.left, node.bit - 1, prefix + "0")
            yield from walk(node.right, node.bit - 1, prefix + "1")

    return walk(bdd, width - 1, "")


@weak_cache
def branch_count(node: BDD) -> int:
    """How many branches `cover` would yield, ie. how many paths reach ACCEPT."""
    if node.bit < 0:
        return int(node is BDD.ACCEPT)
    return branch_count(node.left) + branch_count(node.right)


def group(bits: str) -> str:
    """Write bits as dotted quartets, with the leading one short if need be."""
    if len(bits) <= QUARTET:
        return bits
    return f"{group(bits[:-QUARTET])}.{bits[-QUARTET:]}"


class Pattern(IntSet):
    """An IntSet that reads and writes itself as a bit pattern, like `*1.*.*.0000`.

    Patterns are written most significant bit first, in groups of 4 separated by
    `.`. `0` and `1` are fixed bits, `?` can be either, and `*` is shorthand for
    enough `?` to fill out its group. Only the leading group can be shorter than
    4 bits, so a pattern can be any width.

    >>> p = Pattern("*1.0000")
    >>> p
    Pattern('???1.0000')
    >>> p.width, p.size, p.free
    (8, 8, 3)
    >>> ~Pattern("00??")
    Pattern('01??') | Pattern('1???')

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
        bits = parse(text)
        self.bdd, self.width = build(bits), len(bits)

    @property
    def branches(self) -> tuple[str, ...]:
        """Disjoint single-branch patterns that make up this one."""
        return tuple(map(group, cover(self.bdd, self.width)))

    @property
    def bits(self) -> str:
        """The expanded bits of a single-branch pattern."""
        if (branches := branch_count(self.bdd)) != 1:
            raise ValueError(f"pattern has {branches} branches, not 1")
        return next(cover(self.bdd, self.width))

    @property
    def free(self) -> int:
        """How many bits a single-branch pattern leaves free."""
        return self.bits.count("?")

    def __str__(self) -> str:
        return " | ".join(self.branches)

    def __repr__(self) -> str:
        if not self.width:  # there's no way to write a zero-width pattern
            return f"<Pattern: width=0, size={self.size}>"
        if not (branches := branch_count(self.bdd)):
            return f"~Pattern({group('?' * self.width)!r})"
        # Unions repr as the expression that builds them, unless that's too long.
        if branches <= 8:
            text = " | ".join(f"Pattern({branch!r})" for branch in self.branches)
            if len(text) <= 200:
                return text
        return (
            f"<Pattern: {render_count(branches)} branches, "
            f"width={self.width}, size={render_count(self.size)}>"
        )
