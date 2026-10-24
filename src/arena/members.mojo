"""Membership, `nth` and `rank`. The `_narrow` versions are for sets whose counts fit a
UInt128."""

from bigint import BigUInt

from .arena import ACCEPT, Arena, Id, NARROW_WIDTH, REJECT
from .build import narrow_bit, wide_bit


@always_inline
def bits_between(value: UInt128, *, low: Int32, high: Int32) -> UInt128:
    """`value`'s bits in `[low, high)`."""
    if low >= high:
        return 0
    var mask = UInt128.MAX >> UInt128(128 - (high - low))
    return (value >> UInt128(low)) & mask


def wide_bits_between(value: BigUInt, *, low: Int32, high: Int32) -> BigUInt:
    return (value >> low).low_bits(high - low)


def accepts[
    T: AnyType, //, bit_of: def(T, Int32) thin -> Bool
](arena: Arena, root: Id, value: T) -> Bool:
    var id = root
    while not id.is_leaf():
        var node = arena.decision(id)
        id = node.hi if bit_of(value, node.bit) else node.lo
    return id == ACCEPT


def nth_narrow(arena: Arena, root: Id, n: UInt128) -> UInt128:
    """The caller checks that `n` is in range."""
    var id = root
    var rest = n
    var member: UInt128 = 0
    while not id.is_leaf():
        var node = arena.decision(id)
        # The free bits above the node are the member's highest bits.
        var size = arena.sizes.narrow.unsafe_get(id.index())
        if rest >= size:
            var high: UInt128
            high, rest = divmod(rest, size)
            member |= high << UInt128(node.bit + 1)
        var lo_count = arena.count_narrow(node.lo, node.bit)
        if rest < lo_count:
            id = node.lo
        else:
            member |= UInt128(1) << UInt128(node.bit)
            rest -= lo_count
            id = node.hi
    return member | rest  # the free bits below the last node


def nth(arena: Arena, root: Id, n: BigUInt) -> BigUInt:
    """The caller checks that `n` is in range."""
    var id = root
    var rest = n.copy()
    var member = BigUInt()
    while not id.is_leaf():
        var node = arena.decision(id)
        var size = arena.size(id)
        if rest >= size:
            var high = rest // size
            member |= high << (node.bit + 1)
            rest -= high * size
        if node.bit < NARROW_WIDTH:
            var low = nth_narrow(arena, id, rest.uint128_or_none().value())
            return member | BigUInt(low)
        var lo_count = arena.count(node.lo, node.bit)
        if rest < lo_count:
            id = node.lo
        else:
            member.set_bit(node.bit, True)
            rest -= lo_count
            id = node.hi
    return member | rest


def rank_narrow(
    arena: Arena, root: Id, value: UInt128, width: Int32
) -> UInt128:
    """`value` is below `2 ** width`."""
    var id = root
    var below: UInt128 = 0
    var top = width
    while not id.is_leaf():
        var node = arena.decision(id)
        # Each value of the free bits above the node that's below `value`'s adds
        # `size` members.
        var size = arena.sizes.narrow.unsafe_get(id.index())
        below += size * bits_between(value, low=node.bit + 1, high=top)
        if narrow_bit(value, node.bit):
            below += arena.count_narrow(node.lo, node.bit)
            id = node.hi
        else:
            id = node.lo
        top = node.bit
    return (
        below + bits_between(value, low=0, high=top) if id == ACCEPT else below
    )


def rank(arena: Arena, root: Id, value: BigUInt, width: Int32) -> BigUInt:
    """`value` is below `2 ** width`."""
    var id = root
    var below = BigUInt()
    var top = width
    while id != REJECT:
        if top <= NARROW_WIDTH:
            var low = value.low_bits(top).uint128_or_none().value()
            return below + BigUInt(rank_narrow(arena, id, low, top))
        if id == ACCEPT:
            return below + value.low_bits(top)
        var node = arena.decision(id)
        below += arena.size(id) * wide_bits_between(
            value, low=node.bit + 1, high=top
        )
        if wide_bit(value, node.bit):
            below += arena.count(node.lo, node.bit)
            id = node.hi
        else:
            id = node.lo
        top = node.bit
    return below^
