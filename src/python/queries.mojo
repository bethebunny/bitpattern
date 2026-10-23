"""Questions about diagrams, as Python functions."""

from std.bit import count_leading_zeros
from std.python._cpython import PyObjectPtr

from arena import ACCEPT, Arena, Id, Interrupted, NARROW_WIDTH
from arena import build, compare, members, nodes
from bigint import BigUInt

from .args import Arguments, Signature, node, width
from .capi import Py_DecRef, PyBool_FromLong, PyLong_FromSsize_t, new_string
from .errors import PyError, index_error, value_error
from .ints import (
    count_repr as written_count,
    indexed,
    int_from_big,
    int_from_u128,
)
from .ints import Ints, big_of, i128_of
from .state import Lease, StatePointer, give_back, take_arena


@fieldwise_init
struct Count(Movable):
    """A count, in a UInt128 when it fits."""

    var narrow: UInt128
    var wide: Optional[BigUInt]


@always_inline
def count_fits_u128(id: Id, width: Int32) -> Bool:
    return width < NARROW_WIDTH or (width == NARROW_WIDTH and id != ACCEPT)


def member_count(arena: Arena, id: Id, width: Int32) -> Count:
    if count_fits_u128(id, width):
        return Count(arena.count_narrow(id, width), None)
    return Count(0, arena.count(id, width))


def int_from_count(ints: Ints, count: Count) raises PyError -> PyObjectPtr:
    var int = int_from_big(
        ints, count.wide.value()
    ) if count.wide else int_from_u128(ints, count.narrow)
    if not int:
        raise PyError()
    return int


def checked_width(
    state: StatePointer, id: Id, width: Int32
) raises PyError -> Int32:
    var arena = take_arena(state)
    var bit = Int32(-1) if id.is_leaf() else arena[].decision(id).bit
    give_back(state, arena)
    if bit >= width:
        raise value_error(
            String("node tests bit ", bit, ", outside width ", width)
        )
    return width


comptime COUNT = Signature("count", positional=["node"], keywords=["width"])


def count(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var at = checked_width(state, id, width(state[].ints, args[1]))
    var arena = take_arena(state)
    var total = member_count(arena[], id, at)
    give_back(state, arena)
    return int_from_count(state[].ints, total)


comptime NODE_COUNT = Signature(
    "node_count", positional=["node"], doc="Leaves aren't counted."
)


def node_count(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var lease = Lease(state)
    var found: Int
    try:
        with lease^ as arena:
            found = nodes.node_count(arena[], id)
    except:
        raise PyError()
    return PyLong_FromSsize_t(found)


comptime COUNT_REPR = Signature(
    "count_repr",
    positional=["total"],
    doc=(
        "A count as reprs write it:\n"
        "  - `2**k` for powers of two from `2**10`\n"
        "  - `~2**k` for ints too long to write out\n"
        "  - the digits otherwise"
    ),
)


def count_repr(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var total = indexed(state[].ints, args[0])
    var written: String
    try:
        if big_of(state[].ints, total)[0]:
            raise value_error("counts aren't negative")
        written = written_count(state[].ints, total)
    except e:
        Py_DecRef(total)
        raise e
    Py_DecRef(total)
    return new_string(written)


@fieldwise_init
struct Integer(Movable):
    """An int as its sign and bits: its own if it isn't negative, else its complement's
    (`~int`, which isn't). The bits are in a UInt128 when they fit."""

    var negative: Bool
    var narrow: UInt128
    var wide: Optional[BigUInt]
    """The bits, if they don't fit a UInt128."""

    def bit_length(self) -> Int:
        if self.wide:
            return self.wide.value().bits()
        return 128 - Int(count_leading_zeros(self.narrow))

    def widened(self) -> BigUInt:
        """The bits in a BigUInt."""
        if self.wide:
            return self.wide.value().copy()
        return BigUInt(self.narrow)


def integer(ints: Ints, o: PyObjectPtr) raises PyError -> Integer:
    """Runs `o.__index__` at most once, so its errors come out as they are."""
    var int = indexed(ints, o)
    var small = i128_of(ints, int)
    if small:
        Py_DecRef(int)
        var value = small.value()
        return Integer(value < 0, UInt128(~value if value < 0 else value), None)
    var read = big_of(ints, int)
    Py_DecRef(int)
    # A negative int's complement is its magnitude less one.
    var bits = read[1] - BigUInt(1) if read[0] else read[1].copy()
    var narrow = bits.fitting_uint128()
    if narrow:
        return Integer(read[0], narrow.value(), None)
    return Integer(read[0], 0, bits^)


def integer_bit(value: Integer, bit: Int32) -> Bool:
    """As Python reads ints: two's complement, so a negative int's bits are its
    complement's flipped."""
    if value.wide:
        return build.wide_bit(value.wide.value(), bit) != value.negative
    return build.narrow_bit(value.narrow, bit) != value.negative


comptime ACCEPTS = Signature("accepts", positional=["node", "value"])


def accepts(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var value = integer(state[].ints, args[1])
    var arena = take_arena(state)
    var answer = members.accepts[integer_bit](arena[], id, value)
    give_back(state, arena)
    return PyBool_FromLong(Int(answer))


def member_at(
    arena: Arena, id: Id, width: Int32, index: Integer
) -> Optional[Count]:
    """None if `index` is out of range. `~i` is the `i`th from the end."""
    if not index.wide and count_fits_u128(id, width):
        var total = arena.count_narrow(id, width)
        if index.narrow >= total:
            return None
        var at = total - 1 - index.narrow if index.negative else index.narrow
        return Count(members.nth_narrow(arena, id, at), None)
    var total = arena.count(id, width)
    var at = index.widened()
    if at >= total:
        return None
    if index.negative:
        at = total - (at + 1)
    return Count(0, members.nth(arena, id, at))


comptime NTH = Signature(
    "nth",
    positional=["node", "n"],
    keywords=["width"],
    doc="Negative `n` counts from the end, as for lists.",
)


def nth(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var at = width(state[].ints, args[2])
    var index = integer(state[].ints, args[1])
    at = checked_width(state, id, at)
    var arena = take_arena(state)
    var member = member_at(arena[], id, at, index)
    give_back(state, arena)
    if not member:
        raise index_error(args[1])
    return int_from_count(state[].ints, member.value())


def members_below(arena: Arena, id: Id, width: Int32, value: Integer) -> Count:
    if value.negative:
        return Count(0, None)  # below every member
    if value.bit_length() > Int(width):
        return member_count(arena, id, width)  # above every member
    if width <= NARROW_WIDTH:
        return Count(members.rank_narrow(arena, id, value.narrow, width), None)
    return Count(0, members.rank(arena, id, value.widened(), width))


comptime RANK = Signature(
    "rank",
    positional=["node", "value"],
    keywords=["width"],
    doc="How many members are below `value`.",
)


def rank(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var value = integer(state[].ints, args[1])
    var at = checked_width(state, id, width(state[].ints, args[2]))
    var arena = take_arena(state)
    var below = members_below(arena[], id, at, value)
    give_back(state, arena)
    return int_from_count(state[].ints, below)


def answered[
    walk: def(mut Arena, Id, Id) thin raises Interrupted -> Bool
](state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var a = node(state, args[0])
    var b = node(state, args[1])
    var lease = Lease(state)
    var answer: Bool
    try:
        with lease^ as arena:
            answer = walk(arena[], a, b)
    except:
        raise PyError()
    return PyBool_FromLong(Int(answer))


comptime SUBSET = Signature("subset", positional=["a", "b"])


def subset(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    return answered[compare.subset](state, args)


comptime DISJOINT = Signature("disjoint", positional=["a", "b"])


def disjoint(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    return answered[compare.disjoint](state, args)


comptime PATH_COUNT = Signature(
    "path_count",
    positional=["node"],
    doc="Counted without walking each path to `BDD.ACCEPT`.",
)


def path_count(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var lease = Lease(state)
    var paths: BigUInt
    try:
        with lease^ as arena:
            paths = nodes.path_count(arena[], id)
    except:
        raise PyError()
    var int = int_from_big(state[].ints, paths)
    if not int:
        raise PyError()
    return int
