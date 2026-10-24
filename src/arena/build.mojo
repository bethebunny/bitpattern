"""Diagrams made from values, bit patterns and intervals."""

from bigint import BigUInt

from .arena import ACCEPT, Arena, Id, Interrupted, REJECT
from .paths import ONE, ZERO
from .steps import Span, Step, emptied, pop_branches, take


@always_inline
def narrow_bit(value: UInt128, bit: Int32) -> Bool:
    return bit < 128 and (value >> UInt128(bit)) & 1 == 1


@always_inline
def wide_bit(value: BigUInt, bit: Int32) -> Bool:
    return value.bit(bit)


def exactly_these[
    T: Copyable, //, bit_of: def(T, Int32) thin -> Bool
](mut arena: Arena, values: List[T], width: Int32) raises Interrupted -> Id:
    """`values` is sorted and distinct, and below `2 ** width`."""
    var steps = take(arena.scratch.span_steps)
    var results = take(arena.scratch.results)
    steps.append(Step.visit(Span(0, len(values), width - 1)))
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            var span = step.item
            if span.start == span.end:
                results.append(REJECT)
            elif span.bit < 0:
                results.append(ACCEPT)
            else:
                var split = partition_point[bit_of](values, span)
                steps.append(Step.make(span.bit, span))
                steps.append(Step.visit(Span(split, span.end, span.bit - 1)))
                steps.append(Step.visit(Span(span.start, split, span.bit - 1)))
        else:
            var lo, hi = pop_branches(results)
            results.append(arena.make(step.bit, lo, hi))
    var result = results.pop()
    arena.scratch.span_steps = emptied(steps^)
    arena.scratch.results = emptied(results^)
    return result


def partition_point[
    T: Copyable, //, bit_of: def(T, Int32) thin -> Bool
](values: List[T], span: Span) -> Int:
    """The index of the first value in `span` with `span.bit` set."""
    # A span's values share every bit above `span.bit`, so values with it clear
    # come first.
    var low = span.start
    var high = span.end
    while low < high:
        var middle = (low + high) // 2
        if bit_of(values[middle], span.bit):
            high = middle
        else:
            low = middle + 1
    return low


def diagram(mut arena: Arena, bits: StringSlice[_]) raises Interrupted -> Id:
    """`bits` is most significant first, and any byte but `0` or `1` is a free bit.
    """
    var bytes = bits.as_bytes()
    var id = ACCEPT
    var bit: Int32 = 0
    for symbol in reversed(bytes):
        arena.tick()
        if symbol == ZERO:
            id = arena.make(bit, id, REJECT)
        elif symbol == ONE:
            id = arena.make(bit, REJECT, id)
        bit += 1
    return id


def interval[
    T: Copyable, //, bit_of: def(T, Int32) thin -> Bool
](mut arena: Arena, first: T, last: T, width: Int32) raises Interrupted -> Id:
    """`[first, last]`, for `first <= last < 2 ** width`."""
    var split = width - 1
    while split >= 0 and bit_of(first, split) == bit_of(last, split):
        split -= 1
    var id = ACCEPT
    if split >= 0:
        # Below `split`, `at_least` accepts the low bits of members from `first` up,
        # and `at_most` those up to `last`. Below `low` any bits will do.
        var at_least = ACCEPT
        var at_most = ACCEPT
        var low: Int32 = 0
        while low < split and not bit_of(first, low) and bit_of(last, low):
            low += 1
        for bit in range(low, split):
            arena.tick()
            if bit_of(first, bit):
                at_least = arena.make(bit, REJECT, at_least)
            else:
                at_least = arena.make(bit, at_least, ACCEPT)
            if bit_of(last, bit):
                at_most = arena.make(bit, ACCEPT, at_most)
            else:
                at_most = arena.make(bit, at_most, REJECT)
        id = arena.make(split, at_least, at_most)
    for bit in range(split + 1, width):
        arena.tick()
        if bit_of(first, bit):
            id = arena.make(bit, REJECT, id)
        else:
            id = arena.make(bit, id, REJECT)
    return id
