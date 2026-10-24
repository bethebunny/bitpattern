"""A set's members, written into a DLPack tensor's words."""

from std.math import ceildiv
from std.sys.info import bit_width_of

from .arena import ACCEPT, Arena, Id, Interrupted, REJECT
from .walk import Prefix, Walk


comptime CHUNK = 1 << 12
"""Members written per tick."""

comptime Slots[word: DType] = Pointer[Scalar[word], MutUntrackedOrigin]
"""Room for a tensor's words: unsigned ints of 8, 16, 32 or 64 bits. The tensor's dtype
says how to read them."""


@fieldwise_init
struct WriteStep(TrivialRegisterPassable):
    var id: Id
    """The node to visit, or REJECT for copies."""
    var low: Int32
    var width: Int32
    var at: UInt64
    """A visit's prefix, or where the block to copy starts."""

    @staticmethod
    @always_inline
    def visit(id: Id, width: Int32, prefix: UInt64) -> Self:
        """Members' bits from `width` up are `prefix`'s."""
        return Self(id, 0, width, prefix)

    @staticmethod
    @always_inline
    def copies(*, start: Int, low: Int32, width: Int32) -> Self:
        """Copies the words from `start` on, once for each other value of the free bits
        in `[low, width)` (clear in those words)."""
        return Self(REJECT, low, width, UInt64(start))


def write_words[
    word: DType
](
    mut arena: Arena,
    id: Id,
    width: Int32,
    slots: Slots[word],
    *,
    sign_shift: UInt64,
) raises Interrupted:
    """For widths up to 64. Members under free bits are written once, then copied for
    each other value of the free bits."""
    var steps: List[WriteStep] = []
    if id != REJECT:
        steps.append(WriteStep.visit(id, width, 0))
    var written = 0
    while steps:
        var step = steps.pop()
        if step.id == REJECT:
            written += write_copies(
                arena,
                slots,
                start=Int(step.at),
                written=written,
                low=step.low,
                width=step.width,
                sign_shift=sign_shift,
            )
            continue
        var first, free = descend(
            arena, steps, step.id, step.width, step.at, block_start=written
        )
        var run = 1 << Int(free)
        fill_run(
            arena,
            slots.unsafe_offset(written),
            count=run,
            first=first,
            sign_shift=sign_shift,
        )
        written += run


def descend(
    arena: Arena,
    mut steps: List[WriteStep],
    var id: Id,
    var width: Int32,
    var prefix: UInt64,
    *,
    block_start: Int,
) -> Tuple[UInt64, Int32]:
    """The first run under `id` (not REJECT), as its first member and free bits. Pushes
    what it passes:
      - a visit for each `hi` branch it doesn't take
      - copies for the free bits above each node
    """
    while id != ACCEPT:
        var node = arena.decision(id)
        if node.bit + 1 < width:
            steps.append(
                WriteStep.copies(
                    start=block_start, low=node.bit + 1, width=width
                )
            )
        if node.lo != REJECT and node.hi != REJECT:
            var hi_prefix = prefix | UInt64(1) << UInt64(node.bit)
            steps.append(WriteStep.visit(node.hi, node.bit, hi_prefix))
            id = node.lo
        else:
            prefix |= UInt64(node.lo == REJECT) << UInt64(node.bit)
            id = node.sole_branch()
        width = node.bit
    return (prefix, width)


def write_copies[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    *,
    start: Int,
    written: Int,
    low: Int32,
    width: Int32,
    sign_shift: UInt64,
) raises Interrupted -> Int:
    """Returns how many words it wrote."""
    var block = slots.unsafe_offset(start)
    var size = written - start
    var first = member_of(block[], sign_shift=sign_shift)
    var last = member_of(block[unsafe_offset=size - 1], sign_shift=sign_shift)
    # Writing a run afresh saves reading the block back once it's out of the cache.
    var one_run = last - first == UInt64(size - 1)
    var copies = (1 << Int(width - low)) - 1
    for value in range(1, copies + 1):
        var copy = block.unsafe_offset(value * size)
        var high_bits = UInt64(value) << UInt64(low)
        if one_run:
            fill_run(
                arena,
                copy,
                count=size,
                first=first | high_bits,
                sign_shift=sign_shift,
            )
        else:
            fill_copy(
                arena,
                copy,
                block,
                count=size,
                high_bits=high_bits,
                sign_shift=sign_shift,
            )
    return copies * size


def fill_run[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    *,
    count: Int,
    first: UInt64,
    sign_shift: UInt64,
) raises Interrupted:
    var next = first
    for start in range(0, count, CHUNK):
        arena.tick()
        for at in range(start, min(start + CHUNK, count)):
            slots[unsafe_offset=at] = word_of[word](next, sign_shift=sign_shift)
            next += 1  # past UInt64.MAX, which only the last slot can hold


def fill_copy[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    block: Slots[word],
    *,
    count: Int,
    high_bits: UInt64,
    sign_shift: UInt64,
) raises Interrupted:
    for start in range(0, count, CHUNK):
        arena.tick()
        for at in range(start, min(start + CHUNK, count)):
            var old = member_of(block[unsafe_offset=at], sign_shift=sign_shift)
            slots[unsafe_offset=at] = word_of[word](
                old | high_bits, sign_shift=sign_shift
            )


@always_inline
def word_of[word: DType](member: UInt64, *, sign_shift: UInt64) -> Scalar[word]:
    """Shifting up and back down as an Int64 sign-extends a signed member."""
    return (Int64(member << sign_shift) >> Int64(sign_shift)).cast[word]()


@always_inline
def member_of[
    word: DType
](value: Scalar[word], *, sign_shift: UInt64) -> UInt64:
    return value.cast[DType.uint64]() << sign_shift >> sign_shift


def write_rows[
    V: Prefix, word: DType
](
    mut arena: Arena, id: Id, width: Int32, columns: Int, slots: Slots[word]
) raises Interrupted:
    """For members wider than a word, as rows of words, most significant first.
    """
    comptime bits = bit_width_of[word]()
    var member = List[UInt64](length=ceildiv(columns * bits, 64), fill=0)
    var walk = Walk[V](id, width)
    var row = slots
    while True:
        var run = walk.next_run(arena)
        if run is None:
            return
        # Copied once a run since calling `limb` for each member is slower.
        for i in range(len(member)):
            member.unsafe_get(i) = run.value().first.limb(i)
        var low = member[0]
        for n in range(1 << Int(run.value().free)):
            arena.tick()
            member.unsafe_get(0) = low | UInt64(n)
            for column in range(columns):
                var at = (columns - 1 - column) * bits
                var limb = member.unsafe_get(at >> 6) >> UInt64(at & 63)
                row[unsafe_offset=column] = limb.cast[word]()
            row = row.unsafe_offset(columns)
