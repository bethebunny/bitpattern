"""A set's members, written into a DLPack tensor's words."""

from std.math import ceildiv
from std.sys.info import bit_width_of

from .arena import ACCEPT, Arena, Id, Interrupted, REJECT


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
    def copies(start: Int, low: Int32, width: Int32) -> Self:
        """Copies the words from `start` on, once for each other value of the free bits
        in `[low, width)` (clear in those words)."""
        return Self(REJECT, low, width, UInt64(start))


def write_words[
    word: DType
](
    mut arena: Arena,
    id: Id,
    width: Int32,
    sign_shift: UInt64,
    slots: Slots[word],
) raises Interrupted:
    """For widths up to 64. Members under free bits are written once, then copied for
    each other value of the free bits."""
    var steps = List[WriteStep]()
    if id != REJECT:
        steps.append(WriteStep.visit(id, width, 0))
    var written = 0
    while steps:
        var step = steps.pop()
        if step.id == REJECT:
            written += write_copies(
                arena,
                slots,
                Int(step.at),
                written,
                step.low,
                step.width,
                sign_shift,
            )
            continue
        var first, free = descend(
            arena, steps, written, step.id, step.width, step.at
        )
        var run = 1 << Int(free)
        fill_run(arena, slots.unsafe_offset(written), run, first, sign_shift)
        written += run


def descend(
    arena: Arena,
    mut steps: List[WriteStep],
    block_start: Int,
    var id: Id,
    var width: Int32,
    var prefix: UInt64,
) -> Tuple[UInt64, Int32]:
    """The first run under `id` (not REJECT), as its first member and free bits. Pushes
    what it passes:
      - a visit for each `hi` branch it doesn't take
      - copies for the free bits above each node
    """
    while id != ACCEPT:
        var node = arena.decision(id)
        if node.bit + 1 < width:
            steps.append(WriteStep.copies(block_start, node.bit + 1, width))
        if node.lo != REJECT and node.hi != REJECT:
            var hi_prefix = prefix | UInt64(1) << UInt64(node.bit)
            steps.append(WriteStep.visit(node.hi, node.bit, hi_prefix))
            id = node.lo
        else:
            prefix |= UInt64(Int(node.lo == REJECT)) << UInt64(node.bit)
            id = node.sole_branch()
        width = node.bit
    return (prefix, width)


def write_copies[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    start: Int,
    written: Int,
    low: Int32,
    width: Int32,
    sign_shift: UInt64,
) raises Interrupted -> Int:
    """Returns how many words it wrote."""
    var block = slots.unsafe_offset(start)
    var size = written - start
    var first = member_of(block[], sign_shift)
    var last = member_of(block[unsafe_offset=size - 1], sign_shift)
    # Writing a run afresh saves reading the block back once it's out of the cache.
    var one_run = last - first == UInt64(size - 1)
    var copies = (1 << Int(width - low)) - 1
    for value in range(1, copies + 1):
        var copy = block.unsafe_offset(value * size)
        var high_bits = UInt64(value) << UInt64(low)
        if one_run:
            fill_run(arena, copy, size, first | high_bits, sign_shift)
        else:
            fill_copy(arena, copy, block, size, high_bits, sign_shift)
    return copies * size


def fill_run[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    count: Int,
    first: UInt64,
    sign_shift: UInt64,
) raises Interrupted:
    var next = first
    for start in range(0, count, CHUNK):
        arena.tick()
        for at in range(start, min(start + CHUNK, count)):
            slots[unsafe_offset=at] = word_of[word](next, sign_shift)
            next += 1  # past UInt64.MAX, which only the last slot can hold


def fill_copy[
    word: DType
](
    mut arena: Arena,
    slots: Slots[word],
    block: Slots[word],
    count: Int,
    high_bits: UInt64,
    sign_shift: UInt64,
) raises Interrupted:
    for start in range(0, count, CHUNK):
        arena.tick()
        for at in range(start, min(start + CHUNK, count)):
            var old = member_of(block[unsafe_offset=at], sign_shift)
            slots[unsafe_offset=at] = word_of[word](old | high_bits, sign_shift)


@always_inline
def word_of[word: DType](member: UInt64, sign_shift: UInt64) -> Scalar[word]:
    """Shifting up and back down as an Int64 sign-extends a signed member."""
    return (Int64(member << sign_shift) >> Int64(sign_shift)).cast[word]()


@always_inline
def member_of[word: DType](value: Scalar[word], sign_shift: UInt64) -> UInt64:
    return value.cast[DType.uint64]() << sign_shift >> sign_shift


def write_rows[
    word: DType
](
    mut arena: Arena, id: Id, width: Int32, columns: Int, slots: Slots[word]
) raises Interrupted:
    """For members wider than a word, as rows of words, most significant first.
    """
    comptime bits = bit_width_of[word]()
    var runs = Runs(id, width, columns * bits)
    var row = slots
    while runs.next(arena):
        var low = runs.member[0]
        for n in range(1 << Int(runs.free)):
            arena.tick()
            runs.member.unsafe_get(0) = low | UInt64(n)
            for column in range(columns):
                var at = (columns - 1 - column) * bits
                var limb = runs.member.unsafe_get(at >> 6) >> UInt64(at & 63)
                row[unsafe_offset=column] = limb.cast[word]()
            row = row.unsafe_offset(columns)


comptime VISIT: Int32 = 0
comptime HI_BRANCH: Int32 = 1
comptime FREE_BITS: Int32 = 2


@fieldwise_init
struct Frame(TrivialRegisterPassable):
    """Walks `id` below bit `low`: as it is (a visit), after setting bit `low` (a `hi`
    branch), or after counting up the free bits in `[low, width)`."""

    var kind: Int32
    var id: Id
    var low: Int32
    var width: Int32

    @staticmethod
    @always_inline
    def visit(id: Id, width: Int32) -> Self:
        return Self(VISIT, id, width, width)

    @staticmethod
    @always_inline
    def hi_branch(id: Id, bit: Int32) -> Self:
        return Self(HI_BRANCH, id, bit, bit)

    @staticmethod
    @always_inline
    def free_bits(id: Id, low: Int32, width: Int32) -> Self:
        """Counts the free bits in `[low, width)` up by one and walks `id` again, until
        they're all set."""
        return Self(FREE_BITS, id, low, width)


struct Runs(Movable):
    """A set's members in order, a run of `2 ** free` from `member` at a time. Builds
    each member in `member`, so a walk needs one member's memory and a frame per bit.
    """

    var member: List[UInt64]
    """Least significant limb first."""
    var free: Int32
    var frames: List[Frame]

    def __init__(out self, id: Id, width: Int32, bits: Int):
        """`bits` (at least `width`) is the room the members take."""
        self.member = List[UInt64](length=ceildiv(bits, 64), fill=0)
        self.free = 0
        self.frames = [Frame.visit(id, width)]

    def next(mut self, arena: Arena) -> Bool:
        """Moves on to the next run. False at the end."""
        while self.frames:
            var frame = self.frames.pop()
            if frame.kind == HI_BRANCH:
                self.set(Int(frame.low), True)
            elif frame.kind == FREE_BITS:
                if not self.counted_up(Int(frame.low), Int(frame.width)):
                    continue
                self.frames.append(frame)
            if self.descend(arena, frame.id, frame.low):
                return True
        return False

    def descend(mut self, arena: Arena, var id: Id, var width: Int32) -> Bool:
        """Follows `lo` branches to a leaf, pushing frames for what it passes. False at
        REJECT."""
        while not id.is_leaf():
            var node = arena.decision(id)
            if node.bit + 1 < width:
                self.clear(Int(node.bit) + 1, Int(width))
                self.frames.append(Frame.free_bits(id, node.bit + 1, width))
            if node.hi != REJECT:
                self.frames.append(Frame.hi_branch(node.hi, node.bit))
            self.set(Int(node.bit), False)
            id = node.lo
            width = node.bit
        if id == REJECT:
            return False
        self.clear(0, Int(width))
        self.free = width
        return True

    @always_inline
    def bit(self, bit: Int) -> Bool:
        return (self.member.unsafe_get(bit >> 6) >> UInt64(bit & 63)) & 1 == 1

    @always_inline
    def set(mut self, bit: Int, value: Bool):
        var flag = UInt64(1) << UInt64(bit & 63)
        if value:
            self.member.unsafe_get(bit >> 6) |= flag
        else:
            self.member.unsafe_get(bit >> 6) &= ~flag

    def clear(mut self, low: Int, high: Int):
        for bit in range(low, high):
            self.set(bit, False)

    def counted_up(mut self, low: Int, high: Int) -> Bool:
        """Adds one to the bits in `[low, high)`. False if they overflow to zero.
        """
        for bit in range(low, high):
            var set = self.bit(bit)
            self.set(bit, not set)
            if not set:
                return True
        return False
