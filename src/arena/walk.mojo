"""Walks over a set's members in order, a run at a time."""

from bigint import BigUInt

from .arena import Arena, Id, REJECT
from .steps import VISIT


trait Prefix(Copyable, Defaultable, Deinitable):
    """Narrow for narrow sets, or else Wide. Starts at zero."""

    def bit(self, bit: Int32) -> Bool:
        ...

    def set(mut self, bit: Int32, value: Bool):
        ...

    def cleared_below(self, width: Int32) -> Self:
        ...

    def plus(self, n: UInt64) -> Self:
        ...

    def limb(self, i: Int) -> UInt64:
        """Bits `[64 * i, 64 * i + 64)`."""
        ...


@fieldwise_init
struct Narrow(Prefix, TrivialRegisterPassable):
    var value: UInt128

    @always_inline
    def bit(self, bit: Int32) -> Bool:
        return (self.value >> UInt128(bit)) & 1 == 1

    @always_inline
    def set(mut self, bit: Int32, value: Bool):
        var flag = UInt128(1) << UInt128(bit)
        self.value = self.value | flag if value else self.value & ~flag

    @always_inline
    def cleared_below(self, width: Int32) -> Self:
        # Shifting a UInt128 by 128 isn't defined.
        if width >= 128:
            return Self()
        return Self(self.value >> UInt128(width) << UInt128(width))

    @always_inline
    def plus(self, n: UInt64) -> Self:
        return Self(self.value + UInt128(n))

    @always_inline
    def limb(self, i: Int) -> UInt64:
        return UInt64(self.value >> UInt128(64 * i)) if i < 2 else 0


@fieldwise_init
struct Wide(Copyable, Prefix):
    var value: BigUInt

    def bit(self, bit: Int32) -> Bool:
        return self.value.bit(bit)

    def set(mut self, bit: Int32, value: Bool):
        self.value.set_bit(bit, value)

    def cleared_below(self, width: Int32) -> Self:
        return Self((self.value >> width) << width)

    def plus(self, n: UInt64) -> Self:
        return Self(self.value.plus(n))

    def limb(self, i: Int) -> UInt64:
        return self.value.limb(i)


@fieldwise_init
struct Run[V: Prefix](Copyable):
    """The `2 ** free` integers from `first` on (an aligned block)."""

    var first: Self.V
    var free: Int32


comptime HI_BRANCH: Int32 = -2


@fieldwise_init
struct Frame(TrivialRegisterPassable):
    var id: Id
    var low: Int32
    """VISIT, HI_BRANCH, or the lowest free bit."""
    var width: Int32

    @staticmethod
    @always_inline
    def visit(id: Id, width: Int32) -> Self:
        return Self(id, VISIT, width)

    @staticmethod
    @always_inline
    def hi_branch(id: Id, bit: Int32) -> Self:
        """Sets `bit` and walks `id`, the hi branch of a node that tests it."""
        return Self(id, HI_BRANCH, bit)

    @staticmethod
    @always_inline
    def free_bits(id: Id, *, low: Int32, width: Int32) -> Self:
        """Counts the free bits in `[low, width)` up by one and walks `id` again, until
        they're all set."""
        return Self(id, low, width)


struct Walk[V: Prefix](Movable):
    """Builds each member in `prefix`, so a walk needs one member's memory and a frame
    per bit."""

    var prefix: Self.V
    var frames: List[Frame]

    def __init__(out self, id: Id, width: Int32):
        self.prefix = Self.V()
        self.frames = [Frame.visit(id, width)]

    def next_run(mut self, arena: Arena) -> Optional[Run[Self.V]]:
        while self.frames:
            var frame = self.frames.pop()
            var width = frame.width
            if frame.low == HI_BRANCH:
                self.prefix.set(frame.width, True)
            elif frame.low != VISIT:
                if not count_up(self.prefix, low=frame.low, high=frame.width):
                    continue
                self.frames.append(frame)
                width = frame.low
            var free = self.descend(arena, frame.id, width)
            if free is not None:
                return Run(
                    self.prefix.cleared_below(free.value()), free.value()
                )
        return None

    def descend(
        mut self, arena: Arena, var id: Id, var width: Int32
    ) -> Optional[Int32]:
        """Follows `lo` branches to a leaf, pushing frames for what it passes. Returns
        the free bits below the last node, or None at REJECT."""
        while not id.is_leaf():
            var node = arena.decision(id)
            if node.bit + 1 < width:
                for bit in range(node.bit + 1, width):
                    self.prefix.set(bit, False)
                self.frames.append(
                    Frame.free_bits(id, low=node.bit + 1, width=width)
                )
            if node.hi != REJECT:
                self.frames.append(Frame.hi_branch(node.hi, node.bit))
            self.prefix.set(node.bit, False)
            id = node.lo
            width = node.bit
        if id == REJECT:
            return None
        return width


def count_up[V: Prefix](mut prefix: V, *, low: Int32, high: Int32) -> Bool:
    """Adds one to `prefix`'s bits in `[low, high)`. False if they overflow (to all
    clear)."""
    for bit in range(low, high):
        var set = prefix.bit(bit)
        prefix.set(bit, not set)
        if not set:
            return True
    return False
