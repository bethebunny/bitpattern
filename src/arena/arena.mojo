"""Every diagram's nodes, and the operations every algorithm on them shares.

Terms:
  - a node's _members_ are the integers it accepts
  - a _decision node_ tests one bit and follows `lo` if it's clear or `hi` if it's set
  - a _set_ is a node's members below `2 ** width`
  - a bit is _free_ on a path if no node on the path tests it

Mojo keeps a struct's methods in its definition, so the algorithms are functions of an
`Arena` in this package's other modules.
"""

from bigint import BigUInt

from .collect import MIN_COLLECTION_INTERVAL, Schedule
from .memos import Memos
from .steps import Scratch
from .table import UniqueTable


@fieldwise_init
struct Id(Equatable, TrivialRegisterPassable, Writable):
    """A node's index in the arena."""

    var value: UInt32

    @staticmethod
    def from_index(index: Int) -> Id:
        return Id(UInt32(index))

    @always_inline
    def index(self) -> Int:
        return Int(self.value)

    @always_inline
    def is_leaf(self) -> Bool:
        return self.value < LEAVES


comptime REJECT = Id(0)
comptime ACCEPT = Id(1)
comptime LEAVES = 2
"""REJECT and ACCEPT, below every decision node's id."""

comptime NARROW_WIDTH: Int32 = 128
"""Sets up to this width work in UInt128s rather than BigUInts. Their members, counts and
node sizes all fit (except ACCEPT's count at width 128)."""

comptime MAX_WIDTH: Int32 = Int32.MAX
comptime MAX_BIT: Int32 = MAX_WIDTH - 1

comptime PAUSE_INTERVAL = 1 << 14
"""Steps between a walk's checks for Ctrl-C."""


@fieldwise_init
struct Decision(Equatable, TrivialRegisterPassable):
    """Invariants: `lo != hi`, and both test bits below `bit`."""

    var bit: Int32
    var lo: Id
    var hi: Id

    @always_inline
    def sole_branch(self) -> Id:
        """The branch that isn't REJECT."""
        # REJECT is 0, so an OR has no jump to mispredict.
        return Id(self.lo.value | self.hi.value)


comptime LEAF_SLOT = Decision(-1, REJECT, REJECT)
"""Fills the leaves' unused slots."""


@fieldwise_init
struct Interrupted(TrivialRegisterPassable, Writable):
    """A pause stopped the walk (Python's Ctrl-C handler raised)."""

    pass


@fieldwise_init
struct Misordered(TrivialRegisterPassable, Writable):
    """A node's branch tests `child_bit`, which isn't below the node's `bit`."""

    var bit: Int32
    var child_bit: Int32

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            t"children must test bits below {self.bit}, not {self.child_bit}"
        )


struct Sizes(Movable):
    """How many integers below `2 ** (bit + 1)` each decision node accepts."""

    var narrow: List[UInt128]
    """Nodes below NARROW_WIDTH, by index."""
    var wide: Dict[Int, BigUInt]
    """The rest, by index."""

    def __init__(out self, slots: Int):
        self.narrow = List[UInt128](length=slots, fill=0)
        self.wide = Dict[Int, BigUInt]()


comptime PauseStatePointer = OpaquePointer[MutUntrackedOrigin]
comptime Pause = def(mut Arena, PauseStatePointer) thin -> Bool
"""Runs every PAUSE_INTERVAL steps of a walk with the arena's `pause_state` and
returns whether the walk goes on."""


def never_pause(mut arena: Arena, pause_state: PauseStatePointer) -> Bool:
    return True


struct Arena(Movable):
    """No two nodes share a bit and branches, so two sets are equal exactly when their
    roots are."""

    var nodes: List[Decision]
    var sizes: Sizes
    var unique: UniqueTable
    var free: List[Id]
    """Freed slots, lowest last so it's reused first."""
    var memos: Memos
    var scratch: Scratch
    var schedule: Schedule
    var pause: Pause
    var pause_state: PauseStatePointer
    var countdown: Int
    """Steps until the next pause."""

    def __init__(out self, pause: Pause, pause_state: PauseStatePointer):
        self.nodes = List[Decision](length=LEAVES, fill=LEAF_SLOT)
        self.sizes = Sizes(LEAVES)
        self.unique = UniqueTable()
        self.free = []
        self.memos = Memos()
        self.scratch = Scratch()
        self.schedule = Schedule(made=0, due=MIN_COLLECTION_INTERVAL)
        self.pause = pause
        self.pause_state = pause_state
        self.countdown = PAUSE_INTERVAL

    @staticmethod
    def placeholder() -> Arena:
        """No nodes at all, not even the leaves. A paused walk holds this while Python's
        Ctrl-C handler has the real arena."""
        var arena = Arena(never_pause, PauseStatePointer.unsafe_dangling())
        arena.nodes = []
        arena.sizes = Sizes(0)
        return arena^

    @always_inline
    def decision(self, id: Id) -> Decision:
        assert not id.is_leaf(), "leaves aren't decision nodes"
        return self.nodes.unsafe_get(id.index())

    @always_inline
    def tests_only_below(self, id: Id, bit: Int32) -> Bool:
        return id.is_leaf() or self.decision(id).bit < bit

    def make(mut self, bit: Int32, lo: Id, hi: Id) -> Id:
        if lo == hi:
            return lo
        assert self.tests_only_below(lo, bit) and self.tests_only_below(hi, bit)
        var node = Decision(bit, lo, hi)
        var found = self.unique.get(node)
        if found is not None:
            return found.value()
        var id = self._allocate(node)
        if bit < NARROW_WIDTH:
            var size = self.count_narrow(lo, bit) + self.count_narrow(hi, bit)
            self.sizes.narrow.unsafe_set(id.index(), size)
        else:
            self.sizes.wide[id.index()] = self.count(lo, bit) + self.count(
                hi, bit
            )
        self.unique.insert(node, id)
        self.schedule.made += 1
        return id

    def _allocate(mut self, node: Decision) -> Id:
        if self.free:
            var id = self.free.pop()
            self.nodes.unsafe_set(id.index(), node)
            return id
        self.nodes.append(node)
        self.sizes.narrow.append(0)
        return Id.from_index(len(self.nodes) - 1)

    def make_checked(
        mut self, bit: Int32, lo: Id, hi: Id
    ) raises Misordered -> Id:
        if lo == hi:
            return lo
        for child in [lo, hi]:
            if not self.tests_only_below(child, bit):
                raise Misordered(bit=bit, child_bit=self.decision(child).bit)
        return self.make(bit, lo, hi)

    @always_inline
    def cofactors(self, id: Id, bit: Int32) -> Tuple[Id, Id]:
        """`id`'s branches on `bit`, which is at least `id`'s bit. A node that doesn't
        test `bit` is its own branch either way."""
        var node = self.decision(id)
        if node.bit == bit:
            return (node.lo, node.hi)
        return (id, id)

    @always_inline
    def top_bit(self, a: Id, b: Id) -> Int32:
        return max(self.decision(a).bit, self.decision(b).bit)

    @always_inline
    def tick(mut self) raises Interrupted:
        """Pauses every PAUSE_INTERVAL steps. A pause can run other walks but never a
        collection, so it frees no node the walk holds."""
        self.countdown -= 1
        if self.countdown > 0:
            return
        self.countdown = PAUSE_INTERVAL
        if not self.pause(self, self.pause_state):
            raise Interrupted()

    def size(self, id: Id) -> BigUInt:
        if self.decision(id).bit < NARROW_WIDTH:
            return BigUInt(self.sizes.narrow.unsafe_get(id.index()))
        return self.sizes.wide.get(id.index()).value().copy()

    @always_inline
    def count_narrow(self, id: Id, width: Int32) -> UInt128:
        """The count must fit a UInt128 (so not ACCEPT's at width 128)."""
        if id == REJECT:
            return 0
        if id == ACCEPT:
            return UInt128(1) << UInt128(width)
        # Each free bit above the node doubles its size.
        var shift = UInt128(width - 1 - self.decision(id).bit)
        return self.sizes.narrow.unsafe_get(id.index()) << shift

    def count(self, id: Id, width: Int32) -> BigUInt:
        if id == REJECT:
            return BigUInt()
        if id == ACCEPT:
            return BigUInt(1) << width
        return self.size(id) << (width - 1 - self.decision(id).bit)
