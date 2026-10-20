"""Open-addressing hash tables keyed by node ids.

Both tables probe linearly in a power-of-two array, and grow before they're 3/4 full.
"""

from std.bit import count_leading_zeros

from .arena import Decision, Id, REJECT


comptime EMPTY: UInt64 = 0
comptime MIN_SLOTS = 8


@always_inline
def mixed(key: UInt64) -> UInt64:
    # Fibonacci hashing: the multiply spreads sequential ids over the high bits.
    return key * 0x9E3779B97F4A7C15


def slots_for(entries: Int) -> Int:
    """The fewest slots that hold `entries` under the load limit."""
    if entries == 0:
        return 0
    var needed = max(MIN_SLOTS, (entries * 4 + 2) // 3)
    return 1 << (64 - Int(count_leading_zeros(UInt64(needed - 1))))


@fieldwise_init
struct Entry(ImplicitlyCopyable, TrivialRegisterPassable):
    var key: UInt64
    var value: UInt32


struct Table(Movable, Sized):
    """Maps nonzero UInt64 keys (an id, or two packed by `pair`) to UInt32 values.
    """

    var entries: List[Entry]
    var held: Int

    def __init__(out self):
        self.entries = List[Entry]()
        self.held = 0

    def __len__(self) -> Int:
        return self.held

    def capacity(self) -> Int:
        """How many entries fit before the table grows."""
        return len(self.entries) * 3 // 4

    @always_inline
    def _slot(self, key: UInt64) -> Int:
        var bits = count_leading_zeros(UInt64(len(self.entries)))
        return Int(mixed(key) >> (bits + 1))

    def get(self, key: UInt64) -> Optional[UInt32]:
        if self.held == 0:
            return None
        var mask = len(self.entries) - 1
        var slot = self._slot(key)
        while True:
            var entry = self.entries.unsafe_get(slot)
            if entry.key == key:
                return entry.value
            if entry.key == EMPTY:
                return None
            slot = (slot + 1) & mask

    def insert(mut self, key: UInt64, value: UInt32):
        if (self.held + 1) * 4 > len(self.entries) * 3:
            self._resize(
                slots_for(self.held + 1) * 2 if self.held else MIN_SLOTS
            )
        var mask = len(self.entries) - 1
        var slot = self._slot(key)
        while True:
            ref entry = self.entries.unsafe_get(slot)
            if entry.key == key:
                entry.value = value
                return
            if entry.key == EMPTY:
                entry = Entry(key, value)
                self.held += 1
                return
            slot = (slot + 1) & mask

    def add(mut self, key: UInt64) -> Bool:
        """Adds `key` to a table used as a set. Returns whether it's new."""
        if self.get(key):
            return False
        self.insert(key, 1)
        return True

    def _resize(mut self, slots: Int):
        var old = List[Entry](length=slots, fill=Entry(EMPTY, 0))
        swap(old, self.entries)
        var mask = slots - 1
        for entry in old:
            if entry.key == EMPTY:
                continue
            var slot = self._slot(entry.key)
            while self.entries.unsafe_get(slot).key != EMPTY:
                slot = (slot + 1) & mask
            self.entries.unsafe_set(slot, entry)

    def clear(mut self):
        """Keeps room for twice what it held, so similar walks needn't grow it.
        """
        var room = min(len(self.entries), slots_for(2 * self.held))
        self.held = 0
        if room == len(self.entries):
            for i in range(len(self.entries)):
                self.entries.unsafe_set(i, Entry(EMPTY, 0))
        else:
            self.entries = List[Entry](length=room, fill=Entry(EMPTY, 0))

    def shrink_to_fit(mut self):
        if slots_for(self.held) < len(self.entries):
            self._resize(slots_for(self.held))


@always_inline
def pair(a: Id, b: Id) -> UInt64:
    return UInt64(a.value) << 32 | UInt64(b.value)


@always_inline
def sorted_pair(a: Id, b: Id) -> UInt64:
    """Symmetric operations' key, so `(a, b)` and `(b, a)` share an entry."""
    return pair(a, b) if a.value <= b.value else pair(b, a)


@always_inline
def single(a: Id) -> UInt64:
    return UInt64(a.value)


@always_inline
def node_hash(node: Decision) -> UInt64:
    var key = pair(node.lo, node.hi) ^ (
        UInt64(UInt32(node.bit)) * 0xC2B2AE3D27D4EB4F
    )
    return mixed(key)


@fieldwise_init
struct Unique(ImplicitlyCopyable, TrivialRegisterPassable):
    """A slot of the unique table. `lo == hi` (no node's branches are equal) is empty.
    """

    var node: Decision
    var id: Id


struct UniqueTable(Movable, Sized):
    """Maps each decision node to its id."""

    var slots: List[Unique]
    var held: Int

    def __init__(out self):
        self.slots = List[Unique]()
        self.held = 0

    def __len__(self) -> Int:
        return self.held

    def capacity(self) -> Int:
        return len(self.slots) * 3 // 4

    @always_inline
    def _slot(self, node: Decision) -> Int:
        var bits = count_leading_zeros(UInt64(len(self.slots)))
        return Int(node_hash(node) >> (bits + 1))

    def get(self, node: Decision) -> Optional[Id]:
        if self.held == 0:
            return None
        var mask = len(self.slots) - 1
        var slot = self._slot(node)
        while True:
            var unique = self.slots.unsafe_get(slot)
            if unique.node == node:
                return unique.id
            if unique.node.lo == unique.node.hi:
                return None
            slot = (slot + 1) & mask

    def insert(mut self, node: Decision, id: Id):
        """`node` isn't in the table."""
        if (self.held + 1) * 4 > len(self.slots) * 3:
            self._resize(
                slots_for(self.held + 1) * 2 if self.held else MIN_SLOTS
            )
        self._place(node, id)
        self.held += 1

    def _place(mut self, node: Decision, id: Id):
        var mask = len(self.slots) - 1
        var slot = self._slot(node)
        while (
            self.slots.unsafe_get(slot).node.lo
            != self.slots.unsafe_get(slot).node.hi
        ):
            slot = (slot + 1) & mask
        self.slots.unsafe_set(slot, Unique(node, id))

    def remove(mut self, node: Decision):
        """Backward-shift deletion, so lookups never need tombstones."""
        var mask = len(self.slots) - 1
        var slot = self._slot(node)
        while self.slots.unsafe_get(slot).node != node:
            slot = (slot + 1) & mask
        var hole = slot
        while True:
            slot = (slot + 1) & mask
            var moving = self.slots.unsafe_get(slot)
            if moving.node.lo == moving.node.hi:
                break
            # Move it into the hole unless its home slot lies in (hole, slot].
            var home = self._slot(moving.node)
            if (slot - home) & mask >= (slot - hole) & mask:
                self.slots.unsafe_set(hole, moving)
                hole = slot
        self.slots.unsafe_set(hole, Unique(Decision(0, REJECT, REJECT), REJECT))
        self.held -= 1

    def _resize(mut self, slots: Int):
        var old = List[Unique](
            length=slots, fill=Unique(Decision(0, REJECT, REJECT), REJECT)
        )
        swap(old, self.slots)
        for unique in old:
            if unique.node.lo != unique.node.hi:
                self._place(unique.node, unique.id)

    def shrink_to(mut self, entries: Int):
        """Keeps room for at least `entries`."""
        var slots = slots_for(max(entries, self.held))
        if slots < len(self.slots):
            self._resize(slots)
