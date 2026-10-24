"""Hash tables keyed by node ids, laid out as Swiss tables.

Each slot has a control byte (EMPTY, DELETED, or 7 bits of its key's hash) in a separate
array, and a probe matches GROUP of them at once. A miss reads only control bytes, which
stay in cache when the slots don't.
"""

from std.bit import count_trailing_zeros
from std.memory import pack_bits

from .arena import Decision, Id


comptime GROUP = 16
comptime MIN_SLOTS = GROUP
comptime EMPTY: UInt8 = 0xFF
comptime DELETED: UInt8 = 0x80
comptime Group = SIMD[DType.uint8, GROUP]


@always_inline
def mixed(key: UInt64) -> UInt64:
    # Fibonacci hashing spreads sequential ids over the high bits.
    return key * 0x9E3779B97F4A7C15


def room(slots: Int) -> Int:
    """How many entries fit in `slots` before the table grows."""
    # Probes match a group at a time, so they stay short up to 7/8 full.
    return slots * 7 // 8


def slots_for(entries: Int) -> Int:
    """The fewest slots with room for `entries`."""
    if entries == 0:
        return 0
    var slots = MIN_SLOTS
    while room(slots) < entries:
        slots *= 2
    return slots


@always_inline
def matches(group: Group, byte: UInt8) -> UInt16:
    return pack_bits(group.eq(Group(byte)))


@always_inline
def free_in(group: Group) -> UInt16:
    """EMPTY or DELETED, which have the high bit set."""
    return pack_bits((group & Group(0x80)).eq(Group(0x80)))


@always_inline
def lowest(bits: UInt16) -> Int:
    return Int(count_trailing_zeros(bits))


struct Control(Movable):
    """A byte for each slot, then a copy of the first GROUP, so a group can start at any
    slot."""

    var bytes: List[UInt8]
    var shift: UInt64

    def __init__(out self, slots: Int):
        var count = slots + GROUP if slots else 0
        self.bytes = List[UInt8](length=count, fill=EMPTY)
        self.shift = 64 - UInt64(count_trailing_zeros(slots)) if slots else 64

    @always_inline
    def home(self, hash: UInt64) -> Int:
        return Int(hash >> self.shift)

    @always_inline
    def byte(self, hash: UInt64) -> UInt8:
        # Not the top bits, which keys with the same home share.
        return UInt8((hash >> (self.shift - 7)) & 0x7F)

    @always_inline
    def group(self, slot: Int) -> Group:
        return self.bytes.unsafe_ptr().unsafe_load[width=GROUP](slot)

    @always_inline
    def set(mut self, slot: Int, byte: UInt8):
        self.bytes.unsafe_set(slot, byte)
        if slot < GROUP:
            self.bytes.unsafe_set(len(self.bytes) - GROUP + slot, byte)

    @always_inline
    def is_full(self, slot: Int) -> Bool:
        return self.bytes.unsafe_get(slot) < 0x80

    def empty_slot(self, hash: UInt64) -> Int:
        """The first EMPTY slot from `hash`'s home."""
        var mask = len(self.bytes) - GROUP - 1
        var slot = self.home(hash)
        while self.bytes.unsafe_get(slot) != EMPTY:
            slot = (slot + 1) & mask
        return slot

    def clear(mut self):
        Span(self.bytes).fill(EMPTY)


trait Key(Equatable, TrivialRegisterPassable):
    """How a table keeps its UInt64 keys."""

    def __init__(out self, key: UInt64):
        ...

    def hash(self) -> UInt64:
        ...


struct PairKey(Key):
    """In halves, so an entry takes 12 bytes rather than 16."""

    var high: UInt32
    var low: UInt32

    @always_inline
    def __init__(out self, key: UInt64):
        self.high = UInt32(key >> 32)
        self.low = UInt32(key & 0xFFFFFFFF)

    @always_inline
    def hash(self) -> UInt64:
        return mixed(UInt64(self.high) << 32 | UInt64(self.low))


struct IdKey(Key):
    var id: UInt32

    @always_inline
    def __init__(out self, key: UInt64):
        assert key >> 32 == 0, "an IdTable's keys are single ids"
        self.id = UInt32(key)

    @always_inline
    def hash(self) -> UInt64:
        return mixed(UInt64(self.id))


@fieldwise_init
struct Entry[K: Key](TrivialRegisterPassable):
    var key: Self.K
    var value: UInt32


struct HashTable[K: Key](Defaultable, Movable, Sized):
    """Maps UInt64 keys (an id, or two packed by `pair`) to UInt32 values."""

    var control: Control
    var entries: List[Entry[Self.K]]
    var held: Int

    def __init__(out self):
        self = Self(slots=0)

    def __init__(out self, slots: Int):
        self.control = Control(slots)
        self.entries = List[Entry[Self.K]](unsafe_uninit_length=slots)
        self.held = 0

    def __len__(self) -> Int:
        return self.held

    def capacity(self) -> Int:
        return room(len(self.entries))

    @always_inline
    def get(self, key: UInt64) -> Optional[UInt32]:
        if self.held == 0:
            return None
        var k = Self.K(key)
        var hash = k.hash()
        var byte = self.control.byte(hash)
        var mask = len(self.entries) - 1
        var entries = self.entries.unsafe_ptr()
        var start = self.control.home(hash)
        while True:
            var group = self.control.group(start)
            var found = matches(group, byte)
            while found:
                var slot = (start + lowest(found)) & mask
                var entry = entries[unsafe_offset=slot]
                if entry.key == k:
                    return entry.value
                found &= found - 1
            if matches(group, EMPTY):
                return None
            start = (start + GROUP) & mask

    # The walks of apply and ~ call it for each node they make.
    @always_inline
    def insert(mut self, key: UInt64, value: UInt32):
        if self.held >= room(len(self.entries)):
            self._resize(max(MIN_SLOTS, 2 * len(self.entries)))
        var k = Self.K(key)
        var hash = k.hash()
        var byte = self.control.byte(hash)
        var mask = len(self.entries) - 1
        var entries = self.entries.unsafe_ptr()
        var start = self.control.home(hash)
        while True:
            var group = self.control.group(start)
            var found = matches(group, byte)
            while found:
                var slot = (start + lowest(found)) & mask
                ref entry = entries[unsafe_offset=slot]
                if entry.key == k:
                    entry.value = value
                    return
                found &= found - 1
            var empty = matches(group, EMPTY)
            if empty:
                var slot = (start + lowest(empty)) & mask
                self.control.set(slot, byte)
                entries[unsafe_offset=slot] = Entry(k, value)
                self.held += 1
                return
            start = (start + GROUP) & mask

    def add(mut self, key: UInt64) -> Bool:
        """Adds `key` to a table used as a set. Returns whether it's new."""
        if self.get(key) is not None:
            return False
        self.insert(key, 1)
        return True

    @no_inline
    def _resize(mut self, slots: Int):
        var old = self^
        self = Self(slots)
        self.held = old.held
        for i in range(len(old.entries)):
            if old.control.is_full(i):
                var entry = old.entries.unsafe_get(i)
                var hash = entry.key.hash()
                var slot = self.control.empty_slot(hash)
                self.control.set(slot, self.control.byte(hash))
                self.entries.unsafe_set(slot, entry)

    def clear(mut self):
        """Keeps room for twice what it held, so similar walks needn't grow it.
        """
        var slots = min(len(self.entries), slots_for(2 * self.held))
        if slots < len(self.entries):
            self = Self(slots)
        else:
            self.control.clear()
            self.held = 0

    def shrink_to_fit(mut self):
        if slots_for(self.held) < len(self.entries):
            self._resize(slots_for(self.held))


comptime Table = HashTable[PairKey]
comptime IdTable = HashTable[IdKey]
"""A Table for keys from `single`, in 8-byte entries."""


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
struct Unique(TrivialRegisterPassable):
    var node: Decision
    var id: Id


struct UniqueTable(Movable, Sized):
    """Maps each decision node to its id."""

    var control: Control
    var entries: List[Unique]
    var held: Int
    var deleted: Int

    def __init__(out self, slots: Int = 0):
        self.control = Control(slots)
        self.entries = List[Unique](unsafe_uninit_length=slots)
        self.held = 0
        self.deleted = 0

    def __len__(self) -> Int:
        return self.held

    @always_inline
    def _find(self, node: Decision) -> Int:
        """`node`'s slot, or -1."""
        if self.held == 0:
            return -1
        var hash = node_hash(node)
        var byte = self.control.byte(hash)
        var mask = len(self.entries) - 1
        var entries = self.entries.unsafe_ptr()
        var start = self.control.home(hash)
        while True:
            var group = self.control.group(start)
            var found = matches(group, byte)
            while found:
                var slot = (start + lowest(found)) & mask
                if entries[unsafe_offset=slot].node == node:
                    return slot
                found &= found - 1
            if matches(group, EMPTY):
                return -1
            start = (start + GROUP) & mask

    @always_inline
    def get(self, node: Decision) -> Optional[Id]:
        var slot = self._find(node)
        if slot < 0:
            return None
        return self.entries.unsafe_get(slot).id

    def insert(mut self, node: Decision, id: Id):
        """`node` isn't in the table."""
        if self.held + self.deleted >= room(len(self.entries)):
            var slots = len(self.entries)
            # Dropping the DELETED slots makes room enough unless nodes fill half of it.
            var grow = self.held >= room(slots) // 2
            self._resize(max(MIN_SLOTS, 2 * slots) if grow else slots)
        var hash = node_hash(node)
        var mask = len(self.entries) - 1
        var start = self.control.home(hash)
        while True:
            var free = free_in(self.control.group(start))
            if free:
                var slot = (start + lowest(free)) & mask
                if self.control.bytes.unsafe_get(slot) == DELETED:
                    self.deleted -= 1
                self.control.set(slot, self.control.byte(hash))
                self.entries.unsafe_set(slot, Unique(node, id))
                self.held += 1
                return
            start = (start + GROUP) & mask

    def remove(mut self, node: Decision):
        """`node` is in the table."""
        var slot = self._find(node)
        assert slot >= 0, "removing a node the table doesn't hold"
        # DELETED rather than EMPTY, so lookups go on to the nodes placed past it.
        self.control.set(slot, DELETED)
        self.held -= 1
        self.deleted += 1

    @no_inline
    def _resize(mut self, slots: Int):
        var old = self^
        self = UniqueTable(slots)
        self.held = old.held
        for i in range(len(old.entries)):
            if old.control.is_full(i):
                var unique = old.entries.unsafe_get(i)
                var hash = node_hash(unique.node)
                var slot = self.control.empty_slot(hash)
                self.control.set(slot, self.control.byte(hash))
                self.entries.unsafe_set(slot, unique)

    def shrink_to(mut self, entries: Int):
        """Keeps room for at least `entries` and drops the DELETED slots."""
        var slots = min(len(self.entries), slots_for(max(entries, self.held)))
        if slots < len(self.entries) or self.deleted:
            self._resize(slots)
