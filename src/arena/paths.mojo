"""Walks over the paths from a node to ACCEPT."""

from .arena import ACCEPT, Arena, Id, REJECT

comptime ZERO = UInt8(ord("0"))
comptime ONE = UInt8(ord("1"))
comptime FREE = UInt8(ord("?"))


struct Paths(Movable):
    """Paths are written most significant bit first, with `?` for free bits."""

    var bits: List[UInt8]
    var start: Optional[Id]
    """The root, until the walk starts."""
    var frames: List[Tuple[Id, Int32]]
    """hi branches to come back to, with their parents' bits."""
    var width: Int32

    def __init__(out self, id: Id, width: Int32):
        self.bits = []
        self.start = id
        self.frames = []
        self.width = width

    def next_path(mut self, arena: Arena) -> Bool:
        """Writes the next path to `bits`. False after the last."""
        while True:
            var id: Id
            var width: Int32
            if self.start is not None:
                id = self.start.take()
                width = self.width
            elif self.frames:
                var hi, bit = self.frames.pop()
                self.bits.shrink(Int(self.width - 1 - bit))
                self.bits.append(ONE)
                id = hi
                width = bit
            else:
                return False
            if self.descend(arena, id, width):
                return True

    def descend(mut self, arena: Arena, var id: Id, var width: Int32) -> Bool:
        """Follows `lo` branches to a leaf. Returns whether it's ACCEPT."""
        while not id.is_leaf():
            var node = arena.decision(id)
            self.bits.resize(len(self.bits) + Int(width - 1 - node.bit), FREE)
            if node.hi != REJECT:
                self.frames.append((node.hi, node.bit))
            self.bits.append(ZERO)
            id = node.lo
            width = node.bit
        # A path to REJECT is cut back before it's written out.
        if id == ACCEPT:
            self.bits.resize(len(self.bits) + Int(width), FREE)
        return id == ACCEPT
