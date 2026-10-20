"""Walks keep their own stacks instead of recursing, since a path can test up to
MAX_WIDTH bits."""

from .arena import Arena, Id
from .table import Table


comptime SPARE_ROOM = 1 << 12
"""Room a spare stack keeps between walks."""

comptime VISIT: Int32 = -1


@fieldwise_init
struct Step[T: TrivialRegisterPassable](
    ImplicitlyCopyable, TrivialRegisterPassable
):
    """A walk pushes a make and then a visit for each branch. The visits pop first, so
    the make finds the branches' results on top of the results stack."""

    var bit: Int32
    """VISIT, or the bit of the node to make."""
    var item: Self.T

    @staticmethod
    @always_inline
    def visit(item: Self.T) -> Self:
        return Self(VISIT, item)

    @staticmethod
    @always_inline
    def make(bit: Int32, item: Self.T) -> Self:
        return Self(bit, item)

    @always_inline
    def is_visit(self) -> Bool:
        return self.bit == VISIT


@fieldwise_init
struct Pair(ImplicitlyCopyable, TrivialRegisterPassable):
    var a: Id
    var b: Id


@fieldwise_init
struct Span(ImplicitlyCopyable, TrivialRegisterPassable):
    """`values[start:end]`, which share every bit above `bit`."""

    var start: Int
    var end: Int
    var bit: Int32


struct Spare(Movable):
    """Stacks kept between walks, so short walks allocate nothing. A walk nested in
    another (eg. the `|` in freed_between) finds them taken and allocates its own.
    """

    var pair_steps: List[Step[Pair]]
    var node_steps: List[Step[Id]]
    var span_steps: List[Step[Span]]
    var results: List[Id]
    var node_results: Table
    var pending_pairs: List[Pair]
    var seen_pairs: Table

    def __init__(out self):
        self.pair_steps = List[Step[Pair]]()
        self.node_steps = List[Step[Id]]()
        self.span_steps = List[Step[Span]]()
        self.results = List[Id]()
        self.node_results = Table()
        self.pending_pairs = List[Pair]()
        self.seen_pairs = Table()


def emptied[T: Copyable & Deinitable](var stack: List[T]) -> List[T]:
    """`stack` with nothing in it, keeping at most SPARE_ROOM of room."""
    if stack.capacity() > SPARE_ROOM:
        return List[T](capacity=SPARE_ROOM)
    stack.clear()
    return stack^


def emptied(var table: Table) -> Table:
    if table.capacity() > SPARE_ROOM:
        return Table()
    table.clear()
    return table^


@always_inline
def popped_branches(mut results: List[Id]) -> Tuple[Id, Id]:
    var hi = results.pop()
    var lo = results.pop()
    return (lo, hi)


@always_inline
def push_branches(arena: Arena, mut steps: List[Step[Id]], id: Id):
    var node = arena.decision(id)
    steps.append(Step[Id].make(node.bit, id))
    steps.append(Step[Id].visit(node.hi))
    steps.append(Step[Id].visit(node.lo))
