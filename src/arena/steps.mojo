"""Walks keep their own stacks instead of recursing since a path can test up to
MAX_WIDTH bits."""

from .arena import Arena, Id
from .table import Table


comptime SCRATCH_ROOM = 1 << 12
"""Scratch stacks and tables keep up to this much room between walks."""

comptime VISIT: Int32 = -1


@fieldwise_init
struct Step[T: TrivialRegisterPassable](TrivialRegisterPassable):
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
struct Pair(TrivialRegisterPassable):
    var a: Id
    var b: Id


@fieldwise_init
struct Span(TrivialRegisterPassable):
    """`values[start:end]`, which share every bit above `bit`."""

    var start: Int
    var end: Int
    var bit: Int32


struct Scratch(Defaultable, Movable):
    """Stacks and tables kept between walks, so short walks allocate nothing. Each walk
    takes the ones it uses while it runs. A nested walk (eg. the `|` in freed_between)
    finds them taken and allocates its own.
    """

    var pair_steps: List[Step[Pair]]
    var node_steps: List[Step[Id]]
    var span_steps: List[Step[Span]]
    var results: List[Id]
    var node_results: Table
    var pending_pairs: List[Pair]
    var seen_pairs: Table


def take[T: Movable & Defaultable](mut value: T) -> T:
    """`value`, leaving `T()` in its place."""
    var taken = T()
    swap(taken, value)
    return taken^


def emptied[T: Copyable & Deinitable](var stack: List[T]) -> List[T]:
    """`stack` with nothing in it, keeping at most SCRATCH_ROOM of room."""
    if stack.capacity() > SCRATCH_ROOM:
        return List[T](capacity=SCRATCH_ROOM)
    stack.clear()
    return stack^


def emptied(var table: Table) -> Table:
    if table.capacity() > SCRATCH_ROOM:
        return Table()
    table.clear()
    return table^


@always_inline
def pop_branches(mut results: List[Id]) -> Tuple[Id, Id]:
    var hi = results.pop()
    var lo = results.pop()
    return (lo, hi)


@always_inline
def push_branches(arena: Arena, mut steps: List[Step[Id]], id: Id):
    var node = arena.decision(id)
    steps.append(Step.make(node.bit, id))
    steps.append(Step.visit(node.hi))
    steps.append(Step.visit(node.lo))
