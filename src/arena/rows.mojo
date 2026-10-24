"""The rows that pickles store, and diagrams rebuilt from them."""

from std.utils import Variant

from .arena import ACCEPT, Arena, Id, Interrupted, Misordered, REJECT
from .steps import Step, push_branches
from .table import Table, single


@fieldwise_init
struct Row(TrivialRegisterPassable):
    """A node's bit and branches. A branch is 0 for REJECT, 1 for ACCEPT, or `2 + i`
    for row `i`."""

    var bit: Int32
    var lo: UInt32
    var hi: UInt32


def rows(mut arena: Arena, root: Id) raises Interrupted -> List[Row]:
    """Children first."""
    var numbers = Table()
    var found: List[Row] = []
    var steps: List[Step[Id]] = [Step.visit(root)]
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            if (
                not step.item.is_leaf()
                and numbers.get(single(step.item)) is None
            ):
                push_branches(arena, steps, step.item)
        else:
            var node = arena.decision(step.item)
            found.append(
                Row(
                    step.bit, number(numbers, node.lo), number(numbers, node.hi)
                )
            )
            numbers.insert(single(step.item), UInt32(len(found) + 1))
    return found^


@always_inline
def number(numbers: Table, id: Id) -> UInt32:
    """Leaves are their own numbers."""
    return id.value if id.is_leaf() else numbers.get(single(id)).value()


def rebuilt(
    mut arena: Arena, rows: List[Row]
) raises Variant[Interrupted, Misordered] -> Id:
    """The caller checks that every branch names a leaf or an earlier row."""
    var ids: List[Id] = [REJECT, ACCEPT]
    for row in rows:
        arena.tick()
        ids.append(arena.make_checked(row.bit, ids[row.lo], ids[row.hi]))
    return ids[len(ids) - 1]
