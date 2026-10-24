"""Node and path counts."""

from bigint import BigUInt

from .arena import Arena, Id, Interrupted
from .steps import Step, push_branches
from .table import Table, single


def node_count(mut arena: Arena, root: Id) raises Interrupted -> Int:
    var seen = Table()
    var stack: List[Id] = [root]
    while stack:
        arena.tick()
        var id = stack.pop()
        if not id.is_leaf() and seen.add(single(id)):
            var node = arena.decision(id)
            stack.append(node.lo)
            stack.append(node.hi)
    return len(seen)


def path_count(mut arena: Arena, root: Id) raises Interrupted -> BigUInt:
    # Results are indices into `counts`, so the walk copies no counts. A leaf's id is
    # its count's index.
    var counts: List[BigUInt] = [0, 1]
    var counted = Table()
    var steps: List[Step[Id]] = [Step.visit(root)]
    var results: List[Int] = []
    var found: Optional[UInt32]
    while steps:
        arena.tick()
        var step = steps.pop()
        if not step.is_visit():
            var hi = results.pop()
            var lo = results.pop()
            counted.insert(single(step.item), UInt32(len(counts)))
            results.append(len(counts))
            counts.append(counts[lo] + counts[hi])
        elif step.item.is_leaf():
            results.append(step.item.index())
        elif (found := counted.get(single(step.item))) is not None:
            results.append(Int(found.value()))
        else:
            push_branches(arena, steps, step.item)
    return counts[results.pop()].copy()
