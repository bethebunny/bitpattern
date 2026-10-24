"""Diagrams made from other diagrams by moving, freeing or narrowing their bits."""

from std.utils import Variant

from .apply import union
from .arena import Arena, Id, Interrupted, MAX_BIT
from .steps import Step, emptied, pop_branches, push_branches, take
from .table import single


@fieldwise_init
struct ShiftedOut(TrivialRegisterPassable):
    """Shifting would move a node's `bit` out of range."""

    var bit: Int32


def lshift(
    mut arena: Arena, root: Id, by: Int32
) raises Variant[Interrupted, ShiftedOut] -> Id:
    # The walk makes nodes bottom up, so check the top first to make none in vain.
    if not root.is_leaf() and Int(arena.decision(root).bit) + Int(by) > Int(
        MAX_BIT
    ):
        raise ShiftedOut(arena.decision(root).bit)
    var moved = take(arena.scratch.node_results)
    var steps = take(arena.scratch.node_steps)
    var results = take(arena.scratch.results)
    steps.append(Step.visit(root))
    var found: Optional[UInt32]
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            if step.item.is_leaf():
                results.append(step.item)
            elif (found := moved.get(single(step.item))) is not None:
                results.append(Id(found.value()))
            else:
                push_branches(arena, steps, step.item)
        else:
            if step.bit + by < 0:
                raise ShiftedOut(step.bit)
            var lo, hi = pop_branches(results)
            var result = arena.make(step.bit + by, lo, hi)
            moved.insert(single(step.item), result.value)
            results.append(result)
    var result = results.pop()
    arena.scratch.node_steps = emptied(steps^)
    arena.scratch.results = emptied(results^)
    arena.scratch.node_results = emptied(moved^)
    return result


def freed_between(
    mut arena: Arena, root: Id, *, low_bit: Int32, high_bit: Int32
) raises Interrupted -> Id:
    var freed = take(arena.scratch.node_results)
    var steps = take(arena.scratch.node_steps)
    var results = take(arena.scratch.results)
    steps.append(Step.visit(root))
    var found: Optional[UInt32]
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            if arena.tests_only_below(step.item, low_bit):
                results.append(step.item)
            elif (found := freed.get(single(step.item))) is not None:
                results.append(Id(found.value()))
            else:
                push_branches(arena, steps, step.item)
        else:
            var lo, hi = pop_branches(results)
            var result: Id
            if step.bit >= high_bit:
                result = arena.make(step.bit, lo, hi)
            else:
                result = union(arena, lo, hi)
            freed.insert(single(step.item), result.value)
            results.append(result)
    var result = results.pop()
    arena.scratch.node_steps = emptied(steps^)
    arena.scratch.results = emptied(results^)
    arena.scratch.node_results = emptied(freed^)
    return result


def narrowed(arena: Arena, root: Id, width: Int32) -> Id:
    var id = root
    while not arena.tests_only_below(id, width):
        id = arena.decision(id).lo
    return id
