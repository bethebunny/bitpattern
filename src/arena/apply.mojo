"""Diagrams' `&`, `|`, `^` and `~`."""

from .arena import ACCEPT, Arena, Id, Interrupted, REJECT
from .steps import Pair, Step, emptied, pop_branches, push_branches, take
from .table import Table, single, sorted_pair


@fieldwise_init
struct Op(Equatable, TrivialRegisterPassable):
    var value: Int

    comptime INTERSECTION = Op(0)
    comptime UNION = Op(1)
    comptime SYMMETRIC_DIFFERENCE = Op(2)


def intersection(mut arena: Arena, a: Id, b: Id) raises Interrupted -> Id:
    return apply[Op.INTERSECTION](arena, a, b)


def union(mut arena: Arena, a: Id, b: Id) raises Interrupted -> Id:
    return apply[Op.UNION](arena, a, b)


def symmetric_difference(
    mut arena: Arena, a: Id, b: Id
) raises Interrupted -> Id:
    return apply[Op.SYMMETRIC_DIFFERENCE](arena, a, b)


def difference(mut arena: Arena, a: Id, b: Id) raises Interrupted -> Id:
    var not_b = complement(arena, b)
    return intersection(arena, a, not_b)


def apply[op: Op](mut arena: Arena, a: Id, b: Id) raises Interrupted -> Id:
    """Bryant's apply."""
    # Skips taking the stacks for the many calls that need no walk.
    var answer = known[op](arena, a, b)
    if answer is not None:
        return answer.value()
    var steps = take(arena.scratch.pair_steps)
    var results = take(arena.scratch.results)
    steps.append(Step.visit(Pair(a, b)))
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            var pair = step.item
            var found = known[op](arena, pair.a, pair.b)
            if found is not None:
                results.append(found.value())
                continue
            var bit = arena.top_bit(pair.a, pair.b)
            var a_lo, a_hi = arena.cofactors(pair.a, bit)
            var b_lo, b_hi = arena.cofactors(pair.b, bit)
            steps.append(Step.make(bit, pair))
            steps.append(Step.visit(Pair(a_hi, b_hi)))
            steps.append(Step.visit(Pair(a_lo, b_lo)))
        else:
            var lo, hi = pop_branches(results)
            var made = arena.make(step.bit, lo, hi)
            memoise[op](arena, sorted_pair(step.item.a, step.item.b), made)
            results.append(made)
    var result = results.pop()
    arena.scratch.pair_steps = emptied(steps^)
    arena.scratch.results = emptied(results^)
    return result


@always_inline
def known[
    op: Op
](mut arena: Arena, a: Id, b: Id) raises Interrupted -> Optional[Id]:
    # Every step of apply's walk calls it.
    var answer = trivial[op](arena, a, b)
    if answer is not None:
        return answer
    var memoised: Optional[UInt32]
    comptime if op == Op.INTERSECTION:
        memoised = arena.memos.intersection.get(sorted_pair(a, b))
    elif op == Op.UNION:
        memoised = arena.memos.union.get(sorted_pair(a, b))
    else:
        memoised = arena.memos.symmetric_difference.get(sorted_pair(a, b))
    if memoised is not None:
        return Id(memoised.value())
    return None


@always_inline
def trivial[
    op: Op
](mut arena: Arena, a: Id, b: Id) raises Interrupted -> Optional[Id]:
    comptime if op == Op.INTERSECTION:
        if a == REJECT or b == ACCEPT or a == b:
            return a
        if b == REJECT or a == ACCEPT:
            return b
    elif op == Op.UNION:
        if a == ACCEPT or b == REJECT or a == b:
            return a
        if b == ACCEPT or a == REJECT:
            return b
    else:
        if a == b:
            return REJECT
        if a == REJECT:
            return b
        if b == REJECT:
            return a
        if a == ACCEPT:
            return complement(arena, b)
        if b == ACCEPT:
            return complement(arena, a)
    return None


@always_inline
def memoise[op: Op](mut arena: Arena, key: UInt64, result: Id):
    comptime if op == Op.INTERSECTION:
        arena.memos.intersection.insert(key, result.value)
    elif op == Op.UNION:
        arena.memos.union.insert(key, result.value)
    else:
        arena.memos.symmetric_difference.insert(key, result.value)


def complement(mut arena: Arena, root: Id) raises Interrupted -> Id:
    var answer = known_complement(arena, root)
    if answer is not None:
        return answer.value()
    var steps = take(arena.scratch.node_steps)
    var results = take(arena.scratch.results)
    steps.append(Step.visit(root))
    while steps:
        arena.tick()
        var step = steps.pop()
        if step.is_visit():
            var negation = known_complement(arena, step.item)
            if negation is not None:
                results.append(negation.value())
            else:
                push_branches(arena, steps, step.item)
        else:
            var lo, hi = pop_branches(results)
            var made = arena.make(step.bit, lo, hi)
            arena.memos.complement.insert(single(step.item), made.value)
            results.append(made)
    var result = results.pop()
    arena.scratch.node_steps = emptied(steps^)
    arena.scratch.results = emptied(results^)
    return result


@always_inline
def known_complement(arena: Arena, id: Id) -> Optional[Id]:
    if id == REJECT:
        return ACCEPT
    if id == ACCEPT:
        return REJECT
    var memoised = arena.memos.complement.get(single(id))
    if memoised is not None:
        return Id(memoised.value())
    return None
