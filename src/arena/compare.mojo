"""Subset and disjointness tests, which make no nodes."""

from .arena import ACCEPT, Arena, Id, Interrupted, REJECT
from .steps import Pair, emptied, take
from .table import pair, sorted_pair

comptime WORTH_KEEPING = 8
"""Walks shorter than this cost less to redo than to memoise."""

comptime MAX_ANSWERS = 1 << 16
"""Most questions are asked once, and a table too big for the cache costs about a walk
to add to."""


def trivial_subset(a: Id, b: Id) -> Optional[Bool]:
    if a == REJECT or b == ACCEPT or a == b:
        return True
    if b == REJECT or a == ACCEPT:
        return False
    return None


def trivial_disjoint(a: Id, b: Id) -> Optional[Bool]:
    if a == REJECT or b == REJECT:
        return True
    if a == ACCEPT or b == ACCEPT or a == b:
        return False
    return None


def subset(mut arena: Arena, a: Id, b: Id) raises Interrupted -> Bool:
    var known = trivial_subset(a, b)
    if known is not None:
        return known.value()
    # Unsorted keys since subset isn't symmetric.
    var key = pair(a, b)
    var memoised = arena.memos.subset.get(key)
    if memoised is not None:
        return memoised.value() == 1
    var both = arena.memos.intersection.get(sorted_pair(a, b))
    if both is not None:
        return Id(both.value()) == a
    var answer, walked = walk_pairs[trivial_subset, pair](arena, a, b)
    if walked >= WORTH_KEEPING:
        bound_answers(arena)
        arena.memos.subset.insert(key, UInt32(answer))
    return answer


def disjoint(mut arena: Arena, a: Id, b: Id) raises Interrupted -> Bool:
    var known = trivial_disjoint(a, b)
    if known is not None:
        return known.value()
    var key = sorted_pair(a, b)
    var memoised = arena.memos.disjoint.get(key)
    if memoised is not None:
        return memoised.value() == 1
    var both = arena.memos.intersection.get(key)
    if both is not None:
        return Id(both.value()) == REJECT
    var answer, walked = walk_pairs[trivial_disjoint, sorted_pair](arena, a, b)
    if walked >= WORTH_KEEPING:
        bound_answers(arena)
        arena.memos.disjoint.insert(key, UInt32(answer))
    return answer


def walk_pairs[
    trivial: def(Id, Id) thin -> Optional[Bool], key: def(Id, Id) thin -> UInt64
](mut arena: Arena, a: Id, b: Id) raises Interrupted -> Tuple[Bool, Int]:
    """Walks `a` and `b` together as apply does, until `trivial` answers False. Returns
    the answer and how many pairs the walk visited."""
    var pairs = take(arena.scratch.pending_pairs)
    var seen = take(arena.scratch.seen_pairs)
    pairs.append(Pair(a, b))
    var answer = True
    while pairs:
        arena.tick()
        var next = pairs.pop()
        var known = trivial(next.a, next.b)
        if known is None:
            if seen.add(key(next.a, next.b)):
                var bit = arena.top_bit(next.a, next.b)
                var a_lo, a_hi = arena.cofactors(next.a, bit)
                var b_lo, b_hi = arena.cofactors(next.b, bit)
                pairs.append(Pair(a_hi, b_hi))
                pairs.append(Pair(a_lo, b_lo))
        elif not known.value():
            answer = False
            break
    var walked = len(seen)
    arena.scratch.pending_pairs = emptied(pairs^)
    arena.scratch.seen_pairs = emptied(seen^)
    return (answer, walked)


def bound_answers(mut arena: Arena):
    """Subset and disjoint walks make no nodes, so no collection empties their memos.
    """
    if len(arena.memos.subset) + len(arena.memos.disjoint) >= MAX_ANSWERS:
        arena.memos.subset.clear()
        arena.memos.disjoint.clear()
