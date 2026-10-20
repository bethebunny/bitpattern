"""Garbage collection, and bounds on the memo tables."""

from .arena import ACCEPT, Arena, Id, REJECT

comptime FREED: Int32 = Int32.MIN
"""A freed slot's bit."""

comptime MIN_COLLECTION_INTERVAL = 1 << 20
"""Collections are due after this many new nodes, or as many as the last one kept."""


@fieldwise_init
struct Schedule(ImplicitlyCopyable, TrivialRegisterPassable):
    """Nodes made since the last garbage collection, and how many make the next one
    due."""

    var made: Int
    var due: Int


@fieldwise_init
struct Stats(ImplicitlyCopyable, TrivialRegisterPassable):
    var nodes: Int
    var free: Int
    var room: Int
    var memo: Int
    var memo_room: Int


def collection_due(arena: Arena) -> Bool:
    return arena.schedule.made >= arena.schedule.due


def shrunk[T: Copyable & Deinitable](var list: List[T], room: Int) -> List[T]:
    """`list` with at most `room` of room beyond its items (as Vec::shrink_to).
    """
    var keep = max(len(list), room)
    if list.capacity() <= keep:
        return list^
    var smaller = List[T](capacity=keep)
    smaller.extend(list^)
    return smaller^


def collect(mut arena: Arena, var roots: List[Id]) -> Tuple[Int, Int]:
    """Frees the nodes `roots` don't reach. Returns how many it freed, and the room kept.

    The tables keep room for as many new nodes as last time, and give back the rest. The
    unique table keeps only twice the nodes left, since sparser is slower.
    """
    var made = arena.schedule.made
    var freed = swept(arena, marked(arena, roots^))
    # New nodes take the lowest slots, so the end can come free.
    var end = len(arena.nodes)
    while end > 2 and arena.nodes.unsafe_get(end - 1).bit == FREED:
        end -= 1
    arena.nodes.shrink(end)
    arena.sizes.narrow.shrink(end)
    arena.free.clear()
    for index in range(end - 1, 1, -1):
        if arena.nodes.unsafe_get(index).bit == FREED:
            arena.free.append(Id.from_index(index))
    arena.schedule.made = 0
    arena.schedule.due = max(MIN_COLLECTION_INTERVAL, used(arena))
    var room = used(arena) + max(arena.schedule.due, made)
    arena.nodes = shrunk(arena.nodes^, room)
    arena.sizes.narrow = shrunk(arena.sizes.narrow^, room)
    arena.unique.shrink_to(2 * len(arena.unique))
    # Memos can name freed slots.
    arena.memos.clear()
    return (freed, room)


def marked(arena: Arena, var roots: List[Id]) -> List[Bool]:
    var marks = List[Bool](length=len(arena.nodes), fill=False)
    marks[REJECT.index()] = True
    marks[ACCEPT.index()] = True
    var stack = roots^
    while stack:
        var id = stack.pop()
        if not marks[id.index()]:
            marks[id.index()] = True
            var node = arena.decision(id)
            stack.append(node.lo)
            stack.append(node.hi)
    return marks^


def swept(mut arena: Arena, marks: List[Bool]) -> Int:
    var freed = 0
    for index in range(2, len(arena.nodes)):
        var node = arena.nodes.unsafe_get(index)
        if not marks[index] and node.bit != FREED:
            arena.unique.remove(node)
            if index in arena.sizes.wide:
                try:
                    _ = arena.sizes.wide.pop(index)
                except:
                    pass
            arena.nodes.unsafe_get(index).bit = FREED
            freed += 1
    return freed


def memos_full(arena: Arena) -> Bool:
    """Walks that make no nodes can fill the memos without making a collection due.
    """
    return len(arena.memos) > max(MIN_COLLECTION_INTERVAL, used(arena))


def shrink_to_fit(mut arena: Arena):
    arena.nodes = shrunk(arena.nodes^, 0)
    arena.sizes.narrow = shrunk(arena.sizes.narrow^, 0)
    arena.free = shrunk(arena.free^, 0)
    arena.unique.shrink_to(0)
    arena.memos.shrink_to_fit()


def used(arena: Arena) -> Int:
    return len(arena.nodes) - len(arena.free)


def slots(arena: Arena) -> Int:
    return len(arena.nodes)


def stats(arena: Arena) -> Stats:
    return Stats(
        used(arena),
        len(arena.free),
        arena.nodes.capacity(),
        len(arena.memos),
        arena.memos.capacity(),
    )
