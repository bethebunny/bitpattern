"""Garbage collection, and bounds on the memo tables."""

from bigint import BigUInt

from .arena import ACCEPT, Arena, Id, LEAVES, REJECT

comptime FREED: Int32 = Int32.MIN
"""A freed slot's bit."""

comptime MIN_COLLECTION_INTERVAL = 1 << 20
"""Collections are due after this many new nodes, or as many as the last one kept."""


@fieldwise_init
struct Schedule(TrivialRegisterPassable):
    """Nodes made since the last garbage collection, and how many make the next one
    due."""

    var made: Int
    var due: Int


@fieldwise_init
struct Stats(TrivialRegisterPassable):
    var nodes: Int
    var free: Int
    var room: Int
    var memo: Int
    var memo_room: Int


def collection_due(arena: Arena) -> Bool:
    return arena.schedule.made >= arena.schedule.due


def shrunk[T: Copyable & Deinitable](var list: List[T], room: Int) -> List[T]:
    """`list` with its capacity cut to `room`, or to its length if that's more.
    """
    var keep = max(len(list), room)
    if list.capacity() <= keep:
        return list^
    var smaller = List[T](capacity=keep)
    smaller.extend(list^)
    return smaller^


def collect(mut arena: Arena, var roots: List[Id]) -> Tuple[Int, Int]:
    """Frees the nodes `roots` don't reach. Returns how many it freed, and the room kept.

    The tables keep room for as many new nodes as last time and give back the rest.
    Sparser is slower, so the unique table keeps room for only twice the nodes left.
    """
    var made = arena.schedule.made
    var freed = sweep(arena, marks(arena, roots^))
    # New nodes take the lowest slots, so the end can come free.
    var end = len(arena.nodes)
    while end > LEAVES and arena.nodes.unsafe_get(end - 1).bit == FREED:
        end -= 1
    arena.nodes.shrink(end)
    arena.sizes.narrow.shrink(end)
    arena.free = [
        Id.from_index(index)
        for index in reversed(range(LEAVES, end))
        if arena.nodes.unsafe_get(index).bit == FREED
    ]
    arena.schedule.made = 0
    arena.schedule.due = max(MIN_COLLECTION_INTERVAL, used_slots(arena))
    var room = used_slots(arena) + max(arena.schedule.due, made)
    arena.nodes = shrunk(arena.nodes^, room)
    arena.sizes.narrow = shrunk(arena.sizes.narrow^, room)
    arena.unique.shrink_to(2 * len(arena.unique))
    # Memos can name freed slots.
    arena.memos.clear()
    return (freed, room)


def marks(arena: Arena, var roots: List[Id]) -> List[Bool]:
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


def sweep(mut arena: Arena, marks: List[Bool]) -> Int:
    var freed = 0
    for index in range(LEAVES, len(arena.nodes)):
        var node = arena.nodes.unsafe_get(index)
        if not marks[index] and node.bit != FREED:
            arena.unique.remove(node)
            _ = arena.sizes.wide.pop(index, BigUInt())
            arena.nodes.unsafe_get(index).bit = FREED
            freed += 1
    return freed


def memos_full(arena: Arena) -> Bool:
    """Walks that make no nodes can fill the memos without making a collection due.
    """
    return len(arena.memos) > max(MIN_COLLECTION_INTERVAL, used_slots(arena))


def shrink_to_fit(mut arena: Arena):
    arena.nodes = shrunk(arena.nodes^, 0)
    arena.sizes.narrow = shrunk(arena.sizes.narrow^, 0)
    arena.free = shrunk(arena.free^, 0)
    arena.unique.shrink_to(0)
    arena.memos.shrink_to_fit()


def used_slots(arena: Arena) -> Int:
    return len(arena.nodes) - len(arena.free)


def stats(arena: Arena) -> Stats:
    return Stats(
        nodes=used_slots(arena),
        free=len(arena.free),
        room=arena.nodes.capacity(),
        memo=len(arena.memos),
        memo_room=arena.memos.capacity(),
    )
