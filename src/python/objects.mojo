"""Nodes' Python objects (at most one each), and garbage collection."""

from std.python._cpython import PyObjectPtr

from arena import Arena, Id, Interrupted, collection_due, memos_full
from arena import collect as collected
from arena import shrink_to_fit, slots, stats as arena_stats
from arena.collect import shrunk

from .capi import (
    Py_DecRef,
    Py_NewRef,
    PyDict_New,
    PyDict_SetItemString,
    PyLong_FromSsize_t,
    PyType_GenericAlloc,
    cstr,
)
from .errors import PyError
from .state import (
    Lease,
    StatePointer,
    give_back,
    module_state,
    taken_arena,
)


struct BddObject:
    """A BDD's layout: the object header, its weakrefs, and the node's id."""

    var ob_refcnt: Int
    var ob_type: PyObjectPtr
    var weakrefs: PyObjectPtr
    var id: Id


@always_inline
def node_of(o: PyObjectPtr) -> Id:
    return o.bitcast[BddObject]().unsafe_value()[].id


def wrap(state: StatePointer, id: Id) raises PyError -> PyObjectPtr:
    """A new reference to the node's object, made if it has none. Makes no Python run
    (BDDs aren't tracked by Python's garbage collector, so making one can't start it).
    """
    var index = id.index()
    if index < len(state[].wrappers):
        var existing = state[].wrappers.unsafe_get(index)
        if existing:
            return Py_NewRef(existing)
    var fresh = PyType_GenericAlloc(state[].bdd_type, 0)
    if not fresh:
        raise PyError()
    fresh.bitcast[BddObject]().unsafe_value()[].id = id
    if len(state[].wrappers) <= index:
        state[].wrappers.resize(index + 1, PyObjectPtr())
    state[].wrappers.unsafe_set(index, fresh)
    return fresh


def forget(state: StatePointer, o: PyObjectPtr):
    """Clears the entry of a dying object. A weakref callback can then make the node a
    new object, which the entry names instead."""
    var index = node_of(o).index()
    if (
        index < len(state[].wrappers)
        and state[].wrappers.unsafe_get(index) == o
    ):
        state[].wrappers.unsafe_set(index, PyObjectPtr())


def wrapped_nodes(state: StatePointer) -> List[Id]:
    var roots = List[Id]()
    for index in range(len(state[].wrappers)):
        if state[].wrappers.unsafe_get(index):
            roots.append(Id.from_index(index))
    return roots^


def collect_garbage(state: StatePointer, mut arena: Arena) -> Int:
    var freed, room = collected(arena, wrapped_nodes(state))
    var wrappers = List[PyObjectPtr]()
    swap(wrappers, state[].wrappers)
    wrappers.shrink(min(len(wrappers), slots(arena)))
    state[].wrappers = shrunk(wrappers^, room)
    return freed


struct Call:
    """A call that makes a node, counted in `depth` for a `with` block. The outermost
    call collects first, if due. Calls that make no node just take the arena and never
    collect."""

    var state: StatePointer

    def __init__(out self, state: StatePointer) raises PyError:
        var arena = taken_arena(state)
        if state[].depth == 0:
            if collection_due(arena[]):
                _ = collect_garbage(state, arena[])
            elif memos_full(arena[]):
                arena[].memos.clear()
        give_back(state, arena)
        state[].depth += 1
        self.state = state

    def __enter__(self):
        pass

    def __exit__(self):
        self.state[].depth -= 1


def made[
    F: def(mut Arena) raises Interrupted -> Id
](state: StatePointer, walk: F) raises PyError -> PyObjectPtr:
    """Runs `walk` with the arena and returns its node's object. If a pause stopped
    the walk, Python's error is the handler's."""
    with Call(state):
        var lease = Lease(state)
        var id: Id
        try:
            with lease^ as arena:
                id = walk(arena[])
        except:
            raise PyError()
        # Inside the call, so nothing collects `id` first.
        return wrap(state, id)


def collect(state: StatePointer) raises PyError -> PyObjectPtr:
    """Frees unreachable nodes and gives back all spare room. Returns how many it freed.
    """
    var arena = taken_arena(state)
    var freed = 0
    # An outer call might be about to wrap a node.
    if state[].depth == 0:
        freed = collect_garbage(state, arena[])
        shrink_to_fit(arena[])
        var wrappers = List[PyObjectPtr]()
        swap(wrappers, state[].wrappers)
        state[].wrappers = shrunk(wrappers^, 0)
    give_back(state, arena)
    return PyLong_FromSsize_t(freed)


def stats(state: StatePointer) raises PyError -> PyObjectPtr:
    """The arena's counts:
    - nodes: slots in use
    - free: freed slots
    - wrapped: nodes with a Python object
    - room: slots allocated
    - memo: memo entries
    - memo_room: memo entries allocated
    """
    var arena = taken_arena(state)
    var counts = arena_stats(arena[])
    give_back(state, arena)
    var wrapped = 0
    for index in range(len(state[].wrappers)):
        if state[].wrappers.unsafe_get(index):
            wrapped += 1
    var stats = PyDict_New()
    if not stats:
        raise PyError()
    var names: List[StaticString] = [
        "nodes",
        "free",
        "wrapped",
        "room",
        "memo",
        "memo_room",
    ]
    var values: List[Int] = [
        counts.nodes,
        counts.free,
        wrapped,
        counts.room,
        counts.memo,
        counts.memo_room,
    ]
    for index in range(len(names)):
        var value = PyLong_FromSsize_t(values[index])
        if (
            not value
            or PyDict_SetItemString(stats, cstr(names[index]), value) < 0
        ):
            Py_DecRef(value)
            Py_DecRef(stats)
            raise PyError()
        Py_DecRef(value)
    return stats
