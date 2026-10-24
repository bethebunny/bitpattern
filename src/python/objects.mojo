"""Nodes' Python objects (at most one each), and garbage collection."""

from std.python._cpython import PyObjectPtr

from arena import ACCEPT, Arena, Id, Interrupted
from arena.collect import (
    collect as arena_collect,
    collection_due,
    memos_full,
    shrink_to_fit,
    shrunk,
    stats as arena_stats,
)

from .args import Signature
from .capi import (
    Py_DecRef,
    Py_NewRef,
    PyDict_New,
    PyDict_SetItemString,
    PyLong_FromSsize_t,
    PyType_GenericAlloc,
)
from .errors import PyError
from .state import Lease, StatePointer, give_back, take_arena


struct BddObject:
    """A BDD's layout: the object header, its weakrefs, and the node's id."""

    var ob_refcnt: Int
    var ob_type: PyObjectPtr
    var weakrefs: PyObjectPtr
    var id: Id


@always_inline
def id_of(o: PyObjectPtr) -> Id:
    return o.bitcast[BddObject]().unsafe_value()[].id


def leaf_name(id: Id) -> StaticString:
    return "BDD.ACCEPT" if id == ACCEPT else "BDD.REJECT"


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
    var index = id_of(o).index()
    if (
        index < len(state[].wrappers)
        and state[].wrappers.unsafe_get(index) == o
    ):
        state[].wrappers.unsafe_set(index, PyObjectPtr())


def wrapped_nodes(state: StatePointer) -> List[Id]:
    return [
        Id.from_index(index)
        for index, wrapper in enumerate(state[].wrappers)
        if wrapper
    ]


def collect_garbage(state: StatePointer, mut arena: Arena) -> Int:
    var freed, room = arena_collect(arena, wrapped_nodes(state))
    var wrappers: List[PyObjectPtr] = []
    swap(wrappers, state[].wrappers)
    wrappers.shrink(min(len(wrappers), len(arena.nodes)))
    state[].wrappers = shrunk(wrappers^, room)
    return freed


struct Call:
    """A call that makes a node, counted in `depth` for a `with` block. Calls that make
    no node just take the arena and never collect."""

    var state: StatePointer

    def __init__(out self, state: StatePointer) raises PyError:
        var arena = take_arena(state)
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


def wrap_result[
    F: def(mut Arena) raises Interrupted -> Id
](state: StatePointer, walk: F) raises PyError -> PyObjectPtr:
    """Runs `walk` with the arena and returns its node's object. If a pause stopped
    the walk, Python's error is the handler's."""
    with Call(state):
        var id: Id
        with Lease(state) as arena:
            id = walk(arena[])
        # Inside the call, so nothing collects `id` first.
        return wrap(state, id)


comptime COLLECT = Signature(
    "collect",
    positional=[],
    doc=(
        "Frees unreachable nodes and shrinks the node and memo tables to fit."
        " Returns how many it freed."
    ),
)


def collect(state: StatePointer) raises PyError -> PyObjectPtr:
    var arena = take_arena(state)
    var freed = 0
    if state[].depth == 0:
        freed = collect_garbage(state, arena[])
        shrink_to_fit(arena[])
        var wrappers: List[PyObjectPtr] = []
        swap(wrappers, state[].wrappers)
        state[].wrappers = shrunk(wrappers^, 0)
    give_back(state, arena)
    return PyLong_FromSsize_t(freed)


comptime STATS = Signature(
    "stats",
    positional=[],
    doc=(
        "The arena's counts:\n"
        "  - nodes: slots in use\n"
        "  - free: freed slots\n"
        "  - wrapped: nodes with a Python object\n"
        "  - room: slots allocated\n"
        "  - memo: memo entries\n"
        "  - memo_room: memo entries allocated"
    ),
)


def stats(state: StatePointer) raises PyError -> PyObjectPtr:
    var arena = take_arena(state)
    var counts = arena_stats(arena[])
    give_back(state, arena)
    var wrapped = 0
    for wrapper in state[].wrappers:
        if wrapper:
            wrapped += 1
    var stats = PyDict_New()
    if not stats:
        raise PyError()
    var names = [
        "nodes".as_c_string_span(),
        "free".as_c_string_span(),
        "wrapped".as_c_string_span(),
        "room".as_c_string_span(),
        "memo".as_c_string_span(),
        "memo_room".as_c_string_span(),
    ]
    var values: List[Int] = [
        counts.nodes,
        counts.free,
        wrapped,
        counts.room,
        counts.memo,
        counts.memo_room,
    ]
    for name, value in zip(names, values):
        var int = PyLong_FromSsize_t(value)
        if not int or PyDict_SetItemString(stats, name, int) < 0:
            Py_DecRef(int)
            Py_DecRef(stats)
            raise PyError()
        Py_DecRef(int)
    return stats
