"""`bitpattern.bdd`, bitpattern's decision diagrams.

Invariants:
  - a node has at most one Python object, so `is` compares diagrams
  - no Python code runs while a call has the arena, since it could call back in or
    drop a node's object (eg. from the garbage collector)
  - long walks pause for Ctrl-C, and lend the arena while Python's handler runs
"""

from std.ffi import c_int
from std.memory.alloc import unsafe_alloc
from std.python._cpython import PyObjectPtr
from std.sys import get_defined_string

from arena import ACCEPT, Arena, REJECT

from python.args import keywords_method, no_arguments_method
from python.batches import (
    BLOCKS,
    MEMBERS,
    PATHS,
    batches_type,
    blocks,
    members,
    paths,
)
from python.bdd import bdd_type
from python.builders import (
    DIAGRAM,
    EXACTLY_THESE,
    FREED_BETWEEN,
    INTERVAL,
    LSHIFT,
    NARROWED,
    REBUILT,
    diagram,
    exactly_these,
    freed_between,
    interval,
    lshift,
    narrowed,
    rebuilt,
)
from python.capi import (
    MethodDef,
    ModuleDef,
    ModuleSlot,
    Py_DecRef,
    Py_MOD_PER_INTERPRETER_GIL_SUPPORTED,
    Py_NewRef,
    Py_mod_exec,
    Py_mod_multiple_interpreters,
    PyList_New,
    PyList_SetItem,
    PyModuleDef_Init,
    PyModule_AddObjectRef,
    PyModule_GetState,
    PyObject_SetAttrString,
    PyTuple_New,
    PyTuple_SetItem,
    PyUnicode_FromStringAndSize,
    PyUnicode_InternFromString,
    address,
    cstr,
    pointer_symbol,
    symbol,
    text,
)
from python.dlpack import TENSOR, tensor
from python.errors import PyError, raised
from python.ints import c_function
from python.objects import collect, stats, wrap
from python.queries import (
    ACCEPTS,
    COUNT,
    COUNT_REPR,
    DISJOINT,
    NODE_COUNT,
    NTH,
    PATH_COUNT,
    RANK,
    SUBSET,
    accepts,
    count,
    count_repr,
    disjoint,
    node_count,
    nth,
    path_count,
    rank,
    subset,
)
from python.state import State, StatePointer, paused


comptime VERSION = get_defined_string["VERSION"]()  # from pyproject.toml

comptime DOC = """Implements `BDD`, a reduced, ordered binary decision diagram.

Diagrams and their algorithms are from Bryant's 1986 paper, [Graph-Based
Algorithms for Boolean Function Manipulation](https://doi.org/10.1109/TC.1986.1676819).
"""


def python_version() -> UInt64:
    return Pointer[UInt64, MutUntrackedOrigin](
        unsafe_from_address=symbol("Py_Version")
    )[]


def methods() -> Int:
    """The module's functions, in memory that lives as long as the process."""
    var table: List[MethodDef] = [
        keywords_method[accepts, ACCEPTS]("accepts(node, value)\n--\n\n"),
        keywords_method[count, COUNT]("count(node, *, width)\n--\n\n"),
        keywords_method[nth, NTH](
            "nth(node, n, *, width)\n--\n\nNegative `n` counts from the end, as"
            " for lists."
        ),
        keywords_method[rank, RANK](
            "rank(node, value, *, width)\n--\n\nHow many members are below"
            " `value`."
        ),
        keywords_method[members, MEMBERS]("members(node, *, width)\n--\n\n"),
        keywords_method[blocks, BLOCKS](
            "blocks(node, *, width)\n--\n\nMembers as maximal aligned blocks"
            " (like CIDR blocks). `(first, free)` is the\n`2 ** free` integers"
            " from `first` on."
        ),
        keywords_method[paths, PATHS](
            "paths(node, *, width)\n--\n\nPaths to `BDD.ACCEPT` as `width`"
            " characters of `0`, `1` or `?`, most significant\nfirst. Each"
            " member matches exactly one."
        ),
        keywords_method[tensor, TENSOR](
            "tensor(node, *, width, code, bits, version)\n--\n\n"
        ),
        keywords_method[lshift, LSHIFT]("lshift(node, by)\n--\n\n"),
        keywords_method[freed_between, FREED_BETWEEN](
            "freed_between(node, *, low_bit, high_bit)\n--\n\nAccepts `i` when"
            " `node` accepts any `i'` with the same bits outside\n`[low_bit,"
            " high_bit)`."
        ),
        keywords_method[subset, SUBSET]("subset(a, b)\n--\n\n"),
        keywords_method[disjoint, DISJOINT]("disjoint(a, b)\n--\n\n"),
        keywords_method[narrowed, NARROWED]("narrowed(node, *, width)\n--\n\n"),
        keywords_method[node_count, NODE_COUNT](
            "node_count(node)\n--\n\nLeaves aren't counted."
        ),
        keywords_method[path_count, PATH_COUNT](
            "path_count(node)\n--\n\nCounted without walking each path to"
            " `BDD.ACCEPT`."
        ),
        keywords_method[count_repr, COUNT_REPR](
            "count_repr(total)\n--\n\nA count as reprs write it:\n  - `2**k`"
            " for powers of two from `2**10`\n  - `~2**k` for ints too long to"
            " write out\n  - the digits otherwise"
        ),
        keywords_method[diagram, DIAGRAM](
            "diagram(bits)\n--\n\n`bits` is most significant first, with `?`"
            " for free bits."
        ),
        keywords_method[exactly_these, EXACTLY_THESE](
            "exactly_these(values, *, width)\n--\n\nThe set of `values`, and"
            " its width (at least `width`). Errors are IntSet's."
        ),
        keywords_method[interval, INTERVAL](
            "interval(start, stop, *, width)\n--\n\n"
        ),
        keywords_method[rebuilt, REBUILT]("_rebuilt(rows)\n--\n\n"),
        no_arguments_method[collect](
            "collect",
            (
                "collect()\n--\n\nFrees unreachable nodes and gives back all"
                " spare room. Returns how many it freed."
            ),
        ),
        no_arguments_method[stats](
            "stats",
            (
                "stats()\n--\n\nThe arena's counts:\n  - nodes: slots in use\n "
                " - free: freed slots\n  - wrapped: nodes with a Python"
                " object\n  - room: slots allocated\n  - memo: memo entries\n "
                " - memo_room: memo entries allocated"
            ),
        ),
    ]
    var entries = unsafe_alloc[MethodDef](len(table) + 1)
    for index in range(len(table)):
        entries[unsafe_offset=index] = table[index]
    entries[unsafe_offset=len(table)] = MethodDef(0, 0, 0, 0)
    return Int(entries)


@export
def PyInit_bdd() abi("C") -> PyObjectPtr:
    var slots = unsafe_alloc[ModuleSlot](3)
    slots[unsafe_offset=0] = ModuleSlot(Py_mod_exec, address(module_exec))
    # No globals, so each interpreter gets an arena of its own, even with its own GIL.
    if python_version() >= 0x030C0000:
        slots[unsafe_offset=1] = ModuleSlot(
            Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
        )
        slots[unsafe_offset=2] = ModuleSlot(0, 0)
    else:
        slots[unsafe_offset=1] = ModuleSlot(0, 0)
    var definition = unsafe_alloc[ModuleDef](1)
    definition[] = ModuleDef(
        1,
        0,
        0,
        0,
        0,
        text("bitpattern.bdd"),
        text(DOC),
        8,  # the address of a State
        methods(),
        Int(slots),
        address(module_traverse),
        address(module_clear),
        address(module_free),
    )
    return PyModuleDef_Init(Int(definition))


def module_exec(module: PyObjectPtr) abi("C") -> c_int:
    try:
        var version = python_version()
        var state = unsafe_alloc[State](1)
        state.unsafe_write(State(version))
        Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=PyModule_GetState(module)
        )[] = Int(state)
        if not state[].ints.complete():
            raise raised(
                pointer_symbol("PyExc_ImportError"),
                "bitpattern.bdd can't find the int functions it needs",
            )
        var arena = unsafe_alloc[Arena](1)
        arena.unsafe_write(Arena(paused, Int(state)))
        state[].arena = Int(arena)
        state[].bdd_type = bdd_type(module)
        state[].batches_type = batches_type(module)
        added(module, "BDD", Py_NewRef(state[].bdd_type))
        added(state[].bdd_type, "ACCEPT", wrap(state, ACCEPT))
        added(state[].bdd_type, "REJECT", wrap(state, REJECT))
        added(state[].bdd_type, "__match_args__", match_args())
        added(module, "__version__", string(VERSION))
        var all = PyList_New(1)
        if not all or PyList_SetItem(all, 0, string("BDD")) < 0:
            raise PyError()
        # The functions are private to bitpattern.
        added(module, "__all__", all)
        return 0
    except:
        return -1


def added(
    o: PyObjectPtr, name: StaticString, var value: PyObjectPtr
) raises PyError:
    """Sets `o.name` to `value`, which it steals."""
    if not value:
        raise PyError()
    var failed = PyObject_SetAttrString(o, cstr(name), value) < 0
    Py_DecRef(value)
    if failed:
        raise PyError()


def string(literal: StaticString) -> PyObjectPtr:
    return PyUnicode_FromStringAndSize(cstr(literal), literal.byte_length())


def match_args() -> PyObjectPtr:
    var names = PyTuple_New(3)
    if not names:
        return names
    var fields: List[StaticString] = ["bit", "left", "right"]
    for index in range(3):
        if (
            PyTuple_SetItem(
                names, index, PyUnicode_InternFromString(cstr(fields[index]))
            )
            < 0
        ):
            Py_DecRef(names)
            return PyObjectPtr()
    return names


def module_traverse(
    module: PyObjectPtr, visit: Int, arg: Int
) abi("C") -> c_int:
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyModule_GetState(module)
    )
    if slot[] == 0:
        return 0
    var state = StatePointer(unsafe_from_address=slot[])
    var visited = c_function[def(PyObjectPtr, Int) thin abi("C") -> c_int](
        visit
    )
    for o in [state[].bdd_type, state[].batches_type, state[].from_iterable]:
        if o:
            var failed = visited(o, arg)
            if failed:
                return failed
    return 0


def module_clear(module: PyObjectPtr) abi("C") -> c_int:
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyModule_GetState(module)
    )
    if slot[] == 0:
        return 0
    var state = StatePointer(unsafe_from_address=slot[])
    clear(state[].bdd_type)
    clear(state[].batches_type)
    clear(state[].from_iterable)
    return 0


def clear(mut o: PyObjectPtr):
    """Nulls `o`, then drops it (as Py_CLEAR does)."""
    var dropped = o
    o = PyObjectPtr()
    Py_DecRef(dropped)


def module_free(module: PyObjectPtr) abi("C"):
    _ = module_clear(module)
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyModule_GetState(module)
    )
    if slot[] == 0:
        return
    var state = StatePointer(unsafe_from_address=slot[])
    if state[].arena:
        var arena = Pointer[Arena, MutUntrackedOrigin](
            unsafe_from_address=state[].arena
        )
        arena.unsafe_deinit_pointee()
        arena.unsafe_free()
    state.unsafe_deinit_pointee()
    state.unsafe_free()
    slot[] = 0
