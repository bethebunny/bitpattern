"""`bitpattern.bdd`, bitpattern's decision diagrams.

Invariants:
  - a node has at most one Python object, so `is` compares diagrams
  - no Python code runs while a call has the arena, since it could call back in or
    drop a node's object (eg. from the garbage collector)
  - long walks pause for Ctrl-C and lend the arena while Python's handler runs
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
    PyModule_GetState,
    PyObject_SetAttrString,
    PyTuple_New,
    PyTuple_SetItem,
    PyUnicode_FromStringAndSize,
    PyUnicode_InternFromString,
    address,
    leaked,
    pointer_symbol,
    symbol,
    text,
)
from python.dlpack import TENSOR, tensor
from python.errors import PyError, error
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
from python.state import State, StatePointer, pause


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
        keywords_method[accepts, ACCEPTS](),
        keywords_method[count, COUNT](),
        keywords_method[nth, NTH](),
        keywords_method[rank, RANK](),
        keywords_method[members, MEMBERS](),
        keywords_method[blocks, BLOCKS](),
        keywords_method[paths, PATHS](),
        keywords_method[tensor, TENSOR](),
        keywords_method[lshift, LSHIFT](),
        keywords_method[freed_between, FREED_BETWEEN](),
        keywords_method[subset, SUBSET](),
        keywords_method[disjoint, DISJOINT](),
        keywords_method[narrowed, NARROWED](),
        keywords_method[node_count, NODE_COUNT](),
        keywords_method[path_count, PATH_COUNT](),
        keywords_method[count_repr, COUNT_REPR](),
        keywords_method[diagram, DIAGRAM](),
        keywords_method[exactly_these, EXACTLY_THESE](),
        keywords_method[interval, INTERVAL](),
        keywords_method[rebuilt, REBUILT](),
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
        MethodDef(0, 0, 0, 0),
    ]
    return leaked(table^)


@export
def PyInit_bdd() abi("C") -> PyObjectPtr:
    var slots: List[ModuleSlot] = [
        ModuleSlot(Py_mod_exec, address(module_exec))
    ]
    # No globals, so each interpreter gets an arena of its own, even with its own GIL.
    if python_version() >= 0x030C0000:
        slots.append(
            ModuleSlot(
                Py_mod_multiple_interpreters,
                Py_MOD_PER_INTERPRETER_GIL_SUPPORTED,
            )
        )
    slots.append(ModuleSlot(0, 0))
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
        leaked(slots^),
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
            raise error(
                pointer_symbol("PyExc_ImportError"),
                "bitpattern.bdd can't find the int functions it needs",
            )
        var arena = unsafe_alloc[Arena](1)
        arena.unsafe_write(Arena(pause, Int(state)))
        state[].arena = Int(arena)
        state[].bdd_type = bdd_type(module)
        state[].batches_type = batches_type(module)
        set_attribute(module, "BDD", Py_NewRef(state[].bdd_type))
        set_attribute(state[].bdd_type, "ACCEPT", wrap(state, ACCEPT))
        set_attribute(state[].bdd_type, "REJECT", wrap(state, REJECT))
        set_attribute(state[].bdd_type, "__match_args__", match_args())
        set_attribute(module, "__version__", string(VERSION))
        var name = string("BDD")
        if not name:
            raise PyError()
        var all = PyList_New(1)
        if not all:
            Py_DecRef(name)
            raise PyError()
        _ = PyList_SetItem(all, 0, name)
        # The functions are private to bitpattern.
        set_attribute(module, "__all__", all)
        return 0
    except:
        return -1


def set_attribute(
    o: PyObjectPtr, name: StringLiteral, var value: PyObjectPtr
) raises PyError:
    """Sets `o.name` to `value`, which it steals."""
    if not value:
        raise PyError()
    var failed = PyObject_SetAttrString(o, name.as_c_string_span(), value) < 0
    Py_DecRef(value)
    if failed:
        raise PyError()


def string(literal: StaticString) -> PyObjectPtr:
    return PyUnicode_FromStringAndSize(literal)


def match_args() -> PyObjectPtr:
    var names = PyTuple_New(3)
    if not names:
        return names
    var fields = [
        "bit".as_c_string_span(),
        "left".as_c_string_span(),
        "right".as_c_string_span(),
    ]
    for index, field in enumerate(fields):
        var name = PyUnicode_InternFromString(field)
        if not name:
            Py_DecRef(names)
            return PyObjectPtr()
        _ = PyTuple_SetItem(names, index, name)
    return names


def module_traverse(
    module: PyObjectPtr, visit_address: Int, arg: Int
) abi("C") -> c_int:
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyModule_GetState(module)
    )
    if slot[] == 0:
        return 0
    var state = StatePointer(unsafe_from_address=slot[])
    var visit = c_function[def(PyObjectPtr, Int) thin abi("C") -> c_int](
        visit_address
    )
    for o in [state[].bdd_type, state[].batches_type, state[].from_iterable]:
        if o:
            var failed = visit(o, arg)
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
