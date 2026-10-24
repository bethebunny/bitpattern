"""The module's state: the arena, the BDD type and the nodes' objects.

Mojo has no global variables, so all of it lives in CPython's per-module state, which
also gives each interpreter that imports the module an arena of its own. The module
requires the GIL, which keeps every call into it on one thread at a time.
"""

from std.memory.alloc import Layout, alloc, dealloc
from std.python._cpython import PyObjectPtr

from arena import Arena, PauseStatePointer

from .ints import Ints

from .capi import (
    Py_DecRef,
    PyCallable_Check,
    PyErr_CheckSignals,
    PyEval_GetFrame,
    PyImport_ImportModule,
    PyModule_GetState,
    PyObject_CallFunction,
    PyObject_CallMethod,
    PyOS_InterruptOccurred,
    PyType_GetModuleState,
    SIGINT,
    object_symbol,
    ob_type,
)
from .errors import PyError, runtime_error


struct State(Movable):
    var arena: Optional[ArenaPointer]
    """None while a call has it."""
    var depth: Int
    """Calls under way. An outer call can hold a node it hasn't wrapped yet, so only the
    outermost collects."""
    var wrappers: List[PyObjectPtr]
    """Each node's object by index (borrowed: an object clears its entry as it dies),
    or null. A node has at most one, so `is` compares diagrams."""
    var bdd_type: PyObjectPtr
    var batches_type: PyObjectPtr
    """The iterator type `members`, `blocks` and `paths` return."""
    var from_iterable: PyObjectPtr
    """`itertools.chain.from_iterable`, or null until a walk first needs it."""
    var range_type: PyObjectPtr
    var none: PyObjectPtr
    var not_implemented: PyObjectPtr
    var ints: Ints

    def __init__(out self, version: UInt64):
        self.arena = None
        self.depth = 0
        self.wrappers = []
        self.bdd_type = PyObjectPtr()
        self.batches_type = PyObjectPtr()
        self.from_iterable = PyObjectPtr()
        self.range_type = object_symbol("PyRange_Type")
        self.none = object_symbol("_Py_NoneStruct")
        self.not_implemented = object_symbol("_Py_NotImplementedStruct")
        self.ints = Ints(version)


comptime StatePointer = Pointer[State, MutUntrackedOrigin]
comptime ArenaPointer = Pointer[Arena, MutUntrackedOrigin]


@always_inline
def state_slot(module: PyObjectPtr) -> Pointer[Int, MutUntrackedOrigin]:
    """The module's state, which holds the address of a State."""
    return Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyModule_GetState(module)
    )


@always_inline
def module_state(module: PyObjectPtr) -> StatePointer:
    return StatePointer(unsafe_from_address=state_slot(module)[])


@always_inline
def type_state(type: PyObjectPtr) -> StatePointer:
    """The state of the module that made `type`."""
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=PyType_GetModuleState(type)
    )
    return StatePointer(unsafe_from_address=slot[])


@always_inline
def object_state(o: PyObjectPtr) -> StatePointer:
    return type_state(ob_type(o))


def take_arena(state: StatePointer) raises PyError -> ArenaPointer:
    """Python code that runs while a call has the arena (eg. an `__index__`) could call
    back in, so no Python may run until it's given back. Only Lease, Call and pause take
    and give it back directly. Everything else uses Lease."""
    if state[].arena is None:
        raise runtime_error("a call has the arena")
    return state[].arena.take()


@always_inline
def give_back(state: StatePointer, arena: ArenaPointer):
    state[].arena = arena


struct Lease:
    """The arena, taken for a `with` block and given back however the block ends.

    Mojo 1.2 rejects a `with` whose body raises nothing where the error type is PyError
    (#7206), so such a body starts with a raise that never runs.
    """

    var state: StatePointer
    var arena: ArenaPointer

    def __init__(out self, state: StatePointer) raises PyError:
        self.arena = take_arena(state)
        self.state = state

    def __enter__(self) -> ArenaPointer:
        return self.arena

    def __exit__(self):
        give_back(self.state, self.arena)


def pause(mut arena: Arena, pause_state: PauseStatePointer) -> Bool:
    """Runs Python's SIGINT handler after a Ctrl-C and stops the walk if it raised.

    The handler can use the arena, so the walk lends it the real one and holds a
    placeholder meanwhile. Nothing collects while the handler runs since the walk holds
    nodes nothing else reaches. What the handler raised stays in Python's error
    indicator for the call to return.
    """
    # Clears SIGINT's flag, so Python won't run the handler itself.
    if PyOS_InterruptOccurred() == 0:
        return True
    var state = pause_state.unsafe_bitcast[State]()
    # On the heap since the handler reaches it by an address the compiler can't track.
    var lent = alloc(Layout[Arena].single())
    var real = lent.unsafe_ptr()
    real.unsafe_write(Arena.placeholder())
    swap(arena, real[])
    state[].arena = real.unsafe_origin_cast[MutUntrackedOrigin]()
    state[].depth += 1  # so the handler's calls aren't outermost
    var handled = handle_sigint(state)
    state[].depth -= 1
    # Every call the handler made has given the arena back.
    swap(arena, real[])
    state[].arena = None
    real.unsafe_deinit_pointee()
    dealloc(lent^)
    return handled


def handle_sigint(state: StatePointer) -> Bool:
    """False if the handler raised. Raising SIGINT again instead would look like a second
    Ctrl-C to an event loop."""
    var signal = PyImport_ImportModule("signal".as_c_string_span())
    if not signal:
        return False
    var handler = PyObject_CallMethod(
        signal, "getsignal".as_c_string_span(), "i".as_c_string_span(), SIGINT
    )
    Py_DecRef(signal)
    if not handler:
        return False
    if PyCallable_Check(handler):
        # The frame the signal came in, as Python passes it.
        var frame = PyEval_GetFrame()
        var result = PyObject_CallFunction(
            handler,
            "iO".as_c_string_span(),
            SIGINT,
            frame if frame else state[].none,
        )
        Py_DecRef(handler)
        if not result:
            return False
        Py_DecRef(result)
    else:
        Py_DecRef(handler)
    return PyErr_CheckSignals() == 0
