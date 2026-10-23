"""`BDD`, the class of nodes' Python objects."""

from std.ffi import c_int
from std.memory.alloc import unsafe_alloc
from std.python._cpython import PyObjectPtr
from std.sys.info import size_of

from arena import (
    ACCEPT,
    Arena,
    Id,
    Interrupted,
    REJECT,
    complement,
    difference,
    intersection,
    node_count,
    symmetric_difference,
    union,
)
from bigint import BigUInt

from .args import Signature, bit, node, parsed
from .builders import reduced
from .capi import (
    GetSetDef,
    MemberDef,
    METH_NOARGS,
    METH_O,
    MethodDef,
    Py_DecRef,
    Py_EQ,
    Py_NE,
    Py_NewRef,
    Py_TPFLAGS_DEFAULT,
    Py_nb_and,
    Py_nb_bool,
    Py_nb_invert,
    Py_nb_or,
    Py_nb_subtract,
    Py_nb_xor,
    Py_tp_dealloc,
    Py_tp_doc,
    Py_tp_free,
    Py_tp_getset,
    Py_tp_hash,
    Py_tp_members,
    Py_tp_methods,
    Py_tp_new,
    Py_tp_repr,
    Py_tp_richcompare,
    PyBool_FromLong,
    PyLong_FromLongLong,
    PyObject_ClearWeakRefs,
    PyType_FromModuleAndSpec,
    PyType_GetSlot,
    READONLY,
    T_PYSSIZET,
    TypeSlot,
    TypeSpec,
    address,
    new_string,
    ob_type,
    object_symbol,
    text,
)
from .errors import PyError, value_error
from .ints import c_function, count_repr, int_from_big
from .objects import BddObject, Call, forget, made, node_of, wrap
from .queries import int_from_count, member_count
from .state import (
    Lease,
    StatePointer,
    give_back,
    object_state,
    taken_arena,
    type_state,
)


comptime DOC = """BDD(bit, left, right)
--

A predicate on the bits of an integer as a reduced, ordered decision diagram.

`BDD(bit, left, right)` is an expression on the `bit` boolean variable. The
expression resolves to `left` for inputs where the bit is 0 and `right` where the
bit is 1.

All leaves in the tree must be `BDD.ACCEPT` or `BDD.REJECT`. BDDs are interned
and guaranteed to produce the same object for the same expression.

>>> odd = BDD(0, BDD.REJECT, BDD.ACCEPT)
>>> ~odd is BDD(0, BDD.ACCEPT, BDD.REJECT)
True
>>> odd | ~odd
BDD.ACCEPT

Invariants:
    - **A node tests a higher bit than its children.**
        Every path tests bits in the same order, most significant first, so
        `&` and `|` can walk two diagrams together a bit at a time.
    - `BDD` instances are always reduced by construction.
    - `BDD` instances are interned. Constructing the same diagram will always
        yield exactly the same object.
"""


def leaf_name(id: Id) -> StaticString:
    return "BDD.ACCEPT" if id == ACCEPT else "BDD.REJECT"


comptime NEW = Signature("BDD.__new__", positional=["bit", "left", "right"])


def bdd_new(
    type: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) abi("C") -> PyObjectPtr:
    var state = type_state(type)
    try:
        var given = parsed[NEW](args, kwargs)
        var at = bit(state[].ints, given[0])
        var lo = node(state, given[1])
        var hi = node(state, given[2])
        with Call(state):
            var lease = Lease(state)
            var id: Id
            try:
                with lease^ as arena:
                    id = arena[].make_checked(at, lo, hi)
            except misordered:
                raise value_error(String(misordered))
            return wrap(state, id)
    except:
        return PyObjectPtr()


def bdd_dealloc(o: PyObjectPtr) abi("C"):
    var type = ob_type(o)
    forget(type_state(type), o)
    if o.bitcast[BddObject]().unsafe_value()[].weakrefs:
        PyObject_ClearWeakRefs(o)
    var free = c_function[def(PyObjectPtr) thin abi("C") -> None](
        PyType_GetSlot(type, Py_tp_free)
    )
    free(o)
    Py_DecRef(type)


def bdd_hash(o: PyObjectPtr) abi("C") -> Int:
    # A node's object can come and go while a parent keeps the node, so by node.
    return node_of(o).index()


def bdd_bool(o: PyObjectPtr) abi("C") -> c_int:
    return c_int(Int(node_of(o) != REJECT))


def bdd_richcompare(
    a: PyObjectPtr, b: PyObjectPtr, op: c_int
) abi("C") -> PyObjectPtr:
    var state = object_state(a)
    if ob_type(b) != ob_type(a) or (op != Py_EQ and op != Py_NE):
        return Py_NewRef(state[].not_implemented)
    var equal = node_of(a) == node_of(b)
    return PyBool_FromLong(Int(equal == (op == Py_EQ)))


def bdd_invert(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    var id = node_of(o)

    def negated(mut arena: Arena) raises Interrupted {var id} -> Id:
        return complement(arena, id)

    try:
        return made(object_state(o), negated)
    except:
        return PyObjectPtr()


@always_inline
def binary[
    operation: def(mut Arena, Id, Id) thin raises Interrupted -> Id
](a: PyObjectPtr, b: PyObjectPtr) -> PyObjectPtr:
    var type = ob_type(a)
    # BDD is final, so if the types differ one isn't a BDD.
    if ob_type(b) != type:
        return Py_NewRef(object_symbol("_Py_NotImplementedStruct"))
    var x = node_of(a)
    var y = node_of(b)

    def walk(mut arena: Arena) raises Interrupted {var x, var y} -> Id:
        return operation(arena, x, y)

    try:
        return made(type_state(type), walk)
    except:
        return PyObjectPtr()


def bdd_and(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[intersection](a, b)


def bdd_or(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[union](a, b)


def bdd_sub(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[difference](a, b)


def bdd_xor(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[symmetric_difference](a, b)


def bdd_bit(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    var id = node_of(o)
    if id.is_leaf():
        return PyLong_FromLongLong(-1)
    var state = object_state(o)
    try:
        var arena = taken_arena(state)
        var tested = arena[].decision(id).bit
        give_back(state, arena)
        return PyLong_FromLongLong(Int64(tested))
    except:
        return PyObjectPtr()


@always_inline
def branch[hi: Bool](o: PyObjectPtr) -> PyObjectPtr:
    """A leaf's branches are itself."""
    var id = node_of(o)
    if id.is_leaf():
        return Py_NewRef(o)
    var state = object_state(o)
    try:
        var arena = taken_arena(state)
        var decision = arena[].decision(id)
        give_back(state, arena)
        return wrap(state, decision.hi if hi else decision.lo)
    except:
        return PyObjectPtr()


def bdd_left(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return branch[False](o)


def bdd_right(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return branch[True](o)


def bdd_size(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    var id = node_of(o)
    var state = object_state(o)
    try:
        var arena = taken_arena(state)
        var width = Int32(0) if id.is_leaf() else arena[].decision(id).bit + 1
        var count = member_count(arena[], id, width)
        give_back(state, arena)
        return int_from_count(state[].ints, count)
    except:
        return PyObjectPtr()


def bdd_leaf(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return PyBool_FromLong(Int(node_of(o).is_leaf()))


def bdd_repr(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    # A Ctrl-C handler can drop `o` while the walk is paused, so read the id first.
    var id = node_of(o)
    var state = object_state(o)
    try:
        var written: String
        if id.is_leaf():
            written = leaf_name(id)
        else:
            var lease = Lease(state)
            var nodes: Int
            var tested: Int32
            var size: BigUInt
            try:
                with lease^ as arena:
                    nodes = node_count(arena[], id)
                    tested = arena[].decision(id).bit
                    size = arena[].size(id)
            except:
                raise PyError()
            var total = int_from_big(state[].ints, size)
            if not total:
                raise PyError()
            var counted: String
            try:
                counted = count_repr(state[].ints, total)
            except e:
                Py_DecRef(total)
                raise e
            Py_DecRef(total)
            written = String(
                "<BDD bit=",
                tested,
                ", ",
                nodes,
                " nodes, ",
                counted,
                " members>",
            )
        return new_string(written)
    except:
        return PyObjectPtr()


def bdd_copy(o: PyObjectPtr, unused: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Interned, so a copy is the same object.
    return Py_NewRef(o)


def bdd_reduce(o: PyObjectPtr, unused: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        return reduced(object_state(o), o)
    except:
        return PyObjectPtr()


def bdd_type(module: PyObjectPtr) raises PyError -> PyObjectPtr:
    """Makes the BDD type for `module`. Its specs live as long as the process, as
    PyType_FromSpec needs."""
    var getset = unsafe_alloc[GetSetDef](6)
    getset[unsafe_offset=0] = GetSetDef(
        text("bit"),
        address(bdd_bit),
        0,
        text("The bit the node tests, or -1 for a leaf."),
        0,
    )
    getset[unsafe_offset=1] = GetSetDef(
        text("left"), address(bdd_left), 0, 0, 0
    )
    getset[unsafe_offset=2] = GetSetDef(
        text("right"), address(bdd_right), 0, 0, 0
    )
    getset[unsafe_offset=3] = GetSetDef(
        text("size"),
        address(bdd_size),
        0,
        text("How many integers below `2 ** (bit + 1)` the node accepts."),
        0,
    )
    getset[unsafe_offset=4] = GetSetDef(
        text("leaf"), address(bdd_leaf), 0, 0, 0
    )
    getset[unsafe_offset=5] = GetSetDef(0, 0, 0, 0, 0)

    var methods = unsafe_alloc[MethodDef](4)
    methods[unsafe_offset=0] = MethodDef(
        text("__copy__"), address(bdd_copy), METH_NOARGS, 0
    )
    methods[unsafe_offset=1] = MethodDef(
        text("__deepcopy__"), address(bdd_copy), METH_O, 0
    )
    methods[unsafe_offset=2] = MethodDef(
        text("__reduce__"), address(bdd_reduce), METH_NOARGS, 0
    )
    methods[unsafe_offset=3] = MethodDef(0, 0, 0, 0)

    # Weakrefs live just after the object header.
    var members = unsafe_alloc[MemberDef](2)
    members[unsafe_offset=0] = MemberDef(
        text("__weaklistoffset__"), T_PYSSIZET, 16, READONLY, 0
    )
    members[unsafe_offset=1] = MemberDef(0, 0, 0, 0, 0)

    var slots = unsafe_alloc[TypeSlot](18)
    var filled: List[TypeSlot] = [
        TypeSlot(Py_tp_new, address(bdd_new)),
        TypeSlot(Py_tp_dealloc, address(bdd_dealloc)),
        TypeSlot(Py_tp_hash, address(bdd_hash)),
        TypeSlot(Py_tp_richcompare, address(bdd_richcompare)),
        TypeSlot(Py_tp_repr, address(bdd_repr)),
        TypeSlot(Py_nb_bool, address(bdd_bool)),
        TypeSlot(Py_nb_invert, address(bdd_invert)),
        TypeSlot(Py_nb_and, address(bdd_and)),
        TypeSlot(Py_nb_or, address(bdd_or)),
        TypeSlot(Py_nb_subtract, address(bdd_sub)),
        TypeSlot(Py_nb_xor, address(bdd_xor)),
        TypeSlot(Py_tp_getset, Int(getset)),
        TypeSlot(Py_tp_methods, Int(methods)),
        TypeSlot(Py_tp_members, Int(members)),
        TypeSlot(Py_tp_doc, text(DOC)),
        TypeSlot(0, 0),
    ]
    for index in range(len(filled)):
        slots[unsafe_offset=index] = filled[index]
    var spec = unsafe_alloc[TypeSpec](1)
    spec[] = TypeSpec(
        text("bitpattern.bdd.BDD"),
        c_int(size_of[BddObject]()),
        0,
        Py_TPFLAGS_DEFAULT,
        Int(slots),
    )
    var type = PyType_FromModuleAndSpec(module, Int(spec), PyObjectPtr())
    if not type:
        raise PyError()
    return type
