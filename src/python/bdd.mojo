"""`BDD`, the class of nodes' Python objects."""

from std.ffi import c_int
from std.memory.alloc import unsafe_alloc
from std.python._cpython import PyObjectPtr
from std.sys.info import size_of

from arena import ACCEPT, Arena, Id, Interrupted, REJECT
from arena import apply, nodes
from bigint import BigUInt

from .args import Signature, bit_as_int32, node_as_id, parsed
from .builders import reduction
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
    PyUnicode_FromStringAndSize,
    READONLY,
    T_PYSSIZET,
    TypeSlot,
    TypeSpec,
    address,
    leaked,
    ob_type,
    object_symbol,
    text,
)
from .errors import PyError, value_error
from .ints import c_function, int_from_big, repr_of_count
from .objects import BddObject, Call, forget, id_of, wrap, wrap_result
from .queries import int_from_count, member_count
from .state import (
    Lease,
    StatePointer,
    give_back,
    object_state,
    take_arena,
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
        var bit = bit_as_int32(given[0])
        var lo = node_as_id(state, given[1])
        var hi = node_as_id(state, given[2])
        with Call(state):
            var lease = Lease(state)
            var id: Id
            try:
                with lease^ as arena:
                    id = arena[].make_checked(bit, lo, hi)
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
    # A node's object can come and go while a parent keeps the node, so hash the node.
    return id_of(o).index()


def bdd_bool(o: PyObjectPtr) abi("C") -> c_int:
    return c_int(id_of(o) != REJECT)


def bdd_richcompare(
    a: PyObjectPtr, b: PyObjectPtr, op: c_int
) abi("C") -> PyObjectPtr:
    var state = object_state(a)
    if ob_type(b) != ob_type(a) or (op != Py_EQ and op != Py_NE):
        return Py_NewRef(state[].not_implemented)
    var equal = id_of(a) == id_of(b)
    return PyBool_FromLong(equal == (op == Py_EQ))


def bdd_invert(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    var id = id_of(o)

    def negated(mut arena: Arena) raises Interrupted {var id} -> Id:
        return apply.complement(arena, id)

    try:
        return wrap_result(object_state(o), negated)
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
    var x = id_of(a)
    var y = id_of(b)

    def walk(mut arena: Arena) raises Interrupted {var x, var y} -> Id:
        return operation(arena, x, y)

    try:
        return wrap_result(type_state(type), walk)
    except:
        return PyObjectPtr()


def bdd_and(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[apply.intersection](a, b)


def bdd_or(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[apply.union](a, b)


def bdd_sub(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[apply.difference](a, b)


def bdd_xor(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return binary[apply.symmetric_difference](a, b)


def bdd_bit(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    var id = id_of(o)
    if id.is_leaf():
        return PyLong_FromLongLong(-1)
    var state = object_state(o)
    try:
        var arena = take_arena(state)
        var tested = arena[].decision(id).bit
        give_back(state, arena)
        return PyLong_FromLongLong(Int64(tested))
    except:
        return PyObjectPtr()


@always_inline
def branch[hi: Bool](o: PyObjectPtr) -> PyObjectPtr:
    """A leaf's branches are itself."""
    var id = id_of(o)
    if id.is_leaf():
        return Py_NewRef(o)
    var state = object_state(o)
    try:
        var arena = take_arena(state)
        var decision = arena[].decision(id)
        give_back(state, arena)
        return wrap(state, decision.hi if hi else decision.lo)
    except:
        return PyObjectPtr()


def bdd_left(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return branch[hi=False](o)


def bdd_right(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return branch[hi=True](o)


def bdd_size(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    var id = id_of(o)
    var state = object_state(o)
    try:
        var arena = take_arena(state)
        var width: Int32 = 0 if id.is_leaf() else arena[].decision(id).bit + 1
        var count = member_count(arena[], id, width)
        give_back(state, arena)
        return int_from_count(state[].ints, count)
    except:
        return PyObjectPtr()


def bdd_leaf(o: PyObjectPtr, closure: Int) abi("C") -> PyObjectPtr:
    return PyBool_FromLong(id_of(o).is_leaf())


def bdd_repr(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    # A Ctrl-C handler can drop `o` while the walk is paused, so read the id first.
    var id = id_of(o)
    var state = object_state(o)
    try:
        var written: String
        if id.is_leaf():
            written = leaf_name(id)
        else:
            var lease = Lease(state)
            var found: Int
            var tested: Int32
            var size: BigUInt
            try:
                with lease^ as arena:
                    found = nodes.node_count(arena[], id)
                    tested = arena[].decision(id).bit
                    size = arena[].size(id)
            except:
                raise PyError()
            var total = int_from_big(state[].ints, size)
            if not total:
                raise PyError()
            var counted: String
            try:
                counted = repr_of_count(state[].ints, total)
            except e:
                Py_DecRef(total)
                raise e
            Py_DecRef(total)
            written = String(
                t"<BDD bit={tested}, {found} nodes, {counted} members>"
            )
        return PyUnicode_FromStringAndSize(written)
    except:
        return PyObjectPtr()


def bdd_copy(o: PyObjectPtr, unused: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Interned, so a copy is the same object.
    return Py_NewRef(o)


def bdd_reduce(o: PyObjectPtr, unused: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        return reduction(object_state(o), o)
    except:
        return PyObjectPtr()


def bdd_type(module: PyObjectPtr) raises PyError -> PyObjectPtr:
    var getset: List[GetSetDef] = [
        GetSetDef(
            name=text("bit"),
            get=address(bdd_bit),
            set=0,
            doc=text("The bit the node tests, or -1 for a leaf."),
            closure=0,
        ),
        GetSetDef(
            name=text("left"),
            get=address(bdd_left),
            set=0,
            doc=0,
            closure=0,
        ),
        GetSetDef(
            name=text("right"),
            get=address(bdd_right),
            set=0,
            doc=0,
            closure=0,
        ),
        GetSetDef(
            name=text("size"),
            get=address(bdd_size),
            set=0,
            doc=text(
                "How many integers below `2 ** (bit + 1)` the node accepts."
            ),
            closure=0,
        ),
        GetSetDef(
            name=text("leaf"),
            get=address(bdd_leaf),
            set=0,
            doc=0,
            closure=0,
        ),
        GetSetDef(0, 0, 0, 0, 0),
    ]

    var methods: List[MethodDef] = [
        MethodDef(text("__copy__"), address(bdd_copy), METH_NOARGS, 0),
        MethodDef(text("__deepcopy__"), address(bdd_copy), METH_O, 0),
        MethodDef(text("__reduce__"), address(bdd_reduce), METH_NOARGS, 0),
        MethodDef(0, 0, 0, 0),
    ]

    var members: List[MemberDef] = [
        MemberDef(
            text("__weaklistoffset__"),
            T_PYSSIZET,
            reflect[BddObject].field_offset[name="weakrefs"](),
            READONLY,
            0,
        ),
        MemberDef(0, 0, 0, 0, 0),
    ]

    var slots: List[TypeSlot] = [
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
        TypeSlot(Py_tp_getset, leaked(getset^)),
        TypeSlot(Py_tp_methods, leaked(methods^)),
        TypeSlot(Py_tp_members, leaked(members^)),
        TypeSlot(Py_tp_doc, text(DOC)),
        TypeSlot(0, 0),
    ]
    var spec = unsafe_alloc[TypeSpec](1)
    spec[] = TypeSpec(
        text("bitpattern.bdd.BDD"),
        c_int(size_of[BddObject]()),
        0,
        Py_TPFLAGS_DEFAULT,
        leaked(slots^),
    )
    var type = PyType_FromModuleAndSpec(module, Int(spec), PyObjectPtr())
    if not type:
        raise PyError()
    return type
