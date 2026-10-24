"""The parts of CPython's C API the module calls, and the structs it shares with C.

`external_call` binds each name when Python loads the module, as for any C extension, so
only functions in 3.11's stable ABI are called that way. Data symbols and other functions
are looked up with `symbol`.

Pointers in the structs are plain Ints (addresses), which C reads the same way. A call
through a C function pointer takes memory as a pointer with the memory's own origin, or
as a `mut` argument. Given an Int address, an untracked pointer or a pointer to a
read-only argument, the compiler may drop the store before the call or miss the callee's
write. `external_call` isn't affected.

The one private stdlib name the module uses is `PyObjectPtr`, from
`std.python._cpython`.
"""

from std.ffi import CStringSpan, c_int, c_uint, external_call
from std.python._cpython import PyObjectPtr


# Type slots (Include/typeslots.h).
comptime Py_nb_and = 8
comptime Py_nb_bool = 9
comptime Py_nb_invert = 27
comptime Py_nb_or = 31
comptime Py_nb_subtract = 36
comptime Py_nb_xor = 38
comptime Py_tp_dealloc = 52
comptime Py_tp_doc = 56
comptime Py_tp_hash = 59
comptime Py_tp_iter = 62
comptime Py_tp_iternext = 63
comptime Py_tp_methods = 64
comptime Py_tp_new = 65
comptime Py_tp_repr = 66
comptime Py_tp_richcompare = 67
comptime Py_tp_members = 72
comptime Py_tp_getset = 73
comptime Py_tp_free = 74

# Module slots.
comptime Py_mod_exec = 2
comptime Py_mod_multiple_interpreters = 3
comptime Py_MOD_PER_INTERPRETER_GIL_SUPPORTED = 2

# Type flags (Include/object.h), as type specs hold them.
comptime Py_TPFLAGS_DEFAULT: UInt32 = 1 << 18  # Py_TPFLAGS_HAVE_VERSION_TAG
comptime Py_TPFLAGS_DISALLOW_INSTANTIATION: UInt32 = 1 << 7

# As PyType_GetFlags returns them.
comptime Py_TPFLAGS_LIST_SUBCLASS: UInt64 = 1 << 25
comptime Py_TPFLAGS_TUPLE_SUBCLASS: UInt64 = 1 << 26
comptime Py_TPFLAGS_UNICODE_SUBCLASS: UInt64 = 1 << 28

comptime METH_KEYWORDS: c_int = 0x02
comptime METH_NOARGS: c_int = 0x04
comptime METH_O: c_int = 0x08
comptime METH_FASTCALL: c_int = 0x80

comptime Py_EQ: c_int = 2
comptime Py_NE: c_int = 3

comptime T_PYSSIZET: c_int = 19
comptime READONLY: c_int = 1


@fieldwise_init
struct TypeSlot(TrivialRegisterPassable):
    var slot: c_int
    var pfunc: Int


@fieldwise_init
struct TypeSpec(TrivialRegisterPassable):
    var name: Int
    var basicsize: c_int
    var itemsize: c_int
    var flags: UInt32
    var slots: Int


@fieldwise_init
struct MethodDef(TrivialRegisterPassable):
    var name: Int
    var meth: Int
    var flags: c_int
    var doc: Int


@fieldwise_init
struct GetSetDef(TrivialRegisterPassable):
    var name: Int
    var get: Int
    var set: Int
    var doc: Int
    var closure: Int


@fieldwise_init
struct MemberDef(TrivialRegisterPassable):
    var name: Int
    var type: c_int
    var offset: Int
    var flags: c_int
    var doc: Int


@fieldwise_init
struct ModuleSlot(TrivialRegisterPassable):
    var slot: c_int
    var value: Int


@fieldwise_init
struct ModuleDef(TrivialRegisterPassable):
    # PyModuleDef_HEAD_INIT
    var ob_refcnt: Int
    var ob_type: Int
    var m_init: Int
    var m_index: Int
    var m_copy: Int
    var m_name: Int
    var m_doc: Int
    var m_size: Int
    var m_methods: Int
    var m_slots: Int
    var m_traverse: Int
    var m_clear: Int
    var m_free: Int


struct ObjectHead:
    """The start of every object in a non-free-threaded build."""

    var ob_refcnt: Int
    var ob_type: PyObjectPtr


@always_inline
def address[T: TrivialRegisterPassable](function: T) -> Int:
    """The C address of an `abi("C")` function."""
    return Pointer(to=function).unsafe_bitcast[Int]()[]


@always_inline
def text(literal: StaticString) -> Int:
    """A literal's address as a C string (literals end in a NUL)."""
    return Int(literal.unsafe_ptr())


def leaked[T: Copyable & Deinitable](var items: List[T]) -> Int:
    """The address of `items`, which then live as long as the process (as C's
    tables must)."""
    return Int(items.unsafe_take_allocation().unsafe_leak())


def symbol(name: StaticString) -> Int:
    # RTLD_DEFAULT is null on glibc.
    return external_call["dlsym", Int](Int(0), name.unsafe_ptr())


def object_symbol(name: StaticString) -> PyObjectPtr:
    """A static object, eg. `_Py_NoneStruct`."""
    return PyObjectPtr(
        upcast_from=Pointer[NoneType, MutUntrackedOrigin](
            unsafe_from_address=symbol(name)
        )
    )


def pointer_symbol(name: StaticString) -> PyObjectPtr:
    """The object a `PyObject *` variable points at, eg. `PyExc_ValueError`."""
    return Pointer[PyObjectPtr, MutUntrackedOrigin](
        unsafe_from_address=symbol(name)
    )[]


@always_inline
def ob_type(o: PyObjectPtr) -> PyObjectPtr:
    return o.bitcast[ObjectHead]().unsafe_value()[].ob_type


@always_inline
def Py_IncRef(o: PyObjectPtr):
    external_call["Py_IncRef", NoneType](o)


@always_inline
def Py_DecRef(o: PyObjectPtr):
    external_call["Py_DecRef", NoneType](o)


@always_inline
def Py_NewRef(o: PyObjectPtr) -> PyObjectPtr:
    Py_IncRef(o)
    return o


# Ints.


@always_inline
def PyLong_FromLongLong(value: Int64) -> PyObjectPtr:
    return external_call["PyLong_FromLongLong", PyObjectPtr](value)


@always_inline
def PyLong_FromUnsignedLongLong(value: UInt64) -> PyObjectPtr:
    return external_call["PyLong_FromUnsignedLongLong", PyObjectPtr](value)


@always_inline
def PyLong_FromSsize_t(value: Int) -> PyObjectPtr:
    return external_call["PyLong_FromSsize_t", PyObjectPtr](value)


@always_inline
def PyLong_AsLongLongAndOverflow(o: PyObjectPtr, mut overflow: c_int) -> Int64:
    return external_call["PyLong_AsLongLongAndOverflow", Int64](
        o, Pointer(to=overflow)
    )


@always_inline
def PyNumber_Index(o: PyObjectPtr) -> PyObjectPtr:
    return external_call["PyNumber_Index", PyObjectPtr](o)


@always_inline
def PyBool_FromLong(value: Bool) -> PyObjectPtr:
    return external_call["PyBool_FromLong", PyObjectPtr](Int(value))


# Errors.


@always_inline
def PyErr_Occurred() -> PyObjectPtr:
    return external_call["PyErr_Occurred", PyObjectPtr]()


@always_inline
def PyErr_SetObject(kind: PyObjectPtr, value: PyObjectPtr):
    external_call["PyErr_SetObject", NoneType](kind, value)


@always_inline
def PyErr_Clear():
    external_call["PyErr_Clear", NoneType]()


@always_inline
def PyErr_CheckSignals() -> c_int:
    return external_call["PyErr_CheckSignals", c_int]()


@always_inline
def PyOS_InterruptOccurred() -> c_int:
    return external_call["PyOS_InterruptOccurred", c_int]()


# Strings, tuples, lists and dicts.


@always_inline
def PyUnicode_FromStringAndSize(text: StringSpan) -> PyObjectPtr:
    return external_call["PyUnicode_FromStringAndSize", PyObjectPtr](
        text.unsafe_ptr(), text.byte_length()
    )


@always_inline
def PyUnicode_InternFromString(text: CStringSpan) -> PyObjectPtr:
    return external_call["PyUnicode_InternFromString", PyObjectPtr](text)


@always_inline
def PyUnicode_FromOrdinal(ordinal: c_int) -> PyObjectPtr:
    return external_call["PyUnicode_FromOrdinal", PyObjectPtr](ordinal)


@always_inline
def PyUnicode_CompareWithASCIIString(
    o: PyObjectPtr, text: CStringSpan
) -> c_int:
    return external_call["PyUnicode_CompareWithASCIIString", c_int](o, text)


@always_inline
def PyUnicode_AsUTF8AndSize(o: PyObjectPtr, mut size: Int) -> Int:
    return external_call["PyUnicode_AsUTF8AndSize", Int](o, Pointer(to=size))


@always_inline
def PyObject_Str(o: PyObjectPtr) -> PyObjectPtr:
    return external_call["PyObject_Str", PyObjectPtr](o)


@always_inline
def PyObject_Repr(o: PyObjectPtr) -> PyObjectPtr:
    return external_call["PyObject_Repr", PyObjectPtr](o)


@always_inline
def PyTuple_New(size: Int) -> PyObjectPtr:
    return external_call["PyTuple_New", PyObjectPtr](size)


@always_inline
def PyTuple_Size(o: PyObjectPtr) -> Int:
    return external_call["PyTuple_Size", Int](o)


@always_inline
def PyTuple_GetItem(o: PyObjectPtr, index: Int) -> PyObjectPtr:
    """Borrowed."""
    return external_call["PyTuple_GetItem", PyObjectPtr](o, index)


@always_inline
def PyTuple_SetItem(o: PyObjectPtr, index: Int, item: PyObjectPtr) -> c_int:
    """Steals `item`."""
    return external_call["PyTuple_SetItem", c_int](o, index, item)


@always_inline
def Py_BuildValue(
    format: CStringSpan, first: c_int, second: c_uint, third: c_uint
) -> PyObjectPtr:
    return external_call["Py_BuildValue", PyObjectPtr, num_fixed_args=1](
        format, first, second, third
    )


@always_inline
def Py_BuildValue(
    format: CStringSpan, first: PyObjectPtr, second: PyObjectPtr
) -> PyObjectPtr:
    return external_call["Py_BuildValue", PyObjectPtr, num_fixed_args=1](
        format, first, second
    )


@always_inline
def PyList_New(size: Int) -> PyObjectPtr:
    return external_call["PyList_New", PyObjectPtr](size)


@always_inline
def PyList_SetItem(o: PyObjectPtr, index: Int, item: PyObjectPtr) -> c_int:
    """Steals `item`."""
    return external_call["PyList_SetItem", c_int](o, index, item)


@always_inline
def PyDict_New() -> PyObjectPtr:
    return external_call["PyDict_New", PyObjectPtr]()


@always_inline
def PyDict_SetItemString(
    o: PyObjectPtr, key: CStringSpan, value: PyObjectPtr
) -> c_int:
    return external_call["PyDict_SetItemString", c_int](o, key, value)


@always_inline
def PyDict_Next(
    o: PyObjectPtr,
    mut position: Int,
    mut key: PyObjectPtr,
    mut value: PyObjectPtr,
) -> Bool:
    """Borrows `key` and `value`."""
    return (
        external_call["PyDict_Next", c_int](
            o, Pointer(to=position), Pointer(to=key), Pointer(to=value)
        )
        != 0
    )


# Objects, types and modules.


@always_inline
def PyObject_GetAttrString(o: PyObjectPtr, name: CStringSpan) -> PyObjectPtr:
    return external_call["PyObject_GetAttrString", PyObjectPtr](o, name)


@always_inline
def PyObject_SetAttrString(
    o: PyObjectPtr, name: CStringSpan, value: PyObjectPtr
) -> c_int:
    return external_call["PyObject_SetAttrString", c_int](o, name, value)


@always_inline
def PyObject_CallObject(
    callable: PyObjectPtr, args: PyObjectPtr
) -> PyObjectPtr:
    return external_call["PyObject_CallObject", PyObjectPtr](callable, args)


@always_inline
def PyObject_CallFunction(
    callable: PyObjectPtr,
    format: CStringSpan,
    first: c_int,
    second: PyObjectPtr,
) -> PyObjectPtr:
    return external_call[
        "PyObject_CallFunction", PyObjectPtr, num_fixed_args=2
    ](callable, format, first, second)


@always_inline
def PyObject_CallFunctionObjArgs(
    callable: PyObjectPtr, arg: PyObjectPtr
) -> PyObjectPtr:
    return external_call[
        "PyObject_CallFunctionObjArgs", PyObjectPtr, num_fixed_args=1
    ](callable, arg, PyObjectPtr())


@always_inline
def PyObject_CallMethod(
    o: PyObjectPtr, name: CStringSpan, format: CStringSpan
) -> PyObjectPtr:
    return external_call["PyObject_CallMethod", PyObjectPtr, num_fixed_args=3](
        o, name, format
    )


@always_inline
def PyObject_CallMethod(
    o: PyObjectPtr, name: CStringSpan, format: CStringSpan, arg: c_int
) -> PyObjectPtr:
    return external_call["PyObject_CallMethod", PyObjectPtr, num_fixed_args=3](
        o, name, format, arg
    )


@always_inline
def PyEval_GetFrame() -> PyObjectPtr:
    """Borrowed."""
    return external_call["PyEval_GetFrame", PyObjectPtr]()


@always_inline
def PyCallable_Check(o: PyObjectPtr) -> Bool:
    return external_call["PyCallable_Check", c_int](o) != 0


@always_inline
def PyObject_IsInstance(o: PyObjectPtr, type: PyObjectPtr) -> c_int:
    return external_call["PyObject_IsInstance", c_int](o, type)


@always_inline
def PyObject_GetIter(o: PyObjectPtr) -> PyObjectPtr:
    return external_call["PyObject_GetIter", PyObjectPtr](o)


@always_inline
def PyIter_Next(o: PyObjectPtr) -> PyObjectPtr:
    """A new reference, or null at the end (or with an error set)."""
    return external_call["PyIter_Next", PyObjectPtr](o)


@always_inline
def PyObject_ClearWeakRefs(o: PyObjectPtr):
    external_call["PyObject_ClearWeakRefs", NoneType](o)


@always_inline
def PyType_GenericAlloc(type: PyObjectPtr, items: Int) -> PyObjectPtr:
    return external_call["PyType_GenericAlloc", PyObjectPtr](type, items)


@always_inline
def PyType_GetSlot(type: PyObjectPtr, slot: Int) -> Int:
    return external_call["PyType_GetSlot", Int](type, c_int(slot))


@always_inline
def PyType_GetFlags(type: PyObjectPtr) -> UInt64:
    return external_call["PyType_GetFlags", UInt64](type)


@always_inline
def PyType_GetQualName(type: PyObjectPtr) -> PyObjectPtr:
    return external_call["PyType_GetQualName", PyObjectPtr](type)


@always_inline
def PyType_FromModuleAndSpec(
    module: PyObjectPtr, spec: Int, bases: PyObjectPtr
) -> PyObjectPtr:
    return external_call["PyType_FromModuleAndSpec", PyObjectPtr](
        module, spec, bases
    )


@always_inline
def PyType_GetModule(type: PyObjectPtr) -> PyObjectPtr:
    """Borrowed."""
    return external_call["PyType_GetModule", PyObjectPtr](type)


@always_inline
def PyType_GetModuleState(type: PyObjectPtr) -> Int:
    return external_call["PyType_GetModuleState", Int](type)


@always_inline
def PyModule_GetState(module: PyObjectPtr) -> Int:
    return external_call["PyModule_GetState", Int](module)


@always_inline
def PyImport_ImportModule(name: CStringSpan) -> PyObjectPtr:
    return external_call["PyImport_ImportModule", PyObjectPtr](name)


@always_inline
def PyModuleDef_Init(definition: Int) -> PyObjectPtr:
    return external_call["PyModuleDef_Init", PyObjectPtr](definition)


# Capsules.


@always_inline
def PyCapsule_New(
    pointer: Pointer[NoneType, MutUntrackedOrigin],
    name: CStringSpan,
    destructor: Int,
) -> PyObjectPtr:
    return external_call["PyCapsule_New", PyObjectPtr](
        pointer, name, destructor
    )


@always_inline
def PyCapsule_GetPointer(capsule: PyObjectPtr, name: CStringSpan) -> Int:
    return external_call["PyCapsule_GetPointer", Int](capsule, name)


@always_inline
def PyCapsule_IsValid(capsule: PyObjectPtr, name: CStringSpan) -> Bool:
    return external_call["PyCapsule_IsValid", c_int](capsule, name) != 0
