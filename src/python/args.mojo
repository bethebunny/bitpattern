"""Arguments. Reading one can run Python (eg. `__index__`), so calls read their
arguments before taking the arena."""

from std.ffi import c_int
from std.python._cpython import PyObjectPtr

from arena import Id, MAX_BIT, MAX_WIDTH

from .capi import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    MethodDef,
    Py_DecRef,
    PyDict_Next,
    PyErr_Clear,
    PyLong_AsLongLongAndOverflow,
    PyNumber_Index,
    PyObject_CallMethod,
    PyObject_Str,
    PyTuple_GetItem,
    PyTuple_Size,
    PyType_GetQualName,
    PyUnicode_AsUTF8AndSize,
    PyUnicode_CompareWithASCIIString,
    address,
    leaked,
    ob_type,
    text,
)
from .errors import PyError, out_of_range, type_error
from .ints import Int64OrOverflow
from .objects import node_of
from .state import StatePointer, module_state


comptime MAX_ARGUMENTS = 5


struct Signature(Copyable):
    """A Python function's parameters and doc. Parameters in `positional` can be given
    by position or by name, those in `keywords` only by name. All of them are
    required."""

    var function: StaticString
    var positional: List[StaticString]
    var keywords: List[StaticString]
    var doc: StaticString

    def __init__(
        out self,
        function: StaticString,
        *,
        var positional: List[StaticString],
        var keywords: List[StaticString] = [],
        doc: StaticString = "",
    ):
        self.function = function
        self.positional = positional^
        self.keywords = keywords^
        self.doc = doc

    def arity(self) -> Int:
        return len(self.positional) + len(self.keywords)

    def name(self, index: Int) -> StaticString:
        if index < len(self.positional):
            return self.positional[index]
        return self.keywords[index - len(self.positional)]

    def method_doc(self) -> String:
        """`doc` after the text signature that `inspect.signature` reads."""
        var written = String(self.function, "(")
        for index in range(self.arity()):
            if index > 0:
                written += ", "
            if index == len(self.positional):
                written += "*, "
            written += self.name(index)
        return String(written, ")\n--\n\n", self.doc)


struct Arguments(Movable):
    """A call's arguments by parameter (borrowed)."""

    var items: Array[PyObjectPtr, MAX_ARGUMENTS]

    def __init__(out self):
        self.items = Array[PyObjectPtr, MAX_ARGUMENTS](fill=PyObjectPtr())

    @always_inline
    def __getitem__(self, index: Int) -> PyObjectPtr:
        return self.items[index]


def utf8(o: PyObjectPtr) -> String:
    """A str's text, or "?" if it can't be read."""
    var size = 0
    var data = PyUnicode_AsUTF8AndSize(o, size)
    if data == 0:
        PyErr_Clear()
        return "?"
    return String(
        unsafe_from_utf8_ptr=Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=data
        ).as_imm()
    )


def parsed[
    signature: Signature
](args: Int, nargs: Int, kwnames: PyObjectPtr) raises PyError -> Arguments:
    """METH_FASTCALL | METH_KEYWORDS arguments, matched to `signature`."""
    comptime positional = len(signature.positional)
    if nargs > positional:
        raise too_many[signature](nargs)
    var vector = Pointer[PyObjectPtr, MutUntrackedOrigin](
        unsafe_from_address=args
    )
    var arguments = Arguments()
    for index in range(nargs):
        arguments.items[index] = vector[unsafe_offset=index]
    var keywords = PyTuple_Size(kwnames) if kwnames else 0
    for k in range(keywords):
        var value = vector[unsafe_offset=nargs + k]
        add_keyword[signature](arguments, PyTuple_GetItem(kwnames, k), value)
    return completed[signature](arguments^)


def parsed[
    signature: Signature
](args: PyObjectPtr, kwargs: PyObjectPtr) raises PyError -> Arguments:
    """tp_new's arguments (a tuple and a dict), matched to `signature`."""
    comptime positional = len(signature.positional)
    var given = PyTuple_Size(args)
    if given > positional:
        raise too_many[signature](given)
    var arguments = Arguments()
    for index in range(given):
        arguments.items[index] = PyTuple_GetItem(args, index)
    if kwargs:
        var position = 0
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        while PyDict_Next(kwargs, position, key, value):
            add_keyword[signature](arguments, key, value)
    return completed[signature](arguments^)


def add_keyword[
    signature: Signature
](
    mut arguments: Arguments, key: PyObjectPtr, value: PyObjectPtr
) raises PyError:
    comptime for index in range(signature.arity()):
        comptime name = signature.name(index)
        if PyUnicode_CompareWithASCIIString(key, name.as_c_string_span()) == 0:
            if arguments.items[index]:
                raise type_error(
                    String(
                        signature.function,
                        "() got multiple values for argument '",
                        name,
                        "'",
                    )
                )
            arguments.items[index] = value
            return
    raise type_error(
        String(
            signature.function,
            "() got an unexpected keyword argument '",
            utf8(key),
            "'",
        )
    )


def completed[
    signature: Signature
](var arguments: Arguments) raises PyError -> Arguments:
    comptime assert signature.arity() <= MAX_ARGUMENTS
    comptime for index in range(signature.arity()):
        if not arguments.items[index]:
            raise missing[signature](arguments)
    return arguments^


@no_inline
def too_many[signature: Signature](given: Int) -> PyError:
    comptime positional = len(signature.positional)
    return type_error(
        String(
            signature.function,
            "() takes ",
            positional,
            " positional arguments but ",
            given,
            " were given",
        )
    )


@no_inline
def missing[signature: Signature](arguments: Arguments) -> PyError:
    """A TypeError that names every missing positional argument. If none is missing, it
    names every missing keyword argument."""
    var declared = materialize[signature]()
    var names = List[StaticString]()
    var kind: StaticString = "positional"
    for index in range(declared.arity()):
        if index == len(declared.positional):
            if names:
                break
            kind = "keyword"
        if not arguments[index]:
            names.append(declared.name(index))
    var message = String(
        signature.function, "() missing ", len(names), " required ", kind
    )
    message += " arguments: " if len(names) > 1 else " argument: "
    for index in range(len(names)):
        if index > 0:
            if len(names) > 2:
                message += ","
            message += " and " if index == len(names) - 1 else " "
        message += String("'", names[index], "'")
    return type_error(message)


def node_as_id(state: StatePointer, node: PyObjectPtr) raises PyError -> Id:
    if ob_type(node) != state[].bdd_type:
        raise not_an_instance(state, node, "BDD")
    return node_of(node)


def not_an_instance(
    state: StatePointer, o: PyObjectPtr, type: StaticString
) -> PyError:
    """The TypeError for an argument of the wrong type (eg. "'str' object is not an
    instance of 'BDD'")."""
    if o == state[].none:
        return type_error(String("'None' is not an instance of '", type, "'"))
    var name = PyType_GetQualName(ob_type(o))
    var written = String("?")
    if name:
        written = utf8(name)
        Py_DecRef(name)
    else:
        PyErr_Clear()
    return type_error(
        String("'", written, "' object is not an instance of '", type, "'")
    )


def str_or_size(o: PyObjectPtr) -> String:
    """`str(o)`, or the size of an int too long for `str`."""
    var text = PyObject_Str(o)
    if text:
        var written = utf8(text)
        Py_DecRef(text)
        return written^
    PyErr_Clear()
    var int = PyNumber_Index(o)
    var bit_length = PyObjectPtr()
    if int:
        bit_length = PyObject_CallMethod(
            int, "bit_length".as_c_string_span(), "".as_c_string_span()
        )
        Py_DecRef(int)
    if not bit_length:
        PyErr_Clear()
        return "an int"
    var overflow = c_int(0)
    var bits = PyLong_AsLongLongAndOverflow(bit_length, overflow)
    Py_DecRef(bit_length)
    return String("an int of ", bits, " bits")


def int32_up_to(o: PyObjectPtr, most: Int32) raises PyError -> Optional[Int32]:
    var int_or_overflow = Int64OrOverflow.from_python(o)
    if (
        int_or_overflow.fits()
        and Int64(0) <= int_or_overflow.value
        and int_or_overflow.value <= Int64(most)
    ):
        return Int32(int_or_overflow.value)
    return None


def width_as_int32(width: PyObjectPtr) raises PyError -> Int32:
    """ValueError outside `[0, MAX_WIDTH]`."""
    var width_or_none = int32_up_to(width, MAX_WIDTH)
    if width_or_none is None:
        raise out_of_range("width", width)
    return width_or_none.value()


def bit_as_int32(bit: PyObjectPtr) raises PyError -> Int32:
    """ValueError outside `[0, MAX_BIT]`."""
    var bit_or_none = int32_up_to(bit, MAX_BIT)
    if bit_or_none is None:
        raise out_of_range("bit", bit)
    return bit_or_none.value()


def least_width_as_int32(width: PyObjectPtr) raises PyError -> Int32:
    """A negative width is 0, as for str.ljust."""
    var width_or_overflow = Int64OrOverflow.from_python(width)
    if width_or_overflow.overflow < 0 or (
        width_or_overflow.fits() and width_or_overflow.value < 0
    ):
        return 0
    if width_or_overflow.fits() and width_or_overflow.value <= Int64(MAX_WIDTH):
        return Int32(width_or_overflow.value)
    raise out_of_range("width", width)


def clamped(bit: PyObjectPtr) raises PyError -> Int32:
    """A bit clamped to `[0, MAX_WIDTH]`."""
    var bit_or_overflow = Int64OrOverflow.from_python(bit)
    if bit_or_overflow.overflow < 0:
        return 0
    if bit_or_overflow.overflow > 0:
        return MAX_WIDTH
    return Int32(max(Int64(0), min(bit_or_overflow.value, Int64(MAX_WIDTH))))


# Calling conventions. Each wraps a raising Mojo function in the C function CPython
# calls, which returns NULL once Python's error is set.

comptime Function = def(
    StatePointer, Arguments
) thin raises PyError -> PyObjectPtr
comptime NoArguments = def(StatePointer) thin raises PyError -> PyObjectPtr


def fastcall[
    function: Function, signature: Signature
](module: PyObjectPtr, args: Int, nargs: Int, kwnames: PyObjectPtr) abi(
    "C"
) -> PyObjectPtr:
    try:
        return function(
            module_state(module), parsed[signature](args, nargs, kwnames)
        )
    except:
        return PyObjectPtr()


def no_arguments[
    function: NoArguments
](module: PyObjectPtr, unused: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        return function(module_state(module))
    except:
        return PyObjectPtr()


def keywords_method[function: Function, signature: Signature]() -> MethodDef:
    var doc = List(materialize[signature]().method_doc().as_bytes())
    doc.append(0)  # C strings end in a NUL
    return MethodDef(
        text(signature.function),
        address(fastcall[function, signature]),
        METH_FASTCALL | METH_KEYWORDS,
        leaked(doc^),
    )


def no_arguments_method[
    function: NoArguments
](name: StaticString, doc: StaticString) -> MethodDef:
    return MethodDef(
        text(name), address(no_arguments[function]), METH_NOARGS, text(doc)
    )
