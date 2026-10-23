"""Arguments. Reading one can run Python (eg. `__index__`), so calls read their
arguments before taking the arena."""

from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
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
    PyObject_Str,
    PyTuple_GetItem,
    PyTuple_Size,
    PyUnicode_AsUTF8AndSize,
    PyUnicode_CompareWithASCIIString,
    address,
    cstr,
    ob_type,
    text,
)
from .errors import PyError, type_error, value_error
from .ints import Ints, bits_of, fitting, indexed
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
        if PyUnicode_CompareWithASCIIString(key, cstr(name)) == 0:
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
    """PyO3's TypeError, which names every missing positional argument, or if there are
    none, every missing keyword one."""
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


def node(state: StatePointer, o: PyObjectPtr) raises PyError -> Id:
    if ob_type(o) != state[].bdd_type:
        raise not_an_instance(state, o, "BDD")
    return node_of(o)


def not_an_instance(
    state: StatePointer, o: PyObjectPtr, type: StaticString
) -> PyError:
    """The TypeError for an argument of the wrong type, as PyO3 words it."""
    if o == state[].none:
        return type_error(String("'None' is not an instance of '", type, "'"))
    var name = external_call["PyType_GetQualName", PyObjectPtr](ob_type(o))
    var written = String("?")
    if name:
        written = utf8(name)
        Py_DecRef(name)
    else:
        PyErr_Clear()
    return type_error(
        String("'", written, "' object is not an instance of '", type, "'")
    )


def described(ints: Ints, o: PyObjectPtr) -> String:
    """`str(o)`, or the size of an int too long for `str`."""
    var text = PyObject_Str(o)
    if text:
        var written = utf8(text)
        Py_DecRef(text)
        return written^
    PyErr_Clear()
    try:
        var int = indexed(ints, o)
        var bits = bits_of(ints, int)
        Py_DecRef(int)
        return String("an int of ", bits, " bits")
    except:
        PyErr_Clear()
        return "an int"


def in_range(o: PyObjectPtr, most: Int32) raises PyError -> Optional[Int32]:
    var read = fitting(o)
    if read.fits() and Int64(0) <= read.value and read.value <= Int64(most):
        return Int32(read.value)
    return None


def width(ints: Ints, o: PyObjectPtr) raises PyError -> Int32:
    """ValueError outside `0..=MAX_WIDTH`."""
    var read = in_range(o, MAX_WIDTH)
    if not read:
        raise value_error(
            String("width ", described(ints, o), " is out of range")
        )
    return read.value()


def bit(ints: Ints, o: PyObjectPtr) raises PyError -> Int32:
    """ValueError outside `0..=MAX_BIT`."""
    var read = in_range(o, MAX_BIT)
    if not read:
        raise value_error(
            String("bit ", described(ints, o), " is out of range")
        )
    return read.value()


def least_width(ints: Ints, o: PyObjectPtr) raises PyError -> Int32:
    """A negative width is 0, as for str.ljust."""
    var read = fitting(o)
    if read.overflow < 0 or (read.fits() and read.value < 0):
        return 0
    if read.fits() and read.value <= Int64(MAX_WIDTH):
        return Int32(read.value)
    raise value_error(String("width ", described(ints, o), " is out of range"))


def clamped(o: PyObjectPtr) raises PyError -> Int32:
    """A bit clamped to `0..=MAX_WIDTH`."""
    var read = fitting(o)
    if read.overflow < 0:
        return 0
    if read.overflow > 0:
        return MAX_WIDTH
    return Int32(max(Int64(0), min(read.value, Int64(MAX_WIDTH))))


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
    return MethodDef(
        text(signature.function),
        address(fastcall[function, signature]),
        METH_FASTCALL | METH_KEYWORDS,
        kept(materialize[signature]().method_doc()),
    )


def kept(string: String) -> Int:
    """A copy of `string` as a C string that lives as long as the process."""
    var size = string.byte_length()
    var copy = unsafe_alloc[UInt8](size + 1)
    unsafe_memcpy(dest=copy, src=string.unsafe_ptr(), count=size)
    copy[unsafe_offset=size] = 0
    return Int(copy)


def no_arguments_method[
    function: NoArguments
](name: StaticString, doc: StaticString) -> MethodDef:
    return MethodDef(
        text(name), address(no_arguments[function]), METH_NOARGS, text(doc)
    )
