"""Raising Python exceptions: set Python's error indicator, then raise PyError so the
call returns NULL."""

from std.python._cpython import PyObjectPtr

from .capi import (
    Py_DecRef,
    PyErr_SetObject,
    new_string,
    pointer_symbol,
)


@fieldwise_init
struct PyError(ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Python's error indicator is set."""

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PyError")


def raised(kind: PyObjectPtr, message: String) -> PyError:
    """Sets Python's error to `kind(message)`."""
    var text = new_string(message)
    if text:
        PyErr_SetObject(kind, text)
        Py_DecRef(text)
    return PyError()


# Exception types are looked up when raised, since errors are rare.


def value_error(message: String) -> PyError:
    return raised(pointer_symbol("PyExc_ValueError"), message)


def type_error(message: String) -> PyError:
    return raised(pointer_symbol("PyExc_TypeError"), message)


def runtime_error(message: String) -> PyError:
    return raised(pointer_symbol("PyExc_RuntimeError"), message)


def not_implemented_error(message: String) -> PyError:
    return raised(pointer_symbol("PyExc_NotImplementedError"), message)


def index_error(index: PyObjectPtr) -> PyError:
    """IndexError with the index as given, like a list's."""
    PyErr_SetObject(pointer_symbol("PyExc_IndexError"), index)
    return PyError()
