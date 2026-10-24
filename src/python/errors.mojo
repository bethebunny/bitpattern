"""Raising Python exceptions: set Python's error indicator, then raise PyError so the
call returns NULL."""

from std.python._cpython import PyObjectPtr

from .args import str_or_size
from .capi import (
    Py_DecRef,
    PyErr_SetObject,
    PyUnicode_FromStringAndSize,
    pointer_symbol,
)


@fieldwise_init
struct PyError(TrivialRegisterPassable, Writable):
    """Python's error indicator is set."""

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PyError")


def error(kind: PyObjectPtr, message: String) -> PyError:
    """Sets Python's error to `kind(message)`."""
    var text = PyUnicode_FromStringAndSize(message)
    if text:
        PyErr_SetObject(kind, text)
        Py_DecRef(text)
    return PyError()


# Errors are rare, so exception types are looked up when raised.


def value_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_ValueError"), message)


def out_of_range(name: StaticString, value: PyObjectPtr) -> PyError:
    return value_error(String(t"{name} {str_or_size(value)} is out of range"))


def type_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_TypeError"), message)


def runtime_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_RuntimeError"), message)


def overflow_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_OverflowError"), message)


def memory_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_MemoryError"), message)


def import_error(message: String) -> PyError:
    return error(pointer_symbol("PyExc_ImportError"), message)


def index_error(index: PyObjectPtr) -> PyError:
    """IndexError with the index as given, like a list's."""
    PyErr_SetObject(pointer_symbol("PyExc_IndexError"), index)
    return PyError()
