"""DLPack tensors of a set's members.

https://dmlc.github.io/dlpack/latest/python_spec.html
"""

from std.ffi import c_int, external_call
from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.memory.memory import _free, _malloc
from std.python._cpython import PyObjectPtr

from arena import Arena, Id, Interrupted, NARROW_WIDTH
from arena.export import Slots, write_rows, write_words
from arena.walk import Narrow, Wide

from .args import Arguments, Signature, described, node, not_an_instance, width
from .capi import (
    Py_DecRef,
    PyTuple_GetItem,
    PyTuple_Size,
    PyType_GetFlags,
    address,
    cstr,
    ob_type,
    pointer_symbol,
)
from .errors import PyError, raised, value_error
from .ints import fitting
from .queries import Count, checked_width, int_from_count, member_count
from .state import Lease, StatePointer, give_back, taken_arena


comptime CPU: Int32 = 1

comptime IS_COPIED: UInt64 = 1 << 1

comptime INT: UInt8 = 0
comptime UINT: UInt8 = 1

comptime Py_TPFLAGS_TUPLE_SUBCLASS: UInt64 = 1 << 26


@fieldwise_init
struct Device(TrivialRegisterPassable):
    """DLDevice."""

    var device_type: Int32
    var device_id: Int32


@fieldwise_init
struct DataType(TrivialRegisterPassable):
    """DLDataType."""

    var code: UInt8
    var bits: UInt8
    var lanes: UInt16

    def columns(self, width: Int32) raises PyError -> Int:
        """Words per member. Only UINT takes members wider than a word."""
        if (
            self.bits != 8
            and self.bits != 16
            and self.bits != 32
            and self.bits != 64
        ):
            raise value_error(
                String("words are 8, 16, 32 or 64 bits, not ", self.bits)
            )
        var columns = max(ceildiv(Int(width), Int(self.bits)), 1)
        if columns > 1 and self.code != UINT:
            raise value_error(
                String(
                    width,
                    "-bit members don't fit ",
                    self.bits,
                    "-bit words of type ",
                    self.code,
                )
            )
        return columns


@fieldwise_init
struct Tensor(TrivialRegisterPassable):
    """DLTensor."""

    var data: Int
    var device: Device
    var ndim: Int32
    var dtype: DataType
    var shape: Int
    var strides: Int
    var byte_offset: UInt64


@fieldwise_init
struct Version(TrivialRegisterPassable):
    """DLPackVersion."""

    var major: UInt32
    var minor: UInt32


trait Managed(TrivialRegisterPassable):
    comptime NAME: StaticString
    """The capsule's name."""

    @staticmethod
    def managing(tensor: Tensor, version: Version, deleter: Int) -> Self:
        ...


@fieldwise_init
struct Versioned(Managed):
    """DLManagedTensorVersioned, from DLPack 1.0."""

    comptime NAME: StaticString = "dltensor_versioned"

    var version: Version
    var manager_ctx: Int
    var deleter: Int
    var flags: UInt64
    var dl_tensor: Tensor

    @staticmethod
    def managing(tensor: Tensor, version: Version, deleter: Int) -> Self:
        return Self(version, 0, deleter, IS_COPIED, tensor)


@fieldwise_init
struct Legacy(Managed):
    """DLManagedTensor, for consumers from before DLPack 1.0."""

    comptime NAME: StaticString = "dltensor"

    var dl_tensor: Tensor
    var manager_ctx: Int
    var deleter: Int

    @staticmethod
    def managing(tensor: Tensor, version: Version, deleter: Int) -> Self:
        return Self(tensor, 0, deleter)


comptime Bytes = Pointer[UInt8, MutUntrackedOrigin]


struct Words(Movable):
    """Aligned to 256 bytes, and null when empty, as DLPack asks."""

    var data: Int
    var len: Int

    def __init__(out self, data: Int, len: Int):
        self.data = data
        self.len = len

    def __deinit__(deinit self):
        if self.data:
            _free(Bytes(unsafe_from_address=self.data))

    def slots[word: DType](self) -> Slots[word]:
        """For words that aren't empty, so aren't null."""
        return Slots[word](unsafe_from_address=self.data)


@fieldwise_init
struct Export[M: Managed](Movable):
    """A managed tensor and what it points into, which the deleter frees together.
    """

    var managed: Self.M  # first, so the deleter's pointer is the export's
    var shape: Array[Int64, 2]
    var strides: Array[Int64, 2]
    var words: Words


def delete[M: Managed](managed: Int) abi("C"):
    # It frees only Mojo memory, so any thread can call it.
    var export = Pointer[Export[M], MutUntrackedOrigin](
        unsafe_from_address=managed
    )
    export.unsafe_deinit_pointee()
    export.unsafe_free()


def destructor[M: Managed](capsule: PyObjectPtr) abi("C"):
    """Frees the export unless a consumer renamed the capsule (and so owns the
    tensor)."""
    if external_call["PyCapsule_IsValid", c_int](capsule, cstr(M.NAME)) == 1:
        delete[M](
            external_call["PyCapsule_GetPointer", Int](capsule, cstr(M.NAME))
        )


def capsule[
    M: Managed
](
    var words: Words, dtype: DataType, columns: Int, version: Version
) raises PyError -> PyObjectPtr:
    """1-D for one column, else 2-D."""
    var ndim: Int32 = 1 if columns == 1 else 2
    var shape: Array[Int64, 2] = [Int64(words.len // columns), Int64(columns)]
    var strides: Array[Int64, 2] = [Int64(columns), 1]
    var export = unsafe_alloc[Export[M]](1)
    var tensor = Tensor(
        words.data,
        Device(CPU, 0),
        ndim,
        dtype,
        Int(Pointer(to=export[].shape)),
        Int(Pointer(to=export[].strides)),
        0,
    )
    var managed = M.managing(tensor, version, address(delete[M]))
    export.unsafe_write(Export[M](managed, shape^, strides^, words^))
    var made = external_call["PyCapsule_New", PyObjectPtr](
        export.unsafe_bitcast[NoneType]().unsafe_origin_cast[MutAnyOrigin](),
        cstr(M.NAME),
        address(destructor[M]),
    )
    if not made:
        delete[M](Int(export))
        raise PyError()
    return made


def room(members: Count, columns: Int, size: Int) -> Optional[Words]:
    """None if the words don't fit in memory."""
    var rows = -1
    if members.wide:
        var fits = members.wide.value().fitting_uint64()
        if fits and fits.value() <= UInt64(Int.MAX):
            rows = Int(fits.value())
    elif members.narrow <= UInt128(Int.MAX):
        rows = Int(members.narrow)
    # The bytes, and up to 255 more to align them, must fit an Int.
    if rows >= 0 and rows <= (Int.MAX - 255) // (columns * size):
        if rows == 0:
            return Words(0, 0)
        var data = _malloc[UInt8](rows * columns * size, alignment=256)
        if data:
            return Words(Int(data.unsafe_value()), rows * columns)
    return None


def fill[
    word: DType
](
    mut arena: Arena,
    id: Id,
    width: Int32,
    columns: Int,
    sign_shift: UInt64,
    words: Words,
) raises Interrupted:
    if words.len == 0:  # null, with nothing to write
        return
    if columns == 1:
        write_words(arena, id, width, sign_shift, words.slots[word]())
    elif width <= NARROW_WIDTH:
        write_rows[Narrow](arena, id, width, columns, words.slots[word]())
    else:
        write_rows[Wide](arena, id, width, columns, words.slots[word]())


def written(
    state: StatePointer, id: Id, width: Int32, dtype: DataType, columns: Int
) raises PyError -> Words:
    """`DataType.columns` has checked the word size."""
    _ = checked_width(state, id, width)
    var arena = taken_arena(state)
    var members = member_count(arena[], id, width)
    give_back(state, arena)
    var made = room(members, columns, Int(dtype.bits // 8))
    if not made:
        var total = int_from_count(state[].ints, members)
        var message = String(
            described(state[].ints, total), " members don't fit in memory"
        )
        Py_DecRef(total)
        raise raised(pointer_symbol("PyExc_MemoryError"), message)
    var words = made.take()
    var sign_shift: UInt64 = 0
    if dtype.code == INT:
        sign_shift = UInt64(64 - max(width, 1))
    var lease = Lease(state)
    try:
        with lease^ as arena:
            if dtype.bits == 8:
                fill[DType.uint8](
                    arena[], id, width, columns, sign_shift, words
                )
            elif dtype.bits == 16:
                fill[DType.uint16](
                    arena[], id, width, columns, sign_shift, words
                )
            elif dtype.bits == 32:
                fill[DType.uint32](
                    arena[], id, width, columns, sign_shift, words
                )
            else:
                fill[DType.uint64](
                    arena[], id, width, columns, sign_shift, words
                )
    except:
        raise PyError()  # the Ctrl-C handler's
    return words^


def overflow_error(message: String) -> PyError:
    return raised(pointer_symbol("PyExc_OverflowError"), message)


def unsigned(o: PyObjectPtr, most: Int64) raises PyError -> Int:
    """An int in `0..=most`, with PyO3's errors for a u8 or a u32."""
    var read = fitting(o)
    if not read.fits():
        raise overflow_error("Python int too large to convert to C long")
    if read.value < 0 or read.value > most:
        raise overflow_error("out of range integral type conversion attempted")
    return Int(read.value)


def version_of(
    state: StatePointer, o: PyObjectPtr
) raises PyError -> Optional[Version]:
    """None, or a `(major, minor)` tuple."""
    if o == state[].none:
        return None
    if PyType_GetFlags(ob_type(o)) & Py_TPFLAGS_TUPLE_SUBCLASS == 0:
        raise not_an_instance(state, o, "tuple")
    var length = PyTuple_Size(o)
    if length != 2:
        raise value_error(
            String(
                "expected tuple of length 2, but got tuple of length ", length
            )
        )
    var major = unsigned(PyTuple_GetItem(o, 0), Int64(UInt32.MAX))
    var minor = unsigned(PyTuple_GetItem(o, 1), Int64(UInt32.MAX))
    return Version(UInt32(major), UInt32(minor))


comptime TENSOR = Signature(
    "tensor",
    positional=["node"],
    keywords=["width", "code", "bits", "version"],
    doc=(
        "The members as a capsule for `__dlpack__`, in words of DLPack type\n"
        "`(code, bits)`. Members wider than a word are rows of words, most"
        " significant\nfirst. A legacy capsule if `version` is None."
    ),
)


def tensor(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node(state, args[0])
    var at = width(state[].ints, args[1])
    var code = unsigned(args[2], Int64(UInt8.MAX))
    var bits = unsigned(args[3], Int64(UInt8.MAX))
    var version = version_of(state, args[4])
    var dtype = DataType(UInt8(code), UInt8(bits), 1)
    var columns = dtype.columns(at)
    var words = written(state, id, at, dtype, columns)
    if version:
        return capsule[Versioned](words^, dtype, columns, version.value())
    return capsule[Legacy](words^, dtype, columns, Version(0, 0))
