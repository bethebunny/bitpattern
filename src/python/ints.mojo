"""Python ints to and from Mojo integers, from one .so for CPython 3.11 to 3.14."""

from std.ffi import c_int
from std.math import log2
from std.python._cpython import PyObjectPtr

from bigint import BigUInt, Limbs

from .capi import (
    Buffer,
    Py_DecRef,
    Py_NewRef,
    PyErr_Occurred,
    PyLong_AsLongLongAndOverflow,
    PyLong_FromUnsignedLongLong,
    PyNumber_Index,
    PyObject_Str,
    PyUnicode_AsUTF8AndSize,
    ob_type,
    object_symbol,
    symbol,
)
from .errors import PyError


# Py_ASNATIVEBYTES_* flags.
comptime LITTLE_ENDIAN = c_int(1)
comptime UNSIGNED_BUFFER = c_int(4)

# From 3.13.
comptime AsNativeBytes = def(PyObjectPtr, Buffer, Int, c_int) thin abi(
    "C"
) -> Int
comptime FromNativeBytes = def(Buffer, Int, c_int) thin abi("C") -> PyObjectPtr
# 3.11 and 3.12's private equivalents (3.13 changed _PyLong_AsByteArray's arguments).
comptime AsByteArray = def(PyObjectPtr, Buffer, Int, c_int, c_int) thin abi(
    "C"
) -> c_int
comptime FromByteArray = def(Buffer, Int, c_int, c_int) thin abi(
    "C"
) -> PyObjectPtr


@always_inline
def buffer[T: AnyType](mut value: T) -> Buffer:
    return (
        Pointer(to=value)
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


@always_inline
def limbs_buffer(ref limbs: Limbs) -> Buffer:
    return (
        limbs.unsafe_ptr()
        .unsafe_bitcast[NoneType]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


comptime NumBits = def(PyObjectPtr) thin abi("C") -> Int


@always_inline
def c_function[T: TrivialRegisterPassable](address: Int) -> T:
    """The C function at `address`, which must have type `T`."""
    return Pointer(to=address).unsafe_bitcast[T]()[]


struct Ints(Movable):
    """The running Python's int functions, picked by version at init."""

    var int_type: PyObjectPtr
    var native: Bool
    """Whether PyLong_AsNativeBytes and friends exist (3.13 on)."""
    var as_native_bytes: AsNativeBytes
    var from_unsigned_native_bytes: FromNativeBytes
    var as_byte_array: AsByteArray
    var from_byte_array: FromByteArray
    var num_bits: NumBits

    def __init__(out self, version: UInt64):
        self.int_type = object_symbol("PyLong_Type")
        self.native = version >= 0x030D0000
        var as_native = 0
        var from_unsigned_native = 0
        var as_bytes = 0
        var from_bytes = 0
        if self.native:
            as_native = symbol("PyLong_AsNativeBytes")
            from_unsigned_native = symbol("PyLong_FromUnsignedNativeBytes")
        else:
            as_bytes = symbol("_PyLong_AsByteArray")
            from_bytes = symbol("_PyLong_FromByteArray")
        self.as_native_bytes = c_function[AsNativeBytes](as_native)
        self.from_unsigned_native_bytes = c_function[FromNativeBytes](
            from_unsigned_native
        )
        self.as_byte_array = c_function[AsByteArray](as_bytes)
        self.from_byte_array = c_function[FromByteArray](from_bytes)
        self.num_bits = c_function[NumBits](symbol("_PyLong_NumBits"))

    def complete(self) -> Bool:
        if self.native:
            return Bool(symbol("PyLong_AsNativeBytes")) and Bool(
                symbol("PyLong_FromUnsignedNativeBytes")
            )
        return Bool(symbol("_PyLong_AsByteArray")) and Bool(
            symbol("_PyLong_NumBits")
        )


@fieldwise_init
struct Int64OrOverflow(TrivialRegisterPassable):
    """An int read as an Int64: `value` if `overflow` is 0, else just its sign.
    """

    var value: Int64
    var overflow: c_int

    @staticmethod
    @always_inline
    def from_python(o: PyObjectPtr) raises PyError -> Self:
        """Without the cost of an OverflowError. Runs `__index__` for an int-like, and
        passes on its errors (TypeError for a non-int)."""
        var overflow = c_int(0)
        var value = PyLong_AsLongLongAndOverflow(o, overflow)
        if value == -1 and overflow == 0 and PyErr_Occurred():
            raise PyError()
        return Self(value, overflow)

    @always_inline
    def fits(self) -> Bool:
        return self.overflow == 0


def indexed(ints: Ints, o: PyObjectPtr) raises PyError -> PyObjectPtr:
    """A new reference to `o.__index__()`, so later reads run no Python."""
    if ob_type(o) == ints.int_type:
        return Py_NewRef(o)
    var index = PyNumber_Index(o)
    if not index:
        raise PyError()
    return index


def u128_of(ints: Ints, int: PyObjectPtr) -> Optional[UInt128]:
    """`int` (an int) if it fits a UInt128."""
    var overflow = c_int(0)
    var small = PyLong_AsLongLongAndOverflow(int, overflow)
    if overflow == 0:
        return None if small < 0 else Optional(UInt128(small))
    if overflow < 0:
        return None
    var value = UInt128(0)
    if ints.native:
        var flags = LITTLE_ENDIAN | UNSIGNED_BUFFER
        var needed = ints.as_native_bytes(int, buffer(value), 16, flags)
        return Optional(value) if needed <= 16 else None
    if ints.num_bits(int) > 128:
        return None
    _ = ints.as_byte_array(int, buffer(value), 16, 1, 0)
    return value


def i128_of(ints: Ints, int: PyObjectPtr) -> Optional[Int128]:
    var overflow = c_int(0)
    var small = PyLong_AsLongLongAndOverflow(int, overflow)
    if overflow == 0:
        return Int128(small)
    if ints.native:
        var value = Int128(0)
        var needed = ints.as_native_bytes(int, buffer(value), 16, LITTLE_ENDIAN)
        return Optional(value) if needed <= 16 else None
    if ints.num_bits(int) > 128:
        return None
    # Of the 128-bit ints only -2**127 fits, so read with room for the sign.
    var wide = Int256(0)
    _ = ints.as_byte_array(int, buffer(wide), 32, 1, 1)
    if wide < Int256(Int128.MIN) or wide > Int256(Int128.MAX):
        return None
    return Int128(wide)


@fieldwise_init
struct Signed(Movable):
    """An int as its sign and magnitude."""

    var negative: Bool
    var magnitude: BigUInt

    @staticmethod
    def from_int(ints: Ints, int: PyObjectPtr) -> Self:
        var overflow = c_int(0)
        var small = PyLong_AsLongLongAndOverflow(int, overflow)
        if overflow == 0:
            var magnitude = UInt64(small) if small >= 0 else 0 - UInt64(small)
            return Self(small < 0, BigUInt(UInt128(magnitude)))
        # Two's complement, with room for the sign.
        var bytes: Int
        if ints.native:
            var unused = UInt64(0)
            bytes = ints.as_native_bytes(int, buffer(unused), 0, LITTLE_ENDIAN)
        else:
            bytes = ints.num_bits(int) // 8 + 1
        var limbs = Limbs(length=(bytes + 7) // 8, fill=0)
        if ints.native:
            _ = ints.as_native_bytes(
                int, limbs_buffer(limbs), len(limbs) * 8, LITTLE_ENDIAN
            )
        else:
            _ = ints.as_byte_array(
                int, limbs_buffer(limbs), len(limbs) * 8, 1, 1
            )
        var negative = overflow < 0
        if negative:
            var carry = UInt64(1)
            for i in range(len(limbs)):
                var limb = ~limbs.unsafe_get(i) + carry
                carry = 1 if carry == 1 and limb == 0 else 0
                limbs.unsafe_get(i) = limb
        return Self(negative, BigUInt(limbs=limbs^))

    @staticmethod
    def from_python(ints: Ints, o: PyObjectPtr) raises PyError -> Self:
        """Runs `__index__` for an int-like."""
        var int = indexed(ints, o)
        var signed = Self.from_int(ints, int)
        Py_DecRef(int)
        return signed^


@always_inline
def int_from_u64(value: UInt64) -> PyObjectPtr:
    return PyLong_FromUnsignedLongLong(value)


def int_from_u128(ints: Ints, value: UInt128) -> PyObjectPtr:
    # Most values fit a UInt64, which converts faster.
    if value >> 64 == 0:
        return int_from_u64(UInt64(value))
    var bytes = value
    if ints.native:
        return ints.from_unsigned_native_bytes(buffer(bytes), 16, LITTLE_ENDIAN)
    return ints.from_byte_array(buffer(bytes), 16, 1, 0)


def int_from_big(ints: Ints, value: BigUInt) -> PyObjectPtr:
    var narrow = value.fitting_uint128()
    if narrow:
        return int_from_u128(ints, narrow.value())
    var size = len(value.limbs) * 8
    if ints.native:
        return ints.from_unsigned_native_bytes(
            limbs_buffer(value.limbs), size, LITTLE_ENDIAN
        )
    return ints.from_byte_array(limbs_buffer(value.limbs), size, 1, 0)


comptime WRITABLE_BITS = 14_284
"""Ints of up to this many bits fit Python's limit of 4,300 digits."""


def count_repr(ints: Ints, total: PyObjectPtr) raises PyError -> String:
    """A count (a non-negative int) as reprs write it:
    - `2**k` for powers of two from `2**10`
    - `~2**k` for ints too long to write out
    - the digits otherwise
    """
    var magnitude = Signed.from_int(ints, total).magnitude.copy()
    var bits = magnitude.bits()
    if bits > 10 and magnitude.count_ones() == 1:
        return String("2**", bits - 1)
    # Writing out more digits than Python's limit takes quadratic time.
    if bits > WRITABLE_BITS:
        var shift = bits - 64
        var top = UInt64((magnitude >> shift).fitting_uint128().value())
        var tenths = Int((Float64(shift) + log2(Float64(top))) * 10 + 0.5)
        return String("~2**", tenths // 10, ".", tenths % 10)
    var digits = PyObject_Str(total)
    if not digits:
        raise PyError()
    var size = 0
    var data = PyUnicode_AsUTF8AndSize(digits, size)
    var written = String(
        unsafe_from_utf8_ptr=Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=data
        ).as_imm()
    )
    Py_DecRef(digits)
    return written^
