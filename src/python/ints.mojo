"""Python ints to and from Mojo integers, from one .so for CPython 3.11 to 3.14."""

from std.ffi import c_int
from std.math import log2
from std.python._cpython import PyObjectPtr
from std.sys.info import size_of

from bigint import BigUInt, Limbs

from .capi import (
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
comptime LITTLE_ENDIAN: c_int = 1
comptime UNSIGNED_BUFFER: c_int = 4

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
    var as_native_bytes_address: Int
    var from_unsigned_native_bytes_address: Int
    var as_byte_array_address: Int
    var from_byte_array_address: Int
    var num_bits: NumBits

    def __init__(out self, version: UInt64):
        self.int_type = object_symbol("PyLong_Type")
        self.native = version >= 0x030D0000
        self.as_native_bytes_address = 0
        self.from_unsigned_native_bytes_address = 0
        self.as_byte_array_address = 0
        self.from_byte_array_address = 0
        if self.native:
            self.as_native_bytes_address = symbol("PyLong_AsNativeBytes")
            self.from_unsigned_native_bytes_address = symbol(
                "PyLong_FromUnsignedNativeBytes"
            )
        else:
            self.as_byte_array_address = symbol("_PyLong_AsByteArray")
            self.from_byte_array_address = symbol("_PyLong_FromByteArray")
        self.num_bits = c_function[NumBits](symbol("_PyLong_NumBits"))

    # From 3.13.

    def as_native_bytes[
        T: AnyType
    ](self, int: PyObjectPtr, mut into: T, flags: c_int) -> Int:
        var memory = Pointer(to=into).unsafe_bitcast[NoneType]()
        var call = c_function[
            def(PyObjectPtr, type_of(memory), Int, c_int) thin abi("C") -> Int
        ](self.as_native_bytes_address)
        return call(int, memory, size_of[T](), flags)

    def as_native_bytes(
        self, int: PyObjectPtr, mut into: Limbs, flags: c_int
    ) -> Int:
        var memory = into.unsafe_ptr().unsafe_bitcast[NoneType]()
        var call = c_function[
            def(PyObjectPtr, type_of(memory), Int, c_int) thin abi("C") -> Int
        ](self.as_native_bytes_address)
        return call(int, memory, len(into) * 8, flags)

    # `mut` though C only reads `value` (see capi.mojo).
    def from_unsigned_native_bytes[
        T: AnyType
    ](self, mut value: T, flags: c_int) -> PyObjectPtr:
        var memory = Pointer(to=value).unsafe_bitcast[NoneType]()
        var call = c_function[
            def(type_of(memory), Int, c_int) thin abi("C") -> PyObjectPtr
        ](self.from_unsigned_native_bytes_address)
        return call(memory, size_of[T](), flags)

    def from_unsigned_native_bytes(
        self, limbs: Limbs, flags: c_int
    ) -> PyObjectPtr:
        var memory = limbs.unsafe_ptr().unsafe_bitcast[NoneType]()
        var call = c_function[
            def(type_of(memory), Int, c_int) thin abi("C") -> PyObjectPtr
        ](self.from_unsigned_native_bytes_address)
        return call(memory, len(limbs) * 8, flags)

    # 3.11 and 3.12's private equivalents (3.13 changed _PyLong_AsByteArray's arguments).

    def as_byte_array[
        T: AnyType
    ](
        self,
        int: PyObjectPtr,
        mut into: T,
        little_endian: c_int,
        is_signed: c_int,
    ) -> c_int:
        var memory = Pointer(to=into).unsafe_bitcast[NoneType]()
        var call = c_function[
            def(
                PyObjectPtr, type_of(memory), Int, c_int, c_int
            ) thin abi("C") -> c_int
        ](self.as_byte_array_address)
        return call(int, memory, size_of[T](), little_endian, is_signed)

    def as_byte_array(
        self,
        int: PyObjectPtr,
        mut into: Limbs,
        little_endian: c_int,
        is_signed: c_int,
    ) -> c_int:
        var memory = into.unsafe_ptr().unsafe_bitcast[NoneType]()
        var call = c_function[
            def(
                PyObjectPtr, type_of(memory), Int, c_int, c_int
            ) thin abi("C") -> c_int
        ](self.as_byte_array_address)
        return call(int, memory, len(into) * 8, little_endian, is_signed)

    # `mut` though C only reads `value` (see capi.mojo).
    def from_byte_array[
        T: AnyType
    ](
        self, mut value: T, little_endian: c_int, is_signed: c_int
    ) -> PyObjectPtr:
        var memory = Pointer(to=value).unsafe_bitcast[NoneType]()
        var call = c_function[
            def(type_of(memory), Int, c_int, c_int) thin abi("C") -> PyObjectPtr
        ](self.from_byte_array_address)
        return call(memory, size_of[T](), little_endian, is_signed)

    def from_byte_array(
        self, limbs: Limbs, little_endian: c_int, is_signed: c_int
    ) -> PyObjectPtr:
        var memory = limbs.unsafe_ptr().unsafe_bitcast[NoneType]()
        var call = c_function[
            def(type_of(memory), Int, c_int, c_int) thin abi("C") -> PyObjectPtr
        ](self.from_byte_array_address)
        return call(memory, len(limbs) * 8, little_endian, is_signed)

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
        """Without the cost of an OverflowError. Runs `__index__` for an int-like and
        passes on its errors (TypeError for a non-int)."""
        var overflow: c_int = 0
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
    var overflow: c_int = 0
    var small = PyLong_AsLongLongAndOverflow(int, overflow)
    if overflow == 0:
        return None if small < 0 else UInt128(small)
    if overflow < 0:
        return None
    var value: UInt128 = 0
    if ints.native:
        var flags = LITTLE_ENDIAN | UNSIGNED_BUFFER
        var needed = ints.as_native_bytes(int, value, flags)
        return value if needed <= 16 else None
    if ints.num_bits(int) > 128:
        return None
    _ = ints.as_byte_array(int, value, 1, 0)
    return value


def i128_of(ints: Ints, int: PyObjectPtr) -> Optional[Int128]:
    var overflow: c_int = 0
    var small = PyLong_AsLongLongAndOverflow(int, overflow)
    if overflow == 0:
        return Int128(small)
    if ints.native:
        var value: Int128 = 0
        var needed = ints.as_native_bytes(int, value, LITTLE_ENDIAN)
        return value if needed <= 16 else None
    if ints.num_bits(int) > 128:
        return None
    # Of the 128-bit ints only -2**127 fits, so read with room for the sign.
    var wide: Int256 = 0
    _ = ints.as_byte_array(int, wide, 1, 1)
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
        var overflow: c_int = 0
        var small = PyLong_AsLongLongAndOverflow(int, overflow)
        if overflow == 0:
            var magnitude = UInt64(small) if small >= 0 else 0 - UInt64(small)
            return Self(small < 0, BigUInt(magnitude))
        # Two's complement, with room for the sign.
        var bytes: Int
        if ints.native:
            var unused: UInt64 = 0
            bytes = ints.as_native_bytes(int, unused, LITTLE_ENDIAN)
        else:
            bytes = ints.num_bits(int) // 8 + 1
        var limbs = Limbs(length=(bytes + 7) // 8, fill=0)
        if ints.native:
            _ = ints.as_native_bytes(int, limbs, LITTLE_ENDIAN)
        else:
            _ = ints.as_byte_array(int, limbs, 1, 1)
        var negative = overflow < 0
        if negative:
            var carry: UInt64 = 1
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
        return ints.from_unsigned_native_bytes(bytes, LITTLE_ENDIAN)
    return ints.from_byte_array(bytes, 1, 0)


def int_from_big(ints: Ints, value: BigUInt) -> PyObjectPtr:
    var narrow = value.uint128_or_none()
    if narrow is not None:
        return int_from_u128(ints, narrow.value())
    if ints.native:
        return ints.from_unsigned_native_bytes(value.limbs, LITTLE_ENDIAN)
    return ints.from_byte_array(value.limbs, 1, 0)


comptime WRITABLE_BITS = 14_284
"""Ints of up to this many bits fit Python's limit of 4,300 digits."""


def repr_of_count(ints: Ints, total: PyObjectPtr) raises PyError -> String:
    """A count (a non-negative int) as reprs write it.

    Formats:
      - powers of two from `2**10` as a power, eg. `2**64`
      - ints too long to write out as a rounded power, eg. `~2**20000.3`
      - other ints as their digits, eg. `1000`
    """
    var magnitude = Signed.from_int(ints, total).magnitude.copy()
    var bits = magnitude.bits()
    if bits > 10 and magnitude.count_ones() == 1:
        return String(t"2**{bits - 1}")
    # Writing out more digits than Python's limit takes quadratic time.
    if bits > WRITABLE_BITS:
        var shift = bits - 64
        var top = (magnitude >> shift).limb(0)
        var tenths = Int((Float64(shift) + log2(Float64(top))) * 10 + 0.5)
        return String(t"~2**{tenths // 10}.{tenths % 10}")
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
