"""Unsigned integers of any size, for the counts, members and ranks of sets wider than
128 bits."""

from std.bit import count_leading_zeros, pop_count
from std.ffi import external_call
from std.hashlib import Hasher
from std.memory import unsafe_memcmp, unsafe_memcpy, unsafe_memset_zero
from std.os import abort
from std.sys import inlined_assembly
from std.sys.info import CompilationTarget, is_little_endian

# The largest power of ten in a limb.
comptime DECIMAL_CHUNK = UInt64(10_000_000_000_000_000_000)
comptime DECIMAL_CHUNK_DIGITS = 19
comptime DIGIT_PAIRS = StaticString(
    "00010203040506070809101112131415161718192021222324252627282930313233343536"
    "37383940414243444546474849505152535455565758596061626364656667686970717273"
    "7475767778798081828384858687888990919293949596979899"
)


struct Limbs(Boolable, Copyable, Movable, Sized):
    """A growable array of limbs.

    It allocates with malloc: Mojo's List goes through the runtime's aligned
    allocator, which takes about four times as long for small blocks.
    """

    var _data: Pointer[UInt64, MutUntrackedOrigin]
    var _size: Int
    var _capacity: Int

    def __init__(out self):
        self._data = Pointer[UInt64, MutUntrackedOrigin].unsafe_dangling()
        self._size = 0
        self._capacity = 0

    def __init__(out self, *, capacity: Int):
        self = Self()
        self.reserve(capacity)

    def __init__(out self, *, unsafe_uninit_length: Int):
        self = Self(capacity=unsafe_uninit_length)
        self._size = unsafe_uninit_length

    def __init__(out self, *, length: Int, fill: UInt64):
        self = Self(unsafe_uninit_length=length)
        for i in range(length):
            self._data[unsafe_offset=i] = fill

    def __init__(out self, *, copy: Self):
        self = Self(unsafe_uninit_length=copy._size)
        unsafe_memcpy(dest=self._data, src=copy._data, count=copy._size)

    def __deinit__(deinit self):
        if self._capacity:
            external_call["free", NoneType](self._data)

    @always_inline
    def __len__(self) -> Int:
        return self._size

    @always_inline
    def __bool__(self) -> Bool:
        return self._size != 0

    @always_inline
    def unsafe_ptr[
        origin: Origin, //
    ](ref[origin] self) -> Pointer[UInt64, origin]:
        return self._data.unsafe_mut_cast[origin.mut]().unsafe_origin_cast[
            origin
        ]()

    @always_inline
    def unsafe_get[
        origin: Origin, //
    ](ref[origin] self, i: Int) -> ref[origin] UInt64:
        assert 0 <= i < self._size
        return self.unsafe_ptr()[unsafe_offset=i]

    def reserve(mut self, capacity: Int):
        if capacity <= self._capacity:
            return
        var address: Int
        if self._capacity:
            address = external_call["realloc", Int](self._data, capacity * 8)
        else:
            address = external_call["malloc", Int](capacity * 8)
        if address == 0:
            abort("out of memory")
        self._data = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=address
        )
        self._capacity = capacity

    @always_inline
    def append(mut self, limb: UInt64):
        if self._size == self._capacity:
            self.reserve(max(4, 2 * self._capacity))
        self._data[unsafe_offset=self._size] = limb
        self._size += 1

    def resize(mut self, length: Int, fill: UInt64):
        self.reserve(length)
        for i in range(self._size, length):
            self._data[unsafe_offset=i] = fill
        self._size = length

    def resize(mut self, *, unsafe_uninit_length: Int):
        self.reserve(unsafe_uninit_length)
        self._size = unsafe_uninit_length

    @always_inline
    def shrink(mut self, length: Int):
        assert length <= self._size
        self._size = length

    @always_inline
    def clear(mut self):
        self._size = 0


struct BigUInt(
    Boolable,
    Comparable,
    Copyable,
    Defaultable,
    Hashable,
    Movable,
    Writable,
):
    """An unsigned integer of any size."""

    var limbs: Limbs
    """Least significant first, and never a zero last, so zero has none."""

    def __init__(out self):
        self.limbs = Limbs()

    def __init__(out self, value: UInt64):
        self.limbs = Limbs()
        if value:
            self.limbs.append(value)

    def __init__(out self, value: UInt128):
        self.limbs = Limbs()
        var high = UInt64(value >> 64)
        if high:
            self.limbs.reserve(2)
            self.limbs.append(UInt64(value))
            self.limbs.append(high)
        elif value:
            self.limbs.append(UInt64(value))

    def __init__(out self, value: IntLiteral):
        # Else a literal could be a UInt64 or a UInt128.
        self = Self(UInt128(value))

    def __init__(out self, *, var limbs: Limbs):
        """Least significant first. High zero limbs are dropped."""
        self.limbs = limbs^
        _trim(self.limbs)

    def __init__(out self, *, limbs: List[UInt64]):
        """Least significant first. High zero limbs are dropped."""
        self.limbs = Limbs(unsafe_uninit_length=len(limbs))
        unsafe_memcpy(
            dest=self.limbs.unsafe_ptr(),
            src=limbs.unsafe_ptr(),
            count=len(limbs),
        )
        _trim(self.limbs)

    @staticmethod
    def mask(bits: Int) -> Self:
        """`2**bits - 1`."""
        var limbs = Limbs(capacity=(bits + 63) >> 6)
        limbs.resize(bits >> 6, ~UInt64(0))
        if bits & 63:
            limbs.append(_low_mask(bits & 63))
        return Self(limbs=limbs^)

    @staticmethod
    def from_bytes_le(bytes: Span[UInt8, _]) -> Self:
        comptime assert is_little_endian()
        var n = len(bytes)
        var limbs = Limbs(length=(n + 7) >> 3, fill=0)
        unsafe_memcpy(
            dest=limbs.unsafe_ptr().unsafe_bitcast[UInt8](),
            src=bytes.unsafe_ptr(),
            count=n,
        )
        return Self(limbs=limbs^)

    def to_bytes_le(self) -> List[UInt8]:
        """The fewest bytes that hold it, so none for zero."""
        comptime assert is_little_endian()
        var n = (self.bits() + 7) >> 3
        var bytes = List[UInt8](unsafe_uninit_length=n)
        unsafe_memcpy(
            dest=bytes.unsafe_ptr(),
            src=self.limbs.unsafe_ptr().unsafe_bitcast[UInt8](),
            count=n,
        )
        return bytes^

    def fitting_uint64(self) -> Optional[UInt64]:
        if len(self.limbs) > 1:
            return None
        return self._limb(0)

    def fitting_uint128(self) -> Optional[UInt128]:
        if len(self.limbs) > 2:
            return None
        return UInt128(self._limb(0)) | UInt128(self._limb(1)) << 64

    @always_inline
    def _limb(self, i: Int) -> UInt64:
        return self.limbs.unsafe_get(i) if i < len(self.limbs) else 0

    def is_zero(self) -> Bool:
        return not self.limbs

    def __bool__(self) -> Bool:
        return Bool(self.limbs)

    def bits(self) -> Int:
        """How many bits it takes: 0 for zero."""
        var n = len(self.limbs)
        if n == 0:
            return 0
        return 64 * n - Int(count_leading_zeros(self.limbs.unsafe_get(n - 1)))

    def count_ones(self) -> Int:
        var total = 0
        for i in range(len(self.limbs)):
            total += Int(pop_count(self.limbs.unsafe_get(i)))
        return total

    def bit(self, i: Int) -> Bool:
        assert i >= 0
        var limb = i >> 6
        if limb >= len(self.limbs):
            return False
        return (self.limbs.unsafe_get(limb) >> UInt64(i & 63)) & 1 == 1

    def set_bit(mut self, i: Int, value: Bool):
        assert i >= 0
        var limb = i >> 6
        var flag = UInt64(1) << UInt64(i & 63)
        if value:
            if limb >= len(self.limbs):
                self.limbs.resize(limb + 1, 0)
            self.limbs.unsafe_get(limb) |= flag
        elif limb < len(self.limbs):
            self.limbs.unsafe_get(limb) &= ~flag
            _trim(self.limbs)

    def low_bits(self, bits: Int) -> Self:
        """`self & BigUInt.mask(bits)`."""
        var full = bits >> 6
        if full >= len(self.limbs):
            return self.copy()
        var limbs = Limbs(capacity=full + 1)
        limbs.resize(unsafe_uninit_length=full)
        unsafe_memcpy(
            dest=limbs.unsafe_ptr(), src=self.limbs.unsafe_ptr(), count=full
        )
        if bits & 63:
            limbs.append(self.limbs.unsafe_get(full) & _low_mask(bits & 63))
        return Self(limbs=limbs^)

    def __hash__(self, mut hasher: Some[Hasher]):
        hasher.update(
            Span(
                unsafe_ptr=self.limbs.unsafe_ptr().unsafe_bitcast[UInt8](),
                length=8 * len(self.limbs),
            )
        )

    # Comparisons

    def _order(self, rhs: Self) -> Int:
        """Negative, zero or positive as `self` is below, at or above `rhs`."""
        var n = len(self.limbs)
        if n != len(rhs.limbs):
            return n - len(rhs.limbs)
        for i in range(n - 1, -1, -1):
            var a = self.limbs.unsafe_get(i)
            var b = rhs.limbs.unsafe_get(i)
            if a != b:
                return 1 if a > b else -1
        return 0

    def __eq__(self, rhs: Self) -> Bool:
        var n = len(self.limbs)
        return n == len(rhs.limbs) and (
            unsafe_memcmp(
                self.limbs.unsafe_ptr().as_imm(),
                rhs.limbs.unsafe_ptr().as_imm(),
                n,
            )
            == 0
        )

    def __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    def __lt__(self, rhs: Self) -> Bool:
        return self._order(rhs) < 0

    def __le__(self, rhs: Self) -> Bool:
        return self._order(rhs) <= 0

    def __gt__(self, rhs: Self) -> Bool:
        return self._order(rhs) > 0

    def __ge__(self, rhs: Self) -> Bool:
        return self._order(rhs) >= 0

    # Bitwise

    def __and__(self, rhs: Self) -> Self:
        var n = min(len(self.limbs), len(rhs.limbs))
        var limbs = Limbs(unsafe_uninit_length=n)
        for i in range(n):
            limbs.unsafe_get(i) = self.limbs.unsafe_get(
                i
            ) & rhs.limbs.unsafe_get(i)
        return Self(limbs=limbs^)

    def __iand__(mut self, rhs: Self):
        var n = min(len(self.limbs), len(rhs.limbs))
        self.limbs.shrink(n)
        for i in range(n):
            self.limbs.unsafe_get(i) &= rhs.limbs.unsafe_get(i)
        _trim(self.limbs)

    def __or__(self, rhs: Self) -> Self:
        if len(self.limbs) < len(rhs.limbs):
            return rhs | self
        var limbs = self.limbs.copy()
        for i in range(len(rhs.limbs)):
            limbs.unsafe_get(i) |= rhs.limbs.unsafe_get(i)
        return Self(limbs=limbs^)

    def __ior__(mut self, rhs: Self):
        if len(self.limbs) < len(rhs.limbs):
            self.limbs.resize(len(rhs.limbs), 0)
        for i in range(len(rhs.limbs)):
            self.limbs.unsafe_get(i) |= rhs.limbs.unsafe_get(i)

    def __lshift__(self, shift: Int) -> Self:
        assert shift >= 0
        var n = len(self.limbs)
        if n == 0:
            return Self()
        var whole = shift >> 6
        var limbs = Limbs(capacity=n + whole + 1)
        limbs.resize(unsafe_uninit_length=n + whole)
        var dest = limbs.unsafe_ptr()
        unsafe_memset_zero(dest, whole)
        if shift & 63:
            var top = _shifted_left(
                dest.unsafe_offset(whole),
                self.limbs.unsafe_ptr(),
                n,
                UInt64(shift & 63),
            )
            if top:
                limbs.append(top)
        else:
            unsafe_memcpy(
                dest=dest.unsafe_offset(whole),
                src=self.limbs.unsafe_ptr(),
                count=n,
            )
        return Self(limbs=limbs^)

    def __ilshift__(mut self, shift: Int):
        self = self << shift

    def __rshift__(self, shift: Int) -> Self:
        assert shift >= 0
        var whole = shift >> 6
        var n = len(self.limbs) - whole
        if n <= 0:
            return Self()
        var limbs = Limbs(unsafe_uninit_length=n)
        var low = self.limbs.unsafe_ptr().unsafe_offset(whole)
        if shift & 63:
            _shifted_right(limbs.unsafe_ptr(), low, n, UInt64(shift & 63))
        else:
            unsafe_memcpy(dest=limbs.unsafe_ptr(), src=low, count=n)
        return Self(limbs=limbs^)

    def __irshift__(mut self, shift: Int):
        assert shift >= 0
        var whole = shift >> 6
        var n = len(self.limbs) - whole
        if n <= 0:
            self.limbs.clear()
            return
        var limbs = self.limbs.unsafe_ptr()
        var part = UInt64(shift & 63)
        if part:
            for i in range(n - 1):
                limbs[unsafe_offset=i] = limbs[
                    unsafe_offset=i + whole
                ] >> part | (limbs[unsafe_offset=i + whole + 1] << (64 - part))
            limbs[unsafe_offset=n - 1] = (
                limbs[unsafe_offset=n - 1 + whole] >> part
            )
        elif whole:
            for i in range(n):
                limbs[unsafe_offset=i] = limbs[unsafe_offset=i + whole]
        self.limbs.shrink(n)
        _trim(self.limbs)

    # Arithmetic

    def __add__(self, rhs: Self) -> Self:
        if len(self.limbs) < len(rhs.limbs):
            return Self(limbs=_sum(rhs.limbs, self.limbs))
        return Self(limbs=_sum(self.limbs, rhs.limbs))

    def __add__(self, rhs: UInt64) -> Self:
        var n = len(self.limbs)
        var limbs = Limbs(capacity=n + 1)
        limbs.resize(unsafe_uninit_length=n)
        var dest = limbs.unsafe_ptr()
        var carry = rhs
        for i in range(n):
            var s = self.limbs.unsafe_get(i) + carry
            carry = UInt64(s < carry)
            dest[unsafe_offset=i] = s
        if carry:
            limbs.append(carry)
        return Self(limbs=limbs^)

    def __iadd__(mut self, rhs: Self):
        var n = len(rhs.limbs)
        if len(self.limbs) < n:
            self.limbs.resize(n, 0)
        var carry = _add_to(self.limbs.unsafe_ptr(), rhs.limbs.unsafe_ptr(), n)
        self._carry_from(n, carry)

    def __iadd__(mut self, rhs: UInt64):
        self._carry_from(0, rhs)

    def _carry_from(mut self, i: Int, carry: UInt64):
        """Adds `carry` at limb `i`."""
        var at = i
        var rest = carry
        while rest and at < len(self.limbs):
            var s = self.limbs.unsafe_get(at) + rest
            rest = UInt64(s < rest)
            self.limbs.unsafe_get(at) = s
            at += 1
        if rest:
            self.limbs.append(rest)

    def __sub__(self, rhs: Self) -> Self:
        """`rhs` must not be above `self`."""
        var limbs = self.limbs.copy()
        _subtract(limbs, rhs.limbs)
        return Self(limbs=limbs^)

    def __isub__(mut self, rhs: Self):
        """`rhs` must not be above `self`."""
        _subtract(self.limbs, rhs.limbs)

    def __mul__(self, rhs: Self) -> Self:
        var na = len(self.limbs)
        var nb = len(rhs.limbs)
        if na == 0 or nb == 0:
            return Self()
        if na < nb:
            return rhs * self
        # A row of `self * limb` for each of rhs's limbs.
        var limbs = Limbs(unsafe_uninit_length=na + nb)
        var dest = limbs.unsafe_ptr()
        var a = self.limbs.unsafe_ptr()
        dest[unsafe_offset=na] = _multiplied(
            dest, a, na, rhs.limbs.unsafe_get(0)
        )
        for i in range(1, nb):
            dest[unsafe_offset=i + na] = _added_product(
                dest.unsafe_offset(i), a, na, rhs.limbs.unsafe_get(i)
            )
        return Self(limbs=limbs^)

    def __mul__(self, rhs: UInt64) -> Self:
        var n = len(self.limbs)
        if n == 0 or rhs == 0:
            return Self()
        var limbs = Limbs(capacity=n + 1)
        limbs.resize(unsafe_uninit_length=n)
        var top = _multiplied(
            limbs.unsafe_ptr(), self.limbs.unsafe_ptr(), n, rhs
        )
        if top:
            limbs.append(top)
        return Self(limbs=limbs^)

    def __imul__(mut self, rhs: Self):
        self = self * rhs

    def __imul__(mut self, rhs: UInt64):
        if rhs == 0:
            self.limbs.clear()
            return
        var limbs = self.limbs.unsafe_ptr()
        var carry = UInt64(0)
        for i in range(len(self.limbs)):
            var p = UInt128(limbs[unsafe_offset=i]) * UInt128(rhs) + UInt128(
                carry
            )
            limbs[unsafe_offset=i] = UInt64(p)
            carry = UInt64(p >> 64)
        if carry:
            self.limbs.append(carry)

    def __divmod__(self, divisor: Self) -> Tuple[Self, Self]:
        var quotient = Self()
        var remainder = Self()
        _divide(self.limbs, divisor.limbs, quotient.limbs, remainder.limbs)
        return (quotient^, remainder^)

    def __floordiv__(self, divisor: Self) -> Self:
        var quotient = Self()
        var remainder = Self()
        _divide(self.limbs, divisor.limbs, quotient.limbs, remainder.limbs)
        return quotient^

    def __mod__(self, divisor: Self) -> Self:
        var quotient = Self()
        var remainder = Self()
        _divide(self.limbs, divisor.limbs, quotient.limbs, remainder.limbs)
        return remainder^

    # Text

    def write_to(self, mut writer: Some[Writer]):
        var n = len(self.limbs)
        if n <= 2:
            writer.write(UInt128(self._limb(0)) | UInt128(self._limb(1)) << 64)
            return
        # Chunks of 19 digits, least significant first.
        var rest = self.limbs.copy()
        var chunks = Limbs(capacity=n + n // 64 + 2)
        while rest:
            chunks.append(
                _divide_by(rest.unsafe_ptr(), len(rest), DECIMAL_CHUNK)
            )
            _trim(rest)
        writer.write(chunks.unsafe_get(len(chunks) - 1))
        var digits = Array[UInt8, DECIMAL_CHUNK_DIGITS](uninitialized=True)
        var pairs = DIGIT_PAIRS.unsafe_ptr()
        for i in range(len(chunks) - 2, -1, -1):
            var chunk = chunks.unsafe_get(i)
            digits[0] = UInt8(48) + UInt8(chunk // 10**18)
            chunk %= 10**18
            for at in range(DECIMAL_CHUNK_DIGITS - 1, 0, -2):
                var pair = Int(chunk % 100) * 2
                chunk //= 100
                digits[at] = pairs[unsafe_offset=pair + 1]
                digits[at - 1] = pairs[unsafe_offset=pair]
            writer.write_string(StringSlice(unsafe_from_utf8=Span(digits)))

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("BigUInt(", self, ")")


# Limb kernels. Pointers name `n` limbs, least significant first.


@always_inline
def _low_mask(bits: Int) -> UInt64:
    """The low `bits` set, for 0 < bits < 64."""
    return (UInt64(1) << UInt64(bits)) - 1


def _trim(mut limbs: Limbs):
    var n = len(limbs)
    while n > 0 and limbs.unsafe_get(n - 1) == 0:
        n -= 1
    limbs.shrink(n)


@always_inline
def _added(
    dest: MutPointer[UInt64, _],
    a: Pointer[UInt64, _],
    b: Pointer[UInt64, _],
    n: Int,
) -> UInt64:
    """Writes `a + b`, and returns the carry."""
    var carry = UInt64(0)
    for i in range(n):
        var s = (
            UInt128(a[unsafe_offset=i])
            + UInt128(b[unsafe_offset=i])
            + UInt128(carry)
        )
        dest[unsafe_offset=i] = UInt64(s)
        carry = UInt64(s >> 64)
    return carry


@always_inline
def _add_to(
    dest: MutPointer[UInt64, _], b: Pointer[UInt64, _], n: Int
) -> UInt64:
    """Adds `b` to `dest`, and returns the carry."""
    var carry = UInt64(0)
    for i in range(n):
        var s = (
            UInt128(dest[unsafe_offset=i])
            + UInt128(b[unsafe_offset=i])
            + UInt128(carry)
        )
        dest[unsafe_offset=i] = UInt64(s)
        carry = UInt64(s >> 64)
    return carry


def _sum(a: Limbs, b: Limbs) -> Limbs:
    """`a` has at least as many limbs as `b`."""
    var na = len(a)
    var nb = len(b)
    var limbs = Limbs(capacity=na + 1)
    limbs.resize(unsafe_uninit_length=na)
    var dest = limbs.unsafe_ptr()
    var carry = _added(dest, a.unsafe_ptr(), b.unsafe_ptr(), nb)
    for i in range(nb, na):
        var s = a.unsafe_get(i) + carry
        carry = UInt64(s < carry)
        dest[unsafe_offset=i] = s
    if carry:
        limbs.append(carry)
    return limbs^


def _subtract(mut a: Limbs, b: Limbs):
    """`a -= b`, for `b <= a`."""
    var nb = len(b)
    assert nb <= len(a)
    var dest = a.unsafe_ptr()
    var borrow = UInt64(0)
    for i in range(nb):
        var d = (
            UInt128(dest[unsafe_offset=i])
            - UInt128(b.unsafe_get(i))
            - UInt128(borrow)
        )
        dest[unsafe_offset=i] = UInt64(d)
        borrow = UInt64(d >> 127)
    var i = nb
    while borrow:
        assert i < len(a), "b is above a"
        var limb = dest[unsafe_offset=i]
        dest[unsafe_offset=i] = limb - 1
        borrow = UInt64(limb == 0)
        i += 1
    _trim(a)


@always_inline
def _multiplied(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Writes `a * m`, and returns the limb above."""
    var carry = UInt64(0)
    for i in range(n):
        var p = UInt128(a[unsafe_offset=i]) * UInt128(m) + UInt128(carry)
        dest[unsafe_offset=i] = UInt64(p)
        carry = UInt64(p >> 64)
    return carry


@always_inline
def _added_product(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Adds `a * m` to `dest`, and returns the carry."""
    var carry = UInt64(0)
    for i in range(n):
        # At most (2**64 - 1)**2 + 2 * (2**64 - 1), which fits.
        var p = (
            UInt128(a[unsafe_offset=i]) * UInt128(m)
            + UInt128(dest[unsafe_offset=i])
            + UInt128(carry)
        )
        dest[unsafe_offset=i] = UInt64(p)
        carry = UInt64(p >> 64)
    return carry


@always_inline
def _subtracted_product(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Subtracts `a * m` from `dest`, and returns the borrow."""
    var borrow = UInt64(0)
    for i in range(n):
        var p = UInt128(a[unsafe_offset=i]) * UInt128(m) + UInt128(borrow)
        var low = UInt64(p)
        var limb = dest[unsafe_offset=i]
        dest[unsafe_offset=i] = limb - low
        borrow = UInt64(p >> 64) + UInt64(limb < low)
    return borrow


@always_inline
def _shifted_left(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, shift: UInt64
) -> UInt64:
    """Writes `a << shift`, for 0 < shift < 64, and returns the bits shifted out
    the top."""
    var top = a[unsafe_offset=n - 1] >> (64 - shift)
    for i in range(n - 1, 0, -1):
        dest[unsafe_offset=i] = a[unsafe_offset=i] << shift | (
            a[unsafe_offset=i - 1] >> (64 - shift)
        )
    dest[unsafe_offset=0] = a[unsafe_offset=0] << shift
    return top


@always_inline
def _shifted_right(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, shift: UInt64
):
    """Writes `a >> shift`, for 0 < shift < 64."""
    for i in range(n - 1):
        dest[unsafe_offset=i] = a[unsafe_offset=i] >> shift | (
            a[unsafe_offset=i + 1] << (64 - shift)
        )
    dest[unsafe_offset=n - 1] = a[unsafe_offset=n - 1] >> shift


@fieldwise_init
struct _Division(TrivialRegisterPassable):
    var quotient: UInt64
    var remainder: UInt64


@always_inline
def _two_by_one(high: UInt64, low: UInt64, divisor: UInt64) -> _Division:
    """`(high, low) / divisor`, for high < divisor so the quotient fits a limb.
    """
    comptime if CompilationTarget.is_x86():
        # A UInt128 division calls __udivti3, which takes twice as long.
        return inlined_assembly[
            "divq $4",
            _Division,
            constraints="={rax},={rdx},{rax},{rdx},r,~{flags}",
            has_side_effect=False,
        ](low, high, divisor)
    var q = UInt64(((UInt128(high) << 64) | UInt128(low)) // UInt128(divisor))
    return _Division(q, low - q * divisor)


def _divided(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, divisor: UInt64
) -> UInt64:
    """Writes `a / divisor`, and returns the remainder."""
    var rest = UInt64(0)
    for i in range(n - 1, -1, -1):
        var step = _two_by_one(rest, a[unsafe_offset=i], divisor)
        dest[unsafe_offset=i] = step.quotient
        rest = step.remainder
    return rest


def _divide_by(a: MutPointer[UInt64, _], n: Int, divisor: UInt64) -> UInt64:
    """Divides `a` by `divisor`, and returns the remainder."""
    var rest = UInt64(0)
    for i in range(n - 1, -1, -1):
        var step = _two_by_one(rest, a[unsafe_offset=i], divisor)
        a[unsafe_offset=i] = step.quotient
        rest = step.remainder
    return rest


def _divide(
    a: Limbs,
    b: Limbs,
    mut quotient: Limbs,
    mut remainder: Limbs,
):
    """Knuth's algorithm D (TAOCP vol. 2, 4.3.1)."""
    var n = len(b)
    var na = len(a)
    if n == 0:
        abort("division by zero")
    if na < n or na == n and _less(a, b):
        quotient.clear()
        remainder = a.copy()
        return
    if n == 1:
        quotient.resize(unsafe_uninit_length=na)
        var rest = _divided(
            quotient.unsafe_ptr(), a.unsafe_ptr(), na, b.unsafe_get(0)
        )
        _trim(quotient)
        remainder.clear()
        if rest:
            remainder.append(rest)
        return
    # Shift both so the divisor's top bit is set, which keeps each estimated
    # quotient limb at most 2 too big.
    var shift = count_leading_zeros(b.unsafe_get(n - 1))
    var v = Limbs(unsafe_uninit_length=n)
    var u = Limbs(unsafe_uninit_length=na + 1)
    if shift:
        _ = _shifted_left(v.unsafe_ptr(), b.unsafe_ptr(), n, shift)
        u.unsafe_get(na) = _shifted_left(
            u.unsafe_ptr(), a.unsafe_ptr(), na, shift
        )
    else:
        unsafe_memcpy(dest=v.unsafe_ptr(), src=b.unsafe_ptr(), count=n)
        unsafe_memcpy(dest=u.unsafe_ptr(), src=a.unsafe_ptr(), count=na)
        u.unsafe_get(na) = 0
    var m = na - n
    quotient.resize(unsafe_uninit_length=m + 1)
    var top = v.unsafe_get(n - 1)
    var next = v.unsafe_get(n - 2)
    var up = u.unsafe_ptr()
    for j in range(m, -1, -1):
        # Estimate the quotient limb from the top two limbs of u and the top
        # limb of v. u's top limb is at most v's.
        var high = up[unsafe_offset=j + n]
        var low = up[unsafe_offset=j + n - 1]
        var q: UInt64
        var r: UInt128
        if high >= top:
            q = ~UInt64(0)
            r = UInt128(low) + UInt128(top)
        else:
            var step = _two_by_one(high, low, top)
            q = step.quotient
            r = UInt128(step.remainder)
        # Correct it with v's next limb, unless r has passed a limb.
        while r >> 64 == 0 and UInt128(q) * UInt128(next) > (
            r << 64 | UInt128(up[unsafe_offset=j + n - 2])
        ):
            q -= 1
            r += UInt128(top)
        var borrow = _subtracted_product(
            up.unsafe_offset(j), v.unsafe_ptr(), n, q
        )
        var above = up[unsafe_offset=j + n]
        up[unsafe_offset=j + n] = above - borrow
        if above < borrow:
            # Rarely, q was still one too big.
            q -= 1
            var carry = _add_to(up.unsafe_offset(j), v.unsafe_ptr(), n)
            up[unsafe_offset=j + n] += carry
        quotient.unsafe_get(j) = q
    _trim(quotient)
    remainder.resize(unsafe_uninit_length=n)
    if shift:
        _shifted_right(remainder.unsafe_ptr(), up, n, shift)
    else:
        unsafe_memcpy(dest=remainder.unsafe_ptr(), src=up, count=n)
    _trim(remainder)


def _less(a: Limbs, b: Limbs) -> Bool:
    """For limbs of the same length."""
    for i in range(len(a) - 1, -1, -1):
        var x = a.unsafe_get(i)
        var y = b.unsafe_get(i)
        if x != y:
            return x < y
    return False
