"""Unsigned integers of any size, for the counts, members and ranks of sets wider than
128 bits."""

from std.bit import count_leading_zeros, pop_count
from std.ffi import external_call
from std.hash import Hasher
from std.memory import unsafe_memcmp, unsafe_memcpy, unsafe_memset_zero
from std.os import abort
from std.sys import inlined_assembly
from std.sys.info import CompilationTarget


struct Limbs(Boolable, Copyable, Defaultable, Movable, Sized):
    """A growable array of limbs.

    Allocated with malloc since List's aligned allocator takes about four times as long
    for small blocks.
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
):
    """An unsigned integer of any size."""

    var limbs: Limbs
    """Least significant first, and never a zero last, so zero has none."""

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

    @implicit
    def __init__(out self, value: IntLiteral):
        # Else a literal could be a UInt64 or a UInt128.
        self = Self(UInt128(value))

    def __init__(out self, *, var limbs: Limbs):
        """High zero limbs are dropped."""
        self.limbs = limbs^
        _trim(self.limbs)

    @staticmethod
    def mask(width: Some[Indexer]) -> Self:
        """`2**width - 1`."""
        var bits = index(width)
        var limbs = Limbs(capacity=(bits + 63) >> 6)
        limbs.resize(bits >> 6, UInt64.MAX)
        if bits & 63:
            limbs.append(_low_mask(bits & 63))
        return Self(limbs=limbs^)

    def uint64_or_none(self) -> Optional[UInt64]:
        if len(self.limbs) > 1:
            return None
        return self.limb(0)

    def uint128_or_none(self) -> Optional[UInt128]:
        if len(self.limbs) > 2:
            return None
        return UInt128(self.limb(0)) | UInt128(self.limb(1)) << 64

    @always_inline
    def limb(self, i: Int) -> UInt64:
        return self.limbs.unsafe_get(i) if i < len(self.limbs) else 0

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

    def bit(self, bit: Some[Indexer]) -> Bool:
        var i = index(bit)
        assert i >= 0
        var limb = i >> 6
        if limb >= len(self.limbs):
            return False
        return (self.limbs.unsafe_get(limb) >> UInt64(i & 63)) & 1 == 1

    def set_bit(mut self, bit: Some[Indexer], value: Bool):
        var i = index(bit)
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

    def low_bits(self, width: Some[Indexer]) -> Self:
        """`self & BigUInt.mask(width)`."""
        var bits = index(width)
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

    def __lt__(self, rhs: Self) -> Bool:
        var n = len(self.limbs)
        if n != len(rhs.limbs):
            return n < len(rhs.limbs)
        return _less(self.limbs, rhs.limbs)

    # Bitwise

    def __or__(self, rhs: Self) -> Self:
        var result = self.copy()
        result |= rhs
        return result^

    def __ior__(mut self, rhs: Self):
        if len(self.limbs) < len(rhs.limbs):
            self.limbs.resize(len(rhs.limbs), 0)
        for i in range(len(rhs.limbs)):
            self.limbs.unsafe_get(i) |= rhs.limbs.unsafe_get(i)

    def __lshift__(self, rhs: Some[Indexer]) -> Self:
        var shift = index(rhs)
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
            var top = _shift_left(
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

    def __rshift__(self, rhs: Some[Indexer]) -> Self:
        var shift = index(rhs)
        assert shift >= 0
        var whole = shift >> 6
        var n = len(self.limbs) - whole
        if n <= 0:
            return Self()
        var limbs = Limbs(unsafe_uninit_length=n)
        var low = self.limbs.unsafe_ptr().unsafe_offset(whole)
        if shift & 63:
            _shift_right(limbs.unsafe_ptr(), low, n, UInt64(shift & 63))
        else:
            unsafe_memcpy(dest=limbs.unsafe_ptr(), src=low, count=n)
        return Self(limbs=limbs^)

    # Arithmetic

    def __add__(self, rhs: Self) -> Self:
        if len(self.limbs) < len(rhs.limbs):
            return Self(limbs=_sum(rhs.limbs, self.limbs))
        return Self(limbs=_sum(self.limbs, rhs.limbs))

    def plus(self, rhs: UInt64) -> Self:
        var result = self.copy()
        result._carry_from(0, rhs)
        return result^

    def __iadd__(mut self, rhs: Self):
        var n = len(rhs.limbs)
        if len(self.limbs) < n:
            self.limbs.resize(n, 0)
        var carry = _add_to(self.limbs.unsafe_ptr(), rhs.limbs.unsafe_ptr(), n)
        self._carry_from(n, carry)

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
        var result = self.copy()
        result -= rhs
        return result^

    def __isub__(mut self, rhs: Self):
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
        dest[unsafe_offset=na] = _multiply(dest, a, na, rhs.limbs.unsafe_get(0))
        for i in range(1, nb):
            dest[unsafe_offset=i + na] = _add_product_to(
                dest.unsafe_offset(i), a, na, rhs.limbs.unsafe_get(i)
            )
        return Self(limbs=limbs^)

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
def _add(
    dest: MutPointer[UInt64, _],
    a: Pointer[UInt64, _],
    b: Pointer[UInt64, _],
    n: Int,
) -> UInt64:
    """Writes `a + b` and returns the carry."""
    var carry: UInt64 = 0
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
    """Adds `b` to `dest` and returns the carry."""
    var carry: UInt64 = 0
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
    var carry = _add(dest, a.unsafe_ptr(), b.unsafe_ptr(), nb)
    for i in range(nb, na):
        var s = a.unsafe_get(i) + carry
        carry = UInt64(s < carry)
        dest[unsafe_offset=i] = s
    if carry:
        limbs.append(carry)
    return limbs^


def _subtract(mut a: Limbs, b: Limbs):
    var nb = len(b)
    assert nb <= len(a)
    var dest = a.unsafe_ptr()
    var borrow: UInt64 = 0
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
def _multiply(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Writes `a * m` and returns the limb above."""
    var carry: UInt64 = 0
    for i in range(n):
        var p = UInt128(a[unsafe_offset=i]) * UInt128(m) + UInt128(carry)
        dest[unsafe_offset=i] = UInt64(p)
        carry = UInt64(p >> 64)
    return carry


@always_inline
def _add_product_to(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Adds `a * m` to `dest` and returns the carry."""
    var carry: UInt64 = 0
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
def _subtract_product_from(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, m: UInt64
) -> UInt64:
    """Subtracts `a * m` from `dest` and returns the borrow."""
    var borrow: UInt64 = 0
    for i in range(n):
        var p = UInt128(a[unsafe_offset=i]) * UInt128(m) + UInt128(borrow)
        var low = UInt64(p)
        var limb = dest[unsafe_offset=i]
        dest[unsafe_offset=i] = limb - low
        borrow = UInt64(p >> 64) + UInt64(limb < low)
    return borrow


@always_inline
def _shift_left(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, shift: UInt64
) -> UInt64:
    """Writes `a << shift` (for 0 < shift < 64) and returns the bits shifted out
    the top."""
    var top = a[unsafe_offset=n - 1] >> (64 - shift)
    for i in reversed(range(1, n)):
        dest[unsafe_offset=i] = a[unsafe_offset=i] << shift | (
            a[unsafe_offset=i - 1] >> (64 - shift)
        )
    dest[unsafe_offset=0] = a[unsafe_offset=0] << shift
    return top


@always_inline
def _shift_right(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, shift: UInt64
):
    """Writes `a >> shift` (for 0 < shift < 64)."""
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
def _two_by_one(*, high: UInt64, low: UInt64, divisor: UInt64) -> _Division:
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


def _divide_by_limb(
    dest: MutPointer[UInt64, _], a: Pointer[UInt64, _], n: Int, divisor: UInt64
) -> UInt64:
    """Writes `a / divisor` and returns the remainder."""
    var rest: UInt64 = 0
    for i in reversed(range(n)):
        var step = _two_by_one(
            high=rest, low=a[unsafe_offset=i], divisor=divisor
        )
        dest[unsafe_offset=i] = step.quotient
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
        var rest = _divide_by_limb(
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
        _ = _shift_left(v.unsafe_ptr(), b.unsafe_ptr(), n, shift)
        u.unsafe_get(na) = _shift_left(
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
    for j in reversed(range(m + 1)):
        # Estimate the quotient limb from the top two limbs of u and the top
        # limb of v. u's top limb is at most v's.
        var high = up[unsafe_offset=j + n]
        var low = up[unsafe_offset=j + n - 1]
        var q: UInt64
        var r: UInt128
        if high >= top:
            q = UInt64.MAX
            r = UInt128(low) + UInt128(top)
        else:
            var step = _two_by_one(high=high, low=low, divisor=top)
            q = step.quotient
            r = UInt128(step.remainder)
        # Correct it with v's next limb, unless r has passed a limb.
        while r >> 64 == 0 and UInt128(q) * UInt128(next) > (
            r << 64 | UInt128(up[unsafe_offset=j + n - 2])
        ):
            q -= 1
            r += UInt128(top)
        var borrow = _subtract_product_from(
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
        _shift_right(remainder.unsafe_ptr(), up, n, shift)
    else:
        unsafe_memcpy(dest=remainder.unsafe_ptr(), src=up, count=n)
    _trim(remainder)


def _less(a: Limbs, b: Limbs) -> Bool:
    """For limbs of the same length."""
    for i in reversed(range(len(a))):
        var x = a.unsafe_get(i)
        var y = b.unsafe_get(i)
        if x != y:
            return x < y
    return False
