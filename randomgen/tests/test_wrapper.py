import ctypes

import numpy as np
from numpy.random import Generator
import pytest

from randomgen.pcg64 import PCG64
from randomgen.wrapper import UserBitGenerator


def rotr_64(value, rot):
    value = np.uint64(value)
    rot = np.uint64(rot)
    with np.errstate(over="ignore"):
        return int((value >> rot) | (value << ((-rot) & np.uint(63))))


class _PCG64:
    PCG_DEFAULT_MULTIPLIER = (2549297995355413924 << 64) + 4865540595714422341
    MODULUS = 2**128

    def __init__(self, state, inc):
        self._state = state
        self._inc = inc
        self._has_uint32 = False
        self._uinteger = 0

    def random_raw(self):
        state = self._state * self.PCG_DEFAULT_MULTIPLIER + self._inc
        state = state % self.MODULUS
        self._state = state
        return rotr_64((state >> 64) ^ (state & 0xFFFFFFFFFFFFFFFF), state >> 122)

    def next_64(self):
        def _next_64(void_p):
            return self.random_raw()

        return _next_64

    def next_32(self):
        def _next_32(void_p):
            if self._has_uint32:
                self._has_uint32 = False
                return self._uinteger
            next_value = self.random_raw()
            self._has_uint32 = True
            self._uinteger = next_value >> 32
            return next_value & 0xFFFFFFFF

        return _next_32


PCG64_NATIVE = PCG64(0, None, variant="xsl-rr")
PCG64_INITIAL_STATE = PCG64_NATIVE.state


@pytest.fixture
def python_pcg(request):
    bit_gen = _PCG64(
        PCG64_INITIAL_STATE["state"]["state"],
        PCG64_INITIAL_STATE["state"]["inc"],
    )
    return bit_gen.next_64()


@pytest.fixture
def pcg_native(request):
    return PCG64(0, None, variant="xsl-rr")


@pytest.fixture
def pcg_python(request):
    bit_gen = _PCG64(
        PCG64_INITIAL_STATE["state"]["state"],
        PCG64_INITIAL_STATE["state"]["inc"],
    )
    return UserBitGenerator(bit_gen.next_64(), 64, next_32=bit_gen.next_32())


def test_smoke(python_pcg):
    bg = UserBitGenerator(python_pcg, 64)
    gen = Generator(bg)
    assert isinstance(gen.random(), float)
    assert isinstance(gen.standard_normal(dtype=np.float32), float)
    assert isinstance(gen.integers(0, 2**32, dtype=np.uint32), np.integer)
    assert isinstance(gen.integers(0, 2**64, dtype=np.uint64), np.integer)


def test_state_getter_setter():
    bit_gen = _PCG64(
        PCG64_INITIAL_STATE["state"]["state"],
        PCG64_INITIAL_STATE["state"]["inc"],
    )

    def state_getter():
        return {"state": bit_gen._state, "inc": bit_gen._inc}

    def state_setter(value):
        bit_gen._state = value["state"]
        bit_gen._inc = value["inc"]

    bg = UserBitGenerator(
        bit_gen.next_64(),
        64,
        next_32=bit_gen.next_32(),
        state_getter=state_getter,
        state_setter=state_setter,
    )
    state = bg.state
    first = bg.random_raw(10)
    bg.state = state
    np.testing.assert_equal(bg.random_raw(10), first)
    # The lock is re-entrant, so the state can be used while it is held
    with bg.lock:
        bg.state = bg.state
    native = PCG64(0, None, variant="xsl-rr")
    np.testing.assert_equal(bg.random_raw(), native.random_raw(11)[-1])


@pytest.mark.thread_unsafe(reason="compares draws from shared fixtures")
def test_random_raw(pcg_python, pcg_native):
    np.testing.assert_equal(pcg_python.random_raw(1000), pcg_native.random_raw(1000))


@pytest.mark.parametrize(
    "func",
    [
        lambda bg: bg.random(),
        lambda bg: bg.random(size=10),
        lambda bg: bg.standard_normal(),
        lambda bg: bg.standard_normal(size=(20, 3)),
        lambda bg: bg.standard_normal(dtype=np.float32),
        lambda bg: bg.standard_normal(size=(20, 3), dtype=np.float32),
        lambda bg: bg.integers(0, 2**32, dtype=np.uint32),
        lambda bg: bg.integers(0, 2**32, dtype=np.uint32, size=(7, 5, 3, 2)),
        lambda bg: bg.integers(0, 2**64, dtype=np.uint64),
        lambda bg: bg.integers(0, 2**32, dtype=np.uint64, size=(7, 5, 3, 2)),
    ],
    ids=[
        "base",
        "size-10",
        "standard-normal",
        "standard-normal-size-20-3",
        "standard-normal-float32",
        "standard-normal-size-20-3-float32",
        "integers-32",
        "integers-32-size-7-5-3-2",
        "integers-64",
        "integers-64-size-7-5-3-2",
    ],
)
@pytest.mark.thread_unsafe(reason="compares draws from shared fixtures")
def test_against_ref(func, pcg_python, pcg_native):
    a = func(Generator(pcg_python))
    b = func(Generator(pcg_native))
    np.testing.assert_allclose(a, b)


def test_32():
    def next_raw(vp):
        return np.iinfo(np.uint32).max

    bg = UserBitGenerator(next_raw, 32)
    assert bg.random_raw() == np.iinfo(np.uint32).max
    gen = Generator(bg)
    assert gen.integers(0, 2**64, dtype=np.uint64) == np.iinfo(np.uint64).max
    np.testing.assert_allclose(gen.random(), (2**53 - 1) / (2**53), rtol=1e-14)
    assert "UserBitGenerator(Python)" in repr(bg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"next_raw": 1234},
        {"next_raw": "next_raw"},
        {"next_raw": None},
        {"next_64": 1234},
        {"next_32": 1234},
        {"next_double": 1234},
        {"state_getter": 1234},
        {"state_setter": 1234},
    ],
)
def test_not_callable(kwargs):
    # An int is a function address to ctypes, and used to crash when called
    kwargs = {"next_raw": lambda void_p: 0, **kwargs}
    with pytest.raises(TypeError, match="must be callable"):
        UserBitGenerator(**kwargs)


def test_from_ctypes_not_callable():
    with pytest.raises(TypeError, match="next_raw must be a ctypes function"):
        UserBitGenerator.from_ctypes(1234, 1234, 1234, 1234, None)
    ct_raw = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(lambda void_p: 0)
    ct_32 = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(lambda void_p: 0)
    ct_dbl = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)(lambda void_p: 0.0)
    ct_args = (ct_raw, ct_raw, ct_32, ct_dbl, None)
    with pytest.raises(TypeError, match="state_getter must be callable"):
        UserBitGenerator.from_ctypes(*ct_args, state_getter=1234)
    with pytest.raises(TypeError, match="state_setter must be callable"):
        UserBitGenerator.from_ctypes(*ct_args, state_setter=1234)
