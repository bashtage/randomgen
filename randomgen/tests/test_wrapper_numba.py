import ctypes

import numpy as np
from numpy.random import Generator
import pytest

from randomgen.wrapper import UserBitGenerator

HAS_NUMBA = False
try:
    from numba import carray, cfunc, jit, types

    HAS_NUMBA = True
except ImportError:
    pytestmark = pytest.mark.skip

if HAS_NUMBA:
    murmur_hash_3_sig = types.uint64(types.uint64)

    @jit(signature_or_function=murmur_hash_3_sig, inline="always", nopython=True)
    def murmur_hash_3(z):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

    split_mix_next_sig = types.uint64(types.uint64[:])

    @jit(signature_or_function=split_mix_next_sig, inline="always", nopython=True)
    def splitmix_next(state):
        state[0] += 0x9E3779B97F4A7C15
        return murmur_hash_3(state[0])


class NumbaSplitMix64:
    def __init__(self, state):
        if not isinstance(state, (int, np.integer)) or not (0 <= state < 2**64):
            raise ValueError("state must be a valid uint64")
        # state[0] is the splitmix64 state
        # state[1] contains both the has_uint flag in bit 0
        #   uinteger in bits 32...63
        self._state = np.array([state, 0], dtype=np.uint64)
        self._next_raw = None
        self._next_64 = None
        self._next_32 = None
        self._next_double = None

    @property
    def state_address(self):
        return self._state.ctypes.data_as(ctypes.c_void_p)

    @property
    def next_64(self):
        # Ensure a reference is held
        self._next_64 = self.next_raw

        return self.next_raw

    @property
    def next_32(self):
        sig = types.uint32(types.CPointer(types.uint64))

        @cfunc(sig)
        def next_32(st):
            bit_gen_state = carray(st, (2,), dtype=np.uint64)
            if bit_gen_state[1] & np.uint64(0x1):
                out = bit_gen_state[1] >> np.uint64(32)
                bit_gen_state[1] = 0
                return out
            z = splitmix_next(bit_gen_state)
            bit_gen_state[1] = z | np.uint64(0x1)
            return z & 0xFFFFFFFF

        # Ensure a reference is held
        self._next_32 = next_32

        return next_32

    @property
    def next_double(self):
        sig = types.double(types.CPointer(types.uint64))

        @cfunc(sig)
        def next_double(st):
            bit_gen_state = carray(st, (2,), dtype=np.uint64)
            return (
                np.uint64(splitmix_next(bit_gen_state)) >> np.uint64(11)
            ) / 9007199254740992.0

        # Ensure a reference is held
        self._next_double = next_double

        return next_double

    @property
    def next_raw(self):
        sig = types.uint64(types.CPointer(types.uint64))

        @cfunc(sig)
        def next_64(st):
            bit_gen_state = carray(st, (2,), dtype=np.uint64)
            return splitmix_next(bit_gen_state)

        # Ensure a reference is held
        self._next_64 = next_64

        return next_64

    @property
    def state_getter(self):
        def f() -> dict:
            return {
                "bit_gen": type(self).__name__,
                "state": self._state[0],
                "has_uint": self._state[1] & np.uint64(0x1),
                "uinteger": self._state[1] >> np.uint64(32),
            }

        return f

    @property
    def state_setter(self):
        def f(value: dict):
            name = value.get("bit_gen", None)
            if name != type(self).__name__:
                raise ValueError(f"state must be from a {type(self).__name__}")
            self._state[0] = np.uint64(value["state"])
            temp = np.uint64(value["uinteger"]) << np.uint64(32)
            temp |= np.uint64(value["has_uint"]) & np.uint64(0x1)
            self._state[1] = temp

        return f


@pytest.fixture(scope="module")
def split_mix():
    return NumbaSplitMix64(12345)


def test_ctypes_smoke(split_mix):
    bgf = UserBitGenerator.from_ctypes(
        split_mix.next_raw.ctypes,
        split_mix.next_64.ctypes,
        split_mix.next_32.ctypes,
        split_mix.next_double.ctypes,
        split_mix.state_address,
        state_getter=split_mix.state_getter,
        state_setter=split_mix.state_setter,
    )
    gen = Generator(bgf)
    gen.standard_normal(size=10)
    assert bgf.state == split_mix.state_getter()
    gen.standard_normal(dtype=np.float32)
    assert bgf.state == split_mix.state_getter()
    gen.integers(0, 2**63, dtype=np.uint64, size=10)
    assert bgf.state == split_mix.state_getter()
    old_state = bgf.state.copy()
    old_state["state"] = 1
    bgf.state = old_state
    assert bgf.state == split_mix.state_getter()


def test_cfunc_smoke(split_mix):
    bgf = UserBitGenerator.from_cfunc(
        split_mix.next_raw,
        split_mix.next_64,
        split_mix.next_32,
        split_mix.next_double,
        split_mix.state_address,
        state_getter=split_mix.state_getter,
        state_setter=split_mix.state_setter,
    )
    gen = Generator(bgf)
    gen.standard_normal(size=10)
    assert bgf.state == split_mix.state_getter()
    gen.standard_normal(dtype=np.float32)
    assert bgf.state == split_mix.state_getter()
    gen.integers(0, 2**63, dtype=np.uint64, size=10)
    assert bgf.state == split_mix.state_getter()
    old_state = bgf.state.copy()
    old_state["state"] = 1
    bgf.state = old_state
    assert bgf.state == split_mix.state_getter()


def test_no_setter_getter(split_mix):
    bgf = UserBitGenerator.from_cfunc(
        split_mix.next_raw,
        split_mix.next_64,
        split_mix.next_32,
        split_mix.next_double,
        split_mix.state_address,
    )
    gen = Generator(bgf)
    gen.standard_normal(size=10)
    gen.standard_normal(size=10, dtype=np.float32)
    gen.integers(0, 2**63, dtype=np.uint64, size=10)
    with pytest.raises(NotImplementedError):
        bgf.state
    with pytest.raises(NotImplementedError):
        bgf.state = {"apple"}

    bgf = UserBitGenerator.from_cfunc(
        split_mix.next_raw,
        split_mix.next_64,
        split_mix.next_32,
        split_mix.next_double,
        split_mix.state_address,
        state_getter=split_mix.state_getter,
    )
    assert isinstance(bgf.state, dict)
    with pytest.raises(NotImplementedError):
        bgf.state = {"apple"}

    bgf = UserBitGenerator.from_cfunc(
        split_mix.next_raw,
        split_mix.next_64,
        split_mix.next_32,
        split_mix.next_double,
        split_mix.state_address,
        state_setter=split_mix.state_setter,
    )
    bgf.state = split_mix.state_getter()


def test_invalid():
    with pytest.raises(TypeError, match="next_raw must be"):
        UserBitGenerator.from_cfunc(
            "next_raw", "next_64", "next_32", "next_double", "state"
        )
    with pytest.raises(TypeError, match="next_raw must be"):
        UserBitGenerator.from_ctypes(
            "next_raw", "next_64", "next_32", "next_double", "state"
        )
