from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import os
from os.path import join
import pickle
import sys
import threading
from typing import Any, ClassVar

import numpy as np
from numpy.random import SeedSequence
import numpy.random._pickle as np_pickle
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)
import pytest

from randomgen import (
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    MT64,
    MT19937,
    MWC64X32,
    MWC128X,
    MWC128X32,
    MWC192X,
    PCG32,
    PCG64,
    PCG64DXSM,
    RDRAND,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    BlaBla,
    ChaCha,
    ExtendedGenerator,
    LCG128Mix,
    Philox,
    Romu,
    Squares,
    ThreeFry,
    Tyche,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
    test as rg_tester,
)
import randomgen._pickle as rg_pkl
from randomgen.common import Interface
from randomgen.seed_sequence import ISeedSequence
from randomgen.squares import _test_sentinal, generate_keys
from randomgen.tests._utility import CustomPartial

MISSING_RDRAND = False
try:
    RDRAND()
except RuntimeError:
    MISSING_RDRAND = True

HAS_AESNI = False
aes = AESCounter()
HAS_AESNI = aes.use_aesni

USE_AESNI = [True, False] if HAS_AESNI else [False]

try:
    import cffi  # noqa: F401

    MISSING_CFFI = False
except ImportError:
    MISSING_CFFI = True

try:
    import ctypes  # noqa: F401

    MISSING_CTYPES = False
except ImportError:
    MISSING_CTYPES = False

ISEED_SEQUENCES: tuple[Any, ...] = (
    ISeedSequence,
    np.random.bit_generator.ISeedSequence,
)

pwd = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module", params=(True, False))
def counter_only(request):
    return request.param


@pytest.fixture(scope="module", params=(0, 19813))
def warmup(request):
    return request.param


@pytest.fixture(scope="module", params=(0, 1, 2, 3, 4, 5, 7, 8, 9, 34159))
def step(request):
    return request.param


def assert_state_equal(actual, target):
    for key in actual:
        if isinstance(actual[key], dict):
            assert_state_equal(actual[key], target[key])
        elif isinstance(actual[key], np.ndarray):
            assert_array_equal(actual[key], target[key])
        else:
            assert actual[key] == target[key]


def uniform32_from_uint64(x, reversed=False):
    x = np.uint64(x)
    upper = np.array(x >> np.uint64(32), dtype=np.uint32)
    lower = np.uint64(0xFFFFFFFF)
    lower = np.array(x & lower, dtype=np.uint32)
    if reversed:
        joined = np.column_stack([upper, lower]).ravel()
    else:
        joined = np.column_stack([lower, upper]).ravel()
    out = (joined >> np.uint32(8)) * (np.float32(1.0) / np.float32(2**24))
    return out.astype(np.float32)


def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(0xFFFFFFFF))
    out = (x >> np.uint32(8)) * (np.float32(1.0) / np.float32(2**24))
    return out.astype(np.float32)


def uniform32_from_uint32(x):
    return (x >> np.uint32(8)) * (np.float32(1.0) / np.float32(2**24))


def uniform32_from_uint(x, bits, reversed=False):
    if bits == 64:
        return uniform32_from_uint64(x, reversed)
    elif bits == 53:
        return uniform32_from_uint53(x)
    elif bits == 32:
        return uniform32_from_uint32(x)
    else:
        raise NotImplementedError


def uniform_from_uint(x, bits):
    if bits in (64, 63, 53):
        return uniform_from_uint64(x)
    elif bits == 32:
        return uniform_from_uint32(x)


def uniform_from_uint64(x):
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def uniform_from_uint32(x):
    out = np.empty(len(x) // 2)
    for i in range(0, len(x), 2):
        a = x[i] >> 5
        b = x[i + 1] >> 6
        out[i // 2] = (a * 67108864.0 + b) / 9007199254740992.0
    return out


def uniform_from_dsfmt(x):
    return x.view(np.double) - 1.0


def gauss_from_uint(x, n, bits):
    if bits in (64, 63):
        doubles = uniform_from_uint64(x)
    elif bits == 32:
        doubles = uniform_from_uint32(x)
    elif bits == "dsfmt":
        doubles = uniform_from_dsfmt(x)
    elif bits == "squares32":
        x = x.view(np.uint32)
        doubles = uniform_from_uint32(x)
    gauss = []
    loc = 0
    x1 = x2 = 0.0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            r2 = x1 * x1 + x2 * x2
            loc += 2

        f = np.sqrt(-2.0 * np.log(r2) / r2)
        gauss.append(f * x2)
        gauss.append(f * x1)

    return gauss[:n]


class Base:
    dtype = np.uint64
    data2: ClassVar[dict[str, int | np.ndarray]] = {}
    data1: ClassVar[dict[str, int | np.ndarray]] = {}

    @classmethod
    def setup_class(cls):
        cls.bit_generator = Xoroshiro128
        cls.bits = 64
        cls.dtype = np.uint64
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = []
        cls.seed_sequence_only = False

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(",")
            seed = [int(s.strip(), 0) for s in seed[1:]]
            data = [int(line.split(",")[-1].strip(), 0) for line in csv]
            return {"seed": seed, "data": np.array(data, dtype=cls.dtype)}

    def setup_bitgenerator(self, seed):
        return self.bit_generator(*seed)

    def test_default(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)

    def test_default_sequence(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)
        assert isinstance(bg._get_seed_seq(), ISEED_SEQUENCES)

    def test_raw(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data1["data"])

        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        uints = bit_generator.random_raw()
        assert_equal(uints, self.data1["data"][0])

        bit_generator = self.setup_bitgenerator(self.data2["seed"])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data2["data"])

    def test_random_raw(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        uints = bit_generator.random_raw(output=False)
        assert uints is None
        uints = bit_generator.random_raw(1000, output=False)
        assert uints is None

    def test_gauss_inv(self):
        n = 25
        rs = np.random.RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        bits = getattr(self, "bit_name", self.bits)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, bits), rtol=3e-6)

        rs = np.random.RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss, gauss_from_uint(self.data2["data"], n, bits), rtol=3e-6)

    def test_uniform_double(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        bits = getattr(self, "bit_name", self.bits)
        vals = uniform_from_uint(self.data1["data"], bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform_from_uint(self.data2["data"], bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        vals = uniform32_from_uint(self.data1["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform32_from_uint(self.data2["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

    def test_seed_float(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        match = (
            "seed cannot" if isinstance(rs.bit_generator, RDRAND) else "SeedSequence"
        )
        with pytest.raises(TypeError, match=match):
            rs.bit_generator.seed(np.pi)
        with pytest.raises(TypeError, match=match):
            rs.bit_generator.seed(-np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        with pytest.raises(self.seed_error_type):
            rs.bit_generator.seed(np.array([np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([-np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([np.pi, -np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([0, np.pi]))
        with pytest.raises(TypeError, match="seed must be"):
            rs.bit_generator.seed([np.pi])
        with pytest.raises(TypeError, match="seed must be"):
            rs.bit_generator.seed([0, np.pi])

    def test_seed_out_of_range(self):
        # GH #82
        if self.seed_sequence_only:
            # Not valid on PRNG that only support seed sequence
            return
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed(2 ** (4 * self.bits + 1))
        with pytest.raises(ValueError, match="expected"):
            rs.bit_generator.seed(-1)

    def test_seed_out_of_range_array(self):
        # GH #82
        if self.seed_sequence_only:
            # Not valid on PRNG that only support seed sequence
            return
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed([2 ** (2 * self.bits + 1)])
        with pytest.raises(ValueError, match="expected non-negative"):
            rs.bit_generator.seed([-1])

    def test_repr(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert "Generator" in repr(rs)
        assert type(rs.bit_generator).__name__ in repr(rs)

    def test_str(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert "Generator" in str(rs)
        assert str(type(rs.bit_generator).__name__) in str(rs)
        assert f"{id(rs):#x}".upper().replace("X", "x")[2:] not in str(rs)

    def test_pickle(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        bit_generator_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bit_generator_pkl)
        orig_gen = np.random.Generator(bit_generator)
        reloaded_gen = np.random.Generator(reloaded)
        assert_array_equal(
            orig_gen.standard_normal(1000),
            reloaded_gen.standard_normal(1000),
        )
        assert bit_generator is not reloaded
        assert_state_equal(reloaded.state, bit_generator.state)

    def test_invalid_state_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(TypeError, match="state must be"):
            bit_generator.state = {"1"}

    def test_invalid_state_value(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        state["bit_generator"] = "otherBitGenerator"
        with pytest.raises(ValueError, match="state must be"):
            bit_generator.state = state

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            # No match here as reasons vary by bit gen
            with pytest.raises(TypeError):
                bit_generator.seed(*st)

    def test_invalid_seed_values(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_values:
            with pytest.raises((ValueError, TypeError)):
                bit_generator.seed(*st)

    def test_benchmark(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, "double")
        with pytest.raises(ValueError, match="Unknown"):
            bit_generator._benchmark(1, "int32")

    @pytest.mark.skipif(MISSING_CFFI, reason="cffi not available")
    def test_cffi(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        cffi_interface = bit_generator.cffi
        assert isinstance(cffi_interface, Interface)
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason="ctypes not available")
    def test_ctypes(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        ctypes_interface = bit_generator.ctypes
        assert isinstance(ctypes_interface, Interface)
        other_ctypes_interface = bit_generator.ctypes
        assert other_ctypes_interface is ctypes_interface

    def test_getstate(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        get_state = bit_generator.__getstate__()
        # Legacy numpy compatibility
        alt_state = get_state[0] if isinstance(get_state, tuple) else get_state
        assert_state_equal(state, alt_state)

    def test_uinteger_reset_seed(self):
        bg = self.setup_bitgenerator([None])
        g = np.random.Generator(bg)
        g.integers(0, 2**32, dtype=np.uint32)
        if "has_uint32" not in bg.state or bg.state["has_uint32"] == 0:
            name = bg.__class__.__name__
            pytest.skip(f"bit generator does not cache 32-bit value ({name})")
        bg.seed()
        assert bg.state["has_uint32"] == 0

    def test_uinteger_reset_jump(self):
        bg = self.setup_bitgenerator([None])
        if not hasattr(bg, "jumped"):
            pytest.skip("bit generator does not support jumping")
        g = np.random.Generator(bg)
        seed_seq = bg._get_seed_seq()
        g.integers(0, 2**32, dtype=np.uint32)
        new_bg = bg.jumped()
        jumped = np.random.Generator(new_bg)
        new_seed_seq = new_bg._get_seed_seq()
        assert_equal(seed_seq.entropy, new_seed_seq.entropy)

        if "has_uint32" in jumped.bit_generator.state:
            assert jumped.bit_generator.state["has_uint32"] == 0
            return
        # This next test could fail with prob 1 in 2**32
        next_g = g.integers(0, 2**32, dtype=np.uint32)
        next_jumped = jumped.integers(0, 2**32, dtype=np.uint32)
        assert next_g != next_jumped

    def test_jumped_seed_seq_clone(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        if not hasattr(bg, "jumped"):
            pytest.skip("bit generator does not support jumping")
        seed_seq = bg._get_seed_seq()
        new_bg = bg.jumped()
        new_seed_seq = new_bg._get_seed_seq()
        assert_equal(seed_seq.entropy, new_seed_seq.entropy)
        assert seed_seq.spawn_key == new_seed_seq.spawn_key
        assert seed_seq.n_children_spawned == new_seed_seq.n_children_spawned
        assert not np.all(new_bg.random_raw(10) == bg.random_raw(10))

    def test_uinteger_reset_advance(self):
        bg = self.setup_bitgenerator([None])
        if not hasattr(bg, "advance"):
            pytest.skip("bit generator does not support advancing")
        g = np.random.Generator(bg)
        g.integers(0, 2**32, dtype=np.uint32)
        state = bg.state
        bg.advance(1000)
        if "has_uint32" in bg.state:
            assert bg.state["has_uint32"] == 0
            return
        # This next test could fail with prob 1 in 2**32
        next_advanced = g.integers(0, 2**32, dtype=np.uint32)
        bg.state = state
        next_g = g.integers(0, 2**32, dtype=np.uint32)
        assert next_g != next_advanced

    def test_seed_sequence(self):
        bg = self.setup_bitgenerator([None])
        typ = (
            self.bit_generator.func
            if isinstance(self.bit_generator, CustomPartial)
            else self.bit_generator
        )
        assert isinstance(bg, typ)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        bg = self.setup_bitgenerator([0])
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator(ss)
        assert bg._get_seed_seq().entropy == 0

    def test_bit_generator_repr(self):
        bg = self.setup_bitgenerator([None])
        assert isinstance(repr(bg), str)
        assert type(bg).__name__ in repr(bg)
        assert type(bg).__name__ in str(bg)


class Random123(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Philox
        cls.number = 4
        cls.width = 64

    def setup_bitgenerator(self, seed, **kwargs):
        return self.bit_generator(*seed, **kwargs)

    def test_advance(self, step, counter_only, warmup):
        bg = self.bit_generator()
        bg.random_raw(warmup)
        state0 = bg.state
        n = self.number
        adj_step = step * n if counter_only else step
        bg.random_raw(adj_step)
        state_direct_pre = bg.state
        direct = bg.random_raw()
        bg.state = state0
        bg.advance(step, counter_only)
        advanced = bg.random_raw()
        if counter_only and step:
            bg.state = state_direct_pre
            window = bg.random_raw(self.number + 1)
            assert bool(np.isin(advanced, window))
            return

        # standard case
        assert direct == advanced

    def test_advance_default_counter(self, step, warmup):
        # The default is counter=False, which advances by step draws
        bg = self.bit_generator()
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw(3)
        bg.state = state0
        bg.advance(step)
        assert_array_equal(bg.random_raw(3), direct)

    def test_advance_large(self):
        dtype = np.uint64 if self.width == 64 else np.uint32
        bg = self.bit_generator()
        step = 2**self.width
        bg.advance(step, True)
        state = bg.state
        assert_equal(state["state"]["counter"], np.array([0, 1, 0, 0], dtype=dtype))

        bg = self.bit_generator()
        step = 2**self.width - 1
        bg.advance(step, True)
        state = bg.state
        size_max = np.iinfo(dtype).max
        assert_equal(
            state["state"]["counter"],
            np.array([size_max, 0, 0, 0], dtype=dtype),
        )

        bg = self.bit_generator()
        step = 2 ** (2 * self.width)
        bg.advance(step, True)
        state = bg.state
        assert_equal(state["state"]["counter"], np.array([0, 0, 1, 0], dtype=dtype))

        bg = self.bit_generator()
        step = 2 ** (2 * self.width) - 1
        bg.advance(step, True)
        state = bg.state
        assert_equal(
            state["state"]["counter"],
            np.array([size_max, size_max, 0, 0], dtype=dtype),
        )

    def test_numpy_uint_array(self):
        bg = self.bit_generator(np.uint64(1))
        bg2 = self.bit_generator(1)
        assert_state_equal(bg.state, bg2.state)

    def test_empty_seed(self):
        self.bit_generator(np.array([], dtype=np.uint64))

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], key=0, counter=0)


class TestJSF64(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = JSF
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/jsf64-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/jsf64-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]

    def test_invalid_seed_size(self):
        with pytest.raises(ValueError, match="seed size must be one"):
            self.bit_generator(seed_size=4)
        with pytest.raises(ValueError, match="seed size must be one"):
            self.bit_generator(seed_size=1.0)

    def test_bad_init(self):
        with pytest.raises(ValueError, match="size must be"):
            self.bit_generator(size=33)
        with pytest.raises(ValueError, match="size must be"):
            self.bit_generator(size=0.0 + self.bits)
        with pytest.raises(ValueError, match="p must be"):
            self.bit_generator(p=-1)
        with pytest.raises(ValueError, match="p must be"):
            self.bit_generator(p=31.7)


class TestJSF32(TestJSF64):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator_base = JSF
        cls.bit_generator = CustomPartial(JSF, size=32)
        cls.size = 32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/jsf32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/jsf32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(size=self.size)
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        bg = self.bit_generator_base(0, size=self.size)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, size=self.size)
        assert bg._get_seed_seq().entropy == 1


class TestXoroshiro128(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Xoroshiro128
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/xoroshiro128-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/xoroshiro128-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]


class TestXoroshiro128PlusPlus(TestXoroshiro128):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(Xoroshiro128, plusplus=True)
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, "./data/xoroshiro128plusplus-testset-1.csv"),
        )
        cls.data2 = cls._read_csv(
            join(pwd, "./data/xoroshiro128plusplus-testset-2.csv"),
        )


class TestXoshiro256(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Xoshiro256
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/xoshiro256-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/xoshiro256-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]


class TestXoshiro512(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Xoshiro512
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/xoshiro512-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/xoshiro512-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]


class TestXorshift1024(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Xorshift1024
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/xorshift1024-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/xorshift1024-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]


COUNTER_BASED = [
    pytest.param(cls, number, width, id=f"{cls.__name__}-{number}x{width}")
    for cls in (Philox, ThreeFry)
    for number in (2, 4)
    for width in (32, 64)
]


def _counter_as_int(bg):
    width = bg.state["state"]["counter"].dtype.itemsize * 8
    counter = bg.state["state"]["counter"]
    return sum(int(v) << (width * i) for i, v in enumerate(counter))


@pytest.mark.parametrize(("cls", "number", "width"), COUNTER_BASED)
@pytest.mark.parametrize("warmup", [0, 1, 2, 3, 4, 5, 7, 100])
@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 6, 7, 8, 9, 34159])
def test_advance_counter_false_matches_draws(cls, number, width, warmup, step):
    # Includes a new generator, whose buffer_pos is the buffer size (4), not number
    bg = cls(1234, number=number, width=width)
    bg.random_raw(warmup)
    expected = cls(1234, number=number, width=width)
    expected.random_raw(warmup + step)
    bg.advance(step, counter=False)
    assert_array_equal(bg.random_raw(7), expected.random_raw(7))


@pytest.mark.parametrize(("cls", "number", "width"), COUNTER_BASED)
@pytest.mark.parametrize("warmup", [0, 1, 2, 3, 4, 5, 7, 100])
@pytest.mark.parametrize("step", [1, 2, 3, 4, 5, 34159])
def test_advance_counter_true_matches_draws(cls, number, width, warmup, step):
    # counter=True advances the counter by step blocks of number draws
    bg = cls(1234, number=number, width=width)
    bg.random_raw(warmup)
    counter = _counter_as_int(bg)
    expected = cls(1234, number=number, width=width)
    expected.random_raw((counter + step) * number)
    bg.advance(step, counter=True)
    assert _counter_as_int(bg) == counter + step
    assert_array_equal(bg.random_raw(7), expected.random_raw(7))


@pytest.mark.parametrize(("cls", "number", "width"), COUNTER_BASED)
@pytest.mark.parametrize("counter_only", [True, False])
@pytest.mark.parametrize("warmup", [0, 1, 3])
def test_advance_large(cls, number, width, counter_only, warmup):
    # The counter has number * width bits and holds 2**(number * width) blocks.
    # Check steps near the largest step that can be advanced, which carry
    # across all of the words of the counter.
    bits = number * width
    log2_number = number // 2
    modulus = 2**bits
    for exponent in range(width - 1, bits + log2_number + 1):
        for delta in (-1, 0, 1):
            step = 2**exponent + delta
            bg = cls(1234, number=number, width=width)
            bg.random_raw(warmup)
            counter = _counter_as_int(bg)
            position = min(bg.state["buffer_pos"], number)
            bg.advance(step, counter=counter_only)
            if counter_only:
                assert _counter_as_int(bg) == (counter + step) % modulus
            else:
                new_position = (counter * number + position + step) % (modulus * number)
                assert _counter_as_int(bg) == new_position // number
                assert bg.state["buffer_pos"] == new_position % number


class TestThreeFry(Random123):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = ThreeFry
        cls.number = 4
        cls.width = 64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/threefry-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/threefry-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (1, None, 1),
            (-1,),
            (None, None, 2**257 + 1),
        ]

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        keyed = self.bit_generator(
            counter=state["state"]["counter"],
            key=state["state"]["key"],
        )
        assert_state_equal(bit_generator.state, keyed.state)

    def test_advance(self):
        bg = self.bit_generator()
        state0 = bg.state
        bg.advance(1, True)
        assert_equal(
            bg.state["state"]["counter"],
            np.array([1, 0, 0, 0], dtype=np.uint64),
        )
        bg.advance(1, True)
        assert_equal(
            bg.state["state"]["counter"],
            np.array([2, 0, 0, 0], dtype=np.uint64),
        )
        bg.advance(2**64, True)
        assert_equal(
            bg.state["state"]["counter"],
            np.array([2, 1, 0, 0], dtype=np.uint64),
        )
        bg.state = state0
        bg.advance(2**128, True)
        assert_equal(
            bg.state["state"]["counter"],
            np.array([0, 0, 1, 0], dtype=np.uint64),
        )


class TestPCG64XSLRR(Base):
    bit_generator: type[PCG64]

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(3.2,)]
        cls.invalid_seed_values = [(-2,)]
        cls.large_advance_initial = 262626489767919729675955844831248137855
        cls.large_advance_final = 326675794918500479020985263602132957772

    def setup_bitgenerator(self, seed, inc: int | None = None):
        return self.bit_generator(*seed, variant="xsl-rr", inc=inc)

    def test_seed_float_array(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type,
            rs.bit_generator.seed,
            np.array([np.pi, -np.pi]),
        )
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [np.pi])
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range_array(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed([2 ** (2 * self.bits + 1)])
        with pytest.raises(ValueError, match="expected non"):
            rs.bit_generator.seed([-1])

    def test_advance_symmetry(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        step = -0x9E3779B97F4A7C150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_inc_none(self):
        bg = self.setup_bitgenerator([0], inc=None)
        assert isinstance(bg.random_raw(), int)

        bg = self.setup_bitgenerator([0], inc=None)
        assert isinstance(bg.random_raw(), int)

    def test_inc_alt(self):
        ss = SeedSequence(0)
        bg = self.setup_bitgenerator([ss], inc=0)
        ss2 = SeedSequence(0)
        bg2 = self.setup_bitgenerator([ss2], inc=None)
        assert bg.state["state"]["inc"] == 1
        # Prob 1 in 2**127 of failure
        assert bg2.state["state"]["inc"] != 1

    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="variant"):
            PCG64(0, variant="unknown")
        with pytest.raises(ValueError, match="variant"):
            PCG64(0, variant=3)

    def test_large_advance(self):
        bg = self.setup_bitgenerator([0], inc=1)
        state = bg.state["state"]
        assert state["state"] == self.large_advance_initial
        bg.advance(sum(2**i for i in range(96)))
        state = bg.state["state"]
        assert state["state"] == self.large_advance_final


class TestPhilox(Random123):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Philox
        cls.number = 4
        cls.width = 64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/philox-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/philox-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (1, None, 1),
            (-1,),
            (None, None, 2**257 + 1),
        ]

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        keyed = self.bit_generator(
            counter=state["state"]["counter"],
            key=state["state"]["key"],
        )
        assert_state_equal(bit_generator.state, keyed.state)

    def test_numpy_mode(self):
        ss = SeedSequence(0)
        bg = self.setup_bitgenerator([ss], numpy_seed=True)
        assert isinstance(bg, Philox)

    def test_seed_key(self):
        bg = self.setup_bitgenerator([0])
        with pytest.raises(ValueError, match="seed and key"):
            bg.seed(seed=0, key=0)


class TestPhilox4x32(Random123):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator_base = Philox
        cls.bit_generator = CustomPartial(Philox, number=4, width=32)
        cls.number = 4
        cls.width = 32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/philox4x32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/philox4x32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (1, None, 1),
            (-1,),
            (None, None, 2**257 + 1),
        ]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(number=self.number, width=self.width)
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        bg = self.bit_generator_base(0, number=self.number, width=self.width)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, number=self.number, width=self.width)
        assert bg._get_seed_seq().entropy == 1

    def test_numpy_mode(self):
        ss = SeedSequence(0)
        with pytest.raises(ValueError, match="n must be 4"):
            self.setup_bitgenerator([ss], numpy_seed=True)


class TestAESCounter(TestPhilox):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = AESCounter
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/aesctr-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/aesctr-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (1, None, 1),
            (-1,),
            (None, None, 2**257 + 1),
        ]

    @pytest.fixture(autouse=True, params=USE_AESNI)
    def set_use_aesni(self, request):
        self.use_aesni = request.param

    def setup_bitgenerator(self, seed, **kwargs):
        bg = self.bit_generator(*seed, **kwargs)
        bg.use_aesni = self.use_aesni
        return bg

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        key = state["s"]["seed"][:2]
        key = int(key[0]) + int(key[1]) * 2**64
        counter = state["s"]["counter"][0]
        keyed = self.bit_generator(counter=counter, key=key)
        assert_state_equal(bit_generator.state, keyed.state)

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type,
            rs.bit_generator.seed,
            np.array([np.pi, -np.pi]),
        )
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_bad_state(self):
        bg = self.bit_generator()
        state = bg.state
        state["s"]["seed"] = np.empty((2, 11), dtype=np.uint64)
        with pytest.raises(ValueError, match="seed must be"):
            bg.state = state
        state = bg.state
        state["s"]["counter"] = 4
        with pytest.raises(ValueError, match="counter must be a"):
            bg.state = state

    def test_advance(self, step, warmup):
        bg = self.bit_generator()
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw()
        bg.state = state0
        bg.advance(step)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_one_repeat(self):
        bg = self.bit_generator()
        state0 = bg.state
        bg.random_raw(8)
        direct = bg.random_raw()
        bg.state = state0
        for _ in range(8):
            bg.advance(1)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_large(self):
        bg = self.bit_generator()
        step = 2**65
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["s"]["counter"],
            np.array([4, 1, 5, 1, 6, 1, 7, 1], dtype=np.uint64),
        )

        bg = self.bit_generator()
        step = 2**65 - 7
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["s"]["counter"],
            np.array([0, 1, 1, 1, 2, 1, 3, 1], dtype=np.uint64),
        )

        bg = self.bit_generator()
        step = 2**129 - 16
        bg.advance(step)
        state = bg.state
        m = np.iinfo(np.uint64).max
        assert_equal(
            state["s"]["counter"],
            np.array([m - 3, m, m - 2, m, m - 1, m, m, m], dtype=np.uint64),
        )
        bg.random_raw(9)
        state = bg.state
        assert_equal(
            state["s"]["counter"],
            np.array([0, 0, 1, 0, 2, 0, 3, 0], dtype=np.uint64),
        )

        bg = self.bit_generator()
        state0 = bg.state
        step = 2**129
        bg.advance(step)
        state = bg.state
        assert_state_equal(state0, state)

    @pytest.mark.skip(reason="Not applicable to AESCounter")
    def test_numpy_mode(self):
        pass

    @pytest.mark.skip(reason="Not applicable to AESCounter")
    def test_advance_deprecated(self):
        pass

    @pytest.mark.skipif(HAS_AESNI, reason="Not valid when cpu has AESNI")
    def test_no_aesni(self):
        bg = self.bit_generator()
        with pytest.raises(ValueError, match="CPU does not support AESNI"):
            bg.use_aesni = True

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], counter=0, key=0)

    def test_large_counter(self):
        # GH 267
        bg = self.bit_generator(counter=2**128 - 2)
        state = bg.state
        assert_equal(
            state["s"]["counter"][-4:],
            np.array([0, 0, 1, 0], dtype=np.uint64),
        )


class TestMT19937(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MT19937
        cls.bits = 32
        cls.seed_bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/mt19937-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mt19937-testset-2.csv"))
        cls.seed_error_type = ValueError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,)]
        cls.state_name = "key"

    def test_seed_out_of_range(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed(2 ** (self.seed_bits + 1))
        assert_raises(ValueError, rs.bit_generator.seed, -1)
        rs.bit_generator.seed(2 ** (2 * self.seed_bits + 1))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed([2 ** (self.seed_bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        rs.bit_generator.seed([2 ** (2 * self.seed_bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, bit_generator.seed, [np.pi])
        assert_raises(TypeError, bit_generator.seed, [0, np.pi])

    def test_state_tuple(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        state = bit_generator.state
        desired = rs.integers(2**16)
        # Due to changes
        bg_name = state["bit_generator"].split(".")[-1]
        tup = (bg_name, state["state"]["key"], state["state"]["pos"])
        bit_generator.state = tup
        actual = rs.integers(2**16)
        assert_equal(actual, desired)
        tup = tup + (0, 0.0)
        bit_generator.state = tup
        actual = rs.integers(2**16)
        assert_equal(actual, desired)

    def test_invalid_state(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        state["state"][self.state_name] = state["state"][self.state_name][:10]
        with pytest.raises(ValueError, match="State element"):
            rs.bit_generator.state = state

    def test_negative_jump(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(ValueError, match="iter must be"):
            bg.jumped(-1)

    def test_bad_legacy_state(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(ValueError, match="state is not"):
            bg.state = ("UNKNOWN",)


class TestSFMT(TestMT19937):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = SFMT
        cls.bits = 64
        cls.seed_bits = 32
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/sfmt-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/sfmt-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (-1,),
        ]
        cls.state_name = "state"

    @pytest.mark.skip(reason="Not applicable to SFMT")
    def test_state_tuple(self):
        pass

    @pytest.mark.skip(reason="Not applicable to SFMT")
    def test_bad_legacy_state(self):
        pass


class TestDSFMT(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = DSFMT
        cls.bits = 53
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/dSFMT-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/dSFMT-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (-1,),
        ]

    def test_uniform_double(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_array_equal(uniform_from_dsfmt(self.data1["data"]), rs.random(1000))

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        assert_equal(uniform_from_dsfmt(self.data2["data"]), rs.random(1000))

    def test_gauss_inv(self):
        n = 25
        rs = np.random.RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, "dsfmt"))

        rs = np.random.RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss, gauss_from_uint(self.data2["data"], n, "dsfmt"))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed([2 ** (self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        rs.bit_generator.seed([2 ** (2 * self.bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_uniform_float(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        vals = uniform32_from_uint(self.data1["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform32_from_uint(self.data2["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

    def test_buffer_reset(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.random(1)
        assert rs.bit_generator.state["buffer_loc"] != 382
        rs.bit_generator.seed(*self.data1["seed"])
        assert rs.bit_generator.state["buffer_loc"] == 382

    def test_negative_jump(self):
        bg = self.bit_generator()
        with pytest.raises(ValueError, match="iter must be positive"):
            bg.jumped(-1)


class TestThreeFry4x32(Random123):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator_base = ThreeFry
        cls.bit_generator = CustomPartial(ThreeFry, number=4, width=32)
        cls.number = 4
        cls.width = 32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/threefry32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/threefry32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [
            (1, None, 1),
            (-1,),
            (None, None, 2**129 + 1),
        ]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(number=self.number, width=self.width)
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        bg = self.bit_generator_base(0, number=self.number, width=self.width)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, number=self.number, width=self.width)
        assert bg._get_seed_seq().entropy == 1

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        key = state["state"]["key"]
        power = 32 if key.dtype == np.uint32 else 64
        key = sum([int(key[i]) * 2 ** (power * i) for i in range(len(key))])
        counter = state["state"]["counter"]
        counter = sum([int(counter[i]) * 2 ** (power * i) for i in range(len(counter))])
        keyed = self.bit_generator(counter=counter, key=key)
        assert_state_equal(bit_generator.state, keyed.state)


class TestPCG32(TestPCG64XSLRR):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = PCG32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(3.2,), (None, np.zeros(1))]
        cls.invalid_seed_values = [
            (-1,),
            (None, -1),
            (None, 2**129 + 1),
        ]
        cls.large_advance_initial = 16393846548996684016
        cls.large_advance_final = 15793235383387715777

    def setup_bitgenerator(self, seed, inc=None):
        return self.bit_generator(*seed, inc=inc)

    def test_advance_symmetry(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        step = -0x9E3779B97F4A7C16
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**64 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**64 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos


class TestMT64(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MT64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/mt64-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mt64-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.seed_error_type = ValueError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,)]

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        with pytest.raises(TypeError, match="seed must be"):
            bit_generator.seed(np.array([-np.pi]))
        with pytest.raises(TypeError, match="seed must be"):
            bit_generator.seed(np.array([np.pi, -np.pi]))
        with pytest.raises(TypeError, match="seed must be"):
            bit_generator.seed(np.array([0, np.pi]))
        with pytest.raises(TypeError, match="seed must be"):
            bit_generator.seed([np.pi])
        with pytest.raises(TypeError, match="seed must be"):
            bit_generator.seed([0, np.pi])

    def test_empty_seed(self):
        self.bit_generator(np.array([], dtype=np.uint64))


@pytest.mark.skipif(MISSING_RDRAND, reason="RDRAND is not available")
class TestRDRAND(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = RDRAND
        cls.bits = 64
        cls.dtype = np.uint64
        cls.seed_error_type = TypeError
        cls.data1 = {"seed": [None]}
        cls.invalid_seed_types = [1, np.array([1])]
        cls.invalid_seed_values = []

    def setup_bitgenerator(self, seed):
        return self.bit_generator(*seed)

    def test_initialization(self):
        bit_generator = self.bit_generator()
        state = bit_generator.state
        assert state["buffer_loc"] == 256
        assert state["buffer"].shape == (256,)
        assert (state["buffer"] == np.iinfo(np.uint64).max).all()
        assert state["retries"] == 10
        assert state["status"] == 1
        gen = np.random.Generator(bit_generator)
        gen.integers(0, 2**64, dtype=np.uint64, size=10)
        state = bit_generator.state
        # Incredibly conservative test
        assert (state["buffer"] == np.iinfo(np.uint64).max).sum() < 10
        assert state["status"] == 1

        bit_generator = RDRAND(retries=1000)
        state = bit_generator.state
        assert state["retries"] == 1000

    def test_generator_raises(self):
        bit_generator = self.bit_generator()

        state = bit_generator.state
        state["retries"] = -1
        bit_generator.state = state
        gen = np.random.Generator(bit_generator)
        with pytest.raises(RuntimeError):
            gen.integers(0, 2**64, dtype=np.uint64, size=10)
        assert bit_generator.state["status"] == 0
        bit_generator._reset()
        state = bit_generator.state
        assert state["status"] == 1
        assert state["buffer_loc"] == 256
        assert bit_generator.success is True

    @pytest.mark.thread_unsafe(reason="replaces sys.stdout")
    def test_pending_error_without_stdout(self, monkeypatch):
        # Raising the pending error must not depend on print() doing work
        monkeypatch.setattr(sys, "stdout", None)
        bit_generator = self.bit_generator()
        state = bit_generator.state
        state["retries"] = -1
        bit_generator.state = state
        gen = np.random.Generator(bit_generator)
        with pytest.raises(RuntimeError, match="Unable to get random values"):
            gen.integers(0, 2**64, dtype=np.uint64, size=10)
        # The lock must have been released
        acquired = []

        def try_acquire():
            acquired.append(bit_generator.lock.acquire(False))
            bit_generator.lock.release()

        thread = threading.Thread(target=try_acquire)
        thread.start()
        thread.join()
        assert acquired == [True]

    def test_exception(self):
        with pytest.raises(ValueError, match="retries must be"):
            RDRAND(retries=-1)

    def test_runtimerror(self):
        bit_generator = RDRAND()
        state = bit_generator.state
        state["retries"] = -1
        bit_generator.state = state
        with pytest.raises(RuntimeError):
            bit_generator.random_raw()
        with pytest.raises(RuntimeError):
            bit_generator.random_raw(10)
        with pytest.raises(RuntimeError):
            bit_generator.random_raw(output=False)
        with pytest.raises(RuntimeError):
            bit_generator.random_raw(10, output=False)

    @pytest.mark.parametrize("size", [5, (3, 4), [3, 4], (2, 3, 4)])
    def test_raw_no_output_size(self, size):
        bit_generator = RDRAND()
        assert bit_generator.random_raw(size, output=False) is None
        assert bit_generator.state["status"] == 1

    def test_checked_raw(self):
        bit_generator = RDRAND()
        assert isinstance(bit_generator.random_raw(), int)

    def test_checked_raw_vector(self):
        bit_generator = RDRAND()
        assert isinstance(bit_generator.random_raw(10), np.ndarray)

    def test_raw(self):
        bit_generator = self.bit_generator()
        raw = bit_generator.random_raw(1000)
        assert (raw.max() - raw.min()) > 0

    def test_gauss_inv(self):
        n = 25
        rs = np.random.RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        assert (gauss.max() - gauss.min()) > 0

    def test_uniform_double(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        uniforms = rs.random(1000)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        uniforms = rs.random(1000, dtype=np.float32)
        assert_equal(uniforms.dtype, np.float32)

    @pytest.mark.skip("Bit generator is missing generator attr")
    def test_generator(self):
        pass

    def test_pickle(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        bit_generator_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bit_generator_pkl)
        assert bit_generator is not reloaded
        assert_state_equal(reloaded.state, bit_generator.state)

    @pytest.mark.skip("RDRAND seed accepts only None")
    def test_seed_out_of_range_array(self):
        pass

    @pytest.mark.skip("RDRAND seed accepts only None")
    def test_seed_out_of_range(self):
        pass

    @pytest.mark.skip("RDRAND seed accepts only None")
    def test_seed_float_array(self):
        pass

    def test_jumped(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        new_bg = bg.jumped()
        assert isinstance(new_bg, type(bg))
        assert bg is not new_bg

    def test_jumped_retries(self):
        bg = RDRAND(retries=100)
        assert bg.jumped().state["retries"] == 100
        state = bg.state
        state["retries"] = -1
        bg.state = state
        assert bg.jumped().state["retries"] == -1

    def test_failure_buffer_loc(self):
        bg = RDRAND()
        state = bg.state
        state["retries"] = -1
        bg.state = state
        gen = np.random.Generator(bg)
        # After the first failure the buffer is not refilled
        with pytest.raises(RuntimeError):
            gen.integers(0, 2**64, dtype=np.uint64, size=300)
        assert bg.state["status"] == 0
        assert bg.state["buffer_loc"] <= 256

    def test_seed_sequence(self):
        bg = self.bit_generator()
        assert isinstance(bg, self.bit_generator)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        with pytest.raises(TypeError, match="seed cannot be set"):
            self.bit_generator(0)

        ss = SeedSequence(0)
        with pytest.raises(TypeError, match="seed cannot be set"):
            self.bit_generator(ss)

    def test_default_sequence(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

    def test_seed_sequence_error(self):
        with pytest.raises(TypeError, match="seed cannot be set"):
            self.bit_generator(SeedSequence(0))


class TestChaCha(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = ChaCha
        cls.bits = 64
        cls.seed_bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/chacha-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/chacha-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,)]
        cls.state_name = "key"

    def setup_bitgenerator(self, seed, **kwargs):
        seed = [0] if None in seed else seed
        bg = self.bit_generator(*seed, **kwargs)
        return bg

    def test_set_key(self):
        with pytest.raises(ValueError, match="seed and key cannot"):
            self.bit_generator(seed=0, key=0)

    def test_invalid_rounds(self):
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=3)
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=-4)

    def test_use_simd(self):
        bg = self.bit_generator(0)
        if not bg.use_simd:
            with pytest.raises(ValueError, match="CPU does not support SIMD"):
                bg.use_simd = True
            return
        bg2 = self.bit_generator(0)
        bg2.use_simd = not bg.use_simd
        assert_equal(bg.random_raw(100), bg2.random_raw(100))

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], key=0, counter=0)


class TestHC128(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = HC128
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/hc-128-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/hc-128-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [
            ("apple",),
            (2 + 3j,),
            (3.1,),
            (-2,),
        ]
        cls.invalid_seed_values = []

    def test_seed_float_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type,
            rs.bit_generator.seed,
            np.array([np.pi, -np.pi]),
        )
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, -1)

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            with pytest.raises((TypeError, ValueError)):
                bit_generator.seed(*st)

    def test_key_init(self):
        with pytest.raises(ValueError, match="key is out"):
            self.bit_generator(key=-1)
        with pytest.raises(ValueError, match="key is out"):
            self.bit_generator(key=2**256)
        with pytest.raises(ValueError, match="seed and key"):
            self.bit_generator(seed=1, key=1)


class TestSPECK128(TestHC128):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = SPECK128
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/speck-128-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/speck-128-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [
            ("apple",),
            (2 + 3j,),
            (3.1,),
            (-2,),
        ]
        cls.invalid_seed_values = []

    def test_seed_out_of_range(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed(2**257)
        with pytest.raises(ValueError, match="expected"):
            rs.bit_generator.seed(-1)

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            with pytest.raises((TypeError, ValueError)):
                bit_generator.seed(*st)

    def test_key_init(self):
        with pytest.raises(ValueError, match="key is out"):
            self.bit_generator(key=-1)
        with pytest.raises(ValueError, match="key is out"):
            self.bit_generator(key=2**256)
        with pytest.raises(ValueError, match="seed and key"):
            self.bit_generator(seed=1, key=1)

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.bit_generator.seed([2 ** (4 * self.bits + 1)])
        with pytest.raises(ValueError, match="expected non"):
            rs.bit_generator.seed([-1])

    def test_advance(self, step, warmup):
        bg = self.bit_generator()
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw()
        bg.state = state0
        bg.advance(step)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_one_repeat(self):
        bg = self.bit_generator()
        state0 = bg.state
        bg.random_raw(8)
        direct = bg.random_raw()
        bg.state = state0
        for _ in range(8):
            bg.advance(1)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_large(self):
        bg = self.bit_generator()
        step = 2**65
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"],
            np.array([6, 1, 7, 1, 8, 1, 9, 1, 10, 1, 11, 1], dtype=np.uint64),
        )

        bg = self.bit_generator()
        step = 2**65 - 11
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"],
            np.array([0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1], dtype=np.uint64),
        )

        bg = self.bit_generator()
        step = 2**129 - 24
        bg.advance(step)
        state = bg.state
        m = np.iinfo(np.uint64).max
        assert_equal(
            state["state"]["ctr"],
            np.array(
                [m - 5, m, m - 4, m, m - 3, m, m - 2, m, m - 1, m, m, m],
                dtype=np.uint64,
            ),
        )
        bg.random_raw(13)
        state = bg.state
        assert_equal(
            state["state"]["ctr"],
            np.array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0], dtype=np.uint64),
        )

        bg = self.bit_generator()
        state0 = bg.state
        step = 2**129
        bg.advance(step)
        state = bg.state
        assert_state_equal(state0, state)

    def test_invalid_rounds(self):
        with pytest.raises(ValueError, match="rounds must be"):
            self.bit_generator(rounds=-1)
        with pytest.raises(ValueError, match="rounds must be"):
            self.bit_generator(rounds=35)
        with pytest.raises(ValueError, match="rounds must be"):
            self.bit_generator(rounds=27.5)


class TestLXM(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = LXM
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/lxm-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/lxm-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True

    def setup_bitgenerator(self, seed, **kwargs):
        return self.bit_generator(*seed, **kwargs)

    def test_seed_sequence_error(self):
        pass

    def test_default_sequence(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)
        assert isinstance(bg._get_seed_seq(), ISEED_SEQUENCES)

    def test_seed_sequence(self):
        bg = self.setup_bitgenerator([None])
        typ = (
            self.bit_generator.func
            if isinstance(self.bit_generator, CustomPartial)
            else self.bit_generator
        )
        assert isinstance(bg, typ)
        assert isinstance(bg._get_seed_seq(), SeedSequence)

        bg = self.setup_bitgenerator([0])
        assert bg._get_seed_seq().entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator(ss)
        assert bg._get_seed_seq().entropy == 0


class TestTyche(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Tyche
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/tyche-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/tyche-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True

    def test_gauss_inv(self):
        n = 25
        rs = np.random.RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        bits = getattr(self, "bit_name", self.bits)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, bits), rtol=9e-6)

        rs = np.random.RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss, gauss_from_uint(self.data2["data"], n, bits), rtol=9e-6)

    def test_uniform_float(self):
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        vals = uniform32_from_uint(self.data1["data"], self.bits, reversed=True)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform32_from_uint(self.data2["data"], self.bits, reversed=True)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals, atol=1e-7)
        assert_equal(uniforms.dtype, np.float32)

    def test_idx(self):
        bg1 = self.bit_generator(0, idx=0)
        bg2 = self.bit_generator(0, idx=1)
        assert np.all(bg1.random_raw(10) != bg2.random_raw(10))
        with pytest.raises(ValueError, match="idx must be in"):
            self.bit_generator(0, idx=sum(2**i for i in range(43)))
        with pytest.raises(ValueError, match="idx must be in"):
            self.bit_generator(0, idx=-73)


class TestTycheOpenRand(TestTyche):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(Tyche, original=False)
        cls.data1 = cls._read_csv(join(pwd, "./data/tyche-openrand-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/tyche-openrand-testset-2.csv"))


class TestSFC64(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = SFC64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/sfc-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/sfc-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True

    def test_numpy_mode(self):
        with pytest.raises(ValueError, match="w and k"):
            self.setup_bitgenerator([0], numpy_seed=True, k=3)
        with pytest.raises(ValueError, match="w and k"):
            self.setup_bitgenerator([0], numpy_seed=True, w=3)


class TestPCG64DXSM(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(PCG64, variant="dxsm-128")
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-dxsm-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-dxsm-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True
        cls.large_advance_initial = 262626489767919729675955844831248137855
        cls.large_advance_final = 326675794918500479020985263602132957772

    def setup_bitgenerator(self, seed, inc=0):
        return self.bit_generator(*seed, inc=inc)

    def test_large_advance(self):
        bg = self.setup_bitgenerator([0], inc=1)
        state = bg.state["state"]
        assert state["state"] == self.large_advance_initial
        bg.advance(sum(2**i for i in range(96)))
        state = bg.state["state"]
        assert state["state"] == self.large_advance_final


class TestPCG64CMDXSM(TestPCG64DXSM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(PCG64, variant="dxsm")
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-cm-dxsm-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-cm-dxsm-testset-2.csv"))
        cls.large_advance_initial = 159934576226003702342121456273047082943
        cls.large_advance_final = 43406923282132296644520456716700203596


class TestSquares(TestPCG64DXSM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Squares
        cls.data1 = cls._read_csv(join(pwd, "./data/squares-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/squares-testset-2.csv"))
        cls.large_advance_initial = 11400714819323198485
        cls.large_advance_final = 2177342782468422676

    def setup_bitgenerator(self, seed, counter=None, key=None):
        return self.bit_generator(*seed, counter=counter, key=key)

    def test_large_advance(self):
        bg = self.setup_bitgenerator([0], counter=11400714819323198485)
        state = bg.state["state"]
        assert state["counter"] == self.large_advance_initial
        bg.advance(sum(2**i for i in range(63)))
        state = bg.state["state"]
        assert state["counter"] == self.large_advance_final

    def test_generate_keys(self):
        a = generate_keys(0, 1)
        b = generate_keys(SeedSequence(0), 1)
        assert_equal(a, b)

    def test_generate_keys_errors(self):
        with pytest.raises(ValueError, match="n must be a"):
            generate_keys(0, -1)

    @pytest.mark.thread_unsafe(reason="changes a module-level test flag")
    def test_generate_keys_unique(self):
        a = generate_keys(0, 100, unique=True)

        _test_sentinal.set_testing(True)
        b = generate_keys(0, 100, unique=True)
        _test_sentinal.set_testing(False)
        assert np.any(a != b)
        c = generate_keys(0, 100)
        assert_equal(a, c)

    def test_generate_keys_quality(self):
        keys = generate_keys(0, 1000)
        # First odd
        assert np.all(keys & np.uint64(0x1))
        hexes = np.array([list(hex(v)) for v in keys])[:, 2:]
        for i in range(8):
            assert np.all(hexes[:, i] != hexes[:, i + 1])
        for i in range(len(hexes)):
            assert len({str(s) for s in hexes[i, 8:]}) == 8

    @pytest.mark.thread_unsafe(reason="changes a module-level test flag")
    def test_sentinal(self):
        assert not _test_sentinal.get_testing()
        _test_sentinal.set_testing(True)
        assert _test_sentinal.get_testing()
        _test_sentinal.set_testing(False)
        assert not _test_sentinal.get_testing()

    def test_errors(self):
        with pytest.raises(ValueError, match="variant must be either"):
            self.bit_generator(variant=48)
        with pytest.raises(ValueError, match="variant must be either"):
            self.bit_generator(variant="32")
        with pytest.raises(ValueError, match="counter must be"):
            self.bit_generator(counter=-1)
        with pytest.raises(ValueError, match="key must be"):
            self.bit_generator(key=sum(2**i for i in range(65)))
        with pytest.raises(ValueError, match="key must be odd"):
            self.bit_generator(key=sum(2**i for i in range(4, 62)))

    def test_invalid_state(self):
        bg = self.bit_generator()
        state = bg.state
        state["variant"] = 48
        with pytest.raises(ValueError, match="variant must be"):
            bg.state = state

    def test_state_roundtrip_cached_uint32(self):
        bg = self.setup_bitgenerator([0])
        gen = np.random.Generator(bg)
        gen.integers(0, 2**32, dtype=np.uint32)
        state = bg.state
        # Only the 64-bit variant caches the unused half of a 64-bit draw
        assert state["has_uint32"] == int(state["variant"] == 64)
        other = self.setup_bitgenerator([1])
        other.state = state
        assert_state_equal(other.state, state)
        expected = gen.integers(0, 2**32, size=5, dtype=np.uint32)
        other_gen = np.random.Generator(other)
        assert_equal(other_gen.integers(0, 2**32, size=5, dtype=np.uint32), expected)

    def test_invalid_state_key(self):
        bg = self.bit_generator()
        state = bg.state
        orig = bg.state
        state["state"]["key"] = int(state["state"]["key"]) - 1
        with pytest.raises(ValueError, match="key must be odd"):
            bg.state = state
        state["state"]["key"] = 2**64 + 1
        with pytest.raises(ValueError, match="key must be"):
            bg.state = state
        assert_state_equal(bg.state, orig)

    @pytest.mark.parametrize(
        ("seed", "key"),
        [
            (0, 3661817429494344399),
            (1, 4126033252942326575),
            (2, 8770630966058977947),
            (1234, 9470186258571876535),
            (2**64 - 1, 7173609320913571613),
        ],
    )
    def test_seed_key_pinned(self, seed, key):
        assert self.bit_generator(seed).state["state"]["key"] == key

    def test_generate_keys_pinned(self):
        expected = [3661817429494344399, 13566961089401090923, 6655634076464885827]
        assert_equal(generate_keys(0, 3), np.array(expected, dtype=np.uint64))
        expected = [18012821362677550425, 11190742791071495887]
        assert_equal(generate_keys(1234, 40)[-2:], np.array(expected, dtype=np.uint64))

    def test_bad_jump(self):
        bg = self.bit_generator()
        with pytest.raises(ValueError, match="iter must be"):
            bg.jumped(-1)


class TestSquares32(TestSquares):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(Squares, variant=32)
        cls.data1 = cls._read_csv(join(pwd, "./data/squares-32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/squares-32-testset-2.csv"))

    def test_uniform_double(self):
        # Special case since this ia a 32-bit generator that returns 2 32-bit per pass
        # but uses the better formula discarding the lower bits of both 32 bit
        # values when generating doubles
        rs = np.random.Generator(self.setup_bitgenerator(self.data1["seed"]))
        transformed = self.data1["data"].view(np.uint32)
        vals = uniform_from_uint(transformed, 32)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

        rs = np.random.Generator(self.setup_bitgenerator(self.data2["seed"]))
        transformed = self.data2["data"].view(np.uint32)
        vals = uniform_from_uint(transformed, 32)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

    def test_gauss_inv(self):
        # Special case since this ia a 32-bit generator that returns 2 32-bit per pass
        # but uses the better formula discarding the lower bits of both 32 bit
        # values when generating doubles
        n = 25
        rs = np.random.RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, "squares32"))

        rs = np.random.RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(
            gauss,
            gauss_from_uint(self.data2["data"], n, "squares32"),
            rtol=3e-6,
        )


class TestEFIIX64(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = EFIIX64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/efiix64-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/efiix64-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True


class TestRomuQuad(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = Romu
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/romuquad-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/romuquad-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True

    def test_bad_varainte(self):
        with pytest.raises(TypeError, match="variant must be"):
            self.bit_generator(variant=3)
        with pytest.raises(ValueError, match="variant must be"):
            self.bit_generator(variant="both")


class TestRomuTrio(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = CustomPartial(Romu, variant="trio")
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/romutrio-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/romutrio-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        cls.invalid_seed_types += [("apple",)]
        cls.seed_sequence_only = True


def test_numpy_mode_error_pcg64():
    with pytest.raises(ValueError, match="variant must be"):
        PCG64(variant="dxsm-128", numpy_seed=True)
    with pytest.raises(ValueError, match="inc must be"):
        PCG64(0, inc=1, numpy_seed=True)


@pytest.mark.thread_unsafe(reason="runs pytest")
def test_test_runner():
    val = rg_tester(["--collect-only"], exit=False)
    assert val == 0
    val = rg_tester("--collect-only", exit=False)
    assert val == 0


def test_bitgen_ctor():
    assert isinstance(rg_pkl.__bit_generator_ctor(), MT19937)
    assert isinstance(rg_pkl.__bit_generator_ctor("LXM"), LXM)
    assert isinstance(rg_pkl.__bit_generator_ctor(b"LXM"), LXM)
    with pytest.raises(ValueError, match="NewFangledBG"):
        rg_pkl.__bit_generator_ctor("NewFangledBG")


def test_speck_sse41():
    s = SPECK128(0)
    if not s.use_sse41:
        pytest.skip("Can only test if SSE41 is available")
    assert isinstance(s.use_sse41, bool)
    orig = s.use_sse41
    s.use_sse41 = not orig
    assert s.use_sse41 != orig
    s.use_sse41 = orig


SIMD_FLAGS = pytest.mark.parametrize(
    ("bit_gen", "flag"),
    [
        (AESCounter, "use_aesni"),
        (BlaBla, "use_avx2"),
        (ChaCha, "use_simd"),
        (SPECK128, "use_sse41"),
    ],
    ids=["AESCounter", "BlaBla", "ChaCha", "SPECK128"],
)


@SIMD_FLAGS
def test_simd_flag_per_instance(bit_gen, flag):
    bg = bit_gen(0)
    if not getattr(bg, flag):
        pytest.skip(f"{flag} is not supported by this CPU")
    setattr(bg, flag, False)
    # Neither creating another instance nor reseeding turns it back on
    assert getattr(bit_gen(1), flag)
    bg.seed(0)
    assert not getattr(bg, flag)

    expected = bit_gen(0).random_raw(1000)
    assert_equal(bg.random_raw(1000), expected)
    # Seeded without SIMD, then switched on part way through
    bg.seed(0)
    first = bg.random_raw(500)
    setattr(bg, flag, True)
    assert_equal(np.concatenate([first, bg.random_raw(500)]), expected)


@SIMD_FLAGS
def test_simd_flag_not_pickled(bit_gen, flag):
    # A pickle may be loaded on a CPU without the feature, so the copy must
    # check the CPU it runs on rather than inherit the setting
    bg = bit_gen(0)
    default = getattr(bg, flag)
    setattr(bg, flag, False)
    bg.random_raw(7)
    expected = copy.deepcopy(bg).random_raw(1000)
    for restored in (pickle.loads(pickle.dumps(bg)), copy.deepcopy(bg)):
        assert getattr(restored, flag) == default
        assert_equal(restored.random_raw(1000), expected)
    assert not any(flag in key for key in bg.state)


def test_pcg64_errors():
    with pytest.raises(TypeError, match="variant must be"):
        PCG64(0, variant=("dxsm-128",))
    p_np = PCG64(0, numpy_seed=True)
    with pytest.raises(ValueError, match="inc must be"):
        p_np.seed(seed=0, inc=3)
    p = PCG64(0)
    with pytest.raises(TypeError, match="inc must be a scalar"):
        p.seed(seed=0, inc=[3, 4])
    with pytest.raises(ValueError, match="inc must be a scalar"):
        p.seed(seed=0, inc=-1)
    with pytest.raises(ValueError, match="inc must be a scalar"):
        p.seed(seed=0, inc=sum(2**i for i in range(129)))


@pytest.mark.parametrize(
    ("bg_ctor", "ub"),
    [
        (PCG32, 2**64),
        (PCG64, 2**128),
        (CustomPartial(PCG64, variant="dxsm"), 2**128),
    ],
)
def test_inc_upper_bound(bg_ctor, ub):
    with pytest.raises(ValueError, match="inc must be a scalar"):
        bg_ctor(0, inc=ub)
    with pytest.raises(ValueError, match="inc must be a scalar"):
        bg_ctor(0).seed(0, inc=ub)
    with pytest.raises(ValueError, match="inc must be a scalar"):
        bg_ctor(0, inc=ub + 1)
    bg = bg_ctor(0, inc=ub - 1)
    assert isinstance(bg.random_raw(), int)


@pytest.mark.parametrize("bg_ctor", [PCG32, PCG64, LCG128Mix, PCG64DXSM])
def test_inc_with_seed_none(bg_ctor):
    # inc is used even when seed is None
    bg = bg_ctor(None, inc=5)
    assert bg.state["state"]["inc"] == bg_ctor(0, inc=5).state["state"]["inc"]
    bg.seed(None, inc=7)
    assert bg.state["state"]["inc"] == bg_ctor(0, inc=7).state["state"]["inc"]
    bg.seed(None, inc=None)
    assert bg.state["state"]["inc"] not in (11, 15)


def test_inc_with_seed_none_numpy_seed():
    with pytest.raises(ValueError, match="inc must be"):
        PCG64(None, inc=5, numpy_seed=True)


ADVANCE_STEP_BITS = [(PCG32, 64), (PCG64, 128), (PCG64DXSM, 128), (LCG128Mix, 128)]


def _golden_floors(bits):
    # Integer parts of 2**xx / phi for xx = 1, ..., bits, from 100-digit decimals
    # rather than the integer arithmetic that advance_step uses
    with localcontext() as ctx:
        ctx.prec = 100
        inv_phi = (Decimal(5).sqrt() - 1) / 2
        return [int(Decimal(2) ** xx * inv_phi) for xx in range(1, bits + 1)]


@pytest.mark.parametrize(("bg_cls", "bits"), ADVANCE_STEP_BITS)
def test_advance_step(bg_cls, bits):
    floors = _golden_floors(bits)
    cases = {1, 2, 3, 1000, 10**10, 2**bits // 3}
    for floor in floors:
        cases.update({floor - 1, floor, floor + 1})
    cases = sorted(n for n in cases if 1 <= n <= floors[-1])
    for max_draws in cases:
        # The smallest xx with 2**xx / phi > max_draws, and the closest odd integer
        floor = next(f for f in floors if f >= max_draws)
        expected = floor if floor % 2 else floor + 1
        step = bg_cls.advance_step(max_draws)
        assert step == expected
        assert step % 2 == 1
        assert step >= max_draws


@pytest.mark.parametrize(
    ("max_draws", "step"),
    [
        (1, 1),
        (1000, 0x4F1),
        (10**10, 0x278DDE6E5),
        (10**14, 0x9E3779B97F4B),
        (10**19, 0x9E3779B97F4A7C15),
        (2**63, 0x9E3779B97F4A7C15),
    ],
)
@pytest.mark.parametrize("bg_cls", [PCG32, PCG64, PCG64DXSM, LCG128Mix])
def test_advance_step_values(bg_cls, max_draws, step):
    assert bg_cls.advance_step(max_draws) == step
    assert bg_cls(0).advance_step(max_draws) == step
    assert bg_cls.advance_step(np.uint64(max_draws)) == step


@pytest.mark.parametrize("bg_cls", [PCG64, PCG64DXSM, LCG128Mix])
def test_advance_step_numpy_jumped(bg_cls):
    # The step for XX = 128 is the step of NumPy's PCG64.jumped
    assert bg_cls.advance_step(2**127) == 0x9E3779B97F4A7C15F39CC0605CEDC835


@pytest.mark.parametrize(
    ("bg_cls", "max_draws"),
    [(PCG32, 2**63), (PCG64, 2**127), (PCG64DXSM, 2**127), (LCG128Mix, 2**127)],
)
def test_advance_step_jumped(bg_cls, max_draws):
    # The step for XX = bits is the step of jumped
    step = bg_cls.advance_step(max_draws)
    for i in (1, 2, 5):
        jumped = bg_cls(1234).jumped(i)
        advanced = bg_cls(1234).advance(i * step)
        assert jumped.state == advanced.state


@pytest.mark.parametrize(("bg_cls", "bits"), ADVANCE_STEP_BITS)
def test_advance_step_errors(bg_cls, bits):
    largest = _golden_floors(bits)[-1]
    assert bg_cls.advance_step(largest) < 2**bits
    for max_draws in (0, -1, largest + 1, 2**bits):
        with pytest.raises(ValueError, match=f"between 1 and {largest}"):
            bg_cls.advance_step(max_draws)
    for max_draws in (10.0**10, "10", None):
        with pytest.raises(TypeError):
            bg_cls.advance_step(max_draws)


@pytest.mark.parametrize("bg_cls", [PCG32, PCG64, PCG64DXSM, LCG128Mix])
def test_advance_step_streams_do_not_overlap(bg_cls):
    max_draws = 1000
    step = bg_cls.advance_step(max_draws)
    streams = [bg_cls(1234).advance(i * step) for i in range(4)]
    single = bg_cls(1234).random_raw(4 * step)
    for i, stream in enumerate(streams):
        draws = stream.random_raw(max_draws)
        assert_array_equal(draws, single[i * step : i * step + max_draws])


class TestBlaBla(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = BlaBla
        cls.bits = 64
        cls.seed_bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/blabla-testset-1-avx2.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/blabla-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,)]
        cls.state_name = "key"

    def setup_bitgenerator(self, seed, **kwargs):
        seed = [0] if None in seed else seed
        bg = self.bit_generator(*seed, **kwargs)
        return bg

    def test_set_key(self):
        with pytest.raises(ValueError, match="seed and key cannot"):
            self.bit_generator(seed=0, key=0)

    def test_invalid_rounds(self):
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=3)
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=-4)

    def test_use_avx2(self):
        bg = self.bit_generator(0)
        if not bg.use_avx2:
            with pytest.raises(ValueError, match="CPU does not support AVX2"):
                bg.use_avx2 = True
            return
        bg2 = self.bit_generator(0)
        bg2.use_avx2 = not bg.use_avx2
        assert_equal(bg.random_raw(100), bg2.random_raw(100))

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], key=0, counter=0)


class TestMWC192X(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MWC192X
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/mwc192x-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mwc192x-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,)]
        # log2 of the jump and long-jump distances of the reference code
        cls.jump_log2 = 96
        cls.long_jump_log2 = 144
        cls.jump_n = 0x9E3779B97F4A7C15 + (0x0123456789ABCDEF << 64)
        # States reached from SeedSequence(0xDEADBEAF) by jump(), jump()
        # twice, long_jump() and jump_n(jump_n) of the reference C code
        # (see src/mwcx/mwcx-test-data-gen.c)
        cls.ref_jump = [0x378C18A9317CEFEC, 0xA3122FCAD13E3A74, 0x4CCF8DE02AF05A17]
        cls.ref_jump2 = [0xA402F4824C96FD18, 0x89FE66B5AE1C0A1C, 0x3C7769471B035208]
        cls.ref_long_jump = [
            0x766F5F67BFC844D6,
            0x79E26C48292E3CAE,
            0x738B12B17C7634E1,
        ]
        cls.ref_jump_n = [0x1296749179EBA3A7, 0x1B6682F57E9C91C7, 0x3EB11682E8DEF27]
        # A seed whose SeedSequence output is not a valid state
        cls.fallback_seed = 61

    def words(self, bg):
        return [bg.state["state"][name] for name in bg._names]

    def set_words(self, bg, words):
        state = bg.state
        state["state"] = dict(zip(bg._names, words))
        bg.state = state

    def test_reference_jumps(self):
        seed = self.data1["seed"][0]
        bg = self.bit_generator(seed)
        assert self.words(bg.jumped()) == self.ref_jump
        assert self.words(bg.jumped(2)) == self.ref_jump2
        bg.advance(2**self.long_jump_log2)
        assert self.words(bg) == self.ref_long_jump
        bg = self.bit_generator(seed).advance(self.jump_n)
        assert self.words(bg) == self.ref_jump_n

    def test_advance_symmetry(self):
        bg = self.bit_generator(self.data1["seed"][0])
        bg.random_raw(1000)
        state = self.words(bg)
        bg.advance(-1000)
        assert_equal(bg.random_raw(1000), self.data1["data"])
        assert self.words(bg.jumped(-1).advance(2**self.jump_log2)) == state

    def test_seed_fallback(self):
        bg = self.bit_generator(self.fallback_seed)
        assert bg.state["state"]["c"] == 1

    def test_invalid_state(self):
        bg = self.bit_generator(self.data1["seed"][0])
        words = self.words(bg)
        with pytest.raises(ValueError, match="state is not valid"):
            self.set_words(bg, [0] * len(words))
        with pytest.raises(ValueError, match="state is not valid"):
            self.set_words(bg, words[:-1] + [bg._a])
        with pytest.raises(ValueError, match="must be in"):
            self.set_words(bg, [2**bg._word_bits] + words[1:])


class TestMWC128X(TestMWC192X):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MWC128X
        cls.data1 = cls._read_csv(join(pwd, "./data/mwc128x-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mwc128x-testset-2.csv"))
        cls.jump_log2 = 64
        cls.long_jump_log2 = 96
        cls.jump_n = 0x9E3779B97F4A7C15 + (0x0123456789ABCDEF << 64)
        cls.ref_jump = [0x817813F49D3ED2D1, 0xAA3D8684C281159D]
        cls.ref_jump2 = [0xC607BC0DB8C92DCA, 0x814F18F907E73B2A]
        cls.ref_long_jump = [0x8A0A18B33375C8AC, 0x198ACB1ED1AD5560]
        cls.ref_jump_n = [0x54807E33BCFCCC66, 0xA72CE4A567B8A02E]
        cls.fallback_seed = 920


class TestMWC128X32(TestMWC192X):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MWC128X32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/mwc128x32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mwc128x32-testset-2.csv"))
        cls.jump_log2 = 64
        cls.long_jump_log2 = 96
        cls.jump_n = 0x9E3779B97F4A7C15 + (0x0123456789ABCDEF << 64)
        cls.ref_jump = [0x7A1F8053, 0x11635201, 0xCE3AF6B1, 0x1B6BFF72]
        cls.ref_jump2 = [0x8CF33DEA, 0x866F4BD7, 0x0C8DAB7A, 0x2BA18B81]
        cls.ref_long_jump = [0xA9D9C54D, 0xB4B948C1, 0x2190E965, 0x616E0173]
        cls.ref_jump_n = [0x8ED6AB73, 0x26158A35, 0x59D83BCE, 0x51A9EB84]
        cls.fallback_seed = 1226


class TestMWC64X32(TestMWC128X32):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = MWC64X32
        cls.data1 = cls._read_csv(join(pwd, "./data/mwc64x32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/mwc64x32-testset-2.csv"))
        cls.jump_log2 = 32
        cls.long_jump_log2 = 48
        cls.jump_n = 0x9E3779B97F4A7C15
        cls.ref_jump = [0x524EACF3, 0x72C5D1CA]
        cls.ref_jump2 = [0x9A80E78A, 0xB041A5D8]
        cls.ref_long_jump = [0x919D6B6A, 0x685C7E2B]
        cls.ref_jump_n = [0xBAA48E47, 0xA7823455]
        cls.fallback_seed = 176


# Bit generators with a configuration that selects the output functions. A
# Generator copies these functions when it is created, so the configuration
# cannot be changed through state.
CONFIGURATIONS: dict[Any, list[dict[str, Any]]] = {
    PCG64: [{"variant": v} for v in ("xsl-rr", "dxsm", "dxsm-128")],
    PCG64DXSM: [{}],
    JSF: [{"size": 64}, {"size": 32}],
    Philox: [{"number": n, "width": w} for n in (4, 2) for w in (64, 32)],
    ThreeFry: [{"number": n, "width": w} for n in (4, 2) for w in (64, 32)],
    Romu: [{"variant": "quad"}, {"variant": "trio"}],
    Tyche: [{"original": True}, {"original": False}],
    Squares: [{"variant": 64}, {"variant": 32}],
    Xoroshiro128: [{"plusplus": False}, {"plusplus": True}],
}


def _config_id(cls, kwargs):
    return "-".join([cls.__name__] + [f"{k}={v}" for k, v in kwargs.items()])


ALL_CONFIGURATIONS = [
    pytest.param(cls, kwargs, id=_config_id(cls, kwargs))
    for cls, configs in CONFIGURATIONS.items()
    for kwargs in configs
]
CONFIGURATION_CHANGES = [
    pytest.param(
        cls, kwargs, other, id=f"{_config_id(cls, kwargs)}->{_config_id(cls, other)}"
    )
    for cls, configs in CONFIGURATIONS.items()
    for kwargs in configs
    for other in configs
    if kwargs != other
]


def _draw(gen):
    # Uses the double, 64-bit and 32-bit output functions
    return np.concatenate(
        [
            gen.random(10),
            gen.integers(0, 2**64, 10, dtype=np.uint64).astype(float),
            gen.integers(0, 2**32, 10, dtype=np.uint32).astype(float),
        ]
    )


@pytest.mark.parametrize(("cls", "kwargs", "other"), CONFIGURATION_CHANGES)
def test_state_configuration_change_rejected(cls, kwargs, other):
    bg = cls(1234, **kwargs)
    gen = np.random.Generator(bg)
    extended = ExtendedGenerator(cls(1234, **kwargs))
    other_state = cls(5678, **other).state
    state = bg.state
    with pytest.raises(ValueError, match="Create a new"):
        bg.state = other_state
    with pytest.raises(ValueError, match="Create a new"):
        extended.bit_generator.state = other_state
    # Nothing was written
    assert_state_equal(bg.state, state)
    # The existing Generators still match fresh ones
    fresh = np.random.Generator(cls(1234, **kwargs))
    assert_array_equal(_draw(gen), _draw(fresh))
    fresh_extended = ExtendedGenerator(cls(1234, **kwargs))
    assert_array_equal(
        extended.complex_normal(size=10), fresh_extended.complex_normal(size=10)
    )


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
def test_state_same_configuration(cls, kwargs):
    bg = cls(1234, **kwargs)
    gen = np.random.Generator(bg)
    other = cls(5678, **kwargs)
    bg.state = other.state
    assert_array_equal(_draw(gen), _draw(np.random.Generator(other)))


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
def test_pickle_configurations(cls, kwargs):
    bg = cls(1234, **kwargs)
    gen = np.random.Generator(bg)
    gen.random(3)
    reloaded = pickle.loads(pickle.dumps(bg))
    copied = copy.deepcopy(bg)
    for other in (reloaded, copied):
        assert type(other) is cls
        assert other is not bg
        assert_state_equal(other.state, bg.state)
    expected = _draw(gen)
    assert_array_equal(expected, _draw(np.random.Generator(reloaded)))
    assert_array_equal(expected, _draw(np.random.Generator(copied)))


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
def test_pickle_generators_configurations(cls, kwargs):
    extended = ExtendedGenerator(cls(1234, **kwargs))
    extended.uintegers(3)
    reloaded = pickle.loads(pickle.dumps(extended))
    assert type(reloaded.bit_generator) is cls
    assert_array_equal(
        extended.complex_normal(size=10), reloaded.complex_normal(size=10)
    )

    gen = np.random.Generator(cls(1234, **kwargs))
    reloaded = pickle.loads(pickle.dumps(gen))
    assert type(reloaded.bit_generator) is cls
    assert_array_equal(_draw(gen), _draw(reloaded))


class _Reduced:
    """Pickles as a fixed reduce tuple"""

    def __init__(self, reduced):
        self._reduced = reduced

    def __reduce__(self):
        return self._reduced


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
@pytest.mark.parametrize("style", ["class", "name", "dict"])
def test_unpickle_randomgen_2_3_bit_generator(cls, kwargs, style):
    # randomgen 2.3 used NumPy's reduce, which builds the default configuration
    # and then sets the state
    bg = cls(1234, **kwargs)
    np.random.Generator(bg).random(3)
    ctor = np_pickle.__bit_generator_ctor
    name = f"{cls.__module__}.{cls.__name__}"
    if style == "class":
        try:
            ctor(MT19937)
        except ValueError:
            pytest.skip("NumPy only accepts the name of the bit generator")
        reduced = (ctor, (cls,), (bg.state, bg._seed_seq))
    elif style == "name":
        reduced = (ctor, (name,), (bg.state, bg._seed_seq))
    else:
        reduced = (ctor, (name,), bg.state)
    reloaded = pickle.loads(pickle.dumps(_Reduced(reduced)))
    assert type(reloaded) is cls
    assert_state_equal(reloaded.state, bg.state)
    assert_array_equal(
        _draw(np.random.Generator(bg)), _draw(np.random.Generator(reloaded))
    )


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
def test_unpickle_randomgen_2_3_extended_generator(cls, kwargs):
    extended = ExtendedGenerator(cls(1234, **kwargs))
    extended.uintegers(3)
    ctor = rg_pkl.__extended_generator_ctor
    name = f"{cls.__module__}.{cls.__name__}"
    reduced = (ctor, (name,), extended.bit_generator.state)
    reloaded = pickle.loads(pickle.dumps(_Reduced(reduced)))
    assert type(reloaded.bit_generator) is cls
    assert_array_equal(
        extended.complex_normal(size=10), reloaded.complex_normal(size=10)
    )


@pytest.mark.parametrize(("cls", "kwargs"), ALL_CONFIGURATIONS)
def test_jumped_configurations(cls, kwargs):
    bg = cls(1234, **kwargs)
    if not hasattr(bg, "jumped"):
        pytest.skip("jumped not available")
    jumped = bg.jumped()
    assert type(jumped) is cls
    # Fails if the configuration is different
    same_config = cls(1234, **kwargs)
    same_config.state = jumped.state
    assert_array_equal(
        _draw(np.random.Generator(jumped)),
        _draw(np.random.Generator(same_config)),
    )


ROUND_TRIP_CONFIGURATIONS = [
    *ALL_CONFIGURATIONS,
    *[
        pytest.param(cls, {}, id=cls.__name__)
        for cls in (
            AESCounter,
            BlaBla,
            ChaCha,
            DSFMT,
            EFIIX64,
            HC128,
            LCG128Mix,
            LXM,
            MT19937,
            MT64,
            PCG32,
            SFC64,
            SFMT,
            SPECK128,
            Xorshift1024,
            Xoshiro256,
            Xoshiro512,
        )
    ],
]


@pytest.mark.parametrize(("cls", "kwargs"), ROUND_TRIP_CONFIGURATIONS)
@pytest.mark.parametrize("warmup", [0, 7, 1000])
def test_state_round_trip(cls, kwargs, warmup):
    bg = cls(1234, **kwargs)
    ref = cls(1234, **kwargs)
    bg.random_raw(warmup)
    ref.random_raw(warmup)
    bg.state = bg.state
    assert_state_equal(bg.state, ref.state)
    assert_array_equal(bg.random_raw(100), ref.random_raw(100))


def _replace(state, path, value):
    state = copy.deepcopy(state)
    inner = state
    for key in path[:-1]:
        inner = inner[key]
    inner[path[-1]] = value
    return state


INVALID_STATES = [
    (Philox, {}, ("buffer_pos",), 5, "buffer_pos"),
    (Philox, {}, ("buffer_pos",), -1, "buffer_pos"),
    (Philox, {}, ("number",), 6, "created with"),
    (ThreeFry, {}, ("buffer_pos",), 5, "buffer_pos"),
    (ThreeFry, {}, ("buffer_pos",), -1, "buffer_pos"),
    (ThreeFry, {}, ("number",), 6, "created with"),
    (MT19937, {}, ("state", "pos"), 625, "pos"),
    (MT19937, {}, ("state", "pos"), -1, "pos"),
    (MT64, {}, ("state", "pos"), 313, "pos"),
    (MT64, {}, ("state", "pos"), -1, "pos"),
    (Xorshift1024, {}, ("state", "p"), 16, "p must"),
    (Xorshift1024, {}, ("state", "p"), -1, "p must"),
    (SPECK128, {}, ("state", "rounds"), 0, "rounds"),
    (SPECK128, {}, ("state", "rounds"), 35, "rounds"),
    (SPECK128, {}, ("state", "rounds"), -1, "rounds"),
    (SPECK128, {}, ("state", "offset"), 97, "offset"),
    (SPECK128, {}, ("state", "offset"), -8, "offset"),
    (SPECK128, {}, ("state", "offset"), 12, "multiple of 8"),
    (AESCounter, {}, ("s", "offset"), 65, "offset"),
    (AESCounter, {}, ("s", "offset"), -8, "offset"),
    (AESCounter, {}, ("s", "offset"), 4, "multiple of 8"),
    (BlaBla, {}, ("state", "rounds"), -2, "rounds"),
    (BlaBla, {}, ("state", "rounds"), 0, "rounds"),
    (BlaBla, {}, ("state", "rounds"), 3, "even"),
    (ChaCha, {}, ("state", "rounds"), -2, "rounds"),
    (ChaCha, {}, ("state", "rounds"), 0, "rounds"),
    (ChaCha, {}, ("state", "rounds"), 3, "even"),
    (HC128, {}, ("state", "hc_idx"), 1024, "hc_idx"),
    (HC128, {}, ("state", "hc_idx"), -16, "hc_idx"),
    (HC128, {}, ("state", "hc_idx"), 8, "multiple of 16"),
    (HC128, {}, ("state", "buffer_idx"), 17, "buffer_idx"),
    (HC128, {}, ("state", "buffer_idx"), -1, "buffer_idx"),
    (DSFMT, {}, ("state", "idx"), 383, "idx"),
    (DSFMT, {}, ("state", "idx"), -1, "idx"),
    (DSFMT, {}, ("buffer_loc",), 383, "buffer_loc"),
    (DSFMT, {}, ("buffer_loc",), -1, "buffer_loc"),
    (DSFMT, {}, ("buffered_uniforms",), np.zeros(10), "buffered_uniforms"),
    (SFMT, {}, ("state", "idx"), 625, "idx"),
    (SFMT, {}, ("state", "idx"), -1, "idx"),
    (SFMT, {}, ("buffer_loc",), 313, "buffer_loc"),
    (SFMT, {}, ("buffer_loc",), -1, "buffer_loc"),
    (PCG64, {}, ("variant",), "unknown", "variant unknown is not known"),
    (PCG64, {}, ("variant",), 3, "variant 3 is not known"),
    (JSF, {}, ("state", "p"), 64, "p must"),
    (JSF, {}, ("state", "q"), -1, "q must"),
    (JSF, {}, ("state", "r"), 64, "r must"),
    (JSF, {"size": 32}, ("state", "r"), 32, "r must"),
    (JSF, {}, ("state", "a"), 2**64, "a must"),
    (JSF, {"size": 32}, ("state", "d"), 2**32, "d must"),
    (JSF, {}, ("seed_size",), 0, "seed_size"),
    (JSF, {}, ("seed_size",), 4, "seed_size"),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "path", "value", "match"),
    [
        pytest.param(
            *args, id=f"{args[0].__name__}-{'.'.join(args[2])}={args[3]!r}"[:60]
        )
        for args in INVALID_STATES
    ],
)
def test_invalid_state_rejected(cls, kwargs, path, value, match):
    bg = cls(1234, **kwargs)
    state = bg.state
    invalid = _replace(state, path, value)
    with pytest.raises(ValueError, match=match):
        bg.state = invalid
    # Nothing was written
    assert_state_equal(bg.state, state)
    bg.state = state


def test_mt19937_legacy_invalid_pos():
    bg = MT19937(1234)
    state = bg.state
    legacy = ("MT19937", state["state"]["key"], 625, 0, 0.0)
    with pytest.raises(ValueError, match="pos"):
        bg.state = legacy
    assert_state_equal(bg.state, state)


@pytest.mark.parametrize("cls", [HC128, SPECK128, EFIIX64, Xoshiro256, DSFMT, SFMT])
def test_state_non_contiguous_arrays(cls):
    # Arrays are read through a pointer so must be contiguous
    def spread(value):
        if not isinstance(value, np.ndarray):
            return value
        out = np.empty((value.shape[0], 2), dtype=value.dtype)
        out[:, 0] = value
        out[:, 1] = ~value if value.dtype.kind == "u" else 1
        out = out[:, 0]
        assert not out.flags.c_contiguous
        return out

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(val) for key, val in value.items()}
        return spread(value)

    bg = cls(1234)
    bg.random_raw(5)
    ref = cls(1234)
    ref.random_raw(5)
    bg.state = convert(bg.state)
    assert_state_equal(bg.state, ref.state)
    assert_array_equal(bg.random_raw(50), ref.random_raw(50))


def _seq():
    return SeedSequence(20241103)


# The first 64-bit and double draws below were produced by GCC and MSVC, which
# agree. next64 combines two 32-bit draws, so these values pin the order in
# which the two draws are made.
COMBINED_DRAWS = [
    (lambda: MT19937(_seq()), 0xBA359F020B92E667, 0.7273806932956745),
    (lambda: JSF(_seq(), size=32), 0x2EEF8B101D2FB113, 0.18334263938339634),
    (lambda: PCG32(_seq(), inc=1), 0x8DB8519F2962EDCB, 0.5535937261705531),
    (
        lambda: Philox(_seq(), number=2, width=32),
        0x0585D7598E0C64EB,
        0.02157350297925331,
    ),
    (
        lambda: Philox(_seq(), number=4, width=32),
        0x1382265E1C1729D1,
        0.07620467327332825,
    ),
    (
        lambda: ThreeFry(_seq(), number=2, width=32),
        0x5E3FF57A8B9D45B0,
        0.3681634332639614,
    ),
    (
        lambda: ThreeFry(_seq(), number=4, width=32),
        0x7A6C8547A17BF935,
        0.4782183962374771,
    ),
    (lambda: Tyche(_seq(), original=True), 0xA1DFCBF6FF1BF3ED, 0.6323211192825456),
    (lambda: Tyche(_seq(), original=False), 0x196A969D53A2DA39, 0.0992826545737957),
    (
        lambda: Squares(key=0xF9DB92E5A801E679, variant=32),
        0x80EB62F8E5B545D5,
        0.8972972519963361,
    ),
    (lambda: ChaCha(_seq()), 0xF60C7C2D96091EA5, 0.9611280070585376),
    (lambda: HC128(_seq()), 0xB4EAB755D3BEB319, 0.7067064842738044),
]
COMBINED_IDS = [
    "MT19937",
    "JSF32",
    "PCG32",
    "Philox2x32",
    "Philox4x32",
    "ThreeFry2x32",
    "ThreeFry4x32",
    "Tyche",
    "Tyche-openrand",
    "Squares32",
    "ChaCha",
    "HC128",
]


@pytest.mark.parametrize(
    ("factory", "uint64", "double"), COMBINED_DRAWS, ids=COMBINED_IDS
)
def test_combined_draw_order(factory, uint64, double):
    gen = np.random.Generator(factory())
    assert int(gen.integers(0, 2**64, dtype=np.uint64, endpoint=False)) == uint64
    gen = np.random.Generator(factory())
    assert gen.random() == double
