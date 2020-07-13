from functools import partial
import os
from os.path import join

import numpy as np
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
    PCG32,
    PCG64,
    RDRAND,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    ChaCha,
    Generator,
    Philox,
    RandomState,
    Romu,
    ThreeFry,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
)
from randomgen.common import interface
from randomgen.seed_sequence import ISeedSequence

try:
    from numpy.random import SeedSequence

    NP_SEED_SEQ = True
except ImportError:
    from randomgen import SeedSequence

    NP_SEED_SEQ = False

MISSING_RDRAND = False
try:
    RDRAND()
except RuntimeError:
    MISSING_RDRAND = True

HAS_AESNI = False
aes = AESCounter(mode="sequence")
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

ISEED_SEQUENCES = (ISeedSequence,)
# NumPy 1.17
try:
    ISEED_SEQUENCES += (np.random.bit_generator.ISeedSequence,)
except AttributeError:
    pass
# NumPy 1.18
try:
    ISEED_SEQUENCES += (np.random._bit_generator.ISeedSequence,)
except AttributeError:
    pass

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


def uniform32_from_uint64(x):
    x = np.uint64(x)
    upper = np.array(x >> np.uint64(32), dtype=np.uint32)
    lower = np.uint64(0xFFFFFFFF)
    lower = np.array(x & lower, dtype=np.uint32)
    joined = np.column_stack([lower, upper]).ravel()
    out = (joined >> np.uint32(9)) * (1.0 / 2 ** 23)
    return out.astype(np.float32)


def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(0xFFFFFFFF))
    out = (x >> np.uint32(9)) * (1.0 / 2 ** 23)
    return out.astype(np.float32)


def uniform32_from_uint32(x):
    return (x >> np.uint32(9)) * (1.0 / 2 ** 23)


def uniform32_from_uint(x, bits):
    if bits == 64:
        return uniform32_from_uint64(x)
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


class Base(object):
    dtype = np.uint64
    data2 = data1 = {}

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
            data = []
            for line in csv:
                data.append(int(line.split(",")[-1].strip(), 0))
            return {"seed": seed, "data": np.array(data, dtype=cls.dtype)}

    def setup_bitgenerator(self, seed, mode="legacy"):
        return self.bit_generator(*seed, mode=mode)

    def test_default(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)
        assert self.seed_sequence_only or bg.seed_seq is None

    def test_default_sequence(self):
        bg = self.setup_bitgenerator([None], mode="sequence")
        val = bg.random_raw()
        assert isinstance(val, int)
        assert isinstance(bg.seed_seq, ISEED_SEQUENCES)

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
        rs = RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        bits = getattr(self, "bit_name", self.bits)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, bits), rtol=3e-6)

        rs = RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss, gauss_from_uint(self.data2["data"], n, bits), rtol=3e-6)

    def test_uniform_double(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        bits = getattr(self, "bit_name", self.bits)
        vals = uniform_from_uint(self.data1["data"], bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

        rs = Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform_from_uint(self.data2["data"], bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals, atol=1e-8)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        vals = uniform32_from_uint(self.data1["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform32_from_uint(self.data2["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.pi)
        assert_raises(self.seed_error_type, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        with pytest.raises(self.seed_error_type):
            rs.bit_generator.seed(np.array([np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([-np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([np.pi, -np.pi]))
        with pytest.raises((ValueError, TypeError)):
            rs.bit_generator.seed(np.array([0, np.pi]))
        with pytest.raises(TypeError):
            rs.bit_generator.seed([np.pi])
        with pytest.raises(TypeError):
            rs.bit_generator.seed([0, np.pi])

    def test_seed_out_of_range(self):
        # GH #82
        if self.seed_sequence_only:
            # Not valid on PRNG that only support seed sequence
            return
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, 2 ** (4 * self.bits + 1))
        assert_raises(ValueError, rs.bit_generator.seed, -1)

    def test_seed_out_of_range_array(self):
        # GH #82
        if self.seed_sequence_only:
            # Not valid on PRNG that only support seed sequence
            return
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, [2 ** (2 * self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])

    def test_repr(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert "Generator" in repr(rs)
        assert type(rs.bit_generator).__name__ in repr(rs)

    def test_str(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert "Generator" in str(rs)
        assert str(type(rs.bit_generator).__name__) in str(rs)
        assert "{:#x}".format(id(rs)).upper().replace("X", "x")[2:] not in str(rs)

    def test_generator(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        if "generator" not in dir(bit_generator):
            pytest.skip("generator attribute has been removed")
        with pytest.raises(NotImplementedError):
            bit_generator.generator

    def test_pickle(self):
        import pickle

        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        bit_generator_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bit_generator_pkl)
        orig_gen = Generator(bit_generator)
        reloaded_gen = Generator(reloaded)
        assert_array_equal(
            orig_gen.standard_normal(1000), reloaded_gen.standard_normal(1000)
        )
        assert bit_generator is not reloaded
        assert_state_equal(reloaded.state, bit_generator.state)

    def test_invalid_state_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(TypeError):
            bit_generator.state = {"1"}

    def test_invalid_state_value(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        state["bit_generator"] = "otherBitGenerator"
        with pytest.raises(ValueError):
            bit_generator.state = state

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            with pytest.raises(TypeError):
                bit_generator.seed(*st)

    def test_invalid_seed_values(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_values:
            with pytest.raises(ValueError):
                bit_generator.seed(*st)

    def test_benchmark(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, "double")
        with pytest.raises(ValueError):
            bit_generator._benchmark(1, "int32")

    @pytest.mark.skipif(MISSING_CFFI, reason="cffi not available")
    def test_cffi(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        cffi_interface = bit_generator.cffi
        assert isinstance(cffi_interface, interface)
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason="ctypes not available")
    def test_ctypes(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        ctypes_interface = bit_generator.ctypes
        assert isinstance(ctypes_interface, interface)
        other_ctypes_interface = bit_generator.ctypes
        assert other_ctypes_interface is ctypes_interface

    def test_getstate(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        alt_state = bit_generator.__getstate__()
        assert_state_equal(state, alt_state)

    def test_uinteger_reset_seed(self):
        bg = self.setup_bitgenerator([None])
        g = Generator(bg)
        g.integers(0, 2 ** 32, dtype=np.uint32)
        if "has_uint32" not in bg.state or bg.state["has_uint32"] == 0:
            name = bg.__class__.__name__
            pytest.skip("bit generator does not cache 32-bit value ({0})".format(name))
        bg.seed()
        assert bg.state["has_uint32"] == 0

    def test_uinteger_reset_jump(self):
        bg = self.setup_bitgenerator([None])
        if not hasattr(bg, "jumped"):
            pytest.skip("bit generator does not support jumping")
        g = Generator(bg)
        g.integers(0, 2 ** 32, dtype=np.uint32)
        jumped = Generator(bg.jumped())
        if "has_uint32" in jumped.bit_generator.state:
            assert jumped.bit_generator.state["has_uint32"] == 0
            return
        # This next test could fail with prob 1 in 2**32
        next_g = g.integers(0, 2 ** 32, dtype=np.uint32)
        next_jumped = jumped.integers(0, 2 ** 32, dtype=np.uint32)
        assert next_g != next_jumped

    def test_uinteger_reset_advance(self):
        bg = self.setup_bitgenerator([None])
        if not hasattr(bg, "advance"):
            pytest.skip("bit generator does not support advancing")
        g = Generator(bg)
        g.integers(0, 2 ** 32, dtype=np.uint32)
        state = bg.state
        if isinstance(bg, (Philox, ThreeFry)):
            bg.advance(1000, False)
        else:
            bg.advance(1000)
        if "has_uint32" in bg.state:
            assert bg.state["has_uint32"] == 0
            return
        # This next test could fail with prob 1 in 2**32
        next_advanced = g.integers(0, 2 ** 32, dtype=np.uint32)
        bg.state = state
        next_g = g.integers(0, 2 ** 32, dtype=np.uint32)
        assert next_g != next_advanced

    def test_seed_sequence(self):
        bg = self.setup_bitgenerator([None], mode="sequence")
        typ = (
            self.bit_generator.func
            if isinstance(self.bit_generator, partial)
            else self.bit_generator
        )
        assert isinstance(bg, typ)
        assert isinstance(bg.seed_seq, SeedSequence)

        bg = self.setup_bitgenerator([0], mode="sequence")
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator(ss)
        assert bg.seed_seq.entropy == 0

    def test_seed_sequence_error(self):
        with pytest.raises(ValueError, match="seed is a SeedSequence"):
            self.bit_generator(SeedSequence(0), mode="legacy")

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

    def setup_bitgenerator(self, seed, mode="legacy", **kwargs):
        return self.bit_generator(*seed, mode=mode, **kwargs)

    def test_advance(self, step, counter_only, warmup):
        bg = self.bit_generator(mode="sequence")
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

    def test_advance_large(self):
        dtype = np.uint64 if self.width == 64 else np.uint32
        bg = self.bit_generator(mode="legacy")
        step = 2 ** self.width
        bg.advance(step, True)
        state = bg.state
        assert_equal(state["state"]["counter"], np.array([0, 1, 0, 0], dtype=dtype))

        bg = self.bit_generator(mode="sequence")
        step = 2 ** self.width - 1
        bg.advance(step, True)
        state = bg.state
        size_max = np.iinfo(dtype).max
        assert_equal(
            state["state"]["counter"], np.array([size_max, 0, 0, 0], dtype=dtype)
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** (2 * self.width)
        bg.advance(step, True)
        state = bg.state
        assert_equal(state["state"]["counter"], np.array([0, 0, 1, 0], dtype=dtype))

        bg = self.bit_generator(mode="sequence")
        step = 2 ** (2 * self.width) - 1
        bg.advance(step, True)
        state = bg.state
        assert_equal(
            state["state"]["counter"], np.array([size_max, size_max, 0, 0], dtype=dtype)
        )

    def test_advance_deprecated(self):
        bg = self.bit_generator(mode="sequence")
        with pytest.warns(FutureWarning):
            bg.advance(1)

    def test_0d_array(self):
        bg = self.bit_generator(np.array(1, dtype=np.uint64), mode="legacy")
        bg2 = self.bit_generator(1, mode="legacy")
        assert_state_equal(bg.state, bg2.state)

    def test_empty_seed(self):
        with pytest.raises(ValueError):
            self.bit_generator(np.array([], dtype=np.uint64), mode="legacy")

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], mode="legacy", key=0, counter=0)


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
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]

    def test_bad_init(self):
        with pytest.raises(ValueError):
            self.bit_generator(size=self.bits - 1, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(p=-10, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(q=120, mode="legacy")

    def test_number_seed(self):
        bg1 = self.bit_generator(0, seed_size=1, mode="legacy")
        bg2 = self.bit_generator(0, seed_size=2, mode="legacy")
        bg3 = self.bit_generator(0, seed_size=3, mode="legacy")
        state1 = bg1.state["state"]
        state2 = bg2.state["state"]
        state3 = bg3.state["state"]
        assert state1["c"] != state2["c"]
        assert state1["c"] != state3["c"]
        assert state2["c"] != state3["c"]
        assert state1["d"] != state2["d"]
        assert state1["d"] != state3["d"]
        assert state2["d"] != state3["d"]

    def test_invalid_seed_size(self):
        with pytest.raises(ValueError, match="seed size must be one"):
            self.bit_generator(seed_size=4, mode="legacy")
        with pytest.raises(ValueError, match="seed size must be one"):
            self.bit_generator(seed_size=1.0, mode="legacy")


class TestJSF32(TestJSF64):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator_base = JSF
        cls.bit_generator = partial(JSF, size=32)
        cls.size = 32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/jsf32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/jsf32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [("apple",), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(size=self.size, mode="sequence")
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg.seed_seq, SeedSequence)

        bg = self.bit_generator_base(0, size=self.size, mode="sequence")
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, size=self.size)
        assert bg.seed_seq.entropy == 1


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
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]


class TestXoroshiro128PlusPlus(TestXoroshiro128):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = partial(Xoroshiro128, plusplus=True)
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, "./data/xoroshiro128plusplus-testset-1.csv")
        )
        cls.data2 = cls._read_csv(
            join(pwd, "./data/xoroshiro128plusplus-testset-2.csv")
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
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]

    def test_old_name(self):
        from randomgen.xoshiro256starstar import Xoshiro256StarStar

        with pytest.deprecated_call():
            bitgen = Xoshiro256StarStar()
            assert isinstance(bitgen, Xoshiro256)


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
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]

    def test_old_name(self):
        from randomgen.xoshiro512starstar import Xoshiro512StarStar

        with pytest.deprecated_call():
            bitgen = Xoshiro512StarStar()
            assert isinstance(bitgen, Xoshiro512)


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
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]


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
            (2 ** 257 + 1,),
            (None, None, 2 ** 257 + 1),
        ]

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        keyed = self.bit_generator(
            counter=state["state"]["counter"], key=state["state"]["key"], mode="legacy"
        )
        assert_state_equal(bit_generator.state, keyed.state)

    def test_advance(self):
        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        bg.advance(1, True)
        assert_equal(
            bg.state["state"]["counter"], np.array([1, 0, 0, 0], dtype=np.uint64)
        )
        bg.advance(1, True)
        assert_equal(
            bg.state["state"]["counter"], np.array([2, 0, 0, 0], dtype=np.uint64)
        )
        bg.advance(2 ** 64, True)
        assert_equal(
            bg.state["state"]["counter"], np.array([2, 1, 0, 0], dtype=np.uint64)
        )
        bg.state = state0
        bg.advance(2 ** 128, True)
        assert_equal(
            bg.state["state"]["counter"], np.array([0, 0, 1, 0], dtype=np.uint64)
        )


class TestPCG64(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(np.array([1, 2]),), (3.2,)]
        cls.invalid_seed_values = [
            (-2,),
            (2 ** 129 + 1,),
        ]

    def test_seed_float_array(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type, rs.bit_generator.seed, np.array([np.pi, -np.pi])
        )
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [np.pi])
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range_array(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(
            self.seed_error_type, rs.bit_generator.seed, [2 ** (2 * self.bits + 1)]
        )
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [-1])

    def test_advance_symmetry(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        step = -0x9E3779B97F4A7C150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2 ** 128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2 ** 128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_inc_none(self):
        bg = self.setup_bitgenerator([0, None])
        assert isinstance(bg.random_raw(), int)

        bg = self.setup_bitgenerator([0, None], mode="sequence")
        assert isinstance(bg.random_raw(), int)


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
            (2 ** 257 + 1,),
            (None, None, 2 ** 257 + 1),
        ]

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        keyed = self.bit_generator(
            counter=state["state"]["counter"], key=state["state"]["key"], mode="legacy"
        )
        assert_state_equal(bit_generator.state, keyed.state)


class TestPhilox4x32(Random123):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator_base = Philox
        cls.bit_generator = partial(Philox, number=4, width=32)
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
            (2 ** 257 + 1,),
            (None, None, 2 ** 257 + 1),
        ]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(
            number=self.number, width=self.width, mode="sequence"
        )
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg.seed_seq, SeedSequence)

        bg = self.bit_generator_base(
            0, number=self.number, width=self.width, mode="sequence"
        )
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, number=self.number, width=self.width)
        assert bg.seed_seq.entropy == 1


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
            (2 ** 257 + 1,),
            (None, None, 2 ** 257 + 1),
        ]

    @pytest.fixture(autouse=True, params=USE_AESNI)
    def set_use_aesni(self, request):
        self.use_aesni = request.param

    def setup_bitgenerator(self, seed, mode="legacy", **kwargs):
        bg = self.bit_generator(*seed, mode=mode, **kwargs)
        bg.use_aesni = self.use_aesni
        return bg

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        key = state["s"]["seed"][:2]
        key = int(key[0]) + int(key[1]) * 2 ** 64
        counter = state["s"]["counter"][0]
        keyed = self.bit_generator(counter=counter, key=key, mode="legacy")
        assert_state_equal(bit_generator.state, keyed.state)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type, rs.bit_generator.seed, np.array([np.pi, -np.pi])
        )
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_bad_state(self):
        bg = self.bit_generator(mode="sequence")
        state = bg.state
        state["s"]["seed"] = np.empty((2, 11), dtype=np.uint64)
        with pytest.raises(ValueError):
            bg.state = state
        state = bg.state
        state["s"]["counter"] = 4
        with pytest.raises(ValueError):
            bg.state = state

    def test_advance(self, step, warmup):
        bg = self.bit_generator(mode="sequence")
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw()
        bg.state = state0
        bg.advance(step)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_one_repeat(self):
        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        bg.random_raw(8)
        direct = bg.random_raw()
        bg.state = state0
        for _ in range(8):
            bg.advance(1)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_large(self):
        bg = self.bit_generator(mode="legacy")
        step = 2 ** 65
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["s"]["counter"], np.array([4, 1, 5, 1, 6, 1, 7, 1], dtype=np.uint64)
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 65 - 7
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["s"]["counter"], np.array([0, 1, 1, 1, 2, 1, 3, 1], dtype=np.uint64)
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 129 - 16
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
            state["s"]["counter"], np.array([0, 0, 1, 0, 2, 0, 3, 0], dtype=np.uint64)
        )

        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        step = 2 ** 129
        bg.advance(step)
        state = bg.state
        assert_state_equal(state0, state)

    @pytest.mark.skip(reason="Not applicable to AESCounter")
    def test_advance_deprecated(self):
        pass

    @pytest.mark.skipif(HAS_AESNI, reason="Not valid when cpu has AESNI")
    def test_no_aesni(self):
        bg = self.bit_generator(mode="sequence")
        with pytest.raises(ValueError, match="CPU does not support AESNI"):
            bg.use_aesni = True

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], mode="legacy", counter=0, key=0)


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
        cls.invalid_seed_values = [(-1,), (np.array([2 ** 33]),)]
        cls.state_name = "key"

    def test_seed_out_of_range(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, 2 ** (self.seed_bits + 1))
        assert_raises(ValueError, rs.bit_generator.seed, -1)
        assert_raises(ValueError, rs.bit_generator.seed, 2 ** (2 * self.seed_bits + 1))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, [2 ** (self.seed_bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        assert_raises(TypeError, rs.bit_generator.seed, [2 ** (2 * self.seed_bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, bit_generator.seed, [np.pi])
        assert_raises(TypeError, bit_generator.seed, [0, np.pi])

    def test_state_tuple(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        state = bit_generator.state
        desired = rs.integers(2 ** 16)
        tup = (state["bit_generator"], state["state"]["key"], state["state"]["pos"])
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)
        tup = tup + (0, 0.0)
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)

    def test_invalid_state(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        state["state"][self.state_name] = state["state"][self.state_name][:10]
        with pytest.raises(ValueError):
            rs.bit_generator.state = state

    def test_negative_jump(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(ValueError):
            bg.jumped(-1)

    def test_bad_legacy_state(self):
        bg = self.setup_bitgenerator(self.data1["seed"])
        with pytest.raises(ValueError):
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
            (np.array([2 ** 33]),),
            (np.array([2 ** 33, 2 ** 33]),),
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
            (np.array([2 ** 33]),),
            (np.array([2 ** 33, 2 ** 33]),),
        ]

    def test_uniform_double(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_array_equal(uniform_from_dsfmt(self.data1["data"]), rs.random(1000))

        rs = Generator(self.setup_bitgenerator(self.data2["seed"]))
        assert_equal(uniform_from_dsfmt(self.data2["data"]), rs.random(1000))

    def test_gauss_inv(self):
        n = 25
        rs = RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss, gauss_from_uint(self.data1["data"], n, "dsfmt"))

        rs = RandomState(self.setup_bitgenerator(self.data2["seed"]))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss, gauss_from_uint(self.data2["data"], n, "dsfmt"))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, [2 ** (self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        assert_raises(TypeError, rs.bit_generator.seed, [2 ** (2 * self.bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_uniform_float(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        vals = uniform32_from_uint(self.data1["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.setup_bitgenerator(self.data2["seed"]))
        vals = uniform32_from_uint(self.data2["data"], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_buffer_reset(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        rs.random(1)
        assert rs.bit_generator.state["buffer_loc"] != 382
        rs.bit_generator.seed(*self.data1["seed"])
        assert rs.bit_generator.state["buffer_loc"] == 382

    def test_negative_jump(self):
        bg = self.bit_generator(mode="sequence")
        with pytest.raises(ValueError, match="iter must be positive"):
            bg.jumped(-1)


class TestThreeFry4x32(Random123):
    @classmethod
    def setup_class(cls):
        super(TestThreeFry4x32, cls).setup_class()
        cls.bit_generator_base = ThreeFry
        cls.bit_generator = partial(ThreeFry, number=4, width=32)
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
            (2 ** 257 + 1,),
            (None, None, 2 ** 129 + 1),
        ]

    def test_seed_sequence(self):
        bg = self.bit_generator_base(
            number=self.number, width=self.width, mode="sequence"
        )
        assert isinstance(bg, self.bit_generator_base)
        assert isinstance(bg.seed_seq, SeedSequence)

        bg = self.bit_generator_base(
            0, number=self.number, width=self.width, mode="sequence"
        )
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(0)
        bg = self.bit_generator_base(ss)
        assert bg.seed_seq.entropy == 0

        ss = SeedSequence(1)
        bg = self.bit_generator_base(ss, number=self.number, width=self.width)
        assert bg.seed_seq.entropy == 1

    def test_set_key(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        state = bit_generator.state
        key = state["state"]["key"]
        power = 32 if key.dtype == np.uint32 else 64
        key = sum([int(key[i]) * 2 ** (power * i) for i in range(len(key))])
        counter = state["state"]["counter"]
        counter = sum([int(counter[i]) * 2 ** (power * i) for i in range(len(counter))])
        keyed = self.bit_generator(counter=counter, key=key, mode="legacy")
        assert_state_equal(bit_generator.state, keyed.state)


class TestPCG32(TestPCG64):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = PCG32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg32-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg32-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(np.array([1, 2]),), (3.2,), (None, np.zeros(1))]
        cls.invalid_seed_values = [
            (-1,),
            (2 ** 129 + 1,),
            (None, -1),
            (None, 2 ** 129 + 1),
        ]

    def test_advance_symmetry(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        state = rs.bit_generator.state
        step = -0x9E3779B97F4A7C16
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2 ** 64 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2 ** 64 + step)
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
        cls.invalid_seed_values = [(-1,), (np.array([2 ** 65]),)]

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        bit_generator = rs.bit_generator
        assert_raises(ValueError, bit_generator.seed, np.array([np.pi]))
        assert_raises(ValueError, bit_generator.seed, np.array([-np.pi]))
        assert_raises(ValueError, bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(ValueError, bit_generator.seed, np.array([0, np.pi]))
        assert_raises(ValueError, bit_generator.seed, [np.pi])
        assert_raises(ValueError, bit_generator.seed, [0, np.pi])

    def test_empty_seed(self):
        with pytest.raises(ValueError, match="Seed must be non-empty"):
            self.bit_generator(np.array([], dtype=np.uint64), mode="legacy")


@pytest.mark.skipif(MISSING_RDRAND, reason="RDRAND is not availble")
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

    def setup_bitgenerator(self, seed, mode=None):
        return self.bit_generator(*seed)

    def test_initialization(self):
        bit_generator = self.bit_generator()
        state = bit_generator.state
        assert state["buffer_loc"] == 256
        assert state["buffer"].shape == (256,)
        assert (state["buffer"] == np.iinfo(np.uint64).max).all()
        assert state["retries"] == 10
        assert state["status"] == 1
        gen = Generator(bit_generator)
        gen.integers(0, 2 ** 64, dtype=np.uint64, size=10)
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
        gen = Generator(bit_generator)
        with pytest.raises(RuntimeError):
            gen.integers(0, 2 ** 64, dtype=np.uint64, size=10)
        assert bit_generator.state["status"] == 0
        bit_generator._reset()
        state = bit_generator.state
        assert state["status"] == 1
        assert state["buffer_loc"] == 256
        assert bit_generator.success is True

    def test_exception(self):
        with pytest.raises(ValueError):
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
        rs = RandomState(self.setup_bitgenerator(self.data1["seed"]))
        gauss = rs.standard_normal(n)
        assert (gauss.max() - gauss.min()) > 0

    def test_uniform_double(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        uniforms = rs.random(1000)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        uniforms = rs.random(1000, dtype=np.float32)
        assert_equal(uniforms.dtype, np.float32)

    @pytest.mark.skip("Bit generator is missing generator attr")
    def test_generator(self):
        pass

    def test_pickle(self):
        import pickle

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

    def test_seed_sequence(self):
        bg = self.bit_generator()
        assert isinstance(bg, self.bit_generator)
        assert bg.seed_seq is None

        with pytest.raises(TypeError):
            self.bit_generator(0)

        ss = SeedSequence(0)
        with pytest.raises(TypeError):
            self.bit_generator(ss)

    def test_default_sequence(self):
        bg = self.setup_bitgenerator([None])
        val = bg.random_raw()
        assert isinstance(val, int)
        assert bg.seed_seq is None

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
        cls.invalid_seed_types = [(np.array([2 ** 65]),)]
        cls.invalid_seed_values = [(-1,), [2 ** 257]]
        cls.state_name = "key"

    def setup_bitgenerator(self, seed, mode="legacy", **kwargs):
        stream = 3735928559 * 2 ** 64 + 3735928559 * 2 ** 96
        seed = [0] if None in seed else seed
        key = seed[0] + stream + 2 ** 128 * stream
        if mode == "legacy":
            bg = self.bit_generator(key=key, counter=0, mode=mode)
        else:
            bg = self.bit_generator(*seed, mode=mode, **kwargs)
        return bg

    def test_set_key(self):
        with pytest.raises(ValueError, match="seed and key cannot"):
            self.bit_generator(seed=0, key=0, mode="legacy")

    def test_invalid_rounds(self):
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=3, mode="legacy")
        with pytest.raises(ValueError, match="rounds must be even and"):
            self.bit_generator(rounds=-4, mode="legacy")

    def test_advance(self, step, warmup):
        bg = self.bit_generator(mode="sequence")
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw()
        bg.state = state0
        # Double step since it is a 32 bit gen, but random_raw is 64 bit
        bg.advance(2 * step)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_one_repeat(self):
        steps = 16
        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        # Half step since 32 bit gen using 64 bit raw
        bg.random_raw(steps // 2)
        direct = bg.random_raw()
        bg.state = state0
        for _ in range(steps):
            bg.advance(1)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_large(self):
        bg = self.bit_generator(mode="legacy")
        step = 2 ** 64
        bg.advance(step)
        state = bg.state
        assert_equal(state["state"]["ctr"], np.array([0, 1], dtype=np.uint64))

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 64 - 1
        bg.advance(step)
        state = bg.state
        u64_max = np.iinfo(np.uint64).max
        assert_equal(state["state"]["ctr"], np.array([u64_max, 0], dtype=np.uint64))

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 128 - 16
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"], np.array([u64_max - 15, u64_max], dtype=np.uint64)
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 128 - 1
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"], np.array([u64_max, u64_max], dtype=np.uint64)
        )
        bg.advance(1)
        state = bg.state
        assert_equal(state["state"]["ctr"], np.array([0, 0], dtype=np.uint64))

        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        step = 2 ** 128
        bg.advance(step)
        state = bg.state
        assert_state_equal(state0, state)

    def test_use_simd(self):
        bg = self.bit_generator(0, mode="legacy")
        if not bg.use_simd:
            with pytest.raises(ValueError):
                bg.use_simd = True
            return
        bg2 = self.bit_generator(0, mode="legacy")
        bg2.use_simd = not bg.use_simd
        assert_equal(bg.random_raw(100), bg2.random_raw(100))

    def test_seed_key(self):
        with pytest.raises(ValueError, match="seed and key"):
            self.setup_bitgenerator([0], mode="sequence", key=0, counter=0)


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
            (np.empty((2, 2), dtype=np.int64),),
        ]
        cls.invalid_seed_values = []

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(
            self.seed_error_type, rs.bit_generator.seed, np.array([np.pi, -np.pi])
        )
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, -1)

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            with pytest.raises((TypeError, ValueError)):
                bit_generator.seed(*st)

    def test_key_init(self):
        with pytest.raises(ValueError):
            self.bit_generator(key=-1, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(key=2 ** 256, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(seed=1, key=1, mode="legacy")


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
            (np.empty((2, 2), dtype=np.int64),),
        ]
        cls.invalid_seed_values = []

    def test_seed_out_of_range(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, 2 ** 257)
        assert_raises(ValueError, rs.bit_generator.seed, -1)

    def test_invalid_seed_type(self):
        bit_generator = self.setup_bitgenerator(self.data1["seed"])
        for st in self.invalid_seed_types:
            with pytest.raises((TypeError, ValueError)):
                bit_generator.seed(*st)

    def test_key_init(self):
        with pytest.raises(ValueError):
            self.bit_generator(key=-1, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(key=2 ** 256, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(seed=1, key=1, mode="legacy")

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.setup_bitgenerator(self.data1["seed"]))
        assert_raises(ValueError, rs.bit_generator.seed, [2 ** (4 * self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])

    def test_advance(self, step, warmup):
        bg = self.bit_generator(mode="sequence")
        bg.random_raw(warmup)
        state0 = bg.state
        bg.random_raw(step)
        direct = bg.random_raw()
        bg.state = state0
        bg.advance(step)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_one_repeat(self):
        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        bg.random_raw(8)
        direct = bg.random_raw()
        bg.state = state0
        for _ in range(8):
            bg.advance(1)
        advanced = bg.random_raw()
        assert direct == advanced

    def test_advance_large(self):
        bg = self.bit_generator(mode="legacy")
        step = 2 ** 65
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"],
            np.array([6, 1, 7, 1, 8, 1, 9, 1, 10, 1, 11, 1], dtype=np.uint64),
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 65 - 11
        bg.advance(step)
        state = bg.state
        assert_equal(
            state["state"]["ctr"],
            np.array([0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1], dtype=np.uint64),
        )

        bg = self.bit_generator(mode="sequence")
        step = 2 ** 129 - 24
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

        bg = self.bit_generator(mode="sequence")
        state0 = bg.state
        step = 2 ** 129
        bg.advance(step)
        state = bg.state
        assert_state_equal(state0, state)

    def test_use_sse41(self):
        bg = self.bit_generator(0, mode="legacy")
        if not bg.use_sse41:
            with pytest.raises(ValueError):
                bg.use_sse41 = True
            return
        bg2 = self.bit_generator(0, mode="legacy")
        bg2.use_sse41 = not bg.use_sse41
        assert_equal(bg.random_raw(100), bg2.random_raw(100))

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            self.bit_generator(rounds=-1, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(rounds=35, mode="legacy")
        with pytest.raises(ValueError):
            self.bit_generator(rounds=27.5, mode="legacy")


def test_mode():
    with pytest.warns(FutureWarning, match="mode is None which currently"):
        mt19937 = MT19937(0)
    assert mt19937.seed_seq is None
    with pytest.raises(ValueError, match="mode must be one of None"):
        MT19937(mode="unknown")


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
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
        cls.seed_sequence_only = True

    def setup_bitgenerator(self, seed, mode=None):
        return self.bit_generator(*seed)

    def test_seed_sequence_error(self):
        pass


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
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
        cls.seed_sequence_only = True


class TestPCG64DXSM(Base):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = partial(PCG64, mode="sequence", variant="dxsm-128")
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-dxsm-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-dxsm-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
        cls.seed_sequence_only = True

    def setup_bitgenerator(self, seed, mode=None):
        return self.bit_generator(*seed)


class TestPCG64CMDXSM(TestPCG64DXSM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = partial(PCG64, mode="sequence", variant="dxsm")
        cls.data1 = cls._read_csv(join(pwd, "./data/pcg64-cm-dxsm-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/pcg64-cm-dxsm-testset-2.csv"))


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
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
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
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
        cls.seed_sequence_only = True

    def test_bad_varainte(self):
        with pytest.raises(TypeError):
            self.bit_generator(variant=3)
        with pytest.raises(ValueError):
            self.bit_generator(variant="both")


class TestRomuTrio(TestLXM):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bit_generator = partial(Romu, variant="trio")
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, "./data/romutrio-testset-1.csv"))
        cls.data2 = cls._read_csv(join(pwd, "./data/romutrio-testset-2.csv"))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), ([-2],)]
        if NP_SEED_SEQ:
            cls.invalid_seed_types += [("apple",)]
        else:
            cls.invalid_seed_values += [("apple",)]
        cls.seed_sequence_only = True
