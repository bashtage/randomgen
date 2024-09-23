import numpy as np
import pytest

from randomgen import PCG64, PCG64DXSM, LCG128Mix, SeedSequence
from randomgen.pcg64 import DEFAULT_DXSM_MULTIPLIER, DEFAULT_MULTIPLIER

try:
    from numba import cfunc, types

    MISSING_NUMBA = False
except ImportError:
    MISSING_NUMBA = True

SEED = 12345678909876543321

C_SOURCE = """
#include <inttypes.h>

uint64_t output_upper(uint64_t a, uint64_t b) {
    return a;
}
"""


def test_pcg64_1():
    pcg = PCG64(SeedSequence(12345678909876543321), variant="xsl-rr")
    cpcg = LCG128Mix()
    st = cpcg.state
    st["state"]["state"] = pcg.state["state"]["state"]
    st["state"]["inc"] = pcg.state["state"]["inc"]
    cpcg.state = st
    expected = pcg.random_raw(1000)
    result = cpcg.random_raw(1000)
    np.testing.assert_equal(expected, result)


def test_pcg64_dxsm():
    pcg = PCG64(SeedSequence(12345678909876543321), variant="dxsm-128")
    cpcg = LCG128Mix(output="dxsm")
    st = cpcg.state
    st["state"]["state"] = pcg.state["state"]["state"]
    st["state"]["inc"] = pcg.state["state"]["inc"]
    cpcg.state = st
    expected = pcg.random_raw(1000)
    result = cpcg.random_raw(1000)
    np.testing.assert_equal(expected, result)


def test_pcg64_cm_dxsm():
    pcg = PCG64(SeedSequence(12345678909876543321), variant="dxsm")
    cpcg = LCG128Mix(output="dxsm", post=False, multiplier=DEFAULT_DXSM_MULTIPLIER)
    st = cpcg.state
    st["state"]["state"] = pcg.state["state"]["state"]
    st["state"]["inc"] = pcg.state["state"]["inc"]
    cpcg.state = st
    expected = pcg.random_raw(1000)
    result = cpcg.random_raw(1000)
    np.testing.assert_equal(expected, result)


def test_output_functions_distinct():
    fns = ("xsl-rr", "dxsm", "murmur3", "upper", "lower")
    results = {}
    for of in fns:
        cpcg = LCG128Mix(SEED, output=of)
        results[of] = cpcg.random_raw()
    from itertools import combinations

    for comb in combinations(results.keys(), 2):
        assert np.all(results[comb[0]] != results[comb[1]])


def test_multipliers_distinct():
    mult = DEFAULT_DXSM_MULTIPLIER
    a = LCG128Mix(SEED, multiplier=mult).random_raw(10)
    mult = DEFAULT_MULTIPLIER
    b = LCG128Mix(SEED, multiplier=mult).random_raw(10)
    assert np.all(a != b)


def test_dxsm_multipliers_distinct():
    mult = DEFAULT_DXSM_MULTIPLIER
    a = LCG128Mix(SEED, dxsm_multiplier=mult, output="dxsm").random_raw(10)
    mult = DEFAULT_DXSM_MULTIPLIER + 2
    b = LCG128Mix(SEED, dxsm_multiplier=mult, output="dxsm").random_raw(10)
    assert np.all(a != b)


def test_pre_post():
    a = LCG128Mix(SEED).random_raw(10)
    b = LCG128Mix(SEED, post=False).random_raw(10)
    np.testing.assert_equal(a[:9], b[1:])


@pytest.mark.skipif(MISSING_NUMBA, reason="numba not available")
def test_ouput_ctypes():
    @cfunc(types.uint64(types.uint64, types.uint64))
    def upper(high, low):
        return high

    a = LCG128Mix(SEED, output=upper.ctypes).random_raw(10)
    b = LCG128Mix(SEED, output="upper").random_raw(10)
    np.testing.assert_equal(a, b)

    @cfunc(types.uint64(types.uint64, types.uint64))
    def lower(high, low):
        return low

    a = LCG128Mix(SEED, output=lower.ctypes).random_raw(10)
    b = LCG128Mix(SEED, output="lower").random_raw(10)
    np.testing.assert_equal(a, b)


def test_ctypes():
    import ctypes
    import os
    import subprocess

    base = os.path.split(os.path.abspath(__file__))[0]

    c_loc = os.path.join(base, "ctypes_testing.c")
    with open(c_loc, "w", encoding="utf-8") as c_file:
        c_file.write(C_SOURCE)
    o_loc = os.path.join(base, "ctypes_testing.o")
    so_loc = os.path.join(base, "libctypes_testing.so")
    try:
        cmd = ["gcc", "-c", "-Wall", "-Werror", "-fpic", c_loc, "-o", o_loc]
        subprocess.call(cmd)
        cmd = ["gcc", "-shared", "-o", so_loc, o_loc]
        subprocess.call(cmd)
        if not os.path.exists(so_loc):
            raise FileNotFoundError(f"{so_loc} does not exist")
    except Exception as exc:
        pytest.skip(
            "GCC unavailable or other error compiling the test library" + str(exc)
        )
    libtesting = ctypes.CDLL(so_loc)
    libtesting.output_upper.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    libtesting.output_upper.restype = ctypes.c_uint64
    a = LCG128Mix(SEED, output="upper").random_raw(10)
    rg = LCG128Mix(SEED, output=libtesting.output_upper)
    b = rg.random_raw(10)
    np.testing.assert_equal(a, b)

    c = LCG128Mix()
    c.state = rg.state
    assert c.random_raw() == rg.random_raw()

    libtesting.output_upper.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    with pytest.raises(ValueError):
        LCG128Mix(SEED, output=libtesting.output_upper)
    libtesting.output_upper.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    libtesting.output_upper.restype = ctypes.c_uint32
    with pytest.raises(ValueError):
        LCG128Mix(SEED, output=libtesting.output_upper)


def test_pcg_warnings_and_errors():
    with pytest.raises(ValueError, match="variant unknown is not known"):
        PCG64(0, variant="unknown")


def test_repr():
    cpcg = LCG128Mix(
        0,
        0,
        output="dxsm",
        dxsm_multiplier=DEFAULT_DXSM_MULTIPLIER + 2,
        multiplier=DEFAULT_MULTIPLIER + 2,
        post=False,
    )
    cpcg_repr = repr(cpcg)
    assert "Output Function: dxsm" in cpcg_repr
    assert f"Multiplier: {DEFAULT_MULTIPLIER + 2}" in cpcg_repr
    assert f"DXSM Multiplier: {DEFAULT_DXSM_MULTIPLIER + 2}" in cpcg_repr
    assert "Post: False" in cpcg_repr


def test_bad_state():
    a = LCG128Mix()
    st = a.state
    with pytest.raises(TypeError):
        a.state = ((k, v) for k, v in st.items())
    with pytest.raises(ValueError):
        st["bit_generator"] = "AnotherBG"
        a.state = st


def test_exceptions():
    with pytest.raises(ValueError):
        LCG128Mix(multiplier=DEFAULT_MULTIPLIER + 1)
    with pytest.raises(ValueError):
        LCG128Mix(dxsm_multiplier=DEFAULT_MULTIPLIER + 1)
    with pytest.raises(ValueError):
        LCG128Mix(SEED, output="not-implemented")
    with pytest.raises(TypeError):
        LCG128Mix(SEED, post=3)


@pytest.mark.parametrize("seed", [0, sum([2**i for i in range(1, 128, 2)])])
@pytest.mark.parametrize("inc", [0, sum([2**i for i in range(0, 128, 3)])])
def test_equivalence_pcg64dxsm(seed, inc):
    a = PCG64(seed, inc, variant="dxsm")
    b = PCG64DXSM(seed, inc)
    assert np.all((a.random_raw(10000) - b.random_raw(10000)) == 0)
    assert np.all((a.random_raw(13) - b.random_raw(13)) == 0)
    a_st = a.state
    b_st = b.state
    assert a_st["state"] == b_st["state"]
    a = a.advance(345671)
    b = b.advance(345671)
    a_st = a.state
    b_st = b.state
    assert a_st["state"] == b_st["state"]
