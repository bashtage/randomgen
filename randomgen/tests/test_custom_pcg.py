import numpy as np
import pytest

from randomgen import PCG64, CustomPCG64, SeedSequence
from randomgen.pcg64 import DEFAULT_DXSM_MULTIPLIER, DEFAULT_MULTIPLIER

try:
    from numba import types, cfunc

    MISSING_NUMBA = False
except ImportError:
    MISSING_NUMBA = True

SEED = 12345678909876543321


def test_pcg64_1():
    pcg = PCG64(SeedSequence(12345678909876543321))
    cpcg = CustomPCG64()
    st = cpcg.state
    st["state"]["state"] = pcg.state["state"]["state"]
    st["state"]["inc"] = pcg.state["state"]["inc"]
    cpcg.state = st
    expected = pcg.random_raw(1000)
    result = cpcg.random_raw(1000)
    np.testing.assert_equal(expected, result)


def test_pcg64_dxsm():
    pcg = PCG64(SeedSequence(12345678909876543321), variant="dxsm")
    cpcg = CustomPCG64(output="dxsm")
    st = cpcg.state
    st["state"]["state"] = pcg.state["state"]["state"]
    st["state"]["inc"] = pcg.state["state"]["inc"]
    cpcg.state = st
    expected = pcg.random_raw(1000)
    result = cpcg.random_raw(1000)
    np.testing.assert_equal(expected, result)


def test_pcg64_cm_dxsm():
    pcg = PCG64(SeedSequence(12345678909876543321), variant="cm-dxsm")
    cpcg = CustomPCG64(output="dxsm", post=False, multiplier=DEFAULT_DXSM_MULTIPLIER)
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
        cpcg = CustomPCG64(SEED, output=of)
        results[of] = cpcg.random_raw()
    from itertools import combinations

    for comb in combinations(results.keys(), 2):
        assert np.all(results[comb[0]] != results[comb[1]])


def test_multipliers_distinct():
    mult = DEFAULT_DXSM_MULTIPLIER
    a = CustomPCG64(SEED, multiplier=mult).random_raw(10)
    mult = DEFAULT_MULTIPLIER
    b = CustomPCG64(SEED, multiplier=mult).random_raw(10)
    assert np.all(a != b)


def test_dxsm_multipliers_distinct():
    mult = DEFAULT_DXSM_MULTIPLIER
    a = CustomPCG64(SEED, dxsm_multiplier=mult, output="dxsm").random_raw(10)
    mult = DEFAULT_DXSM_MULTIPLIER + 2
    b = CustomPCG64(SEED, dxsm_multiplier=mult, output="dxsm").random_raw(10)
    assert np.all(a != b)


def test_pre_post():
    a = CustomPCG64(SEED).random_raw(10)
    b = CustomPCG64(SEED, post=False).random_raw(10)
    np.testing.assert_equal(a[:9], b[1:])


@pytest.mark.skipif(MISSING_NUMBA, reason="numba not available")
def test_ouput_ctypes():
    @cfunc(types.uint64(types.uint64, types.uint64))
    def upper(high, low):
        return high

    a = CustomPCG64(SEED, output=upper.ctypes).random_raw(10)
    b = CustomPCG64(SEED, output="upper").random_raw(10)
    np.testing.assert_equal(a, b)

    @cfunc(types.uint64(types.uint64, types.uint64))
    def lower(high, low):
        return low

    a = CustomPCG64(SEED, output=lower.ctypes).random_raw(10)
    b = CustomPCG64(SEED, output="lower").random_raw(10)
    np.testing.assert_equal(a, b)


def test_ctypes():
    import ctypes
    import subprocess
    import os

    base = os.path.split(os.path.abspath(__file__))[0]

    c_loc = os.path.join(base, "data", "ctypes_testing.c")
    o_loc = os.path.join(base, "data", "ctypes_testing.o")
    so_loc = os.path.join(base, "data", "libctypes_testing.so")
    try:
        cmd = ["gcc", "-c", "-Wall", "-Werror", "-fpic", c_loc, "-o", o_loc]
        print(" ".join(cmd))
        subprocess.call(cmd)
        cmd = ["gcc", "-shared", "-o", so_loc, o_loc]
        print(" ".join(cmd))
        subprocess.call(cmd)
    except Exception as exc:
        pytest.skip(
            "GCC unavailable or other error compiling the test library" + str(exc)
        )
    libtesting = ctypes.CDLL(so_loc)
    libtesting.output_upper.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    libtesting.output_upper.restype = ctypes.c_uint64
    a = CustomPCG64(SEED, output="upper").random_raw(10)
    rg = CustomPCG64(SEED, output=libtesting.output_upper)
    b = rg.random_raw(10)
    np.testing.assert_equal(a, b)

    c = CustomPCG64()
    c.state = rg.state
    assert c.random_raw() == rg.random_raw()

    libtesting.output_upper.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    with pytest.raises(ValueError):
        CustomPCG64(SEED, output=libtesting.output_upper)
    libtesting.output_upper.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    libtesting.output_upper.restype = ctypes.c_uint32
    with pytest.raises(ValueError):
        CustomPCG64(SEED, output=libtesting.output_upper)


def test_pcg_warnings_and_errors():
    with pytest.warns(FutureWarning, match="The current default"):
        PCG64(0, mode="sequence", variant=None)
    with pytest.raises(ValueError, match="variant unknown is not known"):
        PCG64(0, mode="sequence", variant="unknown")


def test_repr():
    cpcg = CustomPCG64(
        0,
        0,
        output="dxsm",
        dxsm_multiplier=DEFAULT_DXSM_MULTIPLIER + 2,
        multiplier=DEFAULT_MULTIPLIER + 2,
        post=False,
    )
    cpcg_repr = repr(cpcg)
    assert "Output Function: dxsm" in cpcg_repr
    assert f"Multiplier: {DEFAULT_MULTIPLIER+2}" in cpcg_repr
    assert f"DXSM Multiplier: {DEFAULT_DXSM_MULTIPLIER+2}" in cpcg_repr
    assert "Post: False" in cpcg_repr


def test_bad_state():
    a = CustomPCG64()
    st = a.state
    with pytest.raises(TypeError):
        a.state = ((k, v) for k, v in st.items())
    with pytest.raises(ValueError):
        st["bit_generator"] = "AnotherBG"
        a.state = st


def test_exceptions():
    with pytest.raises(ValueError):
        CustomPCG64(multiplier=DEFAULT_MULTIPLIER + 1)
    with pytest.raises(ValueError):
        CustomPCG64(dxsm_multiplier=DEFAULT_MULTIPLIER + 1)
    with pytest.raises(ValueError):
        CustomPCG64(SEED, output="not-implemented")
    with pytest.raises(TypeError):
        CustomPCG64(SEED, post=3)
