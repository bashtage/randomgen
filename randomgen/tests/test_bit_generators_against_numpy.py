import numpy as np
from packaging.version import parse
import pytest

import randomgen
from randomgen import PCG32, PCG64, PCG64DXSM, SFC64, LCG128Mix, Philox, SeedSequence

NP_LT_119 = parse(np.__version__) < parse("1.19.0")

pytestmark = pytest.mark.skipif(NP_LT_119, reason="Only test Numpy 1.19+")


@pytest.mark.parametrize("bg", ["SFC64", "MT19937", "PCG64", "Philox"])
def test_against_numpy(bg):
    bitgen = getattr(randomgen, bg)
    np_bitgen = getattr(np.random, bg)
    ss = np.random.SeedSequence(1203940)
    np_ss = np.random.SeedSequence(1203940)
    kwargs = {"variant": "xsl-rr"} if bg == "PCG64" else {}
    ref = bitgen(ss, numpy_seed=True, **kwargs)
    exp = np_bitgen(np_ss)
    np.testing.assert_equal(ref.random_raw(1000), exp.random_raw(1000))


def test_pcg_numpy_mode_exception():
    with pytest.raises(ValueError, match="inc must be none"):
        PCG64(SeedSequence(0), numpy_seed=True, inc=3)


@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("w", [1, 3])
def test_sfc_numpy_mode_exception(k, w):
    if k == w == 1:
        return
    with pytest.raises(ValueError, match="w and k"):
        SFC64(SeedSequence(0), numpy_seed=True, w=w, k=k)


@pytest.mark.parametrize("number", [2, 4])
@pytest.mark.parametrize("width", [32, 64])
def test_philox_numpy_mode_exception(number, width):
    if number == 4 and width == 64:
        return
    with pytest.raises(ValueError, match="n must be"):
        Philox(SeedSequence(0), numpy_seed=True, number=number, width=width)


@pytest.mark.parametrize("counter", [0, 1, 2**64 + 5, 2**255])
def test_philox_numpy_counter(counter):
    ss = np.random.SeedSequence(1203940)
    np_ss = np.random.SeedSequence(1203940)
    ref = Philox(ss, counter=counter, numpy_seed=True)
    exp = np.random.Philox(np_ss, counter=counter)
    np.testing.assert_equal(ref.random_raw(1000), exp.random_raw(1000))


JUMPS = [1, 2, 3, 2**20, 2**127 - 1]
GOLDEN_128 = 0x9E3779B97F4A7C15F39CC0605CEDC835
GOLDEN_64 = 0x9E3779B97F4A7C15


@pytest.mark.parametrize("jumps", JUMPS)
def test_pcg64_jumped_against_numpy(jumps):
    ref = PCG64(SeedSequence(1203940), numpy_seed=True).jumped(jumps)
    exp = np.random.PCG64(np.random.SeedSequence(1203940)).jumped(jumps)
    assert ref.state["state"] == exp.state["state"]
    np.testing.assert_equal(ref.random_raw(100), exp.random_raw(100))


@pytest.mark.parametrize("jumps", JUMPS)
def test_pcg64dxsm_jumped_against_numpy(jumps):
    exp = np.random.PCG64DXSM(np.random.SeedSequence(1203940))
    ref = PCG64DXSM(0)
    state = ref.state
    state["state"] = exp.state["state"]
    ref.state = state
    ref = ref.jumped(jumps)
    exp = exp.jumped(jumps)
    assert ref.state["state"] == exp.state["state"]
    np.testing.assert_equal(ref.random_raw(100), exp.random_raw(100))


@pytest.mark.parametrize(
    ("bit_gen", "step", "bits"),
    [
        (PCG64, GOLDEN_128, 128),
        (PCG64DXSM, GOLDEN_128, 128),
        (LCG128Mix, GOLDEN_128, 128),
        (PCG32, GOLDEN_64, 64),
    ],
)
def test_jumped_step_is_odd(bit_gen, step, bits):
    # The step is the odd integer closest to 2**bits / phi
    bg = bit_gen(1203940)
    for jumps in (1, 2, 3, 2**20):
        expected = bit_gen(1203940).advance(jumps * step)
        assert bg.jumped(jumps).state == expected.state
    # An odd step only returns to the starting state after 2**bits jumps
    start = bg.state["state"]
    for k in range(bits):
        assert bg.jumped(2**k).state["state"] != start
    assert bg.jumped(2**bits).state["state"] == start
