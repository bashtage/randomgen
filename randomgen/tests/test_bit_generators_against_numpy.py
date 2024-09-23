import numpy as np
from packaging.version import parse
import pytest

import randomgen
from randomgen import PCG64, SFC64, Philox, SeedSequence

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
    with pytest.raises(ValueError):
        PCG64(SeedSequence(0), numpy_seed=True, inc=3)


@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("w", [1, 3])
def test_sfc_numpy_mode_exception(k, w):
    if k == w == 1:
        return
    with pytest.raises(ValueError):
        SFC64(SeedSequence(0), numpy_seed=True, w=w, k=k)


@pytest.mark.parametrize("number", [2, 4])
@pytest.mark.parametrize("width", [32, 64])
def test_philox_numpy_mode_exception(number, width):
    if number == 4 and width == 64:
        return
    with pytest.raises(ValueError):
        Philox(SeedSequence(0), numpy_seed=True, number=number, width=width)
