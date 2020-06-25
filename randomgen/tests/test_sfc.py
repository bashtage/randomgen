import numpy as np
import pytest

from randomgen import SFC64, SeedSequence

try:
    from numpy.random import Generator
except ImportError:
    from randomgen import Generator


def test_invalid_weyl():
    sfc = SFC64(SeedSequence(0))
    with pytest.raises(ValueError):
        sfc.weyl_increments(1000, 1, 1)
    with pytest.raises(ValueError):
        sfc.weyl_increments(1, 32, 33)
    with pytest.raises(ValueError):
        sfc.weyl_increments(1, 32, -1)
    with pytest.raises(ValueError):
        sfc.weyl_increments(1, 128)
    with pytest.raises(ValueError):
        sfc.weyl_increments(0)
    with pytest.warns(RuntimeWarning):
        sfc.weyl_increments(40, 1, 1)


def test_basic_weyl():
    sfc = SFC64(SeedSequence(0))
    state = sfc.state
    inc = sfc.weyl_increments(1)[0]

    sfc.state = state
    gen = Generator(sfc)
    assert gen.integers(32, 32, endpoint=True, dtype=np.int8) == 32
    bits = set()
    canidates = gen.integers(0, 64, dtype=np.int8, size=64)
    loc = 32
    while len(bits) < 32:
        bits.update(canidates[:loc])
        loc += 1

    for bit in bits:
        assert bool(inc & (np.uint64(0x1) << np.uint64(bit)))


def test_inverse_weyl():
    sfc = SFC64(SeedSequence(0))
    inc = sfc.weyl_increments(1, 16)[0]

    sfc = SFC64(SeedSequence(0))
    inc_inv = sfc.weyl_increments(1, 48)[0]
    assert (inc ^ inc_inv) == 2 ** 64 - 1
