import numpy as np
import pytest

from randomgen import SFC64, SeedSequence

try:
    from numpy.random import Generator
except ImportError:
    from randomgen import Generator


def test_known():
    sfc = SFC64(SeedSequence(0))
    weyl = sfc.weyl_increments(2)
    expected = np.array([5884110806310384263, 5503527272481214085], dtype=np.uint64)
    np.testing.assert_equal(weyl, expected)
    weyl = sfc.weyl_increments(2, 48, 16)
    expected = np.array([315711580115618193, 18433054088229398527], dtype=np.uint64)
    np.testing.assert_equal(weyl, expected)


def test_validity():
    sfc = SFC64(SeedSequence(0))
    weyl = sfc.weyl_increments(10000)
    weyl = weyl.reshape((-1, 1))
    bits = np.unpackbits(weyl.view("u1"), axis=1)
    assert (bits.sum(1) == 32).all()

    weyl = sfc.weyl_increments(10000, 40, 24)
    weyl = weyl.reshape((-1, 1))
    bits = np.unpackbits(weyl.view("u1"), axis=1)
    assert (bits.sum(1) <= 40).all()
    assert (bits.sum(1) >= 24).all()


def test_smoke():
    sfc = SFC64(SeedSequence(0))
    for max_bit in range(1, 64):
        for min_bit in range(1, max_bit + 1):
            weyl = sfc.weyl_increments(1, max_bit, min_bit)
            assert weyl % 2 == 1


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
        sfc.weyl_increments(40, 2)


def test_basic_weyl():
    sfc = SFC64(SeedSequence(0))
    state = sfc.state
    inc = sfc.weyl_increments(1)[0]

    sfc.state = state
    gen = Generator(sfc)
    assert gen.integers(32, 32, endpoint=True, dtype=np.int8) == 32
    bits = set()
    candidates = sfc.random_raw(16).astype("<u8").view("u1") & np.uint8(63)
    loc = 32
    while len(bits) < 32:
        bits.update(candidates[:loc])
        loc += 1

    for bit in bits:
        assert bool(inc & (np.uint64(0x1) << np.uint64(bit)))
