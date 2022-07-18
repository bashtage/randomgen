import numpy as np
import pytest

from randomgen import SFC64, SeedSequence


def test_known():
    sfc = SFC64(SeedSequence(0))
    weyl = sfc.weyl_increments(2)
    expected = np.array([6524879303493105881, 17467897594175157085], dtype=np.uint64)
    np.testing.assert_equal(weyl, expected)
    weyl = sfc.weyl_increments(2, 48, 16)
    expected = np.array([18331436834911646537, 1349527966119344023], dtype=np.uint64)
    np.testing.assert_equal(weyl, expected)


def test_validity():
    sfc = SFC64(SeedSequence())
    weyl = sfc.weyl_increments(10000)
    weyl = weyl.reshape((-1, 1))
    bits = np.unpackbits(weyl.view("u1"), axis=1)
    assert (bits.sum(1) == 32).all()
    assert (weyl % np.uint64(2) == 1).all()

    weyl = sfc.weyl_increments(10000, 40, 24)
    weyl = weyl.reshape((-1, 1))
    bits = np.unpackbits(weyl.view("u1"), axis=1)
    assert (bits.sum(1) <= 40).all()
    assert (bits.sum(1) >= 24).all()
    assert (weyl % np.uint64(2) == 1).all()


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
