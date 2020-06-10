from distutils.version import LooseVersion
import warnings

import numpy as np
from numpy.testing import assert_equal
import pytest

from randomgen import Generator

try:
    from numpy.random import MT19937
except ImportError:
    from randomgen import MT19937


v119 = LooseVersion("1.19")
NP_LT_119 = LooseVersion(np.__version__) < v119


pytestmark = pytest.mark.skipif(NP_LT_119, reason="Only test NumPy 1.19+")


# Catch when using internal MT19937
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    random = Generator(MT19937(1234))


@pytest.mark.parametrize(
    "bound, expected",
    [
        (
            2 ** 32 - 1,
            np.array(
                [
                    517043486,
                    1364798665,
                    1733884389,
                    1353720612,
                    3769704066,
                    1170797179,
                    4108474671,
                ]
            ),
        ),
        (
            2 ** 32,
            np.array(
                [
                    517043487,
                    1364798666,
                    1733884390,
                    1353720613,
                    3769704067,
                    1170797180,
                    4108474672,
                ]
            ),
        ),
        (
            2 ** 32 + 1,
            np.array(
                [
                    517043487,
                    1733884390,
                    3769704068,
                    4108474673,
                    1831631863,
                    1215661561,
                    3869512430,
                ]
            ),
        ),
    ],
)
def test_repeatability_32bit_boundary(bound, expected):
    for size in [None, len(expected)]:
        random = Generator(MT19937(1234))
        x = random.integers(bound, size=size, use_masked=False)
        assert_equal(x, expected if size is not None else expected[0])


def test_dirichelet_alpha():
    # numpy/numpy#15951
    with pytest.raises(ValueError):
        random.dirichlet([[5, 1]])
    with pytest.raises(ValueError):
        random.dirichlet([[5], [1]])
    with pytest.raises(ValueError):
        random.dirichlet([[[5], [1]], [[1], [5]]])
    with pytest.raises(ValueError):
        random.dirichlet(np.array([[5, 1], [1, 5]]))


def test_negative_binomial_p0_exception():
    # numpy/numpy#15913
    # Verify that p=0 raises an exception.
    with pytest.raises(ValueError):
        random.negative_binomial(1, 0)


def test_multivariate_normal_basic_stats():
    # numpy/numpy#15871
    random = Generator(MT19937(12345))
    n_s = 1000
    mean = np.array([1, 2])
    cov = np.array([[2, 1], [1, 2]])
    s = random.multivariate_normal(mean, cov, size=(n_s,))
    s_center = s - mean
    cov_emp = (s_center.T @ s_center) / (n_s - 1)
    # these are pretty loose and are only designed to detect major errors
    assert np.all(np.abs(s_center.mean(-2)) < 0.1)
    assert np.all(np.abs(cov_emp - cov) < 0.2)


# chi2max is the maximum acceptable chi-squared value.
@pytest.mark.parametrize(
    "sample_size,high,dtype,chi2max",
    [
        (5000000, 5, np.int8, 125.0),  # p-value ~4.6e-25
        (5000000, 7, np.uint8, 150.0),  # p-value ~7.7e-30
        (10000000, 2500, np.int16, 3300.0),  # p-value ~3.0e-25
        (50000000, 5000, np.uint16, 6500.0),  # p-value ~3.5e-25
    ],
)
def test_integers_small_dtype_chisquared(sample_size, high, dtype, chi2max):
    # Regression test for gh-14774.
    samples = random.integers(high, size=sample_size, dtype=dtype)

    values, counts = np.unique(samples, return_counts=True)
    expected = sample_size / high
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < chi2max


def test_bad_permuation():
    bad_x_str = "abcd"
    with pytest.raises(IndexError):
        random.permutation(bad_x_str)

    bad_x_float = 1.2
    with pytest.raises(IndexError):
        random.permutation(bad_x_float)
