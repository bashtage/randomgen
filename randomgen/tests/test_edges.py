"""
This file contains test for edge cases missed by other test.  It should be
integrated into the other test modules eventually.
"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from randomgen import RandomGenerator, Xoshiro256StarStar, MT19937

SEED = 1234567890


@pytest.fixture(scope='function')
def random_gen():
    return RandomGenerator(Xoshiro256StarStar(SEED))


@pytest.fixture(scope='function')
def mt19937():
    return RandomGenerator(MT19937(SEED))


def test_noninstantized_brng():
    with pytest.raises(ValueError):
        RandomGenerator(Xoshiro256StarStar)


def test_standard_expoential_type_error(random_gen):
    with pytest.raises(TypeError):
        random_gen.standard_exponential(dtype=np.int32)


def test_standard_normal_type_error(random_gen):
    with pytest.raises(TypeError):
        random_gen.standard_normal(dtype=np.int32)


def test_rand_singleton(mt19937):
    actual = mt19937.rand()
    desired = 0.61879477158567997
    assert_array_almost_equal(actual, desired, decimal=15)


def test_multivariate_normal(random_gen):
    mean = (.123456789, 10)
    cov = [[1, 0], [0, 1]]
    size = (3, 2)
    actual = random_gen.multivariate_normal(mean, cov, size)
    assert actual.shape == (3, 2, 2)

    mu = np.zeros(2)
    cov = np.eye(2)
    with pytest.raises(ValueError):
        random_gen.multivariate_normal(mean, cov, check_valid='other')
    with pytest.raises(ValueError):
        random_gen.multivariate_normal(np.zeros((2, 1, 1)), cov)
    with pytest.raises(ValueError):
        random_gen.multivariate_normal(mu, np.empty((3, 2)))
    with pytest.raises(ValueError):
        random_gen.multivariate_normal(mu, np.eye(3))


def test_dirichlet_no_size(random_gen):
    alpha = np.array([51.72840233779265162, 39.74494232180943953])
    actual = random_gen.dirichlet(alpha)
    assert actual.shape == (2,)


def test_standard_gammma_scalar_float(random_gen):
    actual = random_gen.standard_gamma(10.0, dtype=np.float32)
    assert np.isscalar(actual)


def test_standard_gammma_float_out(random_gen):
    out = np.zeros(10, dtype=np.float32)
    random_gen.standard_gamma(10.0, out=out, dtype=np.float32)
    assert np.all(out != 0)


def test_random_sample_float_scalar(random_gen):
    actual = random_gen.random_sample(dtype=np.float32)
    assert np.isscalar(actual)

# TODO: Complex normal
