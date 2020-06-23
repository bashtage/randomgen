import copy
from distutils.version import LooseVersion
import pickle

import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
    assert_array_almost_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_warns,
)
import pytest

from randomgen import MT19937, PCG64, ExtendedGenerator
from randomgen._testing import suppress_warnings

SEED = 1234567890
MV_SEED = 123456789


@pytest.fixture(scope="module")
def seed():
    return SEED


@pytest.fixture(scope="module")
def mv_seed():
    return MV_SEED


_mt19937 = MT19937(SEED, mode="legacy")
random = ExtendedGenerator(_mt19937)

NP_LT_118 = LooseVersion(np.__version__) < LooseVersion("1.18.0")


@pytest.mark.skipif(NP_LT_118, reason="Can only test with NumPy >= 1.18")
@pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
def test_multivariate_normal_method(seed, method):
    from numpy.random import MT19937 as NPMT19937

    random = ExtendedGenerator(NPMT19937(seed))
    mean = (0.123456789, 10)
    cov = [[1, 0], [0, 1]]
    size = (3, 2)
    actual = random.multivariate_normal(mean, cov, size, method=method)
    desired = np.array(
        [
            [
                [-1.747478062846581, 11.25613495182354],
                [-0.9967333370066214, 10.342002097029821],
            ],
            [
                [0.7850019631242964, 11.181113712443013],
                [0.8901349653255224, 8.873825399642492],
            ],
            [
                [0.7130260107430003, 9.551628690083056],
                [0.7127098726541128, 11.991709234143173],
            ],
        ]
    )

    assert_array_almost_equal(actual, desired, decimal=15)

    # Check for default size, was raising deprecation warning
    actual = random.multivariate_normal(mean, cov, method=method)
    desired = np.array([0.233278563284287, 9.424140804347195])
    assert_array_almost_equal(actual, desired, decimal=15)

    # Check path with scalar size works correctly
    scalar = random.multivariate_normal(mean, cov, 3, method=method)
    tuple1d = random.multivariate_normal(mean, cov, (3,), method=method)
    assert scalar.shape == tuple1d.shape == (3, 2)

    # Check that non symmetric covariance input raises exception when
    # check_valid='raises' if using default svd method.
    mean = [0, 0]
    cov = [[1, 2], [1, 2]]
    assert_raises(
        ValueError, random.multivariate_normal, mean, cov, check_valid="raise"
    )

    # Check that non positive-semidefinite covariance warns with
    # RuntimeWarning
    cov = [[1, 2], [2, 1]]
    assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
    assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov, method="eigh")
    assert_raises(LinAlgError, random.multivariate_normal, mean, cov, method="cholesky")

    # and that it doesn't warn with RuntimeWarning check_valid='ignore'
    assert_no_warnings(random.multivariate_normal, mean, cov, check_valid="ignore")

    # and that it raises with RuntimeWarning check_valid='raises'
    assert_raises(
        ValueError, random.multivariate_normal, mean, cov, check_valid="raise"
    )
    assert_raises(
        ValueError,
        random.multivariate_normal,
        mean,
        cov,
        check_valid="raise",
        method="eigh",
    )

    # check degenerate samples from singular covariance matrix
    cov = [[1, 1], [1, 1]]
    if method in ("svd", "eigh"):
        samples = random.multivariate_normal(mean, cov, size=(3, 2), method=method)
        assert_array_almost_equal(samples[..., 0], samples[..., 1], decimal=6)
    else:
        assert_raises(
            LinAlgError, random.multivariate_normal, mean, cov, method="cholesky"
        )

    cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
    with suppress_warnings() as sup:
        random.multivariate_normal(mean, cov, method=method)
        w = sup.record(RuntimeWarning)
        assert len(w) == 0

    mu = np.zeros(2)
    cov = np.eye(2)
    assert random.multivariate_normal(mu, cov, size=3).shape == (3, 2)
    assert_raises(
        ValueError, random.multivariate_normal, mean, cov, check_valid="other"
    )
    assert_raises(ValueError, random.multivariate_normal, np.zeros((2, 1, 1)), cov)
    assert_raises(ValueError, random.multivariate_normal, mu, np.empty((3, 2)))
    assert_raises(ValueError, random.multivariate_normal, mu, np.eye(3))


@pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
def test_multivariate_normal_basic_stats(seed, method):
    random = ExtendedGenerator(MT19937(seed, mode="sequence"))
    n_s = 1000
    mean = np.array([1, 2])
    cov = np.array([[2, 1], [1, 2]])
    s = random.multivariate_normal(mean, cov, size=(n_s,), method=method)
    s_center = s - mean
    cov_emp = (s_center.T @ s_center) / (n_s - 1)
    # these are pretty loose and are only designed to detect major errors
    assert np.all(np.abs(s_center.mean(-2)) < 0.1)
    assert np.all(np.abs(cov_emp - cov) < 0.2)


@pytest.mark.parametrize("size", [(4, 3, 2), (5, 4, 3, 2)])
@pytest.mark.parametrize("mean", [np.zeros(2), np.zeros((3, 3))])
def test_multivariate_normal_bad_size(mean, size):
    cov = np.eye(4)
    with pytest.raises(ValueError):
        random.multivariate_normal(mean, cov)
    mean = np.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        random.multivariate_normal(mean, cov, size=size)

    with pytest.raises(ValueError):
        random.multivariate_normal(0, [[1]], size=size)
    with pytest.raises(ValueError):
        random.multivariate_normal([0], [1], size=size)


def test_multivariate_normal(seed):
    random.bit_generator.seed(seed)
    mean = (0.123456789, 10)
    cov = [[1, 0], [0, 1]]
    size = (3, 2)
    actual = random.multivariate_normal(mean, cov, size)
    desired = np.array(
        [
            [
                [-3.34929721161096100, 9.891061435770858],
                [-0.12250896439641100, 9.295898449738300],
            ],
            [
                [0.48355927611635563, 10.127832101772366],
                [3.11093021424924300, 10.283109168794352],
            ],
            [
                [-0.20332082341774727, 9.868532121697195],
                [-1.33806889550667330, 9.813657233804179],
            ],
        ]
    )

    assert_array_almost_equal(actual, desired, decimal=15)

    # Check for default size, was raising deprecation warning
    actual = random.multivariate_normal(mean, cov)
    desired = np.array([-1.097443117192574, 10.535787051184261])
    assert_array_almost_equal(actual, desired, decimal=15)

    # Check that non positive-semidefinite covariance warns with
    # RuntimeWarning
    mean = [0, 0]
    cov = [[1, 2], [2, 1]]
    assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)

    # and that it doesn"t warn with RuntimeWarning check_valid="ignore"
    assert_no_warnings(random.multivariate_normal, mean, cov, check_valid="ignore")

    # and that it raises with RuntimeWarning check_valid="raises"
    assert_raises(
        ValueError, random.multivariate_normal, mean, cov, check_valid="raise"
    )

    cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
    with suppress_warnings() as sup:
        random.multivariate_normal(mean, cov)
        w = sup.record(RuntimeWarning)
        assert len(w) == 0

    mu = np.zeros(2)
    cov = np.eye(2)
    assert_raises(
        ValueError, random.multivariate_normal, mean, cov, check_valid="other"
    )
    assert_raises(ValueError, random.multivariate_normal, np.zeros((2, 1, 1)), cov)
    assert_raises(ValueError, random.multivariate_normal, mu, np.empty((3, 2)))
    assert_raises(ValueError, random.multivariate_normal, mu, np.eye(3))


def test_complex_normal(seed):
    random.bit_generator.seed(seed)
    actual = random.complex_normal(loc=1.0, gamma=1.0, relation=0.5, size=(3, 2))
    desired = np.array(
        [
            [
                -2.007493185623132 - 0.05446928211457126j,
                0.7869874090977291 - 0.35205077513085050j,
            ],
            [
                1.3118579018087224 + 0.06391605088618339j,
                3.5872278793967554 + 0.14155458439717636j,
            ],
            [
                0.7170022862582056 - 0.06573393915140235j,
                -0.26571837106621987 - 0.0931713830979103j,
            ],
        ]
    )
    assert_array_almost_equal(actual, desired, decimal=15)

    random.bit_generator.seed(seed)
    actual = random.complex_normal(loc=0, gamma=1.0, relation=0.5, size=3)
    assert_array_almost_equal(actual, desired.flat[:3] - 1.0, decimal=15)

    random.bit_generator.seed(seed)
    actual = random.complex_normal(loc=2.0, gamma=1.0, relation=0.5)
    assert_array_almost_equal(actual, 1.0 + desired[0, 0], decimal=15)


def test_complex_normal_invalid():
    assert_raises(ValueError, random.complex_normal, gamma=1 + 0.5j)
    assert_raises(ValueError, random.complex_normal, relation=2)
    assert_raises(ValueError, random.complex_normal, relation=-3)
    assert_raises(ValueError, random.complex_normal, relation=10j)

    assert_raises(ValueError, random.complex_normal, gamma=[1 + 0.5j])
    assert_raises(ValueError, random.complex_normal, relation=[2])
    assert_raises(ValueError, random.complex_normal, relation=[-3])
    assert_raises(ValueError, random.complex_normal, relation=[10j])


def test_random_uintegers():
    assert len(random.uintegers(10)) == 10
    assert len(random.uintegers(10, bits=32)) == 10
    assert isinstance(random.uintegers(), int)
    assert isinstance(random.uintegers(bits=32), int)
    with pytest.raises(ValueError):
        with pytest.deprecated_call():
            random.uintegers(bits=128)


def test_str_repr():
    assert "ExtendedGenerator" in str(random)
    assert "ExtendedGenerator" in repr(random)
    assert "MT19937" in str(random)


def test_pickle_and_copy(seed):
    gen = ExtendedGenerator(MT19937(seed, mode="legacy"))
    reloaded = pickle.loads(pickle.dumps(gen))
    assert isinstance(reloaded, ExtendedGenerator)
    copied = copy.deepcopy(gen)
    gen_rv = gen.uintegers(10, bits=64)
    reloaded_rv = reloaded.uintegers(10, bits=64)
    copied_rv = copied.uintegers(10, bits=64)
    assert_equal(gen_rv, reloaded_rv)
    assert_equal(gen_rv, copied_rv)


def test_set_get_state(seed):
    state = _mt19937.state
    gen = ExtendedGenerator(MT19937(seed, mode="legacy"))
    gen.state = state
    assert_equal(gen.state["state"]["key"], state["state"]["key"])
    assert_equal(gen.state["state"]["pos"], state["state"]["pos"])


def test_complex_normal_size(mv_seed):
    random = ExtendedGenerator(MT19937(mv_seed, mode="legacy"))
    state = random.state
    loc = np.ones((1, 2))
    gamma = np.ones((3, 1))
    relation = 0.5 * np.ones((3, 2))
    actual = random.complex_normal(loc=loc, gamma=gamma, relation=relation)
    desired = np.array(
        [
            [
                1.393937478212015 - 0.31374589731830593j,
                0.9474905694736895 - 0.16424530802218726j,
            ],
            [
                1.119247463119766 + 0.023956373851168843j,
                0.8776366291514774 + 0.2865220655803411j,
            ],
            [
                0.5515508326417458 - 0.15986016780453596j,
                -0.6803993941303332 + 1.1782711493556892j,
            ],
        ]
    )
    assert_array_almost_equal(actual, desired, decimal=15)

    random.state = state
    actual = random.complex_normal(loc=loc, gamma=1.0, relation=0.5, size=(3, 2))
    assert_array_almost_equal(actual, desired, decimal=15)


def test_invalid_capsule():
    class fake:
        capsule = "capsule"

    with pytest.raises(ValueError):
        ExtendedGenerator(fake())


def test_default_pcg64():
    eg = ExtendedGenerator()
    assert isinstance(eg.bit_generator, PCG64)
    assert eg.bit_generator.variant == "dxsm"
