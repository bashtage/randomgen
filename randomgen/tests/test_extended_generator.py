import copy
from distutils.version import LooseVersion
import pickle

import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_warns,
    suppress_warnings,
)
import pytest

from randomgen import MT19937, PCG64, ExtendedGenerator

try:
    from numpy.random import Generator
except ImportError:
    from randomgen import Generator  # type: ignore[misc]

try:
    from scipy import linalg  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SEED = 1234567890
MV_SEED = 123456789


@pytest.fixture(scope="module")
def seed():
    return SEED


@pytest.fixture(scope="module")
def mv_seed():
    return MV_SEED


@pytest.fixture(scope="function")
def extended_gen():
    pcg = PCG64(0, mode="sequence")
    return ExtendedGenerator(pcg)


@pytest.fixture(scope="function")
def extended_gen_legacy():
    return ExtendedGenerator(MT19937(mode="legacy"))


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


@pytest.mark.parametrize("df", [2, 5, 10])
@pytest.mark.parametrize("dim", [2, 5, 10])
@pytest.mark.parametrize("size", [None, 5, (3, 7)])
def test_standard_wishart_reproduce(df, dim, size):
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    w = eg.standard_wishart(df, dim, size)
    if size is not None:
        sz = size if isinstance(size, tuple) else (size,)
        assert w.shape[:-2] == sz
    else:
        assert w.ndim == 2
    assert w.shape[-2:] == (dim, dim)

    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    w2 = eg.standard_wishart(df, dim, size)
    assert_allclose(w, w2)


@pytest.mark.parametrize("scale_dim", [0, 1, 2])
@pytest.mark.parametrize("df", [8, [10], [[5], [6]]])
def test_wishart_broadcast(df, scale_dim):
    dim = 5
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    scale = np.eye(dim)
    for i in range(scale_dim):
        scale = np.array([scale, scale])
    w = eg.wishart(df, scale)
    assert w.shape[-2:] == (dim, dim)
    if np.isscalar(df) and scale_dim == 0:
        assert w.ndim == 2
    elif scale_dim > 0:
        z = np.zeros(scale.shape[:-2])
        assert w.shape[:-2] == np.broadcast(df, z).shape

    size = w.shape[:-2]
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    w2 = eg.wishart(df, scale, size=size)
    assert_allclose(w, w2)


METHODS = ["svd", "eigh"]
if HAS_SCIPY:
    METHODS += ["cholesky"]


@pytest.mark.parametrize("method", METHODS)
def test_wishart_reduced_rank(method):
    scale = np.eye(3)
    scale[0, 1] = scale[1, 0] = 1.0
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    w = eg.wishart(10, scale, method=method, rank=2)
    assert w.shape == (3, 3)
    assert np.linalg.matrix_rank(w) == 2


@pytest.mark.skipif(HAS_SCIPY, reason="Cannot test with SciPy")
def test_missing_scipy_exception():
    scale = np.eye(3)
    scale[0, 1] = scale[1, 0] = 1.0
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    with pytest.raises(ImportError):
        eg.wishart(10, scale, method="cholesky", rank=2)


def test_wishart_exceptions():
    eg = ExtendedGenerator()
    with pytest.raises(ValueError, match="scale must be non-empty"):
        eg.wishart(10, [10])
    with pytest.raises(ValueError, match="scale must be non-empty"):
        eg.wishart(10, 10)
    with pytest.raises(ValueError, match="scale must be non-empty"):
        eg.wishart(10, np.array([[1, 2]]))
    with pytest.raises(ValueError, match="At least one"):
        eg.wishart([], np.eye(2))
    with pytest.raises(ValueError, match="df must contain strictly"):
        eg.wishart(-1, np.eye(2))
    with pytest.raises(ValueError, match="df must contain strictly"):
        df = np.ones((3, 4, 5))
        df[-1, -1, -1] = -1
        eg.wishart(df, np.eye(2))
    with pytest.raises(ValueError, match="cannot convert float"):
        eg.wishart(np.nan, np.eye(2))
    with pytest.raises(ValueError, match=r"size \(3,\) is not compatible"):
        eg.wishart([[10, 9, 8], [10, 9, 8]], np.eye(2), size=3)


@pytest.mark.parametrize("size", [None, 10, (10,), (10, 10)])
@pytest.mark.parametrize("df", [100, [100] * 10, [[100] * 10] * 10])
@pytest.mark.parametrize("tile", [None, (10,), (10, 10)])
def test_wishart_size(size, df, tile):
    eg = ExtendedGenerator()
    scale = np.eye(3)
    if tile:
        scale = np.tile(scale, tile + (1, 1))

    expected_shape = base_shape = (3, 3)
    sz = (size,) if isinstance(size, int) else size
    if np.asarray(df).ndim or tile:
        tile_shape = () if tile is None else tile
        shape = np.broadcast(np.asarray(df), np.ones(tile_shape)).shape
        expected_shape = shape + base_shape
        if size:
            if len(sz) < len(shape):
                with pytest.raises(ValueError, match=""):
                    eg.wishart(df, scale, size=size)
                return
    if size:
        expected_shape = sz + base_shape

    w = eg.wishart(df, scale, size=size)
    assert w.shape == expected_shape


def test_broadcast_both_paths():
    eg = ExtendedGenerator()
    w = eg.wishart([3, 5], np.eye(4), size=(100, 2))
    assert w.shape == (100, 2, 4, 4)


def test_factor_wishart():
    pcg = PCG64(0, mode="sequence")
    eg = ExtendedGenerator(pcg)
    w = eg.wishart([3, 5], 2 * np.eye(4), size=(10000, 2), method="factor")
    assert_allclose(np.diag((w[:, 0] / 3).mean(0)).mean(), 4, rtol=1e-2)
    assert_allclose(np.diag((w[:, 1] / 5).mean(0)).mean(), 4, rtol=1e-2)


def test_wishart_chi2(extended_gen):
    state = extended_gen.state
    w = extended_gen.standard_wishart(10, 1, 10, rescale=False)
    extended_gen.state = state
    bg = extended_gen.bit_generator
    c = Generator(bg).chisquare(10, size=10)
    assert_allclose(np.squeeze(w), c)


@pytest.mark.parametrize("df", [3, 5])
def test_standard_vs_full_wishart(extended_gen, df):
    state = extended_gen.state
    sw = extended_gen.standard_wishart(df, 4, size=(3, 4, 5), rescale=False)
    extended_gen.state = state
    w = extended_gen.wishart(df, np.eye(4), size=(3, 4, 5))
    assert_allclose(sw, w)


def test_standard_wishart_direct_small(extended_gen):
    bg = extended_gen.bit_generator
    state = bg.state
    w = extended_gen.standard_wishart(3, 4, 2, rescale=False)

    bg.state = state
    gen = Generator(bg)
    for i in range(2):
        n = gen.standard_normal((3, 4))
        direct = n.T @ n
        assert_allclose(direct, w[i])


def test_standard_wishart_direct_large(extended_gen):
    bg = extended_gen.bit_generator
    state = bg.state
    w = extended_gen.standard_wishart(3, 2, 2, rescale=False)
    bg.state = state
    gen = Generator(bg)
    direct = np.zeros((2, 2))
    for i in range(2):
        v11 = gen.chisquare(3)
        n = gen.standard_normal()
        v22 = gen.chisquare(2)
        direct[0, 0] = v11
        direct[1, 1] = n**2 + v22
        direct[1, 0] = direct[0, 1] = n * np.sqrt(v11)
        assert_allclose(direct, w[i])

        upper = np.zeros((2, 2))
        upper[0, 0] = np.sqrt(v11)
        upper[0, 1] = n
        upper[1, 1] = np.sqrt(v22)
        direct2 = upper.T @ upper
        assert_allclose(direct2, w[i])


@pytest.mark.parametrize("ex_gamma", [True, False])
@pytest.mark.parametrize("ex_rel", [True, False])
@pytest.mark.parametrize("ex_loc", [True, False])
@pytest.mark.parametrize("size", [None, (5, 4, 3, 2)])
def test_mv_complex_normal(extended_gen, ex_gamma, ex_rel, ex_loc, size):
    gamma = np.array([[2, 0 + 1.0j], [-0 - 1.0j, 2]])
    if ex_gamma:
        gamma = np.tile(gamma, (2, 1, 1))
    rel = np.array([[0.22, 0 + 0.1j], [+0 + 0.1j, 0.22]])
    if ex_rel:
        rel = np.tile(rel, (3, 1, 1, 1))
    loc = np.zeros(2)
    if ex_loc:
        loc = np.tile(loc, (4, 1, 1, 1))
    extended_gen.multivariate_complex_normal(np.zeros(2), size=size)
    extended_gen.multivariate_complex_normal(np.zeros(2), size=size)
    extended_gen.multivariate_complex_normal(np.zeros(2), gamma, size=size)
    extended_gen.multivariate_complex_normal(np.zeros(2), gamma, rel, size=size)
    extended_gen.multivariate_complex_normal(np.zeros(2), gamma, rel, size=size)


def test_mv_complex_normal_exceptions(extended_gen):
    with pytest.raises(ValueError, match="loc"):
        extended_gen.multivariate_complex_normal(0.0)
    with pytest.raises(ValueError, match="gamma"):
        extended_gen.multivariate_complex_normal([0.0], 1.0)
    with pytest.raises(ValueError, match="gamma"):
        extended_gen.multivariate_complex_normal([0.0], [1.0])
    with pytest.raises(ValueError, match="gamma"):
        extended_gen.multivariate_complex_normal([0.0], np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="relation"):
        extended_gen.multivariate_complex_normal([0.0, 0.0], np.eye(2), 0.0)
    with pytest.raises(ValueError, match="relation"):
        extended_gen.multivariate_complex_normal([0.0, 0.0], np.eye(2), 0.0)
    with pytest.raises(ValueError, match="relation"):
        extended_gen.multivariate_complex_normal([0.0, 0.0], relation=0.0)
    with pytest.raises(ValueError, match="The covariance matrix implied"):
        extended_gen.multivariate_complex_normal(
            [0.0, 0.0], np.array([[1.0, 0 + 1.0j], [0 + 1.0j, 1]])
        )
    with pytest.raises(ValueError, match="The leading dimensions"):
        extended_gen.multivariate_complex_normal(
            [0.0, 0.0], np.ones((4, 1, 3, 2, 2)), np.ones((1, 1, 2, 2, 2))
        )


def test_wishart_edge(extended_gen):
    with pytest.raises(ValueError, match="scale must be non-empty"):
        extended_gen.wishart(5, np.empty((0, 0)))
    with pytest.raises(ValueError, match="scale must be non-empty"):
        extended_gen.wishart(5, np.empty((0, 2, 2)))
    with pytest.raises(ValueError, match="scale must be non-empty"):
        extended_gen.wishart(5, [[]])
    with pytest.raises(ValueError, match="At least one value is required"):
        extended_gen.wishart(np.empty((0, 2, 3)), np.eye(2))


def test_mv_complex_normal_edge(extended_gen):
    with pytest.raises(ValueError, match="loc must be non-empty and at least"):
        extended_gen.multivariate_complex_normal(np.empty((0, 2)))
    with pytest.raises(ValueError, match="gamma must be non-empty"):
        extended_gen.multivariate_complex_normal([0, 0], np.empty((0, 2, 2)))
    with pytest.raises(ValueError, match="relation must be non-empty"):
        extended_gen.multivariate_complex_normal([0, 0], np.eye(2), np.empty((0, 2, 2)))


def test_random_long_double(extended_gen):
    out = extended_gen.random(dtype="longdouble")
    types_equiv = np.empty(1, np.longdouble).dtype == np.empty(1, np.double).dtype
    expected_type = float if types_equiv else np.longdouble
    assert isinstance(out, expected_type)
    out = extended_gen.random(dtype=np.longdouble)
    assert isinstance(out, expected_type)
    out = extended_gen.random(size=10, dtype=np.longdouble)
    expected_type = np.double if types_equiv else np.longdouble
    assert out.dtype == expected_type


def test_random_long_double_out(extended_gen):
    state = extended_gen.state
    out = np.empty((10, 10), dtype=np.longdouble)
    ret = extended_gen.random(out=out, dtype=np.longdouble)
    assert ret is out
    extended_gen.state = state
    alt = extended_gen.random(size=(10, 10), dtype=np.longdouble)
    assert_allclose(out, alt)


def test_random_other_type(extended_gen):
    with pytest.raises(TypeError, match="Unsupported dtype"):
        extended_gen.random(dtype=int)
    f16 = getattr(np, "float16", np.uint32)
    with pytest.raises(TypeError, match="Unsupported dtype"):
        extended_gen.random(dtype=f16)


def test_random(extended_gen_legacy):
    extended_gen_legacy.bit_generator.seed(SEED)
    actual = extended_gen_legacy.random((3, 2))
    desired = np.array(
        [
            [0.61879477158567997, 0.59162362775974664],
            [0.88868358904449662, 0.89165480011560816],
            [0.4575674820298663, 0.7781880808593471],
        ]
    )
    assert_array_almost_equal(actual, desired, decimal=15)

    extended_gen_legacy.bit_generator.seed(SEED)
    actual = extended_gen_legacy.random()
    assert_array_almost_equal(actual, desired[0, 0], decimal=15)


def test_random_float(extended_gen_legacy):
    extended_gen_legacy.bit_generator.seed(SEED)
    actual = extended_gen_legacy.random((3, 2))
    desired = np.array(
        [[0.6187948, 0.5916236], [0.8886836, 0.8916548], [0.4575675, 0.7781881]]
    )
    assert_array_almost_equal(actual, desired, decimal=7)


def test_random_float_scalar(extended_gen_legacy):
    extended_gen_legacy.bit_generator.seed(SEED)
    actual = extended_gen_legacy.random(dtype=np.float32)
    desired = 0.6187948
    assert_array_almost_equal(actual, desired, decimal=7)


def test_random_long_double_direct(extended_gen):
    state = extended_gen.state
    actual = extended_gen.random(10, dtype=np.longdouble)
    extended_gen.state = state
    nmant = np.finfo(np.longdouble).nmant
    denom = np.longdouble(str(2 ** (nmant + 1)))
    c = np.longdouble(1.0) / denom

    def twopart(u1, u2, a, b, c):
        v = int(u1) >> a
        if b:
            v = (v << (64 - b)) + (int(u2) >> b)
        return np.longdouble(str(v)) * c

    if nmant == 52:
        a, b = 11, 0
    elif nmant == 63:
        a, b = 0, 0
    elif nmant == 105:
        a, b = 11, 11
    elif nmant == 112:
        a, b = 7, 8
    else:
        raise NotImplementedError(f"No test for {nmant} bits")
    direct = np.empty(10, dtype=np.longdouble)
    u = extended_gen.bit_generator.random_raw(10 * (1 + (b > 0)))
    for i in range(10):
        if b:
            u1, u2 = u[2 * i : 2 * i + 2]
        else:
            u1, u2 = u[i], None
        direct[i] = twopart(u1, u2, a, b, c)

    assert_allclose(actual, direct)
