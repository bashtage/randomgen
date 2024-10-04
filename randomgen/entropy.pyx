#!python


cimport numpy as np
import numpy as np

from libc.stdint cimport uint32_t, uint64_t

__all__ = ["random_entropy", "seed_by_array"]

np.import_array()

cdef extern from "src/splitmix64/splitmix64.h":
    cdef uint64_t splitmix64_next(uint64_t *state) noexcept nogil

cdef extern from "src/entropy/entropy.h":
    cdef bint entropy_getbytes(void* dest, size_t size)
    cdef bint entropy_fallback_getbytes(void *dest, size_t size)

cdef Py_ssize_t compute_numel(size):
    cdef Py_ssize_t i, n = 1
    if isinstance(size, tuple):
        for i in range(len(size)):
            n *= size[i]
    else:
        n = size
    return n

cdef class TestSentinel(object):
    cdef bint _testing_auto, _testing_fallback

    def __init__(self, object auto=False, fallback=False):
        self._testing_auto = auto
        self._testing_fallback = fallback

    @property
    def testing_auto(self):
        return self._testing_auto

    @testing_auto.setter
    def testing_auto(self, object value):
        self._testing_auto = value

    @property
    def testing_fallback(self):
        return self._testing_fallback

    @testing_fallback.setter
    def testing_fallback(self, object value):
        self._testing_fallback = value


_test_sentinel = TestSentinel()


def seed_by_array(object seed, Py_ssize_t n):
    """
    Transforms a seed array into an initial state using SplitMix64

    Parameters
    ----------
    seed: int, array or uint64
        Array to use. If seed is a scalar, it is promoted to an array.
    n : int
        Number of 64-bit unsigned integers required in the seed

    Returns
    -------
    initial_state : array
        Array of uint64 containing the initial state

    Notes
    -----
    Uses SplitMix64 to transform the input to a seed
    """
    cdef uint64_t seed_copy = 0
    cdef uint64_t[::1] seed_array
    cdef uint64_t[::1] initial_state
    cdef Py_ssize_t seed_size, iter_bound
    cdef int i, loc = 0

    if hasattr(seed, "squeeze"):
        seed = seed.squeeze()
    arr = np.asarray(seed)
    if arr.shape == ():
        err_msg = "Scalar seeds must be integers between 0 and 2**64 - 1"
        if not np.isreal(arr):
            raise TypeError(err_msg)
        int_seed = int(seed)
        if int_seed != seed:
            raise TypeError(err_msg)
        if int_seed < 0 or int_seed > 2**64 - 1:
            raise ValueError(err_msg)
        seed_array = np.array([int_seed], dtype=np.uint64)
    else:
        err_msg = "Seed values must be integers between 0 and 2**64 - 1"
        obj = np.asarray(seed).astype(object)
        if obj.ndim != 1:
            raise ValueError("Array-valued seeds must be 1-dimensional")
        if not all((np.isscalar(v) and np.isreal(v)) for v in obj):
            raise TypeError(err_msg)
        if ((obj > int(2**64 - 1)) | (obj < 0)).any():
            raise ValueError(err_msg)
        obj_int = obj.astype(np.uint64, casting="unsafe")
        if not (obj == obj_int).all():
            raise TypeError(err_msg)
        seed_array = obj_int

    seed_size = seed_array.shape[0]
    iter_bound = n if n > seed_size else seed_size

    initial_state = <np.ndarray>np.empty(n, dtype=np.uint64)
    for i in range(iter_bound):
        if i < seed_size:
            seed_copy ^= seed_array[i]
        initial_state[loc] = splitmix64_next(&seed_copy)
        loc += 1
        if loc == n:
            loc = 0

    return np.array(initial_state)


def random_entropy(size=None, source="system"):
    """
    random_entropy(size=None, source='system')

    Read entropy from the system cryptographic provider

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn. Default is None, in which case a
        single value is returned.
    source : str {'system', 'fallback', 'auto'}
        Source of entropy. 'system' uses system cryptographic pool.
        'fallback' uses a hash of the time and process id. 'auto' attempts
        'system' before automatically falling back to 'fallback'

    Returns
    -------
    entropy : int or ndarray
        Entropy bits in 32-bit unsigned integers. A scalar is returned if size
        is `None`.

    Notes
    -----
    On Unix-like machines, reads from ``/dev/urandom``. On Windows machines
    reads from the RSA algorithm provided by the cryptographic service
    provider.

    This function reads from the system entropy pool and so samples are
    not reproducible. In particular, it does *NOT* make use of a
    bit generator, and so ``seed`` and setting ``state`` have no
    effect.

    Raises RuntimeError if the command fails.
    """
    cdef bint success = True
    cdef Py_ssize_t n = 0
    cdef uint32_t random = 0
    cdef uint32_t [:] randoms

    if source not in ("system", "fallback", "auto"):
        raise ValueError("Unknown value in source.")

    if size is None:
        if source in ("system", "auto"):
            success = entropy_getbytes(<void *>&random, 4)
        else:
            success = entropy_fallback_getbytes(<void *>&random, 4)
    else:
        n = compute_numel(size)
        randoms = np.zeros(n, dtype=np.uint32)
        if source in ("system", "auto"):
            success = entropy_getbytes(<void *>(&randoms[0]), 4 * n)
        else:
            success = entropy_fallback_getbytes(<void *>(&randoms[0]), 4 * n)
    if _test_sentinel.testing_auto and source == "auto":
        success = False
    if _test_sentinel.testing_fallback and source == "fallback":
        success = False
    if not success:
        if source == "auto":
            import warnings
            warnings.warn("Unable to read from system cryptographic provider",
                          RuntimeWarning)
            return random_entropy(size=size, source="fallback")
        else:
            raise RuntimeError("Unable to read from system cryptographic "
                               "provider or use fallback")

    if n == 0:
        return random
    return np.asarray(randoms).reshape(size)
