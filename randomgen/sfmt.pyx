import operator
from cpython.pycapsule cimport PyCapsule_New
from libc.string cimport memcpy

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy

__all__ = ['SFMT']

np.import_array()

DEF SFMT_MEXP = 19937
DEF SFMT_N = 156  # SFMT_MEXP / 128 + 1
DEF SFMT_N64 = SFMT_N * 2

cdef extern from "src/sfmt/sfmt.h":

    union W128_T:
        uint32_t u[4];
        uint64_t u64[2];

    ctypedef W128_T w128_t

    struct SFMT_T:
        w128_t state[SFMT_N]
        int idx

    ctypedef SFMT_T sfmt_t

    struct s_sfmt_state:
        sfmt_t *state
        int has_uint32
        uint32_t uinteger

        uint64_t *buffered_uint64
        int buffer_loc

    ctypedef s_sfmt_state sfmt_state

    uint64_t sfmt_next64(sfmt_state *state)  nogil
    uint32_t sfmt_next32(sfmt_state *state)  nogil

    void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed);
    void sfmt_init_by_array(sfmt_t * sfmt, uint32_t *init_key, int key_length);
    void sfmt_jump(sfmt_state *state)
    void sfmt_jump_n(sfmt_state *state, int count)

cdef uint64_t sfmt_uint64(void* st) nogil:
    return sfmt_next64(<sfmt_state *>st)

cdef uint32_t sfmt_uint32(void *st) nogil:
    return sfmt_next32(<sfmt_state *> st)

cdef uint64_t sfmt_raw(void *st) nogil:
    return sfmt_next64(<sfmt_state *>st)

cdef double sfmt_double(void* st) nogil:
    return uint64_to_double(sfmt_next64(<sfmt_state *>st))


cdef class SFMT:
    u"""
    SFMT(seed=None)

    Container for the SIMD-based Mersenne Twister pseudo RNG.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.  Can
        be any integer between 0 and 2**32 - 1 inclusive, an array (or other
        sequence) of unsigned 32-bit integers, or ``None`` (the default).  If
        `seed` is ``None``, a 32-bit unsigned integer is read from
        ``/dev/urandom`` (or the Windows analog) if available. If unavailable,
        a 32-bit hash of the time and process ID is used.

    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    ``SFMT`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers [1]_ . These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``SFMT`` state vector consists of a 384 element array of 64-bit
    unsigned integers plus a single integer value between 0 and 382
    indicating the current position within the main array. The implementation
    used here augments this with a 382 element array of doubles which are used
    to efficiently access the random numbers produced by the SFMT generator.

    ``SFMT`` is seeded using either a single 32-bit unsigned integer or a
    vector of 32-bit unsigned integers. In either case, the input seed is used
    as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 32-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Parallel Features**

    ``SFMT`` can be used in parallel applications by calling the method
    ``jump`` which advances the state as-if :math:`2^{128}` random numbers
    have been generated [2]_. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be initialized with the same seed to ensure that
    the segments come from the same sequence.

    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import Generator, SFMT
    >>> seed = random_entropy()
    >>> rs = [Generator(SFMT(seed)) for _ in range(10)]
    # Advance each SFMT instance by i jumps
    >>> for i in range(10):
    ...     rs[i].bit_generator.jump()

    **Compatibility Guarantee**

    ``SFMT`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    References
    ----------
    .. [1] Mutsuo Saito and Makoto Matsumoto, "SIMD-oriented Fast Mersenne
           Twister: a 128-bit Pseudorandom Number Generator." Monte Carlo
           and Quasi-Monte Carlo Methods 2006, Springer, pp. 607--622, 2008.
    .. [2] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
           Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
           Sequences and Their Applications - SETA, 290--298, 2008.
    """
    cdef sfmt_state rng_state
    cdef bitgen_t _bitgen
    cdef public object capsule
    cdef object _cffi
    cdef object _ctypes
    cdef public object lock

    def __init__(self, seed=None):
        self.rng_state.state = <sfmt_t *>PyArray_malloc_aligned(sizeof(sfmt_t))
        self.rng_state.buffered_uint64 = <uint64_t *>PyArray_calloc_aligned(SFMT_N64, sizeof(uint64_t))
        self.rng_state.buffer_loc = SFMT_N64
        self.seed(seed)
        self.lock = Lock()

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &sfmt_uint64
        self._bitgen.next_uint32 = &sfmt_uint32
        self._bitgen.next_double = &sfmt_double
        self._bitgen.next_raw = &sfmt_raw
        cdef const char *name = "BitGenerator"
        self.capsule = PyCapsule_New(<void *>&self._bitgen, name, NULL)

        self._cffi = None
        self._ctypes = None

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        from randomgen._pickle import __bit_generator_ctor
        return __bit_generator_ctor, (self.state['bit_generator'],), self.state

    def __dealloc__(self):
        if self.rng_state.state:
            PyArray_free_aligned(self.rng_state.state)
        if self.rng_state.buffered_uint64:
            PyArray_free_aligned(self.rng_state.buffered_uint64)

    cdef _reset_state_variables(self):
        self.rng_state.buffer_loc = SFMT_N64

    def random_raw(self, size=None, output=True):
        """
        random_raw(self, size=None)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values.  Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method directly exposes the the raw underlying pseudo-random
        number generator. All values are returned as unsigned 64-bit
        values irrespective of the number of bits produced by the PRNG.

        See the class docstring for the number of bits returned.
        """
        return random_raw(&self._bitgen, self.lock, size, output)

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        return benchmark(&self._bitgen, self.lock, cnt, method)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**32-1], array of integers in
            [0, 2**32-1] or ``None`` (the default). If `seed` is ``None``,
            then ``SFMT`` will try to read entropy from ``/dev/urandom``
            (or the Windows analog) if available to produce a 32-bit
            seed. If unavailable, a 32-bit hash of the time and process
            ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        cdef np.ndarray obj, seed_arr
        try:
            if seed is None:
                try:
                    seed_arr = random_entropy(2 * SFMT_N64)
                except RuntimeError:
                    seed_arr = random_entropy(2 * SFMT_N64, 'fallback')
                sfmt_init_by_array(self.rng_state.state,
                                   <uint32_t *>np.PyArray_DATA(seed_arr),
                                   <int>np.PyArray_DIM(seed_arr, 0))
            else:
                if hasattr(seed, 'squeeze'):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                sfmt_init_gen_rand(self.rng_state.state, seed)
        except TypeError:
            obj = np.asarray(seed).astype(np.int64, casting='safe').ravel()
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            seed_arr = obj.astype(np.uint32, casting='unsafe', order='C')
            sfmt_init_by_array(self.rng_state.state,
                               <uint32_t *>np.PyArray_DATA(seed_arr),
                               <int>np.PyArray_DIM(seed_arr, 0))
        # Clear the buffer
        self._reset_state_variables()

    cdef jump_inplace(self, iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        if iter < 0:
            raise ValueError('iter must be positive')
        sfmt_jump_n(&self.rng_state, iter)
        # Clear the buffer
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator.

        Returns
        -------
        self : SFMT
            PRNG jumped iter times
        """
        import warnings
        warnings.warn('jump (in-place) has been deprecated in favor of jumped'
                      ', which returns a new instance', DeprecationWarning)

        self.jump_inplace(iter)
        return self

    def jumped(self, np.npy_intp iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(128 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : SFMT
            New instance of generator jumped iter times
        """
        cdef SFMT bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

        return bit_generator

    @property
    def state(self):
        """
        Get or set the PRNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """

        cdef Py_ssize_t i, j, loc = 0
        cdef uint64_t[::1] state
        cdef uint64_t[::1] buffered_uint64

        state = np.empty(SFMT_N64, dtype=np.uint64)
        for i in range(SFMT_N):
            for j in range(2):
                state[loc] = self.rng_state.state.state[i].u64[j]
                loc += 1
        buffered_uint64 = np.empty(SFMT_N64, dtype=np.uint64)
        for i in range(SFMT_N64):
            buffered_uint64[i] = self.rng_state.buffered_uint64[i]
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': np.asarray(state),
                          'idx': self.rng_state.state.idx},
                'buffer_loc': self.rng_state.buffer_loc,
                'buffered_uint64': np.asarray(buffered_uint64),
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef Py_ssize_t i, j, loc = 0
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        state = value['state']['state']
        for i in range(SFMT_N):
            for j in range(2):
                self.rng_state.state.state[i].u64[j] = state[loc]
                loc += 1
        self.rng_state.state.idx = value['state']['idx']
        buffered_uint64 = value['buffered_uint64']
        for i in range(SFMT_N64):
            self.rng_state.buffered_uint64[i] = buffered_uint64[i]
        self.rng_state.buffer_loc = value['buffer_loc']
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    @property
    def ctypes(self):
        """
        ctypes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing ctypes wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._ctypes is None:
            self._ctypes = prepare_ctypes(&self._bitgen)

        return self._ctypes

    @property
    def cffi(self):
        """
        CFFI interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._cffi is not None:
            return self._cffi
        self._cffi = prepare_cffi(&self._bitgen)
        return self._cffi

    @property
    def generator(self):
        """
        Removed, raises NotImplementedError
        """
        raise NotImplementedError('This method for accessing a Generator has'
                                  'been removed.')
