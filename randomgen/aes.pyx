import numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ['AESCounter']

cdef extern from "src/aesctr/aesctr.h":

    # int are placeholders only for
    struct AESCTR_STATE_T:
        int ctr[4]
        int seed[10 + 1]
        uint8_t state[16 * 4]
        size_t offset
        int has_uint32
        uint32_t uinteger

    ctypedef AESCTR_STATE_T aesctr_state_t

    uint64_t aes_next64(aesctr_state_t *aesctr) nogil
    uint32_t aes_next32(aesctr_state_t *aesctr) nogil
    double aes_next_double(aesctr_state_t *aesctr) nogil

    void aesctr_seed(aesctr_state_t *aesctr, uint64_t *seed)
    void aesctr_set_seed_counter(aesctr_state_t *aesctr, uint64_t *seed, uint64_t *counter)
    void aesctr_get_seed_counter(aesctr_state_t *aesctr, uint64_t *seed, uint64_t *counter)
    int aes_capable()
    void aesctr_advance(aesctr_state_t *aesctr, uint64_t *step)
    void aesctr_set_counter(aesctr_state_t *aesctr, uint64_t *counter)

cdef uint64_t aes_uint64(void* st) nogil:
    return aes_next64(<aesctr_state_t *>st)

cdef uint32_t aes_uint32(void *st) nogil:
    return aes_next32(<aesctr_state_t *> st)

cdef double aes_double(void* st) nogil:
    return uint64_to_double(aes_next64(<aesctr_state_t *>st))

cdef class AESCounter(BitGenerator):
    """
    AESCounter(seed=None)

    Container for the AES Counter pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**128-1], or ``None`` (the default).
        If `seed` is ``None``, then  data is read
        from ``/dev/urandom`` (or the Windows analog) if available.  If
        unavailable, a hash of the time and process ID is used.
    counter : {None, int, array_like}, optional
        Counter to use in the AESCounter state. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the AESCounter state.  Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        key and seed cannot both be used.


    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    AESCounter is a 64-bit PRNG that uses a counter-based design based on
    the AES-128 cryptographic function [1]_. Instances using different values
    of the key produce independent sequences.  ``AESCounter`` has a period
    of :math:`2^{128} - 1` and supports arbitrary advancing and
    jumping the sequence in increments of :math:`2^{64}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``AESCounter`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Philox`` and ``ThreeFry`` for a related counter-based PRNG.

    **State and Seeding**

    The ``AESCounter`` state vector consists of a 64-element array of uint8
    that capture buffered draws from the distribution, a 22-element array of
    uint64s holding the seed (11 by 128bits), and an 8-element array of
    uint64 that holds the counters (4 by 129 bits). The first two elements of
    the seed are the value provided by the user (or from the entropy pool).
    The offset varies between 0 and 64 and shows the location in the buffer of
    the next 64 bits.

    ``AESCounter`` is seeded using either a single 128-bit unsigned integer
    or a vector of 2 64-bit unsigned integers.  In either case, the seed is
    used as an input for a second random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial
    state. Using a single 64-bit value for the seed can only initialize a small
    range of the possible initial state values.

    **Parallel Features**

    ``AESCounter`` can be used in parallel applications by calling the ``jump``
    method  to advances the state as-if :math:`2^{64}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**128). When using ``jump``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen import Generator, AESCounter
    >>> rg = [Generator(AESCounter(1234)) for _ in range(10)]
    # Advance each AESCounter instances by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    Alternatively, ``AESCounter`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**93 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(AESCounter(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``AESCounter`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, AESCounter
    >>> rg = Generator(AESCounter(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    """
    cdef aesctr_state_t *rng_state

    def __init__(self, seed=None, counter=None, key=None):
        BitGenerator.__init__(self)
        if not aes_capable():
            raise RuntimeError('AESNI is required to use AESCounter.')  # pragma: no cover
        # Calloc since ctr needs to be 0
        self.rng_state = <aesctr_state_t *>PyArray_calloc_aligned(sizeof(aesctr_state_t), 1)
        self.seed(seed, counter, key)

        self._bitgen.state = <void *>self.rng_state
        self._bitgen.next_uint64 = &aes_uint64
        self._bitgen.next_uint32 = &aes_uint32
        self._bitgen.next_double = &aes_double
        self._bitgen.next_raw = &aes_uint64

    def __dealloc__(self):
        if self.rng_state:
            PyArray_free_aligned(self.rng_state)

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator.

        This method is called when ``AESCounter`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``AESCounter``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``AESCounter``.
        counter : {int array}, optional
            Positive integer less than 2**128 containing the counter position
            or a 2 element array of uint64 containing the counter
        key : {int, array}, options
            Positive integer less than 2**128 containing the key
            or a 2 element array of uint64 containing the key

        Raises
        ------
        ValueError
            If values are out of range for the PRNG.

        Notes
        -----
        The two representation of the counter and key are related through
        array[i] = (value // 2**(64*i)) % 2**64.
        """
        cdef np.ndarray _seed

        if seed is not None and key is not None:
            raise ValueError('seed and key cannot be both used')
        ub = 2 ** 128
        if key is None:
            if seed is None:
                _seed = random_entropy(4, 'auto')
                _seed = _seed.view(np.uint64)
            else:
                seed_arr = np.asarray(seed).squeeze()
                if seed_arr.ndim==0:
                    seed = int_to_array(seed_arr.item(), 'seed', 128, 64)
                _seed = seed_by_array(seed, 2)
        else:
            _seed = <np.ndarray>int_to_array(key, 'key', 128, 64)
        aesctr_seed(self.rng_state, <uint64_t*>np.PyArray_DATA(_seed))
        _counter = np.empty(8, dtype=np.uint64)
        counter = 0 if counter is None else counter
        for i in range(4):
            _counter[2*i:2*i+2] = int_to_array(counter+i, 'counter', 128, 64)
            aesctr_set_counter(self.rng_state,
                               <uint64_t*>np.PyArray_DATA(_counter))
        self._reset_state_variables()

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
        cdef np.ndarray seed, counter
        cdef np.npy_intp i
        cdef size_t offset

        seed = np.empty(2 * (10 + 1), dtype=np.uint64)
        counter = np.empty(2 * 4, dtype=np.uint64)
        aesctr_get_seed_counter(self.rng_state,
                                <uint64_t*>np.PyArray_DATA(seed),
                                <uint64_t*>np.PyArray_DATA(counter))
        state = np.empty(16 * 4, dtype = np.uint8)
        for i in range(16 * 4):
            state[i] = self.rng_state.state[i]
        offset = self.rng_state.offset
        return {'bit_generator': self.__class__.__name__,
                's': {'state': state, 'seed': seed, 'counter': counter,
                      'offset': offset},
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef np.ndarray seed, counter
        cdef np.npy_intp i

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        state =value['s']['state']
        for i in range(16 * 4):
            self.rng_state.state[i] = state[i]
        offset = self.rng_state.offset
        self.rng_state.offset = value['s']['offset']
        seed = np.ascontiguousarray(value['s']['seed'], dtype=np.uint64)
        counter = np.ascontiguousarray(value['s']['counter'], dtype=np.uint64)
        if seed.ndim != 1 or seed.shape[0] != 2 * (10 + 1):
            raise ValueError('seed must be a 1d uint64 array with 22 elements')
        if counter.ndim != 1 or counter.shape[0] != 2 * (4):
            raise ValueError('counter must be a 1d uint64 array with 8 '
                             'elements')
        aesctr_set_seed_counter(self.rng_state,
                                <uint64_t*>np.PyArray_DATA(seed),
                                <uint64_t*>np.PyArray_DATA(counter))
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    cdef jump_inplace(self, iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        self.advance(iter * int(2 ** 64))

    def jump(self, iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**64 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : AESCounter
            PRNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is
        required to ensure exact reproducibility.
        """
        import warnings
        warnings.warn('jump (in-place) has been deprecated in favor of jumped'
                      ', which returns a new instance', DeprecationWarning)
        self.jump_inplace(iter)
        return self

    def jumped(self, iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(64 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : AESCounter
            New instance of generator jumped iter times
        """
        cdef AESCounter bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

        return bit_generator

    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG.

        Returns
        -------
        self : AESCounter
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        cdef np.ndarray step

        delta = wrap_int(delta, 129)
        step = int_to_array(delta, 'delta', 64*3, 64)
        aesctr_advance(self.rng_state, <uint64_t *>np.PyArray_DATA(step))
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        return self