import numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ['SPECK128']

DEF SPECK_UNROLL = 12
DEF SPECK_ROUNDS = 34

cdef extern from "src/speck-128/speck-128.h":

    union SPEC_T:
      uint64_t u64[2]

    ctypedef SPEC_T spec_t

    struct SPECK_STATE_T:
        spec_t ctr[SPECK_UNROLL // 2];
        uint8_t buffer[8 * SPECK_UNROLL];
        uint64_t round_key[SPECK_ROUNDS];
        int offset;
        int has_uint32;
        uint32_t uinteger;

    ctypedef SPECK_STATE_T speck_state_t

    uint64_t speck_next64(speck_state_t *state) nogil
    uint32_t speck_next32(speck_state_t *state) nogil

    void speck_seed(speck_state_t *state, uint64_t *seed)
    void speck_set_counter(speck_state_t *state, uint64_t *ctr)
    void speck_advance(speck_state_t *state, uint64_t *step)
    # void speck_set_seed_counter(speck_state_t *state, uint64_t *seed, uint64_t *counter)
    # void speck_get_seed_counter(speck_state_t *state, uint64_t *seed, uint64_t *counter)
    # void speck_advance(speck_state_t *state, uint64_t *step)
    # void speck_set_counter(speck_state_t *state, uint64_t *counter)

cdef uint64_t speck_uint64(void* st) nogil:
    return speck_next64(<speck_state_t *>st)

cdef uint32_t speck_uint32(void *st) nogil:
    return speck_next32(<speck_state_t *> st)

cdef double speck_double(void* st) nogil:
    return uint64_to_double(speck_next64(<speck_state_t *>st))

cdef class SPECK128(BitGenerator):
    """
    SPECK128(seed=None)

    Container for the SPECK (128 x 256) pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**256), or ``None`` (the default).
        If `seed` is ``None``, then  data is read
        from ``/dev/urandom`` (or the Windows analog) if available.  If
        unavailable, a hash of the time and process ID is used.
    counter : {None, int, array_like}, optional
        Counter to use in the SPECK128 state. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the SPECK128 state.  Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int in [0, 2**256) or a 4-element uint64 array.
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
    SPECK128 is a 64-bit PRNG that uses a counter-based design based on
    the SPECK-128 cryptographic function [1]_. Instances using different values
    of the key produce independent sequences.  ``SPECK128`` has a large period
    :math:`2^{TBD} - 1` and supports arbitrary advancing and
    jumping the sequence in increments of :math:`2^{64}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``SPECK128`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``AESCounter``, ``Philox`` and ``ThreeFry`` for a related
    counter-based PRNG.

    **State and Seeding**

    The ``SPECK128`` state vector consists of a 96-element array of uint8
    that capture buffered draws from the distribution, a 34-element array of
    uint64s holding the round key, and an 12-element array of
    uint64 that holds the 128-bit counters (6 by 128 bits).
    The offset varies between 0 and 96 and shows the location in the buffer of
    the next 64 bits.

    ``SPECK128`` is seeded using either a single 256-bit unsigned integer
    or a vector of 4 64-bit unsigned integers.  In either case, the seed is
    used as an input for a second random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial
    state. Using a single 64-bit value for the seed can only initialize a small
    range of the possible initial state values.

    **Parallel Features**

    ``SPECK128`` can be used in parallel applications by calling the ``jump``
    method  to advances the state as-if :math:`2^{64}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**128). When using ``jump``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen import Generator, SPECK128
    >>> rg = [Generator(SPECK128(1234)) for _ in range(10)]
    # Advance each SPECK128 instances by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    Alternatively, ``SPECK128`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**93 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(SPECK128(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``SPECK128`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, SPECK128
    >>> rg = Generator(SPECK128(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Ray Beaulieu, Douglas Shors, Jason Smith, Stefan Treatman-Clark,
       Bryan Weeks, and Louis Wingers. SIMON and SPECK Implementation Guide.
       National Security Agency. January 15, 2019. from
       https://nsacyber.github.io/simon-speck/implementations/ImplementationGuide1.1.pdf
    """
    cdef speck_state_t *rng_state

    def __init__(self, seed=None, counter=None, key=None):
        BitGenerator.__init__(self)
        # Calloc since ctr needs to be 0
        self.rng_state = <speck_state_t *>PyArray_calloc_aligned(sizeof(speck_state_t), 1)
        self.seed(seed, counter, key)

        self._bitgen.state = <void *>self.rng_state
        self._bitgen.next_uint64 = &speck_uint64
        self._bitgen.next_uint32 = &speck_uint32
        self._bitgen.next_double = &speck_double
        self._bitgen.next_raw = &speck_uint64

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

        This method is called when ``SPECK128`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``SPECK128``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``SPECK128``.
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
        if seed is not None and key is not None:
            raise ValueError('seed and key cannot be both used')
        ub = 2 ** 256
        if key is None:
            if seed is None:
                _seed = random_entropy(8, 'auto')
                _seed = _seed.view(np.uint64)
            else:
                seed_arr = np.asarray(seed).squeeze()
                if seed_arr.ndim==0:
                    seed = int_to_array(seed_arr.item(), 'seed', 256, 64)
                _seed = seed_by_array(seed, 4)
        else:
            _seed = <np.ndarray>int_to_array(key, 'key', 256, 64)
        speck_seed(self.rng_state, <uint64_t *>np.PyArray_DATA(_seed))
        counter = 0 if counter is None else counter
        _counter = int_to_array(counter,'counter', 128, 64)
        speck_set_counter(self.rng_state, <uint64_t *>np.PyArray_DATA(_counter))
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
        cdef int i, j
        cdef uint8_t *arr8
        cdef uint64_t *arr

        ctr = np.empty(SPECK_UNROLL, dtype=np.uint64)
        buffer = np.empty(8 * SPECK_UNROLL, dtype=np.uint8)
        round_key = np.empty(SPECK_ROUNDS, dtype=np.uint64)

        arr = <uint64_t*>np.PyArray_DATA(ctr)
        for i in range(SPECK_UNROLL):
            arr[i] = self.rng_state.ctr[i // 2].u64[i % 2]

        arr8 = <uint8_t*>np.PyArray_DATA(buffer)
        for i in range(8*SPECK_UNROLL):
            arr8[i] = self.rng_state.buffer[i]

        arr = <uint64_t*>np.PyArray_DATA(round_key)
        for i in range(SPECK_ROUNDS):
            arr[i] = self.rng_state.round_key[i]

        return {'bit_generator': self.__class__.__name__,
                'state': {'ctr': ctr,
                          'buffer': buffer,
                          'round_key':round_key,
                          'offset': self.rng_state.offset},
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef i
        cdef uint8_t *arr8
        cdef uint64_t *arr

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))

        state =value['state']

        ctr = check_state_array(state['ctr'], SPECK_UNROLL, 64, 'ctr')
        buffer = check_state_array(state['buffer'], 8 * SPECK_UNROLL, 8,
                                   'buffer')
        round_key = check_state_array(state['round_key'], SPECK_ROUNDS, 64,
                                      'round_key')

        arr = <uint64_t*>np.PyArray_DATA(ctr)
        for i in range(SPECK_UNROLL):
            self.rng_state.ctr[i//2].u64[i%2] = arr[i]

        arr8 = <uint8_t*>np.PyArray_DATA(buffer)
        for i in range(8*SPECK_UNROLL):
            self.rng_state.buffer[i] = arr8[i]

        arr = <uint64_t*>np.PyArray_DATA(round_key)
        for i in range(SPECK_ROUNDS):
            self.rng_state.round_key[i] = arr[i]

        self.rng_state.offset = state['offset']
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
            Jumps the state as-if iter * 2**64 random numbers are generated

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : SPECK128
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
        iter * 2**64 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : SPECK128
            New instance of generator jumped iter times
        """
        cdef SPECK128 bit_generator
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
        self : SPECK128
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
        delta = wrap_int(delta, 129)
        if delta == 0:
            return self

        step = int_to_array(delta, 'delta', 64*3, 64)
        speck_advance(self.rng_state, <uint64_t *>np.PyArray_DATA(step))
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        return self