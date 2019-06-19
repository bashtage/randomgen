import numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ['ThreeFry']

DEF THREEFRY_BUFFER_SIZE=4

cdef extern from 'src/threefry/threefry.h':
    struct s_r123array4x64:
        uint64_t v[4]

    ctypedef s_r123array4x64 r123array4x64

    ctypedef r123array4x64 threefry4x64_key_t
    ctypedef r123array4x64 threefry4x64_ctr_t

    struct s_threefry_state:
        threefry4x64_ctr_t *ctr
        threefry4x64_key_t *key
        int buffer_pos
        uint64_t buffer[THREEFRY_BUFFER_SIZE]
        int has_uint32
        uint32_t uinteger

    ctypedef s_threefry_state threefry_state

    uint64_t threefry_next64(threefry_state *state)  nogil
    uint32_t threefry_next32(threefry_state *state)  nogil
    void threefry_jump(threefry_state *state)
    void threefry_advance(uint64_t *step, threefry_state *state)


cdef uint64_t threefry_uint64(void* st) nogil:
    return threefry_next64(<threefry_state *>st)

cdef uint32_t threefry_uint32(void *st) nogil:
    return threefry_next32(<threefry_state *> st)

cdef double threefry_double(void* st) nogil:
    return uint64_to_double(threefry_next64(<threefry_state *>st))

cdef class ThreeFry(BitGenerator):
    """
    ThreeFry(seed=None, counter=None, key=None)

    Container for the ThreeFry (4x64) pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64-1], array of integers in
        [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
        data will be read from ``/dev/urandom`` (or the Windows analog)
        if available.  If unavailable, a hash of the time and process ID is
        used.
    counter : {None, int, array_like}, optional
        Counter to use in the ThreeFry state. Can be either
        a Python int in [0, 2**256) or a 4-element uint64 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the ThreeFry state.  Unlike seed, which is run through
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
    ThreeFry is a 64-bit PRNG that uses a counter-based design based on
    weaker (and faster) versions of cryptographic functions [1]_. Instances
    using different values of the key produce independent sequences.  ``ThreeFry``
    has a period of :math:`2^{256} - 1` and supports arbitrary advancing and
    jumping the sequence in increments of :math:`2^{128}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``ThreeFry`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Philox`` for a closely related PRNG.

    **State and Seeding**

    The ``ThreeFry`` state vector consists of a 2 256-bit values encoded as
    4-element uint64 arrays. One is a counter which is incremented by 1 for
    every 4 64-bit randoms produced.  The second is a key which determined
    the sequence produced.  Using different keys produces independent
    sequences.

    ``ThreeFry`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers.  In either case, the seed is
    used as an input for a second random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial state.
    Using a single 64-bit value for the seed can only initialize a small range of
    the possible initial state values.

    **Parallel Features**

    ``ThreeFry`` can be used in parallel applications by calling the ``jump``
    method  to advances the state as-if :math:`2^{128}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**256). When using ``jump``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen import Generator, ThreeFry
    >>> rg = [Generator(ThreeFry(1234)) for _ in range(10)]
    # Advance each ThreeFry instance by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    Alternatively, ``ThreeFry`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**196 + 2**132 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(ThreeFry(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``ThreeFry`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, ThreeFry
    >>> rg = Generator(ThreeFry(1234))
    >>> rg.standard_normal()
    0.123  # random

    Identical method using only ThreeFry

    >>> rg = ThreeFry(1234).generator
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,
           "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of
           the International Conference for High Performance Computing,
           Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.
    """
    cdef threefry_state rng_state
    cdef threefry4x64_ctr_t threefry_ctr
    cdef threefry4x64_key_t threefry_key

    def __init__(self, seed=None, counter=None, key=None):
        BitGenerator.__init__(self)
        self.rng_state.ctr = &self.threefry_ctr
        self.rng_state.key = &self.threefry_key
        self.seed(seed, counter, key)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &threefry_uint64
        self._bitgen.next_uint32 = &threefry_uint32
        self._bitgen.next_double = &threefry_double
        self._bitgen.next_raw = &threefry_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        self.rng_state.buffer_pos = THREEFRY_BUFFER_SIZE
        for i in range(THREEFRY_BUFFER_SIZE):
            self.rng_state.buffer[i] = 0

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator.

        This method is called when ``ThreeFry`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``ThreeFry``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``ThreeFry``.
        counter : {None, int array}, optional
            Positive integer less than 2**256 containing the counter position
            or a 4 element array of uint64 containing the counter
        key : {None, int, array}, optional
            Positive integer less than 2**256 containing the key
            or a 4 element array of uint64 containing the key. key and
            seed cannot be simultaneously used.

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
        if key is None:
            if seed is None:
                state = random_entropy(8, 'auto')
                state = state.view(np.uint64)
            else:
                state = seed_by_array(seed, 4)
            for i in range(4):
                self.rng_state.key.v[i] = state[i]
        else:
            key = int_to_array(key, 'key', 256, 64)
            for i in range(4):
                self.rng_state.key.v[i] = key[i]

        counter = 0 if counter is None else counter
        counter = int_to_array(counter, 'counter', 256, 64)
        for i in range(4):
            self.rng_state.ctr.v[i] = counter[i]

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
        ctr = np.empty(4, dtype=np.uint64)
        key = np.empty(4, dtype=np.uint64)
        buffer = np.empty(THREEFRY_BUFFER_SIZE, dtype=np.uint64)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            key[i] = self.rng_state.key.v[i]
        for i in range(THREEFRY_BUFFER_SIZE):
            buffer[i] = self.rng_state.buffer[i]
        state = {'counter': ctr, 'key': key}
        return {'bit_generator': self.__class__.__name__,
                'state': state,
                'buffer': buffer,
                'buffer_pos': self.rng_state.buffer_pos,
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        ctr = check_state_array(value['state']['counter'], 4, 64, 'counter')
        key = check_state_array(value['state']['key'], 4, 64, 'key')
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint64_t>ctr[i]
            self.rng_state.key.v[i] = <uint64_t>key[i]
        buffer = check_state_array(value['buffer'], THREEFRY_BUFFER_SIZE, 64,
                                   'buffer')
        for i in range(THREEFRY_BUFFER_SIZE):
            self.rng_state.buffer[i] = <uint64_t>value['buffer'][i]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
        self.rng_state.buffer_pos = value['buffer_pos']

    cdef jump_inplace(self, iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        self.advance(iter * int(2**128))

    def jump(self, iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : ThreeFry
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
        2**(128 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : ThreeFry
            New instance of generator jumped iter times
        """
        cdef ThreeFry bit_generator

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
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : ThreeFry
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
        delta = wrap_int(delta, 256)

        cdef np.ndarray delta_a
        delta_a = int_to_array(delta, 'step', 256, 64)
        threefry_advance(<uint64_t *>np.PyArray_DATA(delta_a), &self.rng_state)
        self._reset_state_variables()
        return self
