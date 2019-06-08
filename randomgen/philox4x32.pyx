from cpython.pycapsule cimport PyCapsule_New

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

import numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ['Philox4x32']

np.import_array()

DEF PHILOX_BUFFER_SIZE=4

cdef extern from 'src/philox4x32/philox4x32.h':

    struct s_r123array2x32:
        uint32_t v[2];

    struct s_r123array4x32:
        uint32_t v[4]

    ctypedef s_r123array4x32 r123array4x32
    ctypedef s_r123array2x32 r123array2x32

    ctypedef r123array4x32 philox4x32_ctr_t
    ctypedef r123array2x32 philox4x32_key_t

    struct s_philox4x32_state:
        philox4x32_ctr_t *ctr
        philox4x32_key_t *key
        int buffer_pos
        uint32_t buffer[PHILOX_BUFFER_SIZE]

    ctypedef s_philox4x32_state philox4x32_state

    uint64_t philox4x32_next64(philox4x32_state *state)  nogil
    uint32_t philox4x32_next32(philox4x32_state *state)  nogil
    uint32_t philox4x32_next_double(philox4x32_state *state)  nogil
    void philox4x32_advance(uint32_t *step, philox4x32_state *state)

cdef uint64_t philox4x32_uint64(void*st) nogil:
    return philox4x32_next64(<philox4x32_state *> st)

cdef uint32_t philox4x32_uint32(void *st) nogil:
    return philox4x32_next32(<philox4x32_state *> st)

cdef double philox4x32_double(void*st) nogil:
    return philox4x32_next_double(<philox4x32_state *> st)

cdef uint64_t philox4x32_raw(void *st) nogil:
    return <uint64_t>philox4x32_next32(<philox4x32_state *> st)

cdef class Philox4x32:
    """
    Philox4x32(seed=None, counter=None, key=None)

    Container for the Philox (4x32) pseudo-random number generator.

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
        Counter to use in the Philox4x32 state. Can be either
        a Python int (long in 2.x) in [0, 2**128) or a 4-element uint32 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the Philox4x32 state.  Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int (long in 2.x) in [0, 2**64) or a 2-element uint32 array.
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
    Philox4x32 is a 32-bit PRNG that uses a counter-based design based on
    weaker (and faster) versions of cryptographic functions [1]_. Instances
    using different values of the key produce independent sequences.
    ``Philox4x32`` has a period of :math:`2^{128} - 1` and supports arbitrary
    advancing and jumping the sequence in increments of :math:`2^{64}`. These
    features allow multiple non-overlapping sequences to be generated.

    ``Philox4x32`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Philox`` and ``ThreeFry`` for a closely related PRNG.

    **State and Seeding**

    The ``Philox4x32`` state vector consists of a a 128-bit value encoded as
    a 4-element uint32 array and a 64-bit value encoded as a 2-element uint32
    array. One is a counter which is incremented by 1 for every 4 32-bit
    randoms produced.  The second is a key which determines the sequence
    produced.  Using different keys produces independent sequences.

    ``Philox4x32`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers.  In either case, the seed is
    used as an input for a second random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial
    state.

    **Parallel Features**

    ``Philox4x32`` can be used in parallel applications by calling the ``jump``
    method  to advances the state as-if :math:`2^{64}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**128). When using ``jump``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen import Generator, Philox4x32
    >>> rg = [Generator(Philox4x32(1234)) for _ in range(10)]
    # Advance each Philox instance by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    Alternatively, ``Philox4x32`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(Philox4x32(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``Philox4x32`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, Philox4x32
    >>> rg = Generator(Philox4x32(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,
           "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of
           the International Conference for High Performance Computing,
           Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.
    """
    cdef philox4x32_state rng_state
    cdef philox4x32_key_t philox4x32_key
    cdef philox4x32_ctr_t philox4x32_ctr
    cdef bitgen_t _bitgen
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef public object lock

    def __init__(self, seed=None, counter=None, key=None):
        self.rng_state.ctr = &self.philox4x32_ctr
        self.rng_state.key = &self.philox4x32_key
        self.seed(seed, counter, key)
        self.lock = Lock()

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &philox4x32_uint64
        self._bitgen.next_uint32 = &philox4x32_uint32
        self._bitgen.next_double = &philox4x32_double
        self._bitgen.next_raw = &philox4x32_raw

        self._ctypes = None
        self._cffi = None

        cdef const char *name = 'BitGenerator'
        self.capsule = PyCapsule_New(<void *>&self._bitgen, name, NULL)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        from randomgen._pickle import __bit_generator_ctor
        return __bit_generator_ctor, (self.state['bit_generator'],), self.state

    cdef _reset_state_variables(self):
        self.rng_state.buffer_pos = PHILOX_BUFFER_SIZE
        for i in range(PHILOX_BUFFER_SIZE):
            self.rng_state.buffer[i] = 0

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

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator.

        This method is called when ``Philox4x32`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``Philox4x32``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``Philox4x32``.
        counter : {int array}, optional
            Positive integer less than 2**128 containing the counter position
            or a 4 element array of uint32 containing the counter
        key : {int, array}, options
            Positive integer less than 2**64 containing the key
            or a 2 element array of uint32 containing the key

        Raises
        ------
        ValueError
            If values are out of range for the PRNG.

        Notes
        -----
        The two representation of the counter and key are related through
        array[i] = (value // 2**(32*i)) % 2**32.
        """
        if seed is not None and key is not None:
            raise ValueError('seed and key cannot be both used')
        ub = 2 ** 32
        if key is None:
            if seed is None:
                try:
                    state = random_entropy(2)
                except RuntimeError:
                    state = random_entropy(2, 'fallback')
            else:
                state = seed_by_array(seed, 1).view(np.uint32)
            for i in range(2):
                self.rng_state.key.v[i] = state[i]
        else:
            key = int_to_array(key, 'key', 64, 32)
            for i in range(2):
                self.rng_state.key.v[i] = key[i]
        counter = 0 if counter is None else counter
        counter = int_to_array(counter, 'counter', 128, 32)
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
        ctr = np.empty(4, dtype=np.uint32)
        key = np.empty(2, dtype=np.uint32)
        buffer = np.empty(PHILOX_BUFFER_SIZE, dtype=np.uint32)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            if i < 2:
                key[i] = self.rng_state.key.v[i]
        for i in range(PHILOX_BUFFER_SIZE):
            buffer[i] = self.rng_state.buffer[i]

        state = {'counter': ctr, 'key': key}
        return {'bit_generator': self.__class__.__name__,
                'state': state,
                'buffer': buffer,
                'buffer_pos': self.rng_state.buffer_pos}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        ctr = check_state_array(value['state']['counter'], 4, 32, 'counter')
        key = check_state_array(value['state']['key'], 2, 32, 'key')
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint32_t>ctr[i]
            if i < 2:
                self.rng_state.key.v[i] = <uint32_t>key[i]
        buffer = check_state_array(value['buffer'], 4, 32, 'buffer')
        for i in range(PHILOX_BUFFER_SIZE):
            self.rng_state.buffer[i] = <uint32_t> value['buffer'][i]
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
        self.advance(iter * int(2 ** 64))

    def jump(self, jumps=1):
        """
        jump(jumps=1)

        Jumps the state as-if 2**64 random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : Philox4X32
            PRNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is
        required to ensure exact reproducibility.
        """
        import warnings
        warnings.warn('jump (in-place) has been deprecated in favor of jumped'
                      ', which returns a new instance', DeprecationWarning)
        self.jump_inplace(jumps)
        return self

    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(64 * iter) random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Philox4x32
            New instance of generator jumped iter times
        """
        cdef Philox4x32 bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(jumps)

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
        self : Philox4x32
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
        delta = wrap_int(delta, 128)

        cdef np.ndarray delta_a
        delta_a = int_to_array(delta, 'step', 128, 32)
        philox4x32_advance(<uint32_t *>np.PyArray_DATA(delta_a), &self.rng_state)
        self._reset_state_variables()
        return self

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
