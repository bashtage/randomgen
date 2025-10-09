#!python
import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["AESCounter"]

cdef uint64_t aes_uint64(void* st) noexcept nogil:
    return aes_next64(<aesctr_state_t *>st)

cdef uint32_t aes_uint32(void *st) noexcept nogil:
    return aes_next32(<aesctr_state_t *> st)

cdef double aes_double(void* st) noexcept nogil:
    return uint64_to_double(aes_next64(<aesctr_state_t *>st))

cdef class AESCounter(BitGenerator):
    """
    AESCounter(seed=None, *, counter=None, key=None, mode="sequence")

    Container for the AES Counter pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, SeedSequence}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**128), a SeedSequence instance or
        ``None`` (the default). If `seed` is ``None``, then  data is read
        from ``/dev/urandom`` (or the Windows analog) if available. If
        unavailable, a hash of the time and process ID is used.
    counter : {None, int, array_like[uint64]}, optional
        Counter to use in the AESCounter state. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        If not provided, the counter is initialized at 0.
    key : {None, int, array_like[uint64]}, optional
        Key to use in the AESCounter state. Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        key and seed cannot both be used.
    mode : {None, "sequence"}
        Deprecated parameter. Do not use.

        .. deprecated: 2.0.0

           Starting in version 2, only seed sequences are supported.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    AESCounter is a 64-bit PRNG that uses a counter-based design based on
    the AES-128 cryptographic function [1]_. Instances using different values
    of the key produce distinct sequences. ``AESCounter`` has a period
    of :math:`2^{128}` and supports arbitrary advancing and
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
    or a vector of 2 64-bit unsigned integers. In either case, the seed is
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

    >>> from numpy.random import Generator
    >>> from randomgen import AESCounter
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
    >>> from numpy.random import Generator
    >>> from randomgen import AESCounter
    >>> rg = Generator(AESCounter(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Advanced Encryption Standard. (n.d.). In Wikipedia. Retrieved
        June 1, 2019, from https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
    """
    def __init__(self, seed=None, *, counter=None, key=None, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        # Calloc since ctr needs to be 0
        self.rng_state = <aesctr_state_t *>PyArray_calloc_aligned(
            sizeof(aesctr_state_t), 1
        )
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

    @property
    def use_aesni(self):
        """
        Toggle use of AESNI

        Parameters
        ----------
        flag : bool
            Flag indicating whether to use AESNI

        Returns
        -------
        flag : bool
            Current flag value

        Raises
        ------
        ValueError
            If AESNI is not supported
        """
        return RANDOMGEN_USE_AESNI

    @use_aesni.setter
    def use_aesni(self, value):
        capable = aes_capable()
        if value and not capable:
            raise ValueError("CPU does not support AESNI")
        aesctr_use_aesni(bool(value))

    def _seed_from_seq(self, counter=None):
        state = self._get_seed_seq().generate_state(2, np.uint64)
        self.seed(key=state, counter=counter)
        self._reset_state_variables()

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator

        This method is called when ``AESCounter`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``AESCounter``.

        Parameters
        ----------
        seed : {None, int, SeedSequence}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**128), a SeedSequence instance or
            ``None`` (the default). If `seed` is ``None``, then  data is read
            from ``/dev/urandom`` (or the Windows analog) if available. If
            unavailable, a hash of the time and process ID is used.
        counter : {None, int, array_like[uint64]}, optional
            Counter to use in the AESCounter state. Can be either
            a Python int in [0, 2**128) or a 2-element uint64 array.
            If not provided, the counter is initialized at 0.
        key : {None, int, array_like[uint64]}, optional
            Key to use in the AESCounter state. Unlike seed, which is run
            through another RNG before use, the value in key is directly set.
            Can be either a Python int in [0, 2**128) or a 2-element uint64
            array. key and seed cannot both be used.

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
            raise ValueError("seed and key cannot be both used")
        if key is None:
            BitGenerator._seed_with_seed_sequence(self, seed, counter=counter)
            return

        key = object_to_int(key, 128, "key")
        counter = object_to_int(counter, 128, "counter")
        _seed = int_to_array(key, "key", 128, 64)
        # TODO: We have swapped here, but should we always use native in Python?
        aesctr_seed(self.rng_state, <uint64_t*>np.PyArray_DATA(_seed))
        _counter = np.empty(8, dtype=np.uint64)
        counter = 0 if counter is None else counter
        for i in range(4):
            _counter[2*i:2*i+2] = int_to_array(
                (counter + i) % (2**128), "counter", 128, 64
            )
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
        state = np.empty(16 * 4, dtype=np.uint8)
        for i in range(16 * 4):
            state[i] = self.rng_state.state[i]
        offset = self.rng_state.offset
        return {"bit_generator": fully_qualified_name(self),
                "s": {"state": state, "seed": seed, "counter": counter,
                      "offset": offset},
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef np.npy_intp i

        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state =value["s"]["state"]
        for i in range(16 * 4):
            self.rng_state.state[i] = state[i]
        self.rng_state.offset = value["s"]["offset"]
        seed = np.ascontiguousarray(value["s"]["seed"], dtype=np.uint64)
        counter = np.ascontiguousarray(value["s"]["counter"], dtype=np.uint64)
        if seed.ndim != 1 or seed.shape[0] != 2 * (10 + 1):
            raise ValueError("seed must be a 1d uint64 array with 22 elements")
        if counter.ndim != 1 or counter.shape[0] != 2 * (4):
            raise ValueError("counter must be a 1d uint64 array with 8 "
                             "elements")
        aesctr_set_seed_counter(self.rng_state,
                                <uint64_t*>np.PyArray_DATA(seed),
                                <uint64_t*>np.PyArray_DATA(counter))
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]

    cdef jump_inplace(self, object iter):
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
        self : AESCounter
            PRNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is
        required to ensure exact reproducibility.
        """
        import warnings
        warnings.warn("jump (in-place) has been deprecated in favor of jumped"
                      ", which returns a new instance", DeprecationWarning)
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
        bit_generator : AESCounter
            New instance of generator jumped iter times
        """
        cdef AESCounter bit_generator

        bit_generator = self.__class__(seed=self._copy_seed())
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
        draws from the core RNG. This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG. For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        delta = wrap_int(delta, 129)
        if delta == 0:
            return self

        step = int_to_array(delta, "delta", 64*3, 64)
        aesctr_advance(self.rng_state, <uint64_t *>np.PyArray_DATA(step))
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        return self
