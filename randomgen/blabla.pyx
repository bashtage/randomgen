#!python
import numpy as np

__all__ = ["BlaBla"]

cdef uint64_t blabla_uint64(void* st) noexcept nogil:
    return blabla_next64(<blabla_state_t *>st)

cdef uint32_t blabla_uint32(void *st) noexcept nogil:
    return blabla_next32(<blabla_state_t *> st)

cdef double blabla_double(void* st) noexcept nogil:
    return blabla_next_double(<blabla_state_t *>st)

cdef class BlaBla(BitGenerator):
    """
    BlaBla(seed=None, *, counter=None, key=None, rounds=10)

    Container for the BlaBla family of counter pseudo-random number generators

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**256), an array of 4 uint64 values,
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then a new SeedSequence is created usign data is read
        from ``/dev/urandom`` .
    counter : {None, int, array_like[uint64]}, optional
        Counter to use in the BlaBla state. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        If not provided, the counter is initialized at 0.
    key : {None, int, array_like[uint64]}, optional
        Key to use in the BlaBla state. Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int in [0, 2**256) or a 4-element uint64 array.
        key and seed cannot both be used.
    rounds : int, optional
        Number of rounds to run the BlaBla mixer. Must be an even integer.
        The standard number of rounds in 10.

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
    BlaBla is a 64-bit PRNG that uses a counter-based design based on
    the Blake2b hash function [1]_. The idea was first implemented in
    Swift in [2]_, and later reimplemented in C++ in [3]_. The C
    implementation is dervied from the C++ version. Instances using
    different values of the key produce distinct sequences. ``BlaBla`` has
    a period of :math:`2^{128}` and supports arbitrary advancing and
    jumping the sequence in increments of :math:`2^{64}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``BlaBla`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``AESCounter`` and ``ChaCha`` for related counter-based PRNGs.

    **State and Seeding**

    The ``BlaBla`` state vector consists of a 16-element array of uint64s
    that capture buffered draws from the distribution, an 4-element array of
    uint64s holding the seed, and an 2-element array of uint64 that holds the
    counter ([low, high]). The elements of the seed are the value provided by
    the user (or from the entropy pool). The final value rounds contains the
    number of rounds used (typically 10). More rounds can be used to improve
    security (12-16).

    ``BlaBla`` is seeded using either a single 256-bit unsigned integer
    or a vector of 4 64-bit unsigned integers. In either case, the seed is
    used as an input for a second random number generator provided by a
    SeedSequence, and the output of this PRNG function is used as the initial
    state. Using a single 64-bit value for the seed can only initialize a small
    range of the possible initial state values.

    **Parallel Features**

    ``BlaBla`` can be used in parallel applications by calling the ``jumped``
    method  to advances the state as-if :math:`2^{64}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**128). When using ``jumped``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import BlaBla
    # Advance each BlaBla instances by i jumps
    >>> rg = [Generator(BlaBla(1234).jumped(i)) for i in range(10)]

    Alternatively, ``BlaBla`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**93 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(BlaBla(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``BlaBla`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import BlaBla
    >>> rg = Generator(BlaBla(1234))
    >>> rg.standard_normal()
    -0.8632  # random

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/BLAKE_(hash_function)
    .. [2] Aumasson, JP. Swift implrmentation of BlaBla PRNG.
         2017. https://github.com/veorq/blabla/blob/master/BlaBla.swift
    .. [3] Nahdi, Gahtan. BlaBla PRNG. https://github.com/gahtan-syarif/blabla.h
    """
    def __init__(
            self, seed=None, *, counter=None, key=None, rounds=10
    ):
        BitGenerator.__init__(self, seed)
        self.rng_state = <blabla_state_t *>PyArray_malloc_aligned(
            sizeof(blabla_state_t)
        )
        if rounds % 2 != 0 or rounds <= 0:
            raise ValueError("rounds must be even and >= 2")
        self.seed(seed, counter, key)
        self.rng_state.rounds = rounds

        self._bitgen.state = <void *>self.rng_state
        self._bitgen.next_uint64 = &blabla_uint64
        self._bitgen.next_uint32 = &blabla_uint32
        self._bitgen.next_double = &blabla_double
        self._bitgen.next_raw = &blabla_uint64

    def __dealloc__(self):
        if self.rng_state:
            PyArray_free_aligned(self.rng_state)

    def _seed_from_seq(self, counter=None):
        state = self._get_seed_seq().generate_state(4, np.uint64)
        self.seed(key=state, counter=counter)
        self._reset_state_variables()

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    @property
    def use_avx2(self):
        """
        Toggle use of SIMD

        Parameters
        ----------
        flag : bool
            Flag indicating whether to use SIMD

        Returns
        -------
        flag : bool
            Current flag value

        Raises
        ------
        ValueError
            If SIMD is not supported
        """
        return bool(RANDOMGEN_USE_AVX2)

    @use_avx2.setter
    def use_avx2(self, value):
        capable = blabla_avx2_capable()
        if value and not capable:
            raise ValueError("CPU does not support AVX2")
        blabla_use_avx2(bool(value))

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator

        This method is called when ``BlaBla`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``BlaBla``.

        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**256), an array of 4 uint64 values,
            a SeedSequence instance or ``None`` (the default). If `seed` is
            ``None``, then  data is read from ``/dev/urandom`` (or the Windows
            analog) if available. If unavailable, a hash of the time and
            process ID is used.
        counter : {None, int, array_like[uint64]}, optional
            Counter to use in the BlaBla state. Can be either
            a Python int in [0, 2**128) or a 2-element uint64 array.
            If not provided, the counter is initialized at 0.
        key : {None, int, array_like[uint64]}, optional
            Key to use in the BlaBla state. Unlike seed, which is run
            through another RNG before use, the value in key is directly set.
            Can be either a Python int in [0, 2**256) or a 4-element uint64
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
        if seed is not None and key is not None:
            raise ValueError("seed and key cannot be simultaneously used")
        if key is None:
            BitGenerator._seed_with_seed_sequence(self, seed, counter=counter)
            return

        key = object_to_int(key, 256, "key")
        counter = object_to_int(counter, 128, "counter")
        seed = int_to_array(key, "key", 256, 64)
        _seed = seed
        if _seed.dtype != np.uint64:
            _seed = view_little_endian(_seed, np.uint64)
        _stream = _seed[2:]
        counter = 0 if counter is None else counter
        _counter = int_to_array(counter, "counter", 128, 64)

        blabla_seed(self.rng_state,
                    <uint64_t *>np.PyArray_DATA(_seed),
                    <uint64_t *>np.PyArray_DATA(_stream),
                    <uint64_t *>np.PyArray_DATA(_counter))
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
        cdef int i
        block = np.empty(16, dtype=np.uint64)
        keysetup = np.empty(4, dtype=np.uint64)
        block_idx = np.empty(2, dtype=np.uint64)
        ctr = np.empty(2, dtype=np.uint64)

        for i in range(16):
            block[i] = self.rng_state.block[i]
        for i in range(4):
            keysetup[i] = self.rng_state.keysetup[i]
        for i in range(2):
            block_idx[i] = self.rng_state.block_idx[i]
        for i in range(2):
            ctr[i] = self.rng_state.ctr[i]

        return {
            "bit_generator": fully_qualified_name(self),
            "state": {
                "block": block,
                "keysetup": keysetup,
                "block_idx": block_idx,
                "ctr": ctr,
                "rounds": self.rng_state.rounds,
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger
            }
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))

        state = value["state"]
        block = state["block"]
        keysetup = state["keysetup"]
        block_idx = state["block_idx"]
        ctr = state["ctr"]
        for i in range(16):
            self.rng_state.block[i] = block[i]
        for i in range(4):
            self.rng_state.keysetup[i] = keysetup[i]
        for i in range(2):
            self.rng_state.block_idx[i] = block_idx[i]
        for i in range(2):
            self.rng_state.ctr[i] = ctr[i]
        self.rng_state.rounds = state["rounds"]
        self.rng_state.has_uint32 = state["has_uint32"]
        self.rng_state.uinteger = state["uinteger"]

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
        bit_generator : BlaBla
            New instance of generator jumped iter times
        """
        cdef BlaBla bit_generator

        bit_generator = self.__class__(seed=self._copy_seed())
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

        return bit_generator

    def advance(self, delta):
        """
        Advance the state by delta steps.

        Parameters
        ----------
        delta : int
            Number of steps to advance the state. Delta can be any integer value,
            but is wrapped to be in [0, 2**128) which is the size of the counter.
        """
        # Squeeze with wrap into [0, 2**128)
        delta = delta % (1 << 128)
        cdef uint64_t d[2]
        d[0] = delta & 0xFFFFFFFFFFFFFFFF
        d[1] = <uint64_t>(delta >> 64)
        blabla_advance(self.rng_state, d)
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
