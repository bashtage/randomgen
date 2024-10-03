#!python

import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["Xorshift1024"]

cdef uint64_t xorshift1024_uint64(void* st) noexcept nogil:
    return xorshift1024_next64(<xorshift1024_state_t *>st)

cdef uint32_t xorshift1024_uint32(void *st) noexcept nogil:
    return xorshift1024_next32(<xorshift1024_state_t *> st)

cdef double xorshift1024_double(void* st) noexcept nogil:
    return uint64_to_double(xorshift1024_next64(<xorshift1024_state_t *>st))

cdef class Xorshift1024(BitGenerator):
    """
    Xorshift1024(seed=None, *, mode="sequence")

    Container for the xorshift1024*φ pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
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
    xorshift1024*φ is a 64-bit implementation of Saito and Matsumoto's XSadd
    generator [1]_ (see also [2]_, [3]_, [4]_). xorshift1024*φ has a period of
    :math:`2^{1024} - 1` and supports jumping the sequence in increments of
    :math:`2^{512}`, which allows multiple non-overlapping sequences to be
    generated.

    ``Xorshift1024`` provides a capsule containing function pointers that
    produce doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Xoroshiro128`` for a faster bit generator that has a smaller period.

    **State and Seeding**

    The ``Xoroshiro128`` state vector consists of a 16-element array of 64-bit
    unsigned integers.

    ``Xoroshiro1024`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers. In either case, the seed is
    used as an input for another simple random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial
    state. Using a single 64-bit value for the seed can only initialize a
    small range of the possible initial state values.

    **Parallel Features**

    ``Xoroshiro1024`` can be used in parallel applications by calling the
    method ``jump`` which advances the state as-if :math:`2^{512}` random
    numbers have been generated. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be initialized with the same seed to ensure that
    the segments come from the same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import Xorshift1024
    >>> rg = [Generator(Xorshift1024(1234)) for _ in range(10)]
    # Advance each Xorshift1024 instance by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    **Compatibility Guarantee**

    ``Xorshift1024`` makes a guarantee that a fixed seed will always
    produce the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import Xorshift1024
    >>> rg = Generator(Xorshift1024(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] "xorshift*/xorshift+ generators and the PRNG shootout",
           https://prng.di.unimi.it/
    .. [2] Marsaglia, George. "Xorshift RNGs." Journal of Statistical Software
           [Online], 8.14, pp. 1 - 6, 2003.
    .. [3] Sebastiano Vigna. "An experimental exploration of Marsaglia's xorshift
           generators, scrambled." CoRR, abs/1402.6246, 2014.
    .. [4] Sebastiano Vigna. "Further scramblings of Marsaglia's xorshift
           generators." CoRR, abs/1403.0930, 2014.
    """
    def __init__(self, seed=None, *, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &xorshift1024_uint64
        self._bitgen.next_uint32 = &xorshift1024_uint32
        self._bitgen.next_double = &xorshift1024_double
        self._bitgen.next_raw = &xorshift1024_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef int i

        state = self._get_seed_seq().generate_state(16, np.uint64)
        for i in range(16):
            self.rng_state.s[i] = state[i]
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called when ``Xorshift1024`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``Xorshift1024``.

        Parameters
        ----------
        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Entropy initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64), array of integers in
            [0, 2**64), a SeedSequence instance or ``None`` (the default).
            If `seed` is ``None``, then  data is read from ``/dev/urandom``
            (or the Windows analog) if available. If unavailable, a hash of
            the time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)

    cdef jump_inplace(self, np.npy_intp iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        cdef np.npy_intp i
        for i in range(iter):
            xorshift1024_jump(&self.rng_state)
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**512 random numbers have been generated

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : Xorshift1024
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

    def jumped(self, np.npy_intp iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(512 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Xorshift1024
            New instance of generator jumped iter times
        """
        cdef Xorshift1024 bit_generator

        bit_generator = self.__class__(seed=self._copy_seed())
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
        s = np.empty(16, dtype=np.uint64)
        for i in range(16):
            s[i] = self.rng_state.s[i]
        return {"bit_generator": fully_qualified_name(self),
                "state": {"s": s, "p": self.rng_state.p},
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = check_state_array(value["state"]["s"], 16, 64, "s")
        for i in range(16):
            self.rng_state.s[i] = <uint64_t>state[i]
        self.rng_state.p = value["state"]["p"]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
