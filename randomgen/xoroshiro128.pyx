import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ["Xoroshiro128"]

cdef uint64_t xoroshiro128_uint64(void* st) nogil:
    return xoroshiro128_next64(<xoroshiro128_state_t *>st)

cdef uint32_t xoroshiro128_uint32(void *st) nogil:
    return xoroshiro128_next32(<xoroshiro128_state_t *> st)

cdef double xoroshiro128_double(void* st) nogil:
    return uint64_to_double(xoroshiro128_next64(<xoroshiro128_state_t *>st))

cdef class Xoroshiro128(BitGenerator):
    """
    Xoroshiro128(seed=None)

    Container for the xoroshiro128+ pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like, SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64-1], array of integers in [0, 2**64-1],
        a SeedSequence, or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
    mode : {None, "sequence", "legacy"}
        The seeding mode to use. "legacy" uses the legacy
        SplitMix64-based initialization. "sequence" uses a SeedSequence
        to transforms the seed into an initial state. None defaults to "legacy"
        and warns that the default after 1.19 will change to "sequence".

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence. None if mode is "legacy".

    Notes
    -----
    xoroshiro128+ is the successor to xorshift128+ written by David Blackman
    and Sebastiano Vigna. It is a 64-bit PRNG that uses a carefully
    handcrafted shift/rotate-based linear transformation. This change both
    improves speed and statistical quality of the PRNG [1]_. xoroshiro128+ has
    a period of :math:`2^{128} - 1` and supports jumping the sequence in
    increments of :math:`2^{64}`, which allows  multiple non-overlapping
    sequences to be generated.

    ``Xoroshiro128`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Xorshift1024`` for a related PRNG with a larger
    period  (:math:`2^{1024} - 1`) and jump size (:math:`2^{512} - 1`).

    **State and Seeding**

    The ``Xoroshiro128`` state vector consists of a 2-element array of 64-bit
    unsigned integers.

    ``Xoroshiro128`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers. In either case, the seed is
    used as an input for another simple random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial state.
    Using a single 64-bit value for the seed can only initialize a small range of
    the possible initial state values.

    **Parallel Features**

    ``Xoroshiro128`` can be used in parallel applications by calling the method
    ``jump`` which advances the state as-if :math:`2^{64}` random numbers
    have been generated. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be initialized with the same seed to ensure that
    the segments come from the same sequence.

    >>> from randomgen import Generator, Xoroshiro128
    >>> rg = [Generator(Xoroshiro128(1234)) for _ in range(10)]
    # Advance each Xoroshiro128 instance by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    **Compatibility Guarantee**

    ``Xoroshiro128`` makes a guarantee that a fixed seed will always
    produce the same random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, Xoroshiro128
    >>> rg = Generator(Xoroshiro128(1234))
    >>> rg.standard_normal()
    0.123  # random

    Identical method using only Xoroshiro128

    >>> rg = Xoroshiro128(1234).generator
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] "xoroshiro+ / xorshift* / xorshift+ generators and the PRNG shootout",
           http://xorshift.di.unimi.it/
    """
    def __init__(self, seed=None, *, mode=None):
        BitGenerator.__init__(self, seed, mode)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &xoroshiro128_uint64
        self._bitgen.next_uint32 = &xoroshiro128_uint32
        self._bitgen.next_double = &xoroshiro128_double
        self._bitgen.next_raw = &xoroshiro128_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef int i
        cdef uint64_t *state_arr

        state = self.seed_seq.generate_state(2, np.uint64)
        state_arr = <np.uint64_t *>np.PyArray_DATA(state)
        for i in range(2):
            self.rng_state.s[i] = state[i]
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called at initialized. It can be
        called again to re-seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like, SeedSequence}, optional
            Entropy initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64-1], array of integers in
            [0, 2**64-1], a SeedSequence, or ``None`` (the default). If `seed`
            is ``None``, then  data is read from ``/dev/urandom`` (or the
            Windows analog) if available. If unavailable, a hash of the time
            and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)
        if self.seed_seq is not None:
            return
        # Legacy seeding
        ub = 2 ** 64
        if seed is None:
            state = random_entropy(4, "auto")
            state = state.view(np.uint64)
        else:
            state = seed_by_array(seed, 2)
        self.rng_state.s[0] = <uint64_t>int(state[0])
        self.rng_state.s[1] = <uint64_t>int(state[1])
        self._reset_state_variables()

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
            xoroshiro128_jump(&self.rng_state)
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**64 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : Xoroshiro128
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
        2**(64 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Xoroshiro128
            New instance of generator jumped iter times
        """
        cdef Xoroshiro128 bit_generator

        bit_generator = self.__class__(mode=self.mode)
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
        state = np.empty(2, dtype=np.uint64)
        state[0] = self.rng_state.s[0]
        state[1] = self.rng_state.s[1]
        return {"bit_generator": self.__class__.__name__,
                "s": state,
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != self.__class__.__name__:
            raise ValueError("state must be for a {0} "
                             "PRNG".format(self.__class__.__name__))
        state = check_state_array(value["s"], 2, 64, "s")
        self.rng_state.s[0] = <uint64_t>state[0]
        self.rng_state.s[1] = <uint64_t>state[1]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
