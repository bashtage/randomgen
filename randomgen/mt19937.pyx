#!python
# coding=utf-8

import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["MT19937"]

cdef uint64_t mt19937_uint64(void *st) noexcept nogil:
    return mt19937_next64(<mt19937_state_t *> st)

cdef uint32_t mt19937_uint32(void *st) noexcept nogil:
    return mt19937_next32(<mt19937_state_t *> st)

cdef double mt19937_double(void *st) noexcept nogil:
    return mt19937_next_double(<mt19937_state_t *> st)

cdef uint64_t mt19937_raw(void *st) noexcept nogil:
    return <uint64_t>mt19937_next32(<mt19937_state_t *> st)

cdef class MT19937(BitGenerator):
    """
    MT19937(seed=None, *, numpy_seed=False, mode="sequence")

    Container for the Mersenne Twister pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[uint32], SeedSequence}, optional
        Random seed used to initialize the pseudo-random number generator. Can
        be any integer between 0 and 2**32 - 1 inclusive, an array (or other
        sequence) of unsigned 32-bit integers, a SeedSequence instance or
        ``None`` (the default). If `seed` is ``None``, then 624 32-bit
        unsigned integers are read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
    numpy_seed : bool
        Set to True to use  the same seeding mechanism as NumPy and
        so matches NumPy exactly.

        .. versionadded: 2.0.0

    mode : {None, "sequence", "numpy"}, optional
        "sequence" uses a SeedSequence to transforms the seed into an initial
        state. None defaults to "sequence". "numpy" uses the same seeding
        mechanism as NumPy and so matches NumPy exactly.

        .. deprecated: 2.0.0
           mode is deprecated. Use numpy_seed tp enforce numpy-matching seeding

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
    ``MT19937`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers [1]_. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``MT19937`` state vector consists of a 768-element array of
    32-bit unsigned integers plus a single integer value between 0 and 768
    that indexes the current position within the main array.

    ``MT19937`` is seeded using either a single 32-bit unsigned integer
    or a vector of 32-bit unsigned integers. In either case, the input seed is
    used as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 32-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Parallel Features**

    ``MT19937`` can be used in parallel applications by
    calling the method ``jump`` which advances the state as-if :math:`2^{128}`
    random numbers have been generated ([1]_, [2]_). This allows the original
    sequence to be split so that distinct segments can be used in each worker
    process. All generators should be initialized with the same seed to ensure
    that the segments come from the same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import MT19937
    >>> seed = random_entropy()
    >>> rs = [Generator(MT19937(seed)) for _ in range(10)]
    # Advance each MT19937 instance by i jumps
    >>> for i in range(10):
    ...     rs[i].bit_generator.jump(i)

    **Compatibility Guarantee**

    ``MT19937`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    References
    ----------
    .. [1] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
        Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
        Sequences and Their Applications - SETA, 290--298, 2008.
    .. [2] Hiroshi Haramoto, Makoto Matsumoto, Takuji Nishimura, François
        Panneton, Pierre L\'Ecuyer, "Efficient Jump Ahead for F2-Linear
        Random Number Generators", INFORMS JOURNAL ON COMPUTING, Vol. 20,
        No. 3, Summer 2008, pp. 385-390.

    """
    def __init__(self, seed=None, *, numpy_seed=False, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, numpy_seed=numpy_seed, mode=mode)
        self.seed(seed)
        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint64 = &mt19937_uint64
        self._bitgen.next_uint32 = &mt19937_uint32
        self._bitgen.next_double = &mt19937_double
        self._bitgen.next_raw = &mt19937_raw

    def _supported_modes(self):
        return "sequence", "numpy"

    def _seed_from_seq(self):
        state = self._get_seed_seq().generate_state(RK_STATE_LEN, np.uint32)
        mt19937_init_by_array(&self.rng_state,
                              <uint32_t*>np.PyArray_DATA(state),
                              RK_STATE_LEN)

    def _seed_from_seq_numpy_compat(self, inc=None):
        # MSB is 1; assuring non-zero initial array
        val = self._get_seed_seq().generate_state(RK_STATE_LEN, np.uint32)
        # MSB is 1; assuring non-zero initial array
        self.rng_state.key[0] = 0x80000000UL
        for i in range(1, RK_STATE_LEN):
            self.rng_state.key[i] = val[i]
        self.rng_state.pos = i

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint32], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array (or
            other sequence) of unsigned 32-bit integers, a SeedSequence
            instance or ``None`` (the default). If `seed` is ``None``, then
            624 32-bit unsigned integers are read from ``/dev/urandom`` (or
            the Windows analog) if available. If unavailable, a hash of the
            time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)

    cdef jump_inplace(self, int jumps):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the rng.
        """
        if jumps < 0:
            raise ValueError("jumps must be positive")
        mt19937_jump_n(&self.rng_state, jumps)

    def _jump_tester(self):
        """
        Private jump testing function

        Returns
        -------
        jumped : MT19937
            A new instance with a jumped state.

        Notes
        -----
        Used the jump polynomial that ships with the jump program.
        """
        cdef MT19937 bit_generator

        kwargs = {"numpy_seed": True} if self.mode == "numpy" else {}
        bit_generator = self.__class__(**kwargs)
        bit_generator.state = self.state
        mt19937_jump_default(&bit_generator.rng_state)

        return bit_generator

    def jump(self, int jumps=1):
        """
        jump(jumps=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator.

        Returns
        -------
        self : MT19937
            PRNG jumped jumps times
        """
        import warnings
        warnings.warn("jump (in-place) has been deprecated in favor of jumped"
                      ", which returns a new instance", DeprecationWarning)

        self.jump_inplace(jumps)
        return self

    def jumped(self, int jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(128 * jumps) random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : MT19937
            New instance of generator jumped jumps times

        Notes
        -----
        The jump step is computed using a modified version of Matsumoto's
        implementation of Horner's method. The step polynomial is precomputed
        to perform 2**128 steps. The jumped state has been verified to match
        the state produced using Matsumoto's original code. The jump
        implementation is based on code from [1]_ and [2]_.

        References
        ----------
        .. [1] Matsumoto, M, Generating multiple disjoint streams of
           pseudorandom number sequences.  Accessed on: May 6, 2020.
           (online). Available:
           http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/JUMP/

        .. [2] Hiroshi Haramoto, Makoto Matsumoto, Takuji Nishimura, François
           Panneton, Pierre L\'Ecuyer, "Efficient Jump Ahead for F2-Linear
           Random Number Generators", INFORMS JOURNAL ON COMPUTING, Vol. 20,
           No. 3, Summer 2008, pp. 385-390.
        """
        cdef MT19937 bit_generator

        kwargs = {} if self.mode != "numpy" else {"numpy_seed": True}
        bit_generator = self.__class__(seed=self._copy_seed(), **kwargs)
        bit_generator.state = self.state
        bit_generator.jump_inplace(jumps)

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
        key = np.zeros(RK_STATE_LEN, dtype=np.uint32)
        for i in range(RK_STATE_LEN):
            key[i] = self.rng_state.key[i]

        return {"bit_generator": fully_qualified_name(self),
                "state": {"key": key, "pos": self.rng_state.pos}}

    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != "MT19937" or len(value) not in (3, 5):
                raise ValueError("state is not a legacy MT19937 state")
            value ={"bit_generator": "MT19937",
                    "state": {"key": value[1], "pos": value[2]}}

        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        key = check_state_array(value["state"]["key"], RK_STATE_LEN, 32, "key")
        for i in range(RK_STATE_LEN):
            self.rng_state.key[i] = key[i]
        self.rng_state.pos = value["state"]["pos"]
