#!python

# coding=utf-8
import numpy as np

__all__ = ["LXM"]

cdef uint64_t lxm_uint64(void* st) noexcept nogil:
    return lxm_next64(<lxm_state_t *>st)

cdef uint32_t lxm_uint32(void *st) noexcept nogil:
    return lxm_next32(<lxm_state_t *> st)

cdef double lxm_double(void* st) noexcept nogil:
    return uint64_to_double(lxm_next64(<lxm_state_t *>st))

cdef class LXM(BitGenerator):
    r"""
    LXM(seed=None, *, b=3037000493)

    Container for the LXM pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
    b : uint64
        The additive constant in the LCG update. Must be odd, and so 1 is
        added if even. The default value is 3037000493.

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
    The LXM generator combines two simple generators with an optional
    additional hashing of the output. This generator has been proposed
    for inclusion in Java ([1]_). The first generator is a LCG of with
    and update

    .. math::

       s = a s + b \mod 2^{64}

    where a is 2862933555777941757 and b is settable. The default value of
    b is 3037000493 ([5]_). The second is the standard 64-bit xorshift
    generator ([2]_, [3]_). The output of these two is combined using
    addition. This sum is hashed using the Murmur3 hash function using the
    parameters suggested by David Stafford ([4]_). Is pseudo-code, each
    value is computed as Mix(LCG + Xorshift). While the origins of LXM are
    not clear from ([1]_), it appears to be derived from LCG Xorshift Mix.

    ``LXM`` provides a capsule containing function pointers that
    produce doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``LXM`` state vector consists of a 4-element array of 64-bit
    unsigned integers that constraint he state of the Xorshift generator
    and an addition 64-bit unsigned integer that holds the state of the
    LCG.

    The seed value is used to create a ``SeedSequence`` which is then used
    to set the initial state.

    **Parallel Features**

    ``LXM`` can be used in parallel applications by calling the
    method ``jumped`` which provides a new instance with a state that has
    been updated as-if :math:`2^{128}` random numbers have been generated.
    This allows the original sequence to be split so that distinct segments
    can be used in each worker process. All generators should be initialized
    with the same seed to ensure that the segments come from the same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import LXM
    >>> rg = [Generator(LXM(1234))]
    # Advance each LXM instance by i jumps
    >>> for i in range(10):
    ...     rg.append(rg[-1].jumped())

    Note that jumped states only alters the Xorshift state since the jump is a
    full cycle of the LCG.

    **Compatibility Guarantee**

    ``LXM`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import LXM
    >>> rg = Generator(LXM(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Guy Steele (2019, June 21). JEP 356: Enhanced Pseudo-Random
           Number Generators. Retrieved June 01, 2020, from
           https://openjdk.java.net/jeps/356.
    .. [2] Marsaglia, George. "Xorshift RNGs." Journal of Statistical Software
           [Online], 8.14, pp. 1 - 6, 2003.
    .. [3] "xorshift*/xorshift+ generators and the PRNG shootout",
           https://prng.di.unimi.it/.
    .. [4] David Stafford (2011, September 28). Better Bit Mixing - Improving
           on MurmurHash3's 64-bit Finalizer.  Retrieved June 01, 2020, from
           https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html.
    .. [5] Bret R. Beck and Eugene D. Brooks III (2000, December 8). 64-bit
           Linear Congruential Generator. Retrieved June 01, 2020, from
           https://nuclear.llnl.gov/CNP/rng/rngman/node4.html.
    """
    def __init__(self, seed=None, *, b=3037000493):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self.rng_state.b = <uint64_t>b | 1
        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &lxm_uint64
        self._bitgen.next_uint32 = &lxm_uint32
        self._bitgen.next_double = &lxm_double
        self._bitgen.next_raw = &lxm_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef int i
        cdef uint64_t bits = 0

        # Protect against negligible prob of all 0 in Xorshift
        while bits == 0:
            state = self._get_seed_seq().generate_state(5, np.uint64)
            for i in range(4):
                self.rng_state.x[i] = state[i]
                bits |= <uint64_t>state[i]
            self.rng_state.lcg_state = state[4]
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called when ``LXM`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``LXM``.

        Parameters
        ----------
        seed : {None, int, array_like[int], SeedSequence}, optional
            Entropy initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64), array of integers in
            [0, 2**64), a SeedSequence instance or ``None`` (the default).
            The input is passed to SeedSequence which produces the values
            used to initialize the state.

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
            lxm_jump(&self.rng_state)
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : LXM
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
        2**(128 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : LXM
            New instance of generator jumped iter times
        """
        cdef LXM bit_generator

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
        x = np.empty(4, dtype=np.uint64)
        for i in range(4):
            x[i] = self.rng_state.x[i]
        return {"bit_generator": fully_qualified_name(self),
                "state": {"x": x,
                          "lcg_state": self.rng_state.lcg_state,
                          "b": self.rng_state.b,
                          },
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
        state = check_state_array(value["state"]["x"], 4, 64, "x")
        for i in range(4):
            self.rng_state.x[i] = <uint64_t>state[i]
        self.rng_state.lcg_state = value["state"]["lcg_state"]
        self.rng_state.b = value["state"]["b"]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
