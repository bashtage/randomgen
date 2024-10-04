#!python

import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["PCG32"]

cdef uint64_t pcg32_uint64(void* st) noexcept nogil:
    return pcg32_next64(<pcg32_state_t *>st)

cdef uint32_t pcg32_uint32(void *st) noexcept nogil:
    return pcg32_next32(<pcg32_state_t *> st)

cdef double pcg32_double(void* st) noexcept nogil:
    return pcg32_next_double(<pcg32_state_t *>st)

cdef uint64_t pcg32_raw(void* st) noexcept nogil:
    return <uint64_t>pcg32_next32(<pcg32_state_t *> st)


cdef class PCG32(BitGenerator):
    """
    PCG32(seed=None, inc=0, *, mode="sequence")

    Container for the PCG-32 pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, SeedSequence}, optional
        Random seed initializing the pseudo-random number generator. Can be an
        integer in [0, 2**64], a SeedSequence instance or ``None`` (the
        default). If `seed` is ``None``, then ``PCG32`` will try to read data
        from ``/dev/urandom`` (or the Windows analog) if available. If
        unavailable, a 64-bit hash of the time and process ID is used.
    inc : {None, int}, optional
        The increment in the LCG. Can be an integer in [0, 2**64] or ``None``.
        The default is 0. If `inc` is ``None``, then it is initialized using
        entropy.
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
    PCG-32 is a 64-bit implementation of O'Neill's permuted congruential
    generator ([1]_, [2]_). PCG-32 has a period of :math:`2^{64}` and supports
    advancing an arbitrary number of steps.

    ``PCG32`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    Supports the method advance to advance the RNG an arbitrary number of
    steps. The state of the PCG-32 PRNG is represented by 2 64-bit unsigned
    integers.

    See ``PCG64`` for a similar implementation with a smaller period.

    **State and Seeding**

    The ``PCG32`` state vector consists of 2 unsigned 64-bit values.
    ``PCG32`` is seeded using a single 64-bit unsigned integer.

    **Parallel Features**

    ``PCG32`` can be used in parallel applications using ``advance``
    with a different value in  each instance to produce
    non-overlapping sequences.

    >>> from numpy.random import Generator
    >>> from randomgen import PCG32
    >>> rg = [Generator(PCG32(1234, i + 1)) for i in range(10)]
    >>> for i in range(10):
    ...     rg[i].bit_generator.advance(i * 2**32)

    **Compatibility Guarantee**

    ``PCG32`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    References
    ----------
    .. [1] "PCG, A Family of Better Random Number Generators",
           https://www.pcg-random.org/
    .. [2] O'Neill, Melissa E. "PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation"
    """
    def __init__(self, seed=None, inc=None, *, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        self.seed(seed, inc)
        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &pcg32_uint64
        self._bitgen.next_uint32 = &pcg32_uint32
        self._bitgen.next_double = &pcg32_double
        self._bitgen.next_raw = &pcg32_raw

    def _seed_from_seq(self, inc=None):
        cdef uint64_t _inc
        state = self._get_seed_seq().generate_state(2, np.uint64)
        _inc = state[1] if inc is None else <uint64_t>inc
        pcg32_set_seed(&self.rng_state, <uint64_t>state[0], _inc)

    def seed(self, seed=None, inc=None):
        """
        seed(seed=None, inc=None)

        Seed the generator

        This method is called when ``PCG32`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``PCG32``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``PCG64``. Integer between 0 and 2**64-1. If None,
            seeded with entropy.
        inc : int, optional
            The increment in the LCG. Integer between 0 and 2**64-1. If None,
            seeded with entropy.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        ub = 2 ** 64
        if inc is not None:
            err_msg = "inc must be a scalar integer between 0 and " \
                      "{ub}".format(ub=ub)
            if not np.isscalar(inc):
                raise TypeError(err_msg)
            if inc < 0 or inc > ub or int(np.squeeze(inc)) != inc:
                raise ValueError(err_msg)
        BitGenerator._seed_with_seed_sequence(self, seed, inc=inc)

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
        return {"bit_generator": fully_qualified_name(self),
                "state": {"state": self.rng_state.pcg_state.state,
                          "inc": self.rng_state.pcg_state.inc}}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        self.rng_state.pcg_state.state = value["state"]["state"]
        self.rng_state.pcg_state.inc = value["state"]["inc"]

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
        self : PCG32
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
        """
        delta = wrap_int(delta, 64)
        pcg32_advance_state(&self.rng_state, <uint64_t>delta)
        return self

    cdef jump_inplace(self, object iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Notes
        -----
        The step size is phi when divided by the period 2**64
        """
        step = int(0x9e3779b97f4a7c16)
        self.advance(iter * step)

    def jump(self, iter=1):
        """
        jump(iter=1)

        Jump the state a fixed increment

        Jumps the state as-if 11400714819323198486 random numbers have been
        generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : PCG32
            RNG jumped iter times

        Notes
        -----
        The step size is phi when divided by the period 2**64
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
        11400714819323198486 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : PCG32
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi when divided by the period 2**64
        """
        cdef PCG32 bit_generator

        bit_generator = self.__class__(seed=self._copy_seed())
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

        return bit_generator
