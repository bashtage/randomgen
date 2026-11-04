import numpy as np

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
    PCG32(seed=None, inc=0)

    Container for the PCG-32 pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, SeedSequence}, optional
        Random seed initializing the pseudo-random number generator. Can be an
        integer in [0, 2**64], a SeedSequence instance or ``None`` (the
        default). If `seed` is ``None``, a new SeedSequence is created using
        entropy from the operating system.
    inc : {None, int}, optional
        The increment in the LCG. Can be an integer in [0, 2**64) or ``None``.
        The default is 0. If `inc` is ``None``, then it is initialized using
        entropy.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to seed the generator. If `seed` is not
        a SeedSequence, one is created from it.

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

    See ``PCG64`` for a similar implementation with a larger period.

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

    Each instance above uses a different increment ``inc``. To split a single
    stream, advance by multiples of an odd step such as
    ``0x9e3779b97f4a7c15``, or use ``jumped``, rather than by a large power of
    two such as ``2**63``, which gives strongly related streams (see
    ``advance``).

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
    def __init__(self, seed=None, inc=None):
        BitGenerator.__init__(self, seed)
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
            Seed for ``PCG32``. Integer between 0 and 2**64-1. If None,
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
            if inc < 0 or inc >= ub or int(np.squeeze(inc)) != inc:
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
        with self.lock:
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
        with self.lock:
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

        Warnings
        --------
        Do not use ``advance`` with a large power of two, such as ``2**63``
        (half of the period), to split a single stream (the same seed and
        ``inc``) into several. The state is an LCG modulo ``2**64``, whose
        lowest ``e`` bits repeat every ``2**e`` steps, so advancing by
        ``2**e`` leaves them unchanged. After ``advance(2**63)``, the two
        states differ only in their top bit at every step, and each output
        is the other with its 16-bit halves swapped and one bit flipped.

        After ``advance(2**32)``, the states differ in about 16 of their 64
        bits and never in the lowest 32. ``advance_step`` returns a step that
        splits a stream safely, and the Notes explain how to choose one.

        See Also
        --------
        advance_step : Step for splitting one stream into several.

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

        To split one stream into several, advance stream ``i`` by ``i * step``
        with an odd ``step``. With an odd step, the states of two streams
        differ in their lowest bit at every step and their other bits differ as
        if at random: about 32 of the 64 bits differ, with the spread of a
        binomial distribution. Only the lowest few bits follow a fixed pattern,
        since in any LCG they repeat every few steps. This holds for an odd
        step of any size. The size sets how far apart the streams are: stream
        ``i`` reaches the start of stream ``i + 1`` after ``step`` draws, so
        ``step`` must be larger than the number of draws that any stream makes.

        A convenient step is the odd integer closest to ``2**XX / phi``, where
        phi is the golden ratio and ``XX`` is chosen so that the step is larger
        than the number of draws per stream. ``advance_step(max_draws)``
        returns this step for the smallest ``XX`` that works, using only
        integer arithmetic. For example, ``XX = 32`` gives ``0x9e3779b9``,
        about ``2.7 * 10**9`` draws per stream, and ``XX = 64`` gives
        ``0x9e3779b97f4a7c15``, the step of ``jumped``. With ``XX = 64`` the
        multiples of the step wrap around the period, and the golden ratio
        spreads the streams evenly around it.

        >>> step = PCG32.advance_step(10**10)
        >>> hex(step)
        '0x278dde6e5'
        >>> streams = [PCG32(1234).advance(i * step) for i in range(8)]

        Streams ``i`` and ``j`` are ``(j - i) * step`` draws apart, so they
        share the lowest ``v`` bits of their states, where ``2**v`` is the
        largest power of two that divides ``j - i``. ``jumped(iter)`` advances
        by ``iter`` times a golden-ratio step, so it is also a good choice when
        ``iter`` is not a large power of two.
        """
        delta = wrap_int(delta, 64)
        with self.lock:
            pcg32_advance_state(&self.rng_state, <uint64_t>delta)
        return self

    @staticmethod
    def advance_step(max_draws):
        """
        advance_step(max_draws)

        Step for splitting one stream into streams of up to max_draws draws

        .. versionadded:: 2.6.0

        Parameters
        ----------
        max_draws : int
            The largest number of draws that any stream makes. Must be at
            least 1 and at most the integer part of ``2**64 / phi``, about
            ``1.1 * 10**19``, where phi is the golden ratio.

        Returns
        -------
        step : int
            The odd integer closest to ``2**XX / phi``, where ``XX`` is the
            smallest integer for which ``2**XX / phi`` is larger than
            ``max_draws``. The step is never smaller than ``max_draws``.

        Raises
        ------
        ValueError
            If ``max_draws`` is outside the range above.
        TypeError
            If ``max_draws`` is not an integer.

        See Also
        --------
        advance : Advance the state as-if delta draws have occurred.

        Notes
        -----
        Advance stream ``i`` by ``i * step`` to split one stream into several.
        The step is odd, so the states of two streams differ as if at random,
        apart from their lowest few bits. The step is at least ``max_draws``,
        so a stream that makes at most ``max_draws`` draws does not reach the
        start of the next stream. Streams ``0`` to ``n - 1`` do not overlap
        when ``n * step <= 2**64``. The Notes of ``advance`` explain this
        choice of step.

        The step is computed with integer arithmetic only, so it is exact for
        any ``max_draws``.

        Examples
        --------
        >>> from randomgen import PCG32
        >>> step = PCG32.advance_step(10**10)
        >>> hex(step)
        '0x278dde6e5'
        >>> streams = [PCG32(1234).advance(i * step) for i in range(8)]
        """
        return golden_advance_step(max_draws, 64)

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
        The step size is the odd integer closest to 2**64 / phi, where phi is
        the Golden Ratio. As the step is odd, the state only returns to its
        starting value after 2**64 jumps.
        """
        step = 0x9e3779b97f4a7c15
        self.advance(iter * step)

    def jumped(self, iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned bit generator is jumped as-if
        iter * 11400714819323198485 random numbers have been generated,
        modulo 2**64.

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
        The step size is the odd integer closest to 2**64 / phi, where phi is
        the Golden Ratio. As the step is odd, the state only returns to its
        starting value after 2**64 jumps.

        .. versionchanged:: 2.6.0
           The step was the integer closest to ``2**64 / phi``, which is even,
           so ``jumped(2**63)`` returned the original state.
        """
        cdef PCG32 bit_generator

        bit_generator = self.__class__(seed=self._copy_seed())
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

        return bit_generator
