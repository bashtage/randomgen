import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ["SFC64"]

cdef uint64_t sfc_uint64(void* st) nogil:
    return sfc_next64(<sfc_state_t *>st)

cdef uint32_t sfc_uint32(void *st) nogil:
    return sfc_next32(<sfc_state_t *> st)

cdef double sfc_double(void* st) nogil:
    return uint64_to_double(sfc_next64(<sfc_state_t *>st))

cdef class SFC64(BitGenerator):
    """
    SFC64(seed=None, k=None)

    Chris Doty-Humphrey's Small Fast Chaotic PRNG with optional Weyl Sequence

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
    w : {uint64, None}, default 1
        The starting value of the Weyl sequence. If None, then the initial value
        is generated from the `SeedSequence`.
    k : {uint64, None}, default 1
        The increment to the Weyl sequence. Must be odd, and if even, 1 is added.
        If None, then `k` `is generated from the `SeedSequence`.

    Notes
    -----
    ``SFC64`` is a 256-bit implementation of Chris Doty-Humphrey's Small Fast
    Chaotic PRNG ([1]_). ``SFC64`` has a few different cycles that one might be
    on, depending on the seed; the expected period will be about
    :math:`2^{255}` ([2]_). ``SFC64`` incorporates a 64-bit counter which means
    that the absolute minimum cycle length is :math:`2^{64}` and that distinct
    seeds will not run into each other for at least :math:`2^{64}` iterations.
    ``SFC64`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``SFC64`` state vector consists of 4 unsigned 64-bit values. The last
    is a 64-bit counter that increments by 1 each iteration.
    The input seed is processed by `SeedSequence` to generate the first
    3 values, then the ``SFC64`` algorithm is iterated a small number of times
    to mix.

    **Compatibility Guarantee**

    ``SFC64`` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    References
    ----------
    .. [1] "PractRand". http://pracrand.sourceforge.net/RNG_engines.txt.
    .. [2] "Random Invertible Mapping Statistics".
       http://www.pcg-random.org/posts/random-invertible-mapping-statistics.html.
    """
    _seed_seq_len = 4
    _seed_seq_dtype = np.uint64

    def __init__(self, seed=None, w=1, k=1):
        BitGenerator.__init__(self, seed, "sequence")
        self.w = k
        self.k = k
        self.seed(seed)


        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &sfc_uint64
        self._bitgen.next_uint32 = &sfc_uint32
        self._bitgen.next_double = &sfc_double
        self._bitgen.next_raw = &sfc_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef int i, loc, cnt
        cdef uint64_t *state_arr
        cdef uint64_t k, w
        cnt = 3 + (self.k is None) + (self.w is None)

        state = self.seed_seq.generate_state(cnt, np.uint64)
        state_arr = <np.uint64_t *>np.PyArray_DATA(state)
        w = self.w if self.w is not None else state[3]
        loc = 3 if self.w is not None else 4
        k = self.k if self.k is not None else (<uint64_t>state[loc] | 0x1)
        sfc_seed(&self.rng_state, state_arr, w, k)
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called at initialization. It can be called again to
        re-seed the generator.

        seed : {None, int, array_like[ints], SeedSequence}, optional
            A seed to initialize the `BitGenerator`. If None, then fresh,
            unpredictable entropy will be pulled from the OS. If an ``int`` or
            ``array_like[ints]`` is passed, then it will be passed to
            `SeedSequence` to derive the initial `BitGenerator` state. One may also
            pass in a `SeedSequence` instance.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)

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
        return {"bit_generator": type(self).__name__,
                "state": {"a": self.rng_state.a,
                          "b":self.rng_state.b,
                          "c":self.rng_state.c,
                          "w":self.rng_state.w,
                          "k":self.rng_state.k
                          },
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != type(self).__name__:
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        self.rng_state.a = value["state"]["a"]
        self.rng_state.b = value["state"]["b"]
        self.rng_state.c = value["state"]["c"]
        self.rng_state.w = value["state"]["w"]
        self.rng_state.k = value["state"]["k"]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
