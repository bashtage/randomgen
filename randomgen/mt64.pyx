#!python

# coding=utf-8
import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["MT64"]

cdef uint64_t mt64_uint64(void *st) noexcept nogil:
    return mt64_next64(<mt64_state_t *> st)

cdef uint32_t mt64_uint32(void *st) noexcept nogil:
    return mt64_next32(<mt64_state_t *> st)

cdef double mt64_double(void *st) noexcept nogil:
    return uint64_to_double(mt64_next64(<mt64_state_t *> st))

cdef uint64_t mt64_raw(void *st) noexcept nogil:
    return mt64_next64(<mt64_state_t *> st)

cdef class MT64(BitGenerator):
    """
    MT64(seed=None, *, mode="sequence")

    Container for the 64-bit Mersenne Twister pseudo-random number generator

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Random seed used to initialize the pseudo-random number generator. Can
        be any integer between 0 and 2**64 - 1 inclusive, an array (or other
        sequence) of unsigned 64-bit integers, a SeedSequence instance or
        ``None`` (the default). If `seed` is ``None``, then 312 64-bit
        unsigned integers are read from ``/dev/urandom`` (or the Windows
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
    ``MT64`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers ([1]_, [2]_). These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``MT64`` state vector consists of a 312-element array of
    64-bit unsigned integers plus a single integer value between 0 and 312
    that indexes the current position within the main array.

    ``MT64`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers. In either case, the input seed is
    used as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 64-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Compatibility Guarantee**

    ``MT64`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    References
    ----------
    .. [1]  Matsumoto, M.; Nishimura, T. (1998). "Mersenne twister: a
        623-dimensionally equidistributed uniform pseudo-random number
        generator". ACM Transactions on Modeling and Computer Simulation.
        8 (1): 3–30.
    .. [2] Nishimura, T. "Tables of 64-bit Mersenne Twisters" ACM Transactions
        on Modeling and Computer Simulation 10. (2000) 348-357.
    """
    def __init__(self, seed=None, *, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        self.seed(seed)

        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint64 = &mt64_uint64
        self._bitgen.next_uint32 = &mt64_uint32
        self._bitgen.next_double = &mt64_double
        self._bitgen.next_raw = &mt64_raw

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        state = self._get_seed_seq().generate_state(312, np.uint64)
        mt64_init_by_array(&self.rng_state,
                           <uint64_t*>np.PyArray_DATA(state),
                           312)
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Random seed used to initialize the pseudo-random number generator. Can
            be any integer between 0 and 2**64 - 1 inclusive, an array (or other
            sequence) of unsigned 64-bit integers, a SeedSequence instance or
            ``None`` (the default). If `seed` is ``None``, then 312 64-bit
            unsigned integers are read from ``/dev/urandom`` (or the Windows
            analog) if available. If unavailable, a hash of the time and process
            ID is used.

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
        key = np.empty(312, dtype=np.uint64)
        for i in range(312):
            key[i] = self.rng_state.mt[i]

        return {"bit_generator": fully_qualified_name(self),
                "state": {"key": key, "pos": self.rng_state.mti},
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
        key = check_state_array(value["state"]["key"], 312, 64, "key")
        for i in range(312):
            self.rng_state.mt[i] = key[i]
        self.rng_state.mti = value["state"]["pos"]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
