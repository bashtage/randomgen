import operator

import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.entropy import random_entropy

__all__ = ["SFMT"]

DEF SFMT_MEXP = 19937
DEF SFMT_N = 156  # SFMT_MEXP / 128 + 1
DEF SFMT_N64 = SFMT_N * 2

cdef uint64_t sfmt_uint64(void* st) nogil:
    return sfmt_next64(<sfmt_state_t *>st)

cdef uint32_t sfmt_uint32(void *st) nogil:
    return sfmt_next32(<sfmt_state_t *> st)

cdef uint64_t sfmt_raw(void *st) nogil:
    return sfmt_next64(<sfmt_state_t *>st)

cdef double sfmt_double(void* st) nogil:
    return uint64_to_double(sfmt_next64(<sfmt_state_t *>st))


cdef class SFMT(BitGenerator):
    u"""
    SFMT(seed=None, *, mode=None)

    Container for the SIMD-based Mersenne Twister pseudo RNG.

    Parameters
    ----------
    seed : {None, int, array_like[uint32], SeedSequence}, optional
        Entropy used to initialize the pseudo-random number generator. Can
        be any integer between 0 and 2**32 - 1 inclusive, an array (or other
        sequence) of unsigned 32-bit integers, , a SeedSequence instance or
        ``None`` (the default). If `seed` is ``None``, the 624 32-bit
        unsigned integers are read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
    mode : {None, "sequence", "legacy"}, optional
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
    ``SFMT`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers [1]_ . These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``SFMT`` state vector consists of a 384 element array of 64-bit
    unsigned integers plus a single integer value between 0 and 382
    indicating the current position within the main array. The implementation
    used here augments this with a 382 element array of doubles which are used
    to efficiently access the random numbers produced by the SFMT generator.

    ``SFMT`` is seeded using either a single 32-bit unsigned integer or a
    vector of 32-bit unsigned integers. In either case, the input seed is used
    as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 32-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Parallel Features**

    ``SFMT`` can be used in parallel applications by calling the method
    ``jump`` which advances the state as-if :math:`2^{128}` random numbers
    have been generated [2]_. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be initialized with the same seed to ensure that
    the segments come from the same sequence.

    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import Generator, SFMT
    >>> seed = random_entropy()
    >>> rs = [Generator(SFMT(seed)) for _ in range(10)]
    # Advance each SFMT instance by i jumps
    >>> for i in range(10):
    ...     rs[i].bit_generator.jump()

    **Compatibility Guarantee**

    ``SFMT`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    References
    ----------
    .. [1] Mutsuo Saito and Makoto Matsumoto, "SIMD-oriented Fast Mersenne
           Twister: a 128-bit Pseudorandom Number Generator." Monte Carlo
           and Quasi-Monte Carlo Methods 2006, Springer, pp. 607--622, 2008.
    .. [2] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L'Ecuyer, "A Fast
           Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
           Sequences and Their Applications - SETA, 290--298, 2008.
    """
    def __init__(self, seed=None, *, mode=None):
        BitGenerator.__init__(self, seed, mode)
        self.rng_state.state = <sfmt_t *>PyArray_malloc_aligned(sizeof(sfmt_t))
        self.rng_state.buffered_uint64 = <uint64_t *>PyArray_calloc_aligned(SFMT_N64, sizeof(uint64_t))
        self.rng_state.buffer_loc = SFMT_N64
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &sfmt_uint64
        self._bitgen.next_uint32 = &sfmt_uint32
        self._bitgen.next_double = &sfmt_double
        self._bitgen.next_raw = &sfmt_raw

    def __dealloc__(self):
        if self.rng_state.state:
            PyArray_free_aligned(self.rng_state.state)
        if self.rng_state.buffered_uint64:
            PyArray_free_aligned(self.rng_state.buffered_uint64)

    cdef _reset_state_variables(self):
        self.rng_state.buffer_loc = SFMT_N64
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        state = self.seed_seq.generate_state(2 * SFMT_N64, np.uint32)

        sfmt_init_by_array(self.rng_state.state,
                           <uint32_t *>np.PyArray_DATA(state),
                           2 * SFMT_N64)
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like[uint32], SeedSequence}, optional
            Entropy used to initialize the pseudo-random number generator. Can
            be any integer between 0 and 2**32 - 1 inclusive, an array (or
            other sequence) of unsigned 32-bit integers, , a SeedSequence
            instance or ``None`` (the default). If `seed` is ``None``, the
            624 32-bit unsigned integers are read from ``/dev/urandom`` (or
            the Windows analog) if available. If unavailable, a hash of the
            time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        cdef np.ndarray obj, seed_arr

        BitGenerator._seed_with_seed_sequence(self, seed)
        if self.seed_seq is not None:
            return

        try:
            if seed is None:
                seed_arr = random_entropy(2 * SFMT_N64, "auto")
                sfmt_init_by_array(self.rng_state.state,
                                   <uint32_t *>np.PyArray_DATA(seed_arr),
                                   2 * SFMT_N64)
            else:
                if hasattr(seed, "squeeze"):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                sfmt_init_gen_rand(self.rng_state.state, seed)
        except TypeError:
            obj = np.asarray(seed).astype(np.int64, casting="safe").ravel()
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            seed_arr = obj.astype(np.uint32, casting="unsafe", order="C")
            sfmt_init_by_array(self.rng_state.state,
                               <uint32_t *>np.PyArray_DATA(seed_arr),
                               <int>np.PyArray_DIM(seed_arr, 0))
        # Clear the buffer
        self._reset_state_variables()

    cdef jump_inplace(self, object iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        if iter < 0:
            raise ValueError("iter must be positive")
        sfmt_jump_n(&self.rng_state, iter)
        # Clear the buffer
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator.

        Returns
        -------
        self : SFMT
            PRNG jumped iter times
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
        bit_generator : SFMT
            New instance of generator jumped iter times
        """
        cdef SFMT bit_generator

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

        cdef Py_ssize_t i, j, loc = 0
        cdef uint64_t[::1] state
        cdef uint64_t[::1] buffered_uint64

        state = np.empty(SFMT_N64, dtype=np.uint64)
        for i in range(SFMT_N):
            for j in range(2):
                state[loc] = self.rng_state.state.state[i].u64[j]
                loc += 1
        buffered_uint64 = np.empty(SFMT_N64, dtype=np.uint64)
        for i in range(SFMT_N64):
            buffered_uint64[i] = self.rng_state.buffered_uint64[i]
        return {"bit_generator": self.__class__.__name__,
                "state": {"state": np.asarray(state),
                          "idx": self.rng_state.state.idx},
                "buffer_loc": self.rng_state.buffer_loc,
                "buffered_uint64": np.asarray(buffered_uint64),
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef Py_ssize_t i, j, loc = 0
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != self.__class__.__name__:
            raise ValueError("state must be for a {0} "
                             "PRNG".format(self.__class__.__name__))
        state = check_state_array(value["state"]["state"], 2 * SFMT_N, 64,
                                  "state")
        for i in range(SFMT_N):
            for j in range(2):
                self.rng_state.state.state[i].u64[j] = state[loc]
                loc += 1
        self.rng_state.state.idx = value["state"]["idx"]
        buffered_uint64 = check_state_array(value["buffered_uint64"], SFMT_N64,
                                            64,  "buffered_uint64")
        for i in range(SFMT_N64):
            self.rng_state.buffered_uint64[i] = buffered_uint64[i]
        self.rng_state.buffer_loc = value["buffer_loc"]
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
