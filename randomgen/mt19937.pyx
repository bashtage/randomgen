# coding=utf-8
import operator

import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.entropy import random_entropy

__all__ = ["MT19937"]

cdef uint64_t mt19937_uint64(void *st) nogil:
    return mt19937_next64(<mt19937_state_t *> st)

cdef uint32_t mt19937_uint32(void *st) nogil:
    return mt19937_next32(<mt19937_state_t *> st)

cdef double mt19937_double(void *st) nogil:
    return mt19937_next_double(<mt19937_state_t *> st)

cdef uint64_t mt19937_raw(void *st) nogil:
    return <uint64_t>mt19937_next32(<mt19937_state_t *> st)

cdef class MT19937(BitGenerator):
    u"""
    MT19937(seed=None, *, mode=None)

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

    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import Generator, MT19937
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
    def __init__(self, seed=None, *, mode=None):
        BitGenerator.__init__(self, seed, mode)
        self.seed(seed)
        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint64 = &mt19937_uint64
        self._bitgen.next_uint32 = &mt19937_uint32
        self._bitgen.next_double = &mt19937_double
        self._bitgen.next_raw = &mt19937_raw

    def _seed_from_seq(self):
        state = self.seed_seq.generate_state(624, np.uint32)
        mt19937_init_by_array(&self.rng_state,
                              <uint32_t*>np.PyArray_DATA(state),
                              624)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

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
        cdef np.ndarray obj

        BitGenerator._seed_with_seed_sequence(self, seed)
        if self.seed_seq is not None:
            return

        try:
            if seed is None:
                seed = random_entropy(624, "auto")
                mt19937_init_by_array(&self.rng_state,
                                      <uint32_t*>np.PyArray_DATA(seed),
                                      624)
            else:
                if hasattr(seed, "squeeze"):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                mt19937_seed(&self.rng_state, seed)
        except TypeError:
            obj = np.asarray(seed)
            if obj.size == 0:
                raise ValueError("Seed must be non-empty")
            obj = obj.astype(np.int64, casting="safe")
            if np.PyArray_NDIM(obj) != 1:
                raise ValueError("Seed array must be 1-d")
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            obj = obj.astype(np.uint32, casting="unsafe", order="C")
            mt19937_init_by_array(&self.rng_state,
                                  <uint32_t*>np.PyArray_DATA(obj),
                                  <int>np.PyArray_DIM(obj, 0))

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
        """
        cdef MT19937 bit_generator

        bit_generator = self.__class__(mode=self.mode)
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
        key = np.zeros(624, dtype=np.uint32)
        for i in range(624):
            key[i] = self.rng_state.key[i]

        return {"bit_generator": self.__class__.__name__,
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
        if bitgen != self.__class__.__name__:
            raise ValueError("state must be for a {0} "
                             "PRNG".format(self.__class__.__name__))
        key = check_state_array(value["state"]["key"], 624, 32, "key")
        for i in range(624):
            self.rng_state.key[i] = key[i]
        self.rng_state.pos = value["state"]["pos"]
