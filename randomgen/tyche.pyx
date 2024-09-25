#!python
from multiprocessing.managers import Value

import numpy as np
cimport numpy as np

from randomgen.common cimport *

DEF INDIRECTION_SIZE = 16
DEF ITERATION_SIZE = 32


__all__ = ["Tyche"]

cdef uint64_t tyche_uint64(void* st) noexcept nogil:
    return tyche_next64(<tyche_state_t *>st)

cdef uint32_t tyche_uint32(void *st) noexcept nogil:
    return tyche_next32(<tyche_state_t *> st)

cdef double tyche_double(void* st) noexcept nogil:
    return tyche_next_double(<tyche_state_t *>st)

cdef uint64_t tyche_openrand_uint64(void* st) noexcept nogil:
    return tyche_openrand_next64(<tyche_state_t *>st)

cdef uint32_t tyche_openrand_uint32(void *st) noexcept nogil:
    return tyche_openrand_next32(<tyche_state_t *> st)

cdef double tyche_openrand_double(void* st) noexcept nogil:
    return tyche_openrand_next_double(<tyche_state_t *>st)

cdef class Tyche(BitGenerator):
    """
    EFIIX64(seed=None)

    Container for the EFIIX64x384 pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and
        process ID is used.
    idx : {None, int}, optional
        The index to use when seeding from a SeedSequence. If None, the
        default, the index is selected at random.

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
    ``EFIIX64`` (also known as tychex384) is written by Chris Doty-Humphrey.
    It is a 64-bit PRNG that uses a set of tables generate random values.
    This produces a fast PRNG with statistical quality similar to cryptographic
    generators but faster [1]_.

    ``EFIIX64`` provides a capsule containing function pointers that
    produce doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``EFIIX64`` state vector consists of a 16-element array of 64-bit
    unsigned integers and a 32-element array of 64-bit unsigned integers.
    In addition, 3 constant values and a counter are used in the update.

    ``EFIIX64`` is seeded using an integer, a sequence of integer or a
    SeedSequence.  If the seed is not SeedSequence, the seed values are
    passed to a SeedSequence which is then used to produce 4 64-bit unsigned
    integer values which are used to Seed the generator

    **Compatibility Guarantee**

    ``EFIIX64`` makes a guarantee that a fixed seed will always
    produce the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import EFIIX64
    >>> rg = Generator(EFIIX64(1234))
    >>> rg.standard_normal()
    0.123  # random

    **Parallel Features**

    ``EFIIX64`` can be used in parallel when combined with a ``SeedSequence``
    using ``spawn``.

    >>> from randomgen import SeedSequence
    >>> ss = SeedSequence(8509285875904376097169743623867)
    >>> bit_gens = [EFIIX64(child) for child in ss.spawn(1024)]

    References
    ----------
    .. [1] Neves, S., Araujo, F. (2012). Fast and Small Nonlinear Pseudorandom
       Number Generators for Computer Simulation. In: Wyrzykowski, R., Dongarra,
       J., Karczewski, K., Waśniewski, J. (eds) Parallel Processing and Applied
       Mathematics. PPAM 2011. Lecture Notes in Computer Science, vol 7203.
       Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-31464-3_10


    """

    def __init__(self, seed=None, *, idx=None, original=True):
        BitGenerator.__init__(self, seed)
        self.original = bool(original)
        self.seed(seed, idx=idx)

        self._bitgen.state = <void *>&self.rng_state
        if original:
            self._bitgen.next_uint64 = &tyche_uint64
            self._bitgen.next_uint32 = &tyche_uint32
            self._bitgen.next_double = &tyche_double
            self._bitgen.next_raw = &tyche_uint64
        else:
            self._bitgen.next_uint64 = &tyche_openrand_uint64
            self._bitgen.next_uint32 = &tyche_openrand_uint32
            self._bitgen.next_double = &tyche_openrand_double
            self._bitgen.next_raw = &tyche_openrand_uint64

    def _seed_from_seq(self, idx=None):
        cdef uint64_t state
        cdef uint32_t _idx

        try:
            seed_seq = self.seed_seq
        except AttributeError:
            seed_seq = self._seed_seq
        state = seed_seq.generate_state(1, np.uint64)[0]
        if idx is None:
            _idx = seed_seq.generate_state(1, np.uint32)[0]
        else:
            if not 0 <= idx <= np.iinfo(np.uint32).max:
                raise ValueError("idx must be in the interval [0, 2**32).")
            _idx = <uint32_t>idx
        tyche_seed(&self.rng_state, state, _idx, <int>self.original)

    def seed(self, seed=None, *, idx=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called at initialization. It can be called again to
        re-Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Entropy initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64), array of integers in
            [0, 2**64), a SeedSequence instance or ``None`` (the default).
            If `seed` is ``None``, then  data is read from ``/dev/urandom``
            (or the Windows analog) if available. If unavailable, a hash
            of the time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed, idx=idx)

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
        state = {"a": self.rng_state.a,
                 "b": self.rng_state.b,
                 "c": self.rng_state.c,
                 "d": self.rng_state.d,
                 "original": self.original}
        return {"bit_generator": fully_qualified_name(self),
                "state": state}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = value["state"]
        if bool(self.original) != bool(state["original"]):
            raise ValueError("state must be for the same variant of the PRNG")
        self.rng_state.a = state["a"]
        self.rng_state.b = state["b"]
        self.rng_state.c = state["c"]
        self.rng_state.d = state["d"]

