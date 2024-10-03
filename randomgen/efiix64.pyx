#!python
import numpy as np

DEF INDIRECTION_SIZE = 16
DEF ITERATION_SIZE = 32


__all__ = ["EFIIX64"]

cdef uint64_t efiix64_uint64(void* st) noexcept nogil:
    return efiix64_next64(<efiix64_state_t *>st)

cdef uint32_t efiix64_uint32(void *st) noexcept nogil:
    return efiix64_next32(<efiix64_state_t *> st)

cdef double efiix64_double(void* st) noexcept nogil:
    return uint64_to_double(efiix64_next64(<efiix64_state_t *>st))

cdef class EFIIX64(BitGenerator):
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
    ``EFIIX64`` (also known as efiix64x384) is written by Chris Doty-Humphrey.
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
    .. [1] Random, P., 2020. Practically Random / Discussion / Open Discussion:
       Is Too Low A Chi-Squared Sum Really A Problem?. [online] Sourceforge.net.
       Available at:
       https://sourceforge.net/p/pracrand/discussion/366935/thread/c73ddb7b/#d0fc
       [Accessed 22 June 2020].
    """
    _seed_seq_len = 4
    _seed_seq_dtype = np.uint64

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &efiix64_uint64
        self._bitgen.next_uint32 = &efiix64_uint32
        self._bitgen.next_double = &efiix64_double
        self._bitgen.next_raw = &efiix64_uint64

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef uint64_t *state_arr

        state = self._get_seed_seq().generate_state(4, np.uint64)
        state_arr = <np.uint64_t *>np.PyArray_DATA(state)
        efiix64_seed(&self.rng_state, state_arr)
        self._reset_state_variables()

    def seed(self, seed=None):
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
        cdef Py_ssize_t i
        cdef uint64_t *arr
        indirection_table = np.empty(INDIRECTION_SIZE, dtype=np.uint64)
        arr = <np.uint64_t *>np.PyArray_DATA(indirection_table)
        for i in range(0, INDIRECTION_SIZE):
            arr[i] = self.rng_state.indirection_table[i]

        iteration_table = np.empty(ITERATION_SIZE, dtype=np.uint64)
        arr = <np.uint64_t *>np.PyArray_DATA(iteration_table)
        for i in range(0, ITERATION_SIZE):
            arr[i] = self.rng_state.iteration_table[i]

        state = {"indirection_table": indirection_table,
                 "iteration_table": iteration_table,
                 "i": self.rng_state.i,
                 "a": self.rng_state.a,
                 "b": self.rng_state.b,
                 "c": self.rng_state.c}
        return {"bit_generator": fully_qualified_name(self),
                "state": state,
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        cdef Py_ssize_t i
        cdef uint64_t *arr

        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = value["state"]

        indirection_table = check_state_array(state["indirection_table"],
                                              INDIRECTION_SIZE,
                                              64,
                                              "indirection_table")
        arr = <np.uint64_t *>np.PyArray_DATA(indirection_table)
        for i in range(0, INDIRECTION_SIZE):
            self.rng_state.indirection_table[i] = arr[i]

        iteration_table = check_state_array(state["iteration_table"],
                                            ITERATION_SIZE,
                                            64,
                                            "iteration_table")
        arr = <np.uint64_t *>np.PyArray_DATA(iteration_table)
        for i in range(0, ITERATION_SIZE):
            self.rng_state.iteration_table[i] = arr[i]
        self.rng_state.i = state["i"]
        self.rng_state.a = state["a"]
        self.rng_state.b = state["b"]
        self.rng_state.c = state["c"]

        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
