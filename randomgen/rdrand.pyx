import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ["RDRAND"]

cdef extern from "src/rdrand/rdrand.h":

    struct s_rdrand_state:
        int status

    ctypedef s_rdrand_state rdrand_state

    uint64_t rdrand_next64(rdrand_state* state)  nogil
    uint32_t rdrand_next32(rdrand_state* state)  nogil
    int rdrand_capable()

cdef uint64_t rdrand_uint64(void* st) nogil:
    return rdrand_next64(<rdrand_state*>st)

cdef uint32_t rdrand_uint32(void *st) nogil:
    return rdrand_next32(<rdrand_state*>st)

cdef double rdrand_double(void* st) nogil:
    return uint64_to_double(rdrand_next64(<rdrand_state*>st))

cdef class RDRAND(BitGenerator):
    """
    RDRAND(seed=None)

    Container for the hardware RDRAND random number generator.

    Parameters
    ----------
    seed : None
        Must be None. Raises if any other value is passed.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : None
        Always None since RDRAND cannot be seeded.

    Notes
    -----
    RDRAND is a hardware random number generator that is available on Intel
    processors from the Ivy Bridge line (2012) or later, and AMD processors
    starting in 2015.

    RDRAND has been audited and is reported to be a secure generator. It is
    **much slower** than software BitGenerators and so is only useful in
    applications where security is more important than performance.

    **State and Seeding**

    Results from ``RDRAND`` are **NOT** reproducible.

    ``RDRAND`` uses a hardware generated seed and so cannot be seeded. The
    state contains a single integer value ``status`` that takes the value 1
    if all calls have succeeded and 0 if any fail. A failure of a call to
    RDRAND does not propagate, and so users much check this value to determine
    if results are random.

    **Parallel Features**

    ``RDRAND`` is stateless and so multiple instances can be used in parallel.

    >>> from randomgen import Generator, RDRAND
    >>> rg = [Generator(RDRAND()) for _ in range(10)]

    **No Compatibility Guarantee**

    ``RDRAND`` is hardware dependent and not reproducible, and so there is no
    stream guarantee.

    Raises
    ------
    RuntimeError
        If RDRAND is not available on the platform.

    Examples
    --------
    >>> from randomgen import Generator, RDRAND
    >>> rg = Generator(RDRAND())
    >>> rg.standard_normal()
    0.123  # random
    """
    cdef rdrand_state rng_state

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed, mode="sequence")
        if not rdrand_capable():
            raise RuntimeError("The RDRAND instruction is not available")   # pragma: no cover
        self.rng_state.status = 1
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &rdrand_uint64
        self._bitgen.next_uint32 = &rdrand_uint32
        self._bitgen.next_double = &rdrand_double
        self._bitgen.next_raw = &rdrand_uint64

    def _seed_from_seq(self):
        pass

    def seed(self, seed=None):
        """
        seed(seed=None)

        Parameters
        ----------
        seed : None
            Must be None. Raises if any other value is passed.

        Raises
        ------
        ValueError
            If seed is not None
        """
        if seed is not None:
            raise TypeError("seed cannot be set and so must be None")

    def jumped(self, iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator

        ``RDRAND`` is stateless and so the new generator can be used in a
        parallel applications.

        Parameters
        ----------
        iter : integer, positive
            This parameter is ignored

        Returns
        -------
        bit_generator : RDRAND
            New instance of generator jumped iter times

        Notes
        -----
        Provided for API compatibility
        """
        cdef RDRAND bit_generator
        bit_generator = self.__class__()

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
        return {"bit_generator": self.__class__.__name__,
                "status": self.rng_state.status}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen != self.__class__.__name__:
            raise ValueError("state must be for a {0} "
                             "PRNG".format(self.__class__.__name__))
