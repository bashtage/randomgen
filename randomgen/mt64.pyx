import operator

import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy

__all__ = ['MT64']

cdef extern from "src/mt64/mt64.h":

    struct MT64_T:
        uint64_t mt[312]
        int mti
        int has_uint32
        uint32_t uinteger

    ctypedef MT64_T mt64_t

    uint64_t mt64_next64(mt64_t *state)  nogil
    uint32_t mt64_next32(mt64_t *state)  nogil
    double mt64_next_double(mt64_t *state)  nogil
    void mt64_init_by_array(mt64_t *state, uint64_t *init_key, int key_length)
    void mt64_seed(mt64_t *state, uint64_t seed)

cdef uint64_t mt64_uint64(void *st) nogil:
    return mt64_next64(<mt64_t *> st)

cdef uint32_t mt64_uint32(void *st) nogil:
    return mt64_next32(<mt64_t *> st)

cdef double mt64_double(void *st) nogil:
    return uint64_to_double(mt64_next64(<mt64_t *> st))

cdef uint64_t mt64_raw(void *st) nogil:
    return mt64_next64(<mt64_t *> st)

cdef class MT64(BitGenerator):
    """
    MT64(seed=None)

    Container for the 64-bit Mersenne Twister pseudo-random number generator

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.  Can
        be any integer between 0 and 2**64 - 1 inclusive, an array (or other
        sequence) of unsigned 64-bit integers, or ``None`` (the default).  If
        `seed` is ``None``, a 64-bit unsigned integer is read from
        ``/dev/urandom`` (or the Windows analog) if available. If unavailable,
        a 64-bit hash of the time and process ID is used.

    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    ``MT64`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers [1]_. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``MT64`` state vector consists of a 312-element array of
    64-bit unsigned integers plus a single integer value between 0 and 312
    that indexes the current position within the main array.

    ``MT64`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers.  In either case, the input seed is
    used as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 64-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Compatibility Guarantee**

    ``MT64`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.
    """
    cdef mt64_t rng_state

    def __init__(self, seed=None):
        BitGenerator.__init__(self)
        self.seed(seed)

        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint64 = &mt64_uint64
        self._bitgen.next_uint32 = &mt64_uint32
        self._bitgen.next_double = &mt64_double
        self._bitgen.next_raw = &mt64_raw

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64-1], array of integers in
            [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
            then ``MT64`` will try to read entropy from ``/dev/urandom``
            (or the Windows analog) if available to produce a 64-bit
            seed. If unavailable, a 64-bit hash of the time and process
            ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        cdef np.ndarray obj
        try:
            if seed is None:
                seed = random_entropy(2, 'auto')
                mt64_seed(&self.rng_state, seed.astype(np.uint64)[0])
            else:
                if hasattr(seed, 'squeeze'):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**64 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**64 - 1")
                mt64_seed(&self.rng_state, seed)
        except TypeError:
            obj = np.asarray(seed)
            if obj.size == 0:
                raise ValueError("Seed must be non-empty")
            obj = obj.astype(np.object, casting='safe')
            if np.PyArray_NDIM(obj) != 1:
                raise ValueError("Seed array must be 1-d")
            if ((obj > int(2**64 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**64 - 1")
            for val in obj:
                if np.floor(val) != val:
                    raise ValueError("Seed must contains integers")
            obj = obj.astype(np.uint64, casting='unsafe', order='C')
            mt64_init_by_array(&self.rng_state,
                               <uint64_t*>np.PyArray_DATA(obj),
                               np.PyArray_DIM(obj, 0))

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

        return {'bit_generator': self.__class__.__name__,
                'state': {'key': key, 'pos': self.rng_state.mti},
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = check_state_array(value['state']['key'], 312, 64, 'key')
        for i in range(312):
            self.rng_state.mt[i] = key[i]
        self.rng_state.mti = value['state']['pos']
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
