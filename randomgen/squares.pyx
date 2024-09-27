#!python
# cython: binding=True, language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from numpy.random.bit_generator import ISeedSequence
from randomgen.common cimport BitGenerator
from randomgen.common cimport *

from libc.stdint cimport (
    uint8_t,
    uint32_t,
    uint64_t,
    intptr_t
)

import numpy as np
cimport numpy as np
from numpy.random import SeedSequence

from randomgen.src.squares.squares import seed_seq

cdef uint64_t squares_uint64(void* st) noexcept nogil:
    return squares_next64(<squares_state_t *>st)

cdef uint32_t squares_uint32(void *st) noexcept nogil:
    return squares_next32(<squares_state_t *> st)

cdef double squares_double(void* st) noexcept nogil:
    return squares_next_double(<squares_state_t *>st)

cdef uint64_t squares_32_uint64(void* st) noexcept nogil:
    return squares_32_next64(<squares_state_t *>st)

cdef uint32_t squares_32_uint32(void *st) noexcept nogil:
    return squares_32_next32(<squares_state_t *> st)

cdef double squares_32_double(void* st) noexcept nogil:
    return squares_32_next_double(<squares_state_t *>st)


cdef class SequenceSampler:
    cdef int randoms_drawm, bit_well_loc, index
    cdef uint64_t[::1] random_well
    cdef uint64_t bit_well
    cdef object seed_seq

    def __init__(self, seed_seq: SeedSequence, int initial_draw=10):
        self.seed_seq = seed_seq
        self.randoms_drawm = initial_draw
        self.random_well = self.seed_seq.generate_state(self.randoms_drawm, dtype=np.uint64)
        self.bit_well = 0
        self.bit_well_loc = 0
        self.index = 0
        self.refill()

    cdef void refill(self):
        self.bit_well = <uint64_t>self.random_well[self.index]
        self.bit_well_loc = 0
        self.index += 1
        if self.index > self.randoms_drawm:
            self.randoms_drawm *= 2
            self.random_well = self.seed_seq.generate_state(self.randoms_drawm, dtype=np.uint64)

    cdef uint64_t gen_bits(self, int nbits):
        cdef uint64_t out

        if self.bit_well_loc + nbits > 64:
            self.refill()
        out = self.bit_well & ((1 << nbits) - 1)
        self.bit_well >>= nbits
        self.bit_well_loc += nbits
        return out

    cdef uint64_t random_inverval(self, uint64_t max_value):
        cdef int bits
        cdef uint64_t out

        if max_value == 1:
            return 0
        elif max_value == 2:
            return self.gen_bits(1)
        elif max_value < 4:
            bits = 2
        elif max_value < 8:
            bits = 3
        else:
            bits = 4
        while True:
            out = self.gen_bits(bits)
            if out < max_value:
                return out

    cdef void shuffle(self, uint8_t[::1] array):
        cdef intptr_t i, n
        cdef uint64_t j
        n = array.shape[0]
        for i in range(n - 1):
            j = i + self.random_inverval(n - i)
            tmp = array[i]
            array[i] = array[j]
            array[j] = tmp

    cdef uint8_t next_word(self):
        return <uint8_t>self.gen_bits(4)


cdef uint64_t generate_key(SequenceSampler seq_sampler):
    cdef intptr_t pos
    cdef uint64_t out, first, i
    cdef uint8_t[::1] tmp
    cdef uint8_t last_value, next_value, k, old

    out = 0
    tmp = np.arange(16, dtype=np.uint8)
    seq_sampler.shuffle(tmp)
    if not (tmp[0] & 0x1):
        old = tmp[0]
        tmp[0] += 1
        for i in range(1,8):
            if tmp[i] == tmp[0]:
                tmp[i] = old
    for i in range(8):
        out |= <uint64_t> tmp[i] << (4 * i)
    last_value = tmp[7]
    # 1. Choose a random byte from 1, 3, ..., 15, call a
    # 2. Store in byte 0
    # 3. Populate an array with bytes 1...15 ex a
    # 4. Shuffle the array
    # 5. Store in Bytes 1...7 with first 7 locations form shuffled array
    # 7. Choose a random number from 1,2,...,15 ex position 7
    # 8. Store in byte 8
    # 9. Populate bytes 9...15 with random bytes
    #    using the rule that byte[i] must be different from byte[i-1]
    #    and byte[i] != 0
    for i in range(8, 16):
        next_value = 0
        while next_value == last_value or next_value == 0:
            next_value = seq_sampler.next_word()
        last_value = next_value
        out |= <uint64_t>next_value << (4 * i)

    return out


def _key_gen_test_harness(seed_seq):
    return generate_key(SequenceSampler(seed_seq))


def generate_keys(seed=None, intptr_t n=1):
    cdef intptr_t i
    cdef uint64_t[::1] out
    cdef np.ndarray out_arr
    cdef SequenceSampler seq_sampler
    if isinstance(seed, ISeedSequence):
        seed_seq = seed
    else:
        seed_seq = SeedSequence(seed)
    seq_sampler = SequenceSampler(seed_seq, 10 * n)
    out_arr = np.empty(n, dtype=np.uint64)
    out = out_arr
    for i in range(n):
        out[i] = generate_key(seq_sampler)
    return np.array(out)


cdef class Squares(BitGenerator):
    def __init__(self, seed=None, counter=None, key=None):
        BitGenerator.__init__(self, seed)
        if counter is None:
            self.rng_state.counter = 0
        else:
            self.rng_state.counter = self._check_value(counter, "counter", False)
        if key is not None:
            self.rng_state.key =self._check_value(key, "key", True)

    cdef void _setup_bitgen(self):
        if self.original:
            self._bitgen.next_uint64 = &squares_uint64
            self._bitgen.next_uint32 = &squares_uint32
            self._bitgen.next_double = &squares_double
            self._bitgen.next_raw = &squares_uint64
        else:
            self._bitgen.next_uint64 = &squares_32_uint64
            self._bitgen.next_uint32 = &squares_32_uint32
            self._bitgen.next_double = &squares_32_double
            self._bitgen.next_raw = &squares_32_uint64

    cdef uint64_t _check_value(self, object val, object name, bint odd):
        val = int(val)
        if not 0 <= val <= np.iinfo(np.uint64).max:
            raise ValueError(f"{name} must be positive and less than 2**64.")

        if odd and not (val & 0x1):
            raise ValueError(f"{name} must be odd.")

        return <uint64_t>val

    def _seed_from_seq(self, uint64_t counter=0):
        cdef uint64_t key

        try:
            seed_seq = self.seed_seq
        except AttributeError:
            seed_seq = self._seed_seq
        state = seed_seq.generate_state(1, np.uint64)[0]
        self.rng_state.key = generate_key(SequenceSampler(seed_seq))
        self.rng_state.ctr = counter

    cdef void _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        self._reset_state_variables()
    
    def seed(self, seed=None, *, counter=None, key=None):
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
        if counter is not None:
            self.rng_state.ctr = self._check_value(counter, "counter", False)
        if key is not None:
            self.rng_state.key = self._check_value(key, "key", True)
        else:
            BitGenerator._seed_with_seed_sequence(self, seed, counter=counter)

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
                "state": {"key": self.rng_state.key, "counter": self.rng_state.counter},
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger,
                }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = value["state"]
        self.rng_state.key = state["key"]
        self.rng_state.counter = state["counter"]
        self._setup_bitgen()

