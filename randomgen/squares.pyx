#!python
# cython: binding=True, language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from randomgen.common cimport BitGenerator

from libc.stdint cimport (
uint8_t,
    uint64_t,
    intptr_t
)

import numpy as np
cimport numpy as np
from numpy.random import SeedSequence

cdef class SequenceSampled:
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


def generate_key(seed_seq):
    cdef intptr_t pos
    cdef uint64_t out, first, i
    cdef uint8_t[::1] tmp
    cdef uint8_t last_value, next_value, k

    ss = SequenceSampled(seed_seq, 20)
    out = 0
    tmp = np.arange(16, dtype=np.uint8)
    ss.shuffle(tmp)
    if not (tmp[0] & 0x1):
        old = tmp[0]
        tmp[0] += 1
        for i in range(1,8):
            if tmp[i] == tmp[0]:
                tmp[i] = old
    for i in range(8):
        out |= <uint64_t> tmp[i - 1] << (4 * i)
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
            next_value = ss.next_word()
        last_value = next_value
        out |= <uint64_t>next_value << (4 * i)

    return out

cdef class Squares(BitGenerator):
    def __init__(self, seed=None, counter=None, key=None):
        BitGenerator.__init__(self, seed)
