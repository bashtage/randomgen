cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name, wrap_int


cdef extern from "src/pcg32/pcg32.h":

    cdef struct pcg_state_setseq_64:
        uint64_t state
        uint64_t inc

    ctypedef pcg_state_setseq_64 pcg32_random_t

    struct PCG32_STATE_T:
        pcg32_random_t pcg_state

    ctypedef PCG32_STATE_T pcg32_state_t

    uint64_t pcg32_next64(pcg32_state_t *state) noexcept nogil
    uint32_t pcg32_next32(pcg32_state_t *state) noexcept nogil
    double pcg32_next_double(pcg32_state_t *state) noexcept nogil
    void pcg32_jump(pcg32_state_t *state)
    void pcg32_advance_state(pcg32_state_t *state, uint64_t step)
    void pcg32_set_seed(pcg32_state_t *state, uint64_t seed, uint64_t inc)


cdef class PCG32(BitGenerator):
    cdef pcg32_state_t rng_state
    cdef jump_inplace(self, object iter)
