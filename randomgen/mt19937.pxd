cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, check_state_array, fully_qualified_name


cdef extern from "src/mt19937/mt19937.h":

    struct MT19937_STATE_T:
        uint32_t key[624]
        int pos

    ctypedef MT19937_STATE_T mt19937_state_t

    uint64_t mt19937_next64(mt19937_state_t *state) noexcept nogil
    uint32_t mt19937_next32(mt19937_state_t *state) noexcept nogil
    double mt19937_next_double(mt19937_state_t *state) noexcept nogil
    void mt19937_init_by_array(mt19937_state_t *state, uint32_t *init_key, int key_length)
    void mt19937_seed(mt19937_state_t *state, uint32_t seed)
    void mt19937_jump(mt19937_state_t *state)
    void mt19937_jump_default(mt19937_state_t *state)
    void mt19937_jump_n(mt19937_state_t *state, int count)

    enum:
        RK_STATE_LEN

cdef class MT19937(BitGenerator):

    cdef mt19937_state_t rng_state
    cdef jump_inplace(self, int jumps)
