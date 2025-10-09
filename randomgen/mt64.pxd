

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/mt64/mt64.h":

    struct MT64_STATE_T:
        uint64_t mt[312]
        int mti
        int has_uint32
        uint32_t uinteger

    ctypedef MT64_STATE_T mt64_state_t

    uint64_t mt64_next64(mt64_state_t *state) noexcept nogil
    uint32_t mt64_next32(mt64_state_t *state) noexcept nogil
    double mt64_next_double(mt64_state_t *state) noexcept nogil
    void mt64_init_by_array(mt64_state_t *state, uint64_t *init_key, int key_length)
    void mt64_seed(mt64_state_t *state, uint64_t seed)

cdef class MT64(BitGenerator):

    cdef mt64_state_t rng_state
    cdef _reset_state_variables(self)
