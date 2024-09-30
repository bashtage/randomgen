cimport numpy as np
from libc.stdint cimport uint8_t, uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_calloc_aligned,
    PyArray_free_aligned,
    check_state_array,
    fully_qualified_name,
    int_to_array,
    object_to_int,
    uint64_to_double,
    wrap_int,
)

DEF SPECK_UNROLL = 12
DEF SPECK_ROUNDS = 34

cdef extern from "src/speck-128/speck-128.h":

    union SPEC_T:
        uint64_t u64[2]

    ctypedef SPEC_T spec_t

    struct SPECK_STATE_T:
        spec_t round_key[SPECK_ROUNDS]
        spec_t ctr[SPECK_UNROLL // 2]
        uint8_t buffer[8 * SPECK_UNROLL]
        int rounds

        int offset
        int has_uint32
        uint32_t uinteger

    ctypedef SPECK_STATE_T speck_state_t

    uint64_t speck_next64(speck_state_t *state) noexcept nogil
    uint32_t speck_next32(speck_state_t *state) noexcept nogil

    int RANDOMGEN_USE_SSE41
    int speck_sse41_capable()
    void speck_use_sse41(int val)
    void speck_seed(speck_state_t *state, uint64_t *seed)
    void speck_set_counter(speck_state_t *state, uint64_t *ctr)
    void speck_advance(speck_state_t *state, uint64_t *step)


cdef class SPECK128(BitGenerator):

    cdef speck_state_t *rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)
