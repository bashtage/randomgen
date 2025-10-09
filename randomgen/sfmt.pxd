cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_calloc_aligned,
    PyArray_free_aligned,
    PyArray_malloc_aligned,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)

DEF SFMT_MEXP = 19937
DEF SFMT_N = 156  # SFMT_MEXP / 128 + 1
DEF SFMT_N64 = SFMT_N * 2

cdef extern from "src/sfmt/sfmt.h":

    union W128_T:
        uint32_t u[4]
        uint64_t u64[2]

    ctypedef W128_T w128_t

    struct SFMT_T:
        w128_t state[SFMT_N]
        int idx

    ctypedef SFMT_T sfmt_t

    struct SFMT_STATE_T:
        sfmt_t *state
        int has_uint32
        uint32_t uinteger

        uint64_t *buffered_uint64
        int buffer_loc

    ctypedef SFMT_STATE_T sfmt_state_t

    uint64_t sfmt_next64(sfmt_state_t *state) noexcept nogil
    uint32_t sfmt_next32(sfmt_state_t *state) noexcept nogil

    void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed)
    void sfmt_init_by_array(sfmt_t * sfmt, uint32_t *init_key, int key_length)
    void sfmt_jump(sfmt_state_t *state)
    void sfmt_jump_n(sfmt_state_t *state, int count)

cdef class SFMT(BitGenerator):

    cdef sfmt_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)
