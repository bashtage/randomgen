
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_calloc_aligned,
    PyArray_free_aligned,
    PyArray_malloc_aligned,
    check_state_array,
    fully_qualified_name,
)

DEF DSFMT_MEXP = 19937
DEF DSFMT_N = 191  # ((DSFMT_MEXP - 128) / 104 + 1)
DEF DSFMT_N_PLUS_1 = 192  # DSFMT_N + 1
DEF DSFMT_N64 = DSFMT_N * 2

cdef extern from "src/dsfmt/dsfmt.h":

    union W128_T:
        uint64_t u[2]
        uint32_t u32[4]
        double d[2]

    ctypedef W128_T w128_t

    struct DSFMT_T:
        w128_t status[DSFMT_N_PLUS_1]
        int idx

    ctypedef DSFMT_T dsfmt_t

    struct DSFMT_STATE_T:
        dsfmt_t *state
        double *buffered_uniforms
        int buffer_loc

    ctypedef DSFMT_STATE_T dsfmt_state_t

    double dsfmt_next_double(dsfmt_state_t *state) noexcept nogil
    uint64_t dsfmt_next64(dsfmt_state_t *state) noexcept nogil
    uint32_t dsfmt_next32(dsfmt_state_t *state) noexcept nogil
    uint64_t dsfmt_next_raw(dsfmt_state_t *state) noexcept nogil

    void dsfmt_init_gen_rand(dsfmt_t *dsfmt, uint32_t seed)
    void dsfmt_init_by_array(dsfmt_t *dsfmt, uint32_t init_key[], int key_length)
    void dsfmt_jump(dsfmt_state_t *state)
    void dsfmt_jump_n(dsfmt_state_t *state, int count)

cdef class DSFMT(BitGenerator):

    cdef dsfmt_state_t rng_state
    cdef jump_inplace(self, object iter)
    cdef _reset_state_variables(self)
