

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_free_aligned,
    PyArray_malloc_aligned,
    fully_qualified_name,
    int_to_array,
    object_to_int,
    view_little_endian,
)


cdef extern from "src/blabla/blabla.h":

    int RANDOMGEN_USE_AVX2

    struct BLABLA_STATE_T:
        uint64_t block[16]
        uint64_t keysetup[4]
        uint64_t block_idx[2]
        uint64_t ctr[2]
        int rounds
        int has_uint32
        uint32_t next_uint32


    ctypedef BLABLA_STATE_T blabla_state_t

    uint32_t blabla_next32(blabla_state_t *state) noexcept nogil
    uint64_t blabla_next64(blabla_state_t *state) noexcept nogil
    double blabla_next_double(blabla_state_t *state) noexcept nogil

    void blabla_seed(
            blabla_state_t *state,
            uint64_t *seedval,
            uint64_t *stream,
            uint64_t *ctr
    )
    void blabla_advance(blabla_state_t *state, uint64_t *delta)
    int blabla_avx2_capable()
    void blabla_use_avx2(int value)

cdef class BlaBla(BitGenerator):

    cdef blabla_state_t *rng_state
