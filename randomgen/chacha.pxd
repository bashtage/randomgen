

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


cdef extern from "src/chacha/chacha.h":

    int RANDOMGEN_USE_SIMD

    struct CHACHA_STATE_T:
        uint32_t block[16]
        uint32_t keysetup[8]
        uint64_t ctr[2]
        int rounds

    ctypedef CHACHA_STATE_T chacha_state_t

    uint32_t chacha_next32(chacha_state_t *state) noexcept nogil
    uint64_t chacha_next64(chacha_state_t *state) noexcept nogil
    double chacha_next_double(chacha_state_t *state) noexcept nogil

    void chacha_seed(
            chacha_state_t *state,
            uint64_t *seedval,
            uint64_t *stream,
            uint64_t *ctr
    )
    void chacha_advance(chacha_state_t *state, uint64_t *delta)
    int chacha_simd_capable()
    void chacha_use_simd(int value)

cdef class ChaCha(BitGenerator):

    cdef chacha_state_t *rng_state
