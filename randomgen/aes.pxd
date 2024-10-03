

cimport numpy as np
from libc.stdint cimport uint8_t, uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_calloc_aligned,
    PyArray_free_aligned,
    fully_qualified_name,
    int_to_array,
    object_to_int,
    uint64_to_double,
    wrap_int,
)


cdef extern from "src/aesctr/aesctr.h":

    # int are placeholders only for
    struct AESCTR_STATE_T:
        int ctr[4]
        int seed[10 + 1]
        uint8_t state[16 * 4]
        size_t offset
        int has_uint32
        uint32_t uinteger

    ctypedef AESCTR_STATE_T aesctr_state_t

    uint64_t aes_next64(aesctr_state_t *aesctr) noexcept nogil
    uint32_t aes_next32(aesctr_state_t *aesctr) noexcept nogil
    double aes_next_double(aesctr_state_t *aesctr) noexcept nogil

    int RANDOMGEN_USE_AESNI
    void aesctr_use_aesni(int val)
    void aesctr_seed(aesctr_state_t *aesctr, uint64_t *seed)
    void aesctr_set_seed_counter(
            aesctr_state_t *aesctr, uint64_t *seed, uint64_t *counter
    )
    void aesctr_get_seed_counter(
            aesctr_state_t *aesctr, uint64_t *seed, uint64_t *counter
    )
    int aes_capable()
    void aesctr_advance(aesctr_state_t *aesctr, uint64_t *step)
    void aesctr_set_counter(aesctr_state_t *aesctr, uint64_t *counter)


cdef class AESCounter(BitGenerator):

    cdef aesctr_state_t *rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)
