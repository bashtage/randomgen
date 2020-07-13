from randomgen.common cimport *


cdef extern from "src/sfc/sfc.h":

    struct SFC_STATE_T:
        uint64_t a
        uint64_t b
        uint64_t c
        uint64_t w
        uint64_t k
        int has_uint32
        uint32_t uinteger

    ctypedef SFC_STATE_T sfc_state_t

    uint64_t sfc_next64(sfc_state_t *state) nogil
    uint32_t sfc_next32(sfc_state_t *state) nogil
    void sfc_seed(sfc_state_t *state, uint64_t *seed, uint64_t w, uint64_t k)

cdef class SFC64(BitGenerator):

    cdef object k, w
    cdef sfc_state_t rng_state
    cdef _reset_state_variables(self)
    cdef uint64_t generate_bits(self, int8_t bits)
