# cython: binding=True, language_level=3

from randomgen.common cimport *

DEF INDIRECTION_SIZE = 16
DEF ITERATION_SIZE = 32

cdef extern from "src/efiix64/efiix64.h":

    struct EFIIX64_STATE_T:
        uint64_t indirection_table[INDIRECTION_SIZE]
        uint64_t iteration_table[ITERATION_SIZE]
        uint64_t i, a, b, c
        int has_uint32
        uint32_t uinteger

    ctypedef EFIIX64_STATE_T efiix64_state_t

    uint64_t efiix64_next64(efiix64_state_t *state) nogil
    uint32_t efiix64_next32(efiix64_state_t *state) nogil
    void efiix64_seed(efiix64_state_t *state, uint64_t seed[4])


cdef class EFIIX64(BitGenerator):

    cdef efiix64_state_t rng_state
    cdef _reset_state_variables(self)
