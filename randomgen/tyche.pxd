# cython: binding=True, language_level=3

from randomgen.common cimport *

cdef extern from "src/tyche/tyche.h":

    struct TYCHE_STATE_T:
        uint32_t a
        uint32_t b
        uint32_t c
        uint32_t d

    ctypedef TYCHE_STATE_T TYCHE_state_t

    uint64_t tyche_next64(tyche_state_t *state) noexcept nogil
    uint32_t tyche_next32(tyche_state_t *state) noexcept nogil
    double tyche_next_double(tyche_state_t *state)
    void tyche_seed(efiix64_state_t *state, uint64_t seed, uint32_t inc)


cdef class Tyche(BitGenerator):

    cdef tyche_state_t rng_state
    cdef _reset_state_variables(self)
