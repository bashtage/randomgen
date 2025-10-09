

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, check_state_array, fully_qualified_name


cdef extern from "src/tyche/tyche.h":

    struct TYCHE_STATE_T:
        uint32_t a
        uint32_t b
        uint32_t c
        uint32_t d

    ctypedef TYCHE_STATE_T tyche_state_t

    uint64_t tyche_next64(tyche_state_t *state) noexcept nogil
    uint32_t tyche_next32(tyche_state_t *state) noexcept nogil
    double tyche_next_double(tyche_state_t *state) noexcept nogil
    uint64_t tyche_openrand_next64(tyche_state_t *state) noexcept nogil
    uint32_t tyche_openrand_next32(tyche_state_t *state) noexcept nogil
    double tyche_openrand_next_double(tyche_state_t *state) noexcept nogil

    void tyche_seed(tyche_state_t *state, uint64_t seed, uint32_t inc, int openrand) noexcept nogil



cdef class Tyche(BitGenerator):
    cdef bint original
    cdef tyche_state_t rng_state
    cdef void _setup_bitgen(self)
