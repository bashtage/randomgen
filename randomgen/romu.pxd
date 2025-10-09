cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name, uint64_to_double


cdef extern from "src/romu/romu.h":

    struct ROMU_STATE_T:
        uint64_t w
        uint64_t x
        uint64_t y
        uint64_t z
        int has_uint32
        uint32_t uinteger

    ctypedef ROMU_STATE_T romu_state_t

    uint64_t romuquad_next64(romu_state_t *state) noexcept nogil
    uint32_t romuquad_next32(romu_state_t *state) noexcept nogil
    uint64_t romutrio_next64(romu_state_t *state) noexcept nogil
    uint32_t romutrio_next32(romu_state_t *state) noexcept nogil
    void romu_seed(romu_state_t *state, uint64_t w, uint64_t x, uint64_t y, uint64_t z, int quad)


cdef class Romu(BitGenerator):

    cdef readonly object variant
    cdef romu_state_t rng_state
    cdef _reset_state_variables(self)
    cdef _check_variant(self, variant)
    cdef _setup_bitgen(self)
