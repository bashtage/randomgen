cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name, uint64_to_double


cdef extern from "src/mwcx/mwcx.h":

    struct MWC192X_STATE_T:
        uint64_t x
        uint64_t y
        uint64_t c
        int has_uint32
        uint32_t uinteger

    ctypedef MWC192X_STATE_T mwc192x_state_t

    struct MWC128X_STATE_T:
        uint64_t x
        uint64_t c
        int has_uint32
        uint32_t uinteger

    ctypedef MWC128X_STATE_T mwc128x_state_t

    struct MWC128X32_STATE_T:
        uint32_t x
        uint32_t y
        uint32_t z
        uint32_t c

    ctypedef MWC128X32_STATE_T mwc128x32_state_t

    struct MWC64X32_STATE_T:
        uint32_t x
        uint32_t c

    ctypedef MWC64X32_STATE_T mwc64x32_state_t

    uint64_t mwc192x_next64(mwc192x_state_t *state) noexcept nogil
    uint32_t mwc192x_next32(mwc192x_state_t *state) noexcept nogil
    uint64_t mwc128x_next64(mwc128x_state_t *state) noexcept nogil
    uint32_t mwc128x_next32(mwc128x_state_t *state) noexcept nogil
    uint32_t mwc128x32_next32(mwc128x32_state_t *state) noexcept nogil
    uint64_t mwc128x32_next64(mwc128x32_state_t *state) noexcept nogil
    double mwc128x32_next_double(mwc128x32_state_t *state) noexcept nogil
    uint32_t mwc64x32_next32(mwc64x32_state_t *state) noexcept nogil
    uint64_t mwc64x32_next64(mwc64x32_state_t *state) noexcept nogil
    double mwc64x32_next_double(mwc64x32_state_t *state) noexcept nogil


cdef class _MWCX(BitGenerator):
    pass

cdef class MWC192X(_MWCX):
    cdef mwc192x_state_t rng_state

cdef class MWC128X(_MWCX):
    cdef mwc128x_state_t rng_state

cdef class MWC128X32(_MWCX):
    cdef mwc128x32_state_t rng_state

cdef class MWC64X32(_MWCX):
    cdef mwc64x32_state_t rng_state
