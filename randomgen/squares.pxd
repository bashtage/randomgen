

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name


cdef extern from "src/squares/squares.h":

    struct SQUARES_STATE_T:
        uint64_t key
        uint64_t counter
        int has_uint32
        uint32_t uinteger

    ctypedef SQUARES_STATE_T squares_state_t

    uint64_t squares_next64(squares_state_t *state) noexcept nogil
    uint32_t squares_next32(squares_state_t *state) noexcept nogil
    double squares_next_double(squares_state_t *state) noexcept nogil
    uint64_t squares_32_next64(squares_state_t *state) noexcept nogil
    uint32_t squares_32_next32(squares_state_t *state) noexcept nogil
    double squares_32_next_double(squares_state_t *state) noexcept nogil

cdef class Squares(BitGenerator):
    cdef squares_state_t rng_state
    cdef void _setup_bitgen(self)
    cdef int variant
    cdef bint _use64
    cdef uint64_t _check_value(self, object val, object name, bint odd)
    cdef void _reset_state_variables(self)

