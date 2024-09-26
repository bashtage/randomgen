# cython: binding=True, language_level=3

from randomgen.common cimport *


cdef extern from "src/squares/squares.h":

    struct SQUARES_STATE_T:
        uint64_t cnt
        uint64_t key
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
