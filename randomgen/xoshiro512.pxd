cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/xoshiro512/xoshiro512.h":

    struct XOSHIRO512_STATE_T:
        uint64_t s[8]
        int has_uint32
        uint32_t uinteger

    ctypedef XOSHIRO512_STATE_T xoshiro512_state_t

    uint64_t xoshiro512_next64(xoshiro512_state_t *state) noexcept nogil
    uint32_t xoshiro512_next32(xoshiro512_state_t *state) noexcept nogil
    void xoshiro512_jump(xoshiro512_state_t *state)


cdef class Xoshiro512(BitGenerator):

    cdef xoshiro512_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, np.npy_intp iter)
