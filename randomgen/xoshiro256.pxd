cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/xoshiro256/xoshiro256.h":

    struct XOSHIRO256_STATE_T:
        uint64_t s[8]
        int has_uint32
        uint32_t uinteger

    ctypedef XOSHIRO256_STATE_T xoshiro256_state_t

    uint64_t xoshiro256_next64(xoshiro256_state_t *state) noexcept nogil
    uint32_t xoshiro256_next32(xoshiro256_state_t *state) noexcept nogil
    void xoshiro256_jump(xoshiro256_state_t *state)


cdef class Xoshiro256(BitGenerator):

    cdef xoshiro256_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, np.npy_intp iter)
