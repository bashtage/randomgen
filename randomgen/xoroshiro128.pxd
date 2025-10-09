cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/xoroshiro128/xoroshiro128.h":

    struct XOROSHIRO128_STATE_T:
        uint64_t s[2]
        int has_uint32
        uint32_t uinteger

    ctypedef XOROSHIRO128_STATE_T xoroshiro128_state_t

    uint64_t xoroshiro128_next64(xoroshiro128_state_t *state) noexcept nogil
    uint32_t xoroshiro128_next32(xoroshiro128_state_t *state) noexcept nogil
    void xoroshiro128_jump(xoroshiro128_state_t *state)

    uint64_t xoroshiro128plusplus_next64(xoroshiro128_state_t *state) noexcept nogil
    uint32_t xoroshiro128plusplus_next32(xoroshiro128_state_t *state) noexcept nogil
    void xoroshiro128plusplus_jump(xoroshiro128_state_t *state)


cdef class Xoroshiro128(BitGenerator):

    cdef xoroshiro128_state_t rng_state
    cdef bint _plusplus
    cdef _reset_state_variables(self)
    cdef _set_generators(self)
    cdef jump_inplace(self, np.npy_intp iter)
