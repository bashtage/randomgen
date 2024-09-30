cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/xorshift1024/xorshift1024.h":

    struct XORSHIFT1024_STATE_T:
        uint64_t s[16]
        int p
        int has_uint32
        uint32_t uinteger

    ctypedef XORSHIFT1024_STATE_T xorshift1024_state_t

    uint64_t xorshift1024_next64(xorshift1024_state_t *state) noexcept nogil
    uint32_t xorshift1024_next32(xorshift1024_state_t *state) noexcept nogil
    void xorshift1024_jump(xorshift1024_state_t *state)


cdef class Xorshift1024(BitGenerator):

    cdef xorshift1024_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, np.npy_intp iter)
