

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    uint64_to_double,
)


cdef extern from "src/lxm/lxm.h":

    struct LXM_STATE_T:
        uint64_t x[4]
        uint64_t lcg_state
        uint64_t b
        int has_uint32
        uint32_t uinteger

    ctypedef LXM_STATE_T lxm_state_t

    uint64_t lxm_next64(lxm_state_t *state) noexcept nogil
    uint32_t lxm_next32(lxm_state_t *state) noexcept nogil
    double lxm_next_double(lxm_state_t *state) noexcept nogil
    void lxm_jump(lxm_state_t *state) noexcept nogil

cdef class LXM(BitGenerator):

    cdef lxm_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, np.npy_intp iter)
