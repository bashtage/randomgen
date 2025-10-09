

cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name, uint64_to_double


cdef extern from "src/jsf/jsf.h":

    union JSF_UINT_T:
        uint64_t u64
        uint32_t u32
    ctypedef JSF_UINT_T jsf_uint_t

    struct JSF_STATE_T:
        jsf_uint_t a
        jsf_uint_t b
        jsf_uint_t c
        jsf_uint_t d
        int p
        int q
        int r
        int has_uint32
        uint32_t uinteger

    ctypedef JSF_STATE_T jsf_state_t

    uint64_t jsf64_next64(jsf_state_t *state) noexcept nogil
    uint32_t jsf64_next32(jsf_state_t *state) noexcept nogil
    double jsf64_next_double(jsf_state_t *state) noexcept nogil
    void jsf64_seed(jsf_state_t *state, uint64_t *seed, int size)

    uint64_t jsf32_next64(jsf_state_t *state) noexcept nogil
    uint32_t jsf32_next32(jsf_state_t *state) noexcept nogil
    double jsf32_next_double(jsf_state_t *state) noexcept nogil
    void jsf32_seed(jsf_state_t *state, uint32_t *seed, int size)


cdef class JSF(BitGenerator):

    cdef jsf_state_t rng_state
    cdef int size
    cdef int seed_size
    cdef setup_generator(self, object p, object q, object r)
    cdef _reset_state_variables(self)
