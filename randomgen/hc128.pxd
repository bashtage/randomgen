# cython: binding=True, language_level=3
# coding=utf-8
from randomgen.common cimport *


cdef extern from "src/hc-128/hc-128.h":

    struct HC128_STATE_T:
        uint32_t p[512]
        uint32_t q[512]
        uint32_t buffer[16]
        int hc_idx
        int buffer_idx

    ctypedef HC128_STATE_T hc128_state_t

    uint32_t hc128_next32(hc128_state_t *state) nogil
    uint64_t hc128_next64(hc128_state_t *state) nogil
    double hc128_next_double(hc128_state_t *state) nogil
    void hc128_seed(hc128_state_t *state, uint32_t *seed)


cdef class HC128(BitGenerator):

    cdef hc128_state_t rng_state
