cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport BitGenerator, fully_qualified_name, uint64_to_double

DEF BUFFER_SIZE = 256

cdef extern from "src/rdrand/rdrand.h":

    struct s_rdrand_state:
        uint64_t buffer[BUFFER_SIZE]
        int buffer_loc
        int status
        int retries
        uint64_t weyl_seq

    ctypedef s_rdrand_state rdrand_state

    int rdrand_fill_buffer(rdrand_state *state) noexcept nogil
    int rdrand_next64(rdrand_state *state, uint64_t *val) noexcept nogil
    int rdrand_capable()