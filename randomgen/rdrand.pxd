from randomgen.common cimport *

DEF BUFFER_SIZE = 256

cdef extern from "src/rdrand/rdrand.h":

    struct s_rdrand_state:
        uint64_t buffer[BUFFER_SIZE]
        int buffer_loc
        int status
        int retries
        uint64_t weyl_seq

    ctypedef s_rdrand_state rdrand_state

    int rdrand_fill_buffer(rdrand_state *state) nogil
    int rdrand_next64(rdrand_state *state, uint64_t *val) nogil
    int rdrand_capable()