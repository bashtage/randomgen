from randomgen.common cimport *


cdef extern from "src/pcg64/pcg64.h":
    # Use int as generic type, actual type read from pcg64.h and is platform dependent
    ctypedef int pcg64_random_t

    struct PCG64_STATE_T:
        pcg64_random_t *pcg_state
        int has_uint32
        uint32_t uinteger

    ctypedef PCG64_STATE_T pcg64_state_t

    uint64_t pcg64_next64(pcg64_state_t *state)  nogil
    uint32_t pcg64_next32(pcg64_state_t *state)  nogil
    void pcg64_jump(pcg64_state_t *state)
    void pcg64_advance(pcg64_state_t *state, uint64_t *step)
    void pcg64_set_seed(pcg64_state_t *state, uint64_t *seed, uint64_t *inc)
    void pcg64_get_state(pcg64_state_t *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void pcg64_set_state(pcg64_state_t *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)


cdef class PCG64(BitGenerator):

    cdef pcg64_state_t rng_state
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)
