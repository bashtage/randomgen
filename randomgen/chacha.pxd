from randomgen.common cimport *

cdef extern from "src/chacha/chacha.h":

    int RANDOMGEN_USE_SIMD

    struct CHACHA_STATE_T:
        uint32_t block[16]
        uint32_t keysetup[8]
        uint64_t ctr[2]
        int rounds

    ctypedef CHACHA_STATE_T chacha_state_t

    uint32_t chacha_next32(chacha_state_t *state) nogil
    uint64_t chacha_next64(chacha_state_t *state) nogil
    double chacha_next_double(chacha_state_t *state) nogil

    void chacha_seed(chacha_state_t *state, uint64_t *seedval, uint64_t *stream, uint64_t *ctr)
    void chacha_advance(chacha_state_t *state, uint64_t *delta)
    int chacha_simd_capable()
    void chacha_use_simd(int value)

cdef class ChaCha(BitGenerator):
    cdef chacha_state_t *rng_state
    cdef jump_inplace(self, object iter)
