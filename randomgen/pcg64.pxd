cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    PyArray_free_aligned,
    PyArray_malloc_aligned,
    fully_qualified_name,
    uint64_to_double,
    wrap_int,
)


cdef extern from "src/pcg64/pcg64-common.h":
    ctypedef uint64_t (* pcg_output_func_t)(uint64_t high, uint64_t low) noexcept nogil

cdef extern from "src/pcg64/pcg64-v2.h":
    # Use int as generic type, actual type read from pcg64.h and is platform dependent
    ctypedef int pcg64_random_t

    struct PCG64_STATE_T:
        pcg64_random_t *pcg_state
        int use_dxsm
        int has_uint32
        uint32_t uinteger

    ctypedef PCG64_STATE_T pcg64_state_t

    uint64_t pcg64_next64(pcg64_state_t *state) noexcept nogil
    uint32_t pcg64_next32(pcg64_state_t *state) noexcept nogil
    uint64_t pcg64_cm_dxsm_next64(pcg64_state_t *state) noexcept nogil
    uint32_t pcg64_cm_dxsm_next32(pcg64_state_t *state) noexcept nogil

    void pcg64_advance(pcg64_state_t *state, uint64_t *step, int cheap_multiplier)
    void pcg64_set_seed(pcg64_state_t *state, uint64_t *seed, uint64_t *inc, int cheap_multiplier)
    void pcg64_get_state(pcg64_state_t *state, uint64_t *state_arr, int *use_dxsm, int *has_uint32, uint32_t *uinteger)
    void pcg64_set_state(pcg64_state_t *state, uint64_t *state_arr, int use_dxsm, int has_uint32, uint32_t uinteger)

cdef extern from "src/pcg64/lcg128mix.h":

    ctypedef int pcg128_t

    struct lcg128mix_RANDOM_T:
        pcg128_t state
        pcg128_t inc
        pcg128_t multiplier
        uint64_t dxsm_multiplier
        int post
        int output_idx
        pcg_output_func_t output_func

    ctypedef lcg128mix_RANDOM_T lcg128mix_random_t

    struct lcg128mix_STATE_T:
      lcg128mix_random_t *pcg_state
      int use_dxsm
      int has_uint32
      uint32_t uinteger

    ctypedef lcg128mix_STATE_T lcg128mix_state_t

    uint64_t lcg128mix_next64(lcg128mix_state_t *state) noexcept nogil
    uint64_t lcg128mix_next32(lcg128mix_state_t *state) noexcept nogil

    void lcg128mix_set_state(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[], uint64_t multiplier[]) noexcept nogil
    void lcg128mix_get_state(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[], uint64_t multiplier[]) noexcept nogil
    void lcg128mix_seed(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[], uint64_t multiplier[]) noexcept nogil
    void lcg128mix_advance(lcg128mix_state_t *rng, uint64_t step[]) noexcept nogil

cdef class PCG64(BitGenerator):

    cdef pcg64_state_t rng_state
    cdef readonly str variant
    cdef bint use_dxsm
    cdef bint cheap_multiplier
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)

cdef class LCG128Mix(BitGenerator):

    cdef lcg128mix_state_t rng_state
    cdef object multiplier, _default_multiplier, _default_dxsm_multiplier
    cdef object output_function_name, _output_lookup, _inv_output_lookup, _cfunc
    cdef bint post
    cdef size_t output_function_address
    cdef uint64_t dxsm_multiplier
    cdef int output_function
    cdef _reset_state_variables(self)
    cdef jump_inplace(self, object iter)

cdef class PCG64DXSM(PCG64):
    pass
