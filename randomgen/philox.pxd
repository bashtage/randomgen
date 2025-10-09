cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t

from randomgen.common cimport (
    BitGenerator,
    check_state_array,
    fully_qualified_name,
    int_to_array,
    object_to_int,
    uint64_to_double,
    view_little_endian,
    wrap_int,
)


cdef extern from "src/philox/philox.h":
    struct s_r123array1x32:
        uint32_t v[1]
    ctypedef s_r123array1x32 r123array1x32
    struct s_r123array2x32:
        uint32_t v[2]
    ctypedef s_r123array2x32 r123array2x32
    struct s_r123array4x32:
        uint32_t v[4]
    ctypedef s_r123array4x32 r123array4x32
    struct s_r123array1x64:
        uint64_t v[1]
    ctypedef s_r123array1x64 r123array1x64
    struct s_r123array2x64:
        uint64_t v[2]
    ctypedef s_r123array2x64 r123array2x64
    struct s_r123array4x64:
        uint64_t v[4]
    ctypedef s_r123array4x64 r123array4x64
    #######################################################
    # 2x32
    #######################################################
    ctypedef r123array2x32 philox2x32_ctr_t
    ctypedef r123array1x32 philox2x32_key_t
    struct s_philox2x32_state:
        philox2x32_ctr_t ctr
        philox2x32_key_t key

    ctypedef s_philox2x32_state philox2x32_state
    #######################################################
    # 4x32
    #######################################################
    ctypedef r123array4x32 philox4x32_ctr_t
    ctypedef r123array2x32 philox4x32_key_t
    struct s_philox4x32_state:
        philox4x32_ctr_t ctr
        philox4x32_key_t key

    ctypedef s_philox4x32_state philox4x32_state
    #######################################################
    # 2x64
    #######################################################
    ctypedef r123array2x64 philox2x64_ctr_t
    ctypedef r123array1x64 philox2x64_key_t
    struct s_philox2x64_state:
        philox2x64_ctr_t ctr
        philox2x64_key_t key

    ctypedef s_philox2x64_state philox2x64_state
    #######################################################
    # 4x64
    #######################################################
    ctypedef r123array4x64 philox4x64_ctr_t
    ctypedef r123array2x64 philox4x64_key_t
    struct s_philox4x64_state:
        philox4x64_ctr_t ctr
        philox4x64_key_t key

    ctypedef s_philox4x64_state philox4x64_state

    union R123_UINT_T:
        uint64_t u64
        uint32_t u32

    ctypedef R123_UINT_T r123_uint_t

    union PHILOX_STATE_T:
        philox2x32_state state2x32
        philox4x32_state state4x32
        philox2x64_state state2x64
        philox4x64_state state4x64

    ctypedef PHILOX_STATE_T philox_state_t

    struct PHILOX_ALL_T:
        philox_state_t state
        int buffer_pos
        r123_uint_t buffer[4]
        int has_uint32
        uint32_t uinteger
        int width
        int number

    ctypedef PHILOX_ALL_T philox_all_t
    uint64_t philox2x32_next64(philox_all_t *state) noexcept nogil
    uint32_t philox2x32_next32(philox_all_t *state) noexcept nogil
    double philox2x32_next_double(philox_all_t *state) noexcept nogil
    void philox2x32_advance(philox_all_t *state, uint32_t *step, int use_carry)
    uint64_t philox4x32_next64(philox_all_t *state) noexcept nogil
    uint32_t philox4x32_next32(philox_all_t *state) noexcept nogil
    double philox4x32_next_double(philox_all_t *state) noexcept nogil
    void philox4x32_advance(philox_all_t *state, uint32_t *step, int use_carry)
    uint64_t philox2x64_next64(philox_all_t *state) noexcept nogil
    uint32_t philox2x64_next32(philox_all_t *state) noexcept nogil
    double philox2x64_next_double(philox_all_t *state) noexcept nogil
    void philox2x64_advance(philox_all_t *state, uint64_t *step, int use_carry)
    uint64_t philox4x64_next64(philox_all_t *state) noexcept nogil
    uint32_t philox4x64_next32(philox_all_t *state) noexcept nogil
    double philox4x64_next_double(philox_all_t *state) noexcept nogil
    void philox4x64_advance(philox_all_t *state, uint64_t *step, int use_carry)
