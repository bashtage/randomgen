from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from numpy.random cimport bitgen_t

from randomgen.broadcasting cimport (
    constraint_type,
    cont,
    cont_f,
    disc,
    float_fill_from_double,
)
from randomgen.common cimport (
    byteswap_little_endian,
    int_to_array,
    object_to_int,
    view_little_endian,
)

from Cython.Includes.cpython.datetime import noexcept

from libc.stdint cimport int64_t, uint64_t


cdef extern from "_shim_dist.h":
    double double0_func(bitgen_t *state) noexcept nogil
    double double1_func(bitgen_t *state, double a) noexcept nogil
    double double2_func(bitgen_t *state, double a, double b) noexcept nogil
    double double3_func(bitgen_t *state, double a, double b, double c) noexcept nogil

    float float_0(bitgen_t *state) noexcept nogil
    float float_1(bitgen_t *state, float a) noexcept nogil

    int64_t int_0(bitgen_t *state) noexcept nogil
    int64_t int_d(bitgen_t *state, double a) noexcept nogil
    int64_t int_dd(bitgen_t *state, double a, double b) noexcept nogil
    int64_t int_di(bitgen_t *state, double a, uint64_t b) noexcept nogil
    int64_t int_i(bitgen_t *state, int64_t a) noexcept nogil
    int64_t int_iii(bitgen_t *state, int64_t a, int64_t b, int64_t c) noexcept nogil
