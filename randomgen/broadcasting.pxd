cimport numpy as np
from cpython cimport PyFloat_AsDouble
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from numpy.random cimport bitgen_t

cimport randomgen.api as api


cdef double POISSON_LAM_MAX
cdef double LEGACY_POISSON_LAM_MAX

cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_POSITIVE_NOT_NAN
    CONS_BOUNDED_0_1
    CONS_BOUNDED_GT_0_1
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON
    LEGACY_CONS_POISSON

ctypedef ConstraintType constraint_type

cdef int check_constraint(double val, object name, constraint_type cons) except -1
cdef int check_array_constraint(
        np.ndarray val, object name, constraint_type cons
) except -1
cdef check_output(object out, object dtype, object size, bint require_c_array)

ctypedef double (*random_double_fill)(
        bitgen_t *state, np.npy_intp count, double* out
) noexcept nogil
ctypedef double (*random_double_0)(void *state) noexcept nogil
ctypedef double (*random_double_1)(void *state, double a) noexcept nogil
ctypedef double (*random_double_2)(void *state, double a, double b) noexcept nogil
ctypedef double (*random_double_3)(void *state, double a, double b, double c) noexcept nogil

ctypedef float (*random_float_0)(bitgen_t *state) noexcept nogil
ctypedef float (*random_float_1)(bitgen_t *state, float a) noexcept nogil

ctypedef int64_t (*random_uint_0)(void *state) noexcept nogil
ctypedef int64_t (*random_uint_d)(void *state, double a) noexcept nogil
ctypedef int64_t (*random_uint_dd)(void *state, double a, double b) noexcept nogil
ctypedef int64_t (*random_uint_di)(void *state, double a, uint64_t b) noexcept nogil
ctypedef int64_t (*random_uint_i)(void *state, int64_t a) noexcept nogil
ctypedef int64_t (*random_uint_iii)(void *state, int64_t a, int64_t b, int64_t c) noexcept nogil

ctypedef uint32_t (*random_uint_0_32)(bitgen_t *state) noexcept nogil
ctypedef uint32_t (*random_uint_1_i_32)(bitgen_t *state, uint32_t a) noexcept nogil

ctypedef int32_t (*random_int_2_i_32)(bitgen_t *state, int32_t a, int32_t b) noexcept nogil
ctypedef int64_t (*random_int_2_i)(bitgen_t *state, int64_t a, int64_t b) noexcept nogil

cdef object double_fill(
        void *func, bitgen_t *state, object size, object lock, object out
)

cdef object float_fill(
        void *func, bitgen_t *state, object size, object lock, object out
)

cdef object float_fill_from_double(
        void *func, bitgen_t *state, object size, object lock, object out
)

cdef object cont(void *func, void *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out)

cdef object disc(void *func, void *state, object size, object lock,
                 int narg_double, int narg_int64,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

cdef object cont_f(void *func, bitgen_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out)

cdef object cont_broadcast_3(void *func,
                             void *state,
                             object size, object lock,
                             np.ndarray a_arr,
                             object a_name,
                             constraint_type a_constraint,
                             np.ndarray b_arr,
                             object b_name,
                             constraint_type b_constraint,
                             np.ndarray c_arr,
                             object c_name,
                             constraint_type c_constraint)

cdef object discrete_broadcast_iii(void *func,
                                   void *state,
                                   object size,
                                   object lock,
                                   np.ndarray a_arr,
                                   object a_name,
                                   constraint_type a_constraint,
                                   np.ndarray b_arr,
                                   object b_name,
                                   constraint_type b_constraint,
                                   np.ndarray c_arr,
                                   object c_name,
                                   constraint_type c_constraint)
