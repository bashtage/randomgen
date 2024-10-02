# cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

from cpython.pycapsule cimport PyCapsule_New
from libc.math cimport sqrt
from libc.stdint cimport uint64_t, uintptr_t

import numpy as np

cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random.bit_generator cimport BitGenerator as _BitGenerator


cdef bint RANDOMGEN_BIG_ENDIAN
cdef bint RANDOMGEN_LITTLE_ENDIAN
cdef uint64_t MAXSIZE

cdef class BitGenerator(_BitGenerator):
    cdef object mode
    cdef object _copy_seed(self)

cdef object benchmark(bitgen_t *bitgen, object lock, Py_ssize_t cnt, object method)
cdef object random_raw(bitgen_t *bitgen, object lock, object size, object output)
cdef object prepare_cffi(bitgen_t *bitgen)
cdef object prepare_ctypes(bitgen_t *bitgen)
cdef object wrap_int(object val, object bits)
cdef object check_state_array(object arr, np.npy_intp required_len,
                              int required_bits, object name)
cpdef object object_to_int(object val, object bits, object name,
                           int default_bits=*, object allowed_sizes=*)

cdef extern from "src/aligned_malloc/aligned_malloc.h":
    cdef void *PyArray_realloc_aligned(void *p, size_t n)
    cdef void *PyArray_malloc_aligned(size_t n)
    cdef void *PyArray_calloc_aligned(size_t n, size_t s)
    cdef void PyArray_free_aligned(void *p)

cdef inline double uint64_to_double(uint64_t rnd) noexcept nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef np.ndarray int_to_array(
        object value, object name, object bits, object uint_size
)

cdef view_little_endian(arr, dtype)

cdef byteswap_little_endian(arr)

cdef inline void compute_complex(double *rv_r,
                                 double *rv_i,
                                 double loc_r,
                                 double loc_i,
                                 double var_r,
                                 double var_i,
                                 double rho) noexcept nogil:
    cdef double scale_c, scale_i, scale_r

    scale_c = sqrt(1 - rho * rho)
    scale_r = sqrt(var_r)
    scale_i = sqrt(var_i)

    rv_i[0] = loc_i + scale_i * (rho * rv_r[0] + scale_c * rv_i[0])
    rv_r[0] = loc_r + scale_r * rv_r[0]


cdef object fully_qualified_name(instance)
