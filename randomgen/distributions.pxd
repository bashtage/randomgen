# cython: language_level=3

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intptr_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

import numpy as np

cimport numpy as np
from numpy.random cimport bitgen_t

ctypedef uint64_t (*next_uint64_t)(void *st) noexcept nogil
ctypedef uint32_t (*next_uint32_t)(void *st) noexcept nogil
ctypedef double (*next_double_t)(void *st) noexcept nogil

cdef extern from "src/distributions/rg-distributions.h":
    void random_double_fill(bitgen_t * bitgen_state, np.npy_intp cnt, double *out) noexcept nogil
    void random_long_double_fill(bitgen_t * bitgen_state, np.npy_intp cnt, long double *out) noexcept nogil
    void random_wishart_large_df(bitgen_t *bitgen_state, int64_t df, np.npy_intp dim, np.npy_intp num, double *w, double *n) noexcept nogil

    long double random_long_double(bitgen_t *bitgen_state) noexcept nogil

    float random_float(bitgen_t *bitgen_state) noexcept nogil

    int random_long_double_size() noexcept nogil
