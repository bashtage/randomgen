# cython: language_level=3, boundscheck=False, wraparound=False
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from libc.stdint cimport uint32_t

import numpy as np

cimport cython
cimport numpy as np

from randomgen.common cimport bitgen_t, uint64_to_double
from randomgen.xoshiro256 cimport Xoshiro256, xoshiro256_next64

from randomgen.xoshiro256 import Xoshiro256

np.import_array()


def uniform_using_bitgen(Py_ssize_t n):
    """
    Example showing how to use the standarzed interface provided by bitgen
    """

    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double sum = 0.0

    x = Xoshiro256()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <bitgen_t *>PyCapsule_GetPointer(capsule, capsule_name)

    for i in range(n):
        sum += rng.next_double(rng.state)
    return sum / n


def uniform_using_lowlevel(Py_ssize_t n):
    """
    Example showing how to use the low-level interface provided by the pxd file
    """
    cdef Py_ssize_t i
    cdef double[::1] random_values
    cdef double sum = 0.0

    x = Xoshiro256()

    for i in range(n):
        sum += uint64_to_double(xoshiro256_next64(&(<Xoshiro256>x).rng_state))
    return sum / n
