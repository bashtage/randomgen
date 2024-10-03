# cython: language_level=3
import numpy as np

cimport cython
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from randomgen.common cimport *
from randomgen.distributions cimport random_gauss_zig

from randomgen.xoroshiro128 import Xoroshiro128


@cython.boundscheck(False)
@cython.wraparound(False)
def normals_zig(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double[::1] random_values

    x = Xoroshiro128()
    capsule = x.capsule
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n)
    for i in range(n):
        random_values[i] = random_gauss_zig(rng)
    randoms = np.asarray(random_values)
    return randoms


@cython.boundscheck(False)
@cython.wraparound(False)
def uniforms(Py_ssize_t n):
    cdef Py_ssize_t i
    cdef bitgen_t *rng
    cdef const char *capsule_name = "BitGenerator"
    cdef double[::1] random_values

    x = Xoroshiro128()
    capsule = x.capsule
    # Optional check that the capsule if from a BitGenerator
    if not PyCapsule_IsValid(capsule, capsule_name):
        raise ValueError("Invalid pointer to anon_func_state")
    # Cast the pointer
    rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
    random_values = np.empty(n)
    for i in range(n):
        # Call the function
        random_values[i] = rng.next_double(rng.state)
    randoms = np.asarray(random_values)
    return randoms
