from randomgen.common cimport (
    byteswap_little_endian,
    int_to_array,
    object_to_int,
    view_little_endian,
)

from randomgen.chacha import ChaCha

from numpy.random cimport bitgen_t

import numpy as np

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from randomgen.broadcasting cimport constraint_type, cont


def view_little_endian_shim(arr, dtype):
    return view_little_endian(arr, dtype)


def int_to_array_shim(value, name, bits, uint_size):
    return np.asarray(int_to_array(value, name, bits, uint_size))


def byteswap_little_endian_shim(arr):
    return byteswap_little_endian(arr)

def object_to_int_shim(val, bits, name, default_bits=64, allowed_sizes=(64,)):
    return object_to_int(val, bits, name, default_bits, allowed_sizes)

cdef double double0_func(void *state):
    return 3.141592

cdef double double1_func(void *state, double a):
    return a

cdef double double2_func(void *state, double a, double b):
    return a+b

cdef double double3_func(void *state, double a, double b, double c):
    return a+b+c

cdef bitgen_t _bitgen
cdef const char *name = "BitGenerator"

chacha = ChaCha()
capsule = chacha.capsule
_bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]

def cont_0(size=None, out=None):
    return cont(&double0_func, &_bitgen ,size, chacha.lock, 0,
               0.0, "", constraint_type.CONS_NONE,
               0.0, "", constraint_type.CONS_NONE,
               0.0, "", constraint_type.CONS_NONE,
               out)
def cont_1(a, size=None, out=None):
    return cont(&double1_func, &_bitgen ,size, chacha.lock, 1,
               a, "a", constraint_type.CONS_NON_NEGATIVE,
               0.0, "", constraint_type.CONS_NONE,
               0.0, "", constraint_type.CONS_NONE,
               out)

def cont_2(a, b, size=None, out=None):
    return cont(&double2_func, &_bitgen ,size, chacha.lock, 2,
               a, "a", constraint_type.CONS_POSITIVE,
               b, "b", constraint_type.CONS_POSITIVE_NOT_NAN,
               0.0, "", constraint_type.CONS_NONE,
               out)

def cont_3(a, b, c, size=None, out=None):
    return cont(&double3_func, &_bitgen ,size, chacha.lock, 3,
               a, "a", constraint_type.CONS_BOUNDED_0_1,
               b, "b", constraint_type.CONS_BOUNDED_GT_0_1,
               c, "c", constraint_type.CONS_GT_1,
               out)

def cont_3_alt_cons(a, b, c, size=None, out=None):
    return cont(&double3_func, &_bitgen, size, chacha.lock, 3,
                a, "a", constraint_type.CONS_GTE_1,
                b, "b", constraint_type.CONS_POISSON,
                c, "c", constraint_type.LEGACY_CONS_POISSON,
                out)

# Iterations
# Scalar or vector parameters (must be broadcastable)
# size
# out, if used with size, and parameters, must be compatible
