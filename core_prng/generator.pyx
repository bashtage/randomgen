import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from common cimport *

from core_prng.core_prng import CorePRNG

np.import_array()

cdef class RandomGenerator:
    cdef public object __core_prng
    cdef anon_func_state anon_rng_func_state

    def __init__(self, prng=None):
        if prng is None:
            prng = CorePRNG()
        self.__core_prng = prng

        capsule = prng._anon_func_state
        cdef const char *anon_name = "Anon CorePRNG func_state"
        if not PyCapsule_IsValid(capsule, anon_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self.anon_rng_func_state = (<anon_func_state *>PyCapsule_GetPointer(capsule, anon_name))[0]

    def random_integer(self):
        cdef random_uint64_anon f = <random_uint64_anon>self.anon_rng_func_state.f
        return f(self.anon_rng_func_state.state)
