import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from common cimport *

np.import_array()

cdef struct state:
    uint64_t state

ctypedef state state_t

ctypedef uint64_t (*random_uint64)(state* st)

cdef struct func_state:
    state st
    random_uint64 f

ctypedef func_state func_state_t


cdef uint64_t _splitmix64(state* st):
    cdef uint64_t z
    cdef uint64_t c1 = 11400714819323198485
    cdef uint64_t c2 = 13787848793156543929
    cdef uint64_t c3 = 10723151780598845931
    st[0].state += c1 # 0x9E3779B97F4A7C15
    z = <uint64_t>st[0].state
    z = (z ^ (z >> 30)) * c2 # 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * c3 # 0x94D049BB133111EB
    return z ^ (z >> 31)

cdef uint64_t _splitmix64_anon(void* st):
    return _splitmix64(<state *> st)

cdef class CorePRNG:
    cdef state rng_state
    cdef func_state rng_func_state
    cdef anon_func_state anon_func_state
    cdef public object _func_state
    cdef public object _anon_func_state

    def __init__(self):
        self.rng_state.state = 17013192669731687406
        self.rng_func_state.st = self.rng_state
        self.rng_func_state.f = &_splitmix64
        cdef const char *name = "CorePRNG func_state"
        self._func_state = PyCapsule_New(<void *>&self.rng_func_state,
                                         name, NULL)
        self.anon_func_state.state = <void *>&self.rng_state
        self.anon_func_state.f = <void *>&_splitmix64_anon
        cdef const char *anon_name = "Anon CorePRNG func_state"
        self._anon_func_state = PyCapsule_New(<void *>&self.anon_func_state,
                                              anon_name, NULL)

    def random(self):
        return _splitmix64(&self.rng_state)

    def get_state(self):
        return self.rng_state.state

    def get_state2(self):
        return (<uint64_t *>self.anon_func_state.state)[0]

    def set_state(self, uint64_t value):
        self.rng_state.state = value

    @staticmethod
    cdef uint64_t c_random(state *st):
        return _splitmix64(st)

    @staticmethod
    cdef uint64_t c_random_void(void *st):
        return _splitmix64(<state *> st)

    def random_using_c(self):
        return CorePRNG.c_random(&self.rng_state)

    def random_using_c_random_void(self):
        return CorePRNG.c_random_void(<void *> &self.rng_state)

    def random_using_struct(self):
        return self.rng_func_state.f(&self.rng_func_state.st)

# cdef class RandomGenerator:
#     cdef object __core_prng
#     cdef func_state rng_func_state
#     cdef anon_func_state anon_rng_func_state
#
#     def __init__(self, prng=None):
#         if prng is None:
#             prng = CorePRNG()
#         self.__core_prng = prng
#         capsule = prng._func_state
#         cdef const char *name = "CorePRNG func_state"
#         if not PyCapsule_IsValid(capsule, name):
#             raise ValueError("Invalid pointer to func_state")
#         self.rng_func_state = (<func_state *>PyCapsule_GetPointer(capsule, name))[0]
#
#         capsule = prng._anon_func_state
#         cdef const char *anon_name = "Anon CorePRNG func_state"
#         if not PyCapsule_IsValid(capsule, anon_name):
#             raise ValueError("Invalid pointer to anon_func_state")
#         self.anon_rng_func_state = (<anon_func_state *>PyCapsule_GetPointer(capsule, anon_name))[0]
#
#     def random_integer(self):
#         return self.__core_prng.random()
#
#     def random_using_struct(self):
#         return self.rng_func_state.f(&self.rng_func_state.st)
#
#     def random_using_anon_struct(self):
#         cdef random_uint64_anon f = <random_uint64_anon>self.anon_rng_func_state.f
#         return f(self.anon_rng_func_state.state)
