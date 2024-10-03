
cimport numpy as np
from libc.stdint cimport uint32_t


cdef class SeedSequence(object):
    cdef readonly object entropy
    cdef readonly tuple spawn_key
    cdef readonly Py_ssize_t pool_size
    cdef readonly object pool
    cdef readonly uint32_t n_children_spawned

    cdef mix_entropy(self, np.ndarray[np.npy_uint32, ndim=1] mixer,
                     np.ndarray[np.npy_uint32, ndim=1] entropy_array)
    cdef get_assembled_entropy(self)

cdef class SeedlessSequence(object):
    pass
