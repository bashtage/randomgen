cimport numpy as np

cdef class SeedSequence(object):
    cdef readonly object entropy
    cdef readonly tuple spawn_key
    cdef readonly int pool_size
    cdef readonly object pool
    cdef readonly int n_children_spawned

    cdef mix_entropy(self, np.ndarray[np.npy_uint32, ndim=1] mixer,
                     np.ndarray[np.npy_uint32, ndim=1] entropy_array)
    cdef get_assembled_entropy(self)

cdef class SeedlessSequence(object):
    pass
