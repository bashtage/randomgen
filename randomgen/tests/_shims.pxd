from numpy.random cimport bitgen_t
from randomgen.tests.data.compute_hashes import bit_gen
cdef extern from "_shim_dist.h":
    double double0_func(bitgen_t *state);
    double double1_func(bitgen_t *state, double a);
    double double2_func(bitgen_t *state, double a, double b);
    double double3_func(bitgen_t *state, double a, double b, double c);
