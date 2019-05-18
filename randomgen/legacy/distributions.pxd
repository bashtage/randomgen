#cython: language_level=3

from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t)

import numpy as np
cimport numpy as np

from randomgen.distributions cimport brng_t, binomial_t

cdef extern from "../src/legacy/distributions-boxmuller.h":

    struct aug_brng:
        brng_t *basicrng
        int has_gauss
        double gauss

    ctypedef aug_brng aug_brng_t

    double legacy_gauss(aug_brng_t *aug_state) nogil
    double legacy_pareto(aug_brng_t *aug_state, double a) nogil
    double legacy_weibull(aug_brng_t *aug_state, double a) nogil
    double legacy_standard_gamma(aug_brng_t *aug_state, double shape) nogil
    double legacy_normal(aug_brng_t *aug_state, double loc, double scale) nogil
    double legacy_standard_t(aug_brng_t *aug_state, double df) nogil

    double legacy_standard_exponential(aug_brng_t *aug_state) nogil
    double legacy_power(aug_brng_t *aug_state, double a) nogil
    double legacy_gamma(aug_brng_t *aug_state, double shape, double scale) nogil
    double legacy_power(aug_brng_t *aug_state, double a) nogil
    double legacy_chisquare(aug_brng_t *aug_state, double df) nogil
    double legacy_noncentral_chisquare(aug_brng_t *aug_state, double df,
                                    double nonc) nogil
    double legacy_noncentral_f(aug_brng_t *aug_state, double dfnum, double dfden,
                            double nonc) nogil
    double legacy_wald(aug_brng_t *aug_state, double mean, double scale) nogil
    double legacy_lognormal(aug_brng_t *aug_state, double mean, double sigma) nogil
    int64_t legacy_negative_binomial(aug_brng_t *aug_state, double n, double p) nogil
    int64_t legacy_random_hypergeometric(brng_t *bitgen_state, int64_t good, int64_t bad, int64_t sample) nogil
    int64_t legacy_random_logseries(brng_t *bitgen_state, double p) nogil
    int64_t legacy_random_poisson(brng_t *bitgen_state, double lam) nogil
    int64_t legacy_random_zipf(brng_t *bitgen_state, double a) nogil
    int64_t legacy_random_geometric(brng_t *bitgen_state, double p) nogil
    void legacy_random_multinomial(brng_t *bitgen_state, long n, long *mnix, double *pix, np.npy_intp d, binomial_t *binomial) nogil
    double legacy_standard_cauchy(aug_brng_t *state) nogil
    double legacy_beta(aug_brng_t *aug_state, double a, double b) nogil
    double legacy_f(aug_brng_t *aug_state, double dfnum, double dfden) nogil
    double legacy_exponential(aug_brng_t *aug_state, double scale) nogil
    double legacy_power(aug_brng_t *state, double a) nogil

    void legacy_random_bounded_uint64_fill(aug_brng_t *state, uint64_t off, uint64_t rng, np.npy_intp cnt, uint64_t *out) nogil
    void legacy_random_bounded_uint32_fill(aug_brng_t *state, uint32_t off, uint32_t rng, np.npy_intp cnt, uint32_t *out) nogil
    void legacy_random_bounded_uint16_fill(aug_brng_t *state, uint16_t off, uint16_t rng, np.npy_intp cnt, uint16_t *out) nogil
    void legacy_random_bounded_uint8_fill(aug_brng_t *state, uint8_t off, uint8_t rng, np.npy_intp cnt, uint8_t *out) nogil
    void legacy_random_bounded_bool_fill(aug_brng_t *state, np.npy_bool off, np.npy_bool rng, np.npy_bool cnt, np.npy_bool*out) nogil
