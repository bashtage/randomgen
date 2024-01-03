#cython: language_level=3

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

ctypedef uint64_t (*next_uint64_t)(void *st) noexcept nogil
ctypedef uint32_t (*next_uint32_t)(void *st) noexcept nogil
ctypedef double (*next_double_t)(void *st) noexcept nogil

cdef extern from "src/distributions/distributions.h":

    struct s_binomial_t:
        int has_binomial
        double psave
        int64_t nsave
        double r
        double q
        double fm
        int64_t m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    ctypedef s_binomial_t binomial_t

    struct bitgen:
        void *state
        next_uint64_t next_uint64
        next_uint32_t next_uint32
        next_double_t next_double
        next_uint64_t next_raw

    ctypedef bitgen bitgen_t

    double random_double(bitgen_t *bitgen_state) noexcept nogil
    void random_double_fill(bitgen_t* bitgen_state, np.npy_intp cnt, double *out) noexcept nogil
    double random_standard_exponential(bitgen_t *bitgen_state) noexcept nogil
    void random_standard_exponential_fill(bitgen_t *bitgen_state, np.npy_intp cnt, double *out) noexcept nogil
    double random_standard_exponential_zig(bitgen_t *bitgen_state) noexcept nogil
    void random_standard_exponential_zig_fill(bitgen_t *bitgen_state, np.npy_intp cnt, double *out) noexcept nogil
    double random_gauss_zig(bitgen_t* bitgen_state) noexcept nogil
    void random_gauss_zig_fill(bitgen_t *bitgen_state, np.npy_intp count, double *out) noexcept nogil
    double random_standard_gamma_zig(bitgen_t *bitgen_state, double shape) noexcept nogil

    float random_float(bitgen_t *bitgen_state) noexcept nogil
    float random_standard_exponential_f(bitgen_t *bitgen_state) noexcept nogil
    float random_standard_exponential_zig_f(bitgen_t *bitgen_state) noexcept nogil
    float random_gauss_zig_f(bitgen_t* bitgen_state) noexcept nogil
    float random_standard_gamma_f(bitgen_t *bitgen_state, float shape) noexcept nogil
    float random_standard_gamma_zig_f(bitgen_t *bitgen_state, float shape) noexcept nogil

    int64_t random_positive_int64(bitgen_t *bitgen_state) noexcept nogil
    int32_t random_positive_int32(bitgen_t *bitgen_state) noexcept nogil
    int64_t random_positive_int(bitgen_t *bitgen_state) noexcept nogil
    uint64_t random_uint(bitgen_t *bitgen_state) noexcept nogil

    double random_normal_zig(bitgen_t *bitgen_state, double loc, double scale) noexcept nogil

    double random_gamma(bitgen_t *bitgen_state, double shape, double scale) noexcept nogil
    float random_gamma_float(bitgen_t *bitgen_state, float shape, float scale) noexcept nogil

    double random_exponential(bitgen_t *bitgen_state, double scale) noexcept nogil
    double random_uniform(bitgen_t *bitgen_state, double lower, double range) noexcept nogil
    double random_beta(bitgen_t *bitgen_state, double a, double b) noexcept nogil
    double random_chisquare(bitgen_t *bitgen_state, double df) noexcept nogil
    double random_f(bitgen_t *bitgen_state, double dfnum, double dfden) noexcept nogil
    double random_standard_cauchy(bitgen_t *bitgen_state) noexcept nogil
    double random_pareto(bitgen_t *bitgen_state, double a) noexcept nogil
    double random_weibull(bitgen_t *bitgen_state, double a) noexcept nogil
    double random_power(bitgen_t *bitgen_state, double a) noexcept nogil
    double random_laplace(bitgen_t *bitgen_state, double loc, double scale) noexcept nogil
    double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) noexcept nogil
    double random_logistic(bitgen_t *bitgen_state, double loc, double scale) noexcept nogil
    double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma) noexcept nogil
    double random_rayleigh(bitgen_t *bitgen_state, double mode) noexcept nogil
    double random_standard_t(bitgen_t *bitgen_state, double df) noexcept nogil
    double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                       double nonc) noexcept nogil
    double random_noncentral_f(bitgen_t *bitgen_state, double dfnum,
                               double dfden, double nonc) noexcept nogil
    double random_wald(bitgen_t *bitgen_state, double mean, double scale) noexcept nogil
    double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa) noexcept nogil
    double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                             double right) noexcept nogil

    int64_t random_poisson(bitgen_t *bitgen_state, double lam) noexcept nogil
    int64_t random_negative_binomial(bitgen_t *bitgen_state, double n, double p) noexcept nogil
    int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n, binomial_t *binomial) noexcept nogil
    int64_t random_logseries(bitgen_t *bitgen_state, double p) noexcept nogil
    int64_t random_geometric_search(bitgen_t *bitgen_state, double p) noexcept nogil
    int64_t random_geometric_inversion(bitgen_t *bitgen_state, double p) noexcept nogil
    int64_t random_geometric(bitgen_t *bitgen_state, double p) noexcept nogil
    int64_t random_zipf(bitgen_t *bitgen_state, double a) noexcept nogil
    int64_t random_hypergeometric(bitgen_t *bitgen_state, int64_t good,
                                  int64_t bad, int64_t sample) noexcept nogil

    uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) noexcept nogil

    # Generate random uint64 numbers in closed interval [off, off + rng].
    uint64_t random_bounded_uint64(bitgen_t *bitgen_state,
                                   uint64_t off, uint64_t rng,
                                   uint64_t mask, bint use_masked) noexcept nogil

    # Generate random uint32 numbers in closed interval [off, off + rng].
    uint32_t random_buffered_bounded_uint32(bitgen_t *bitgen_state,
                                            uint32_t off, uint32_t rng,
                                            uint32_t mask, bint use_masked,
                                            int *bcnt, uint32_t *buf) noexcept nogil
    uint16_t random_buffered_bounded_uint16(bitgen_t *bitgen_state,
                                            uint16_t off, uint16_t rng,
                                            uint16_t mask, bint use_masked,
                                            int *bcnt, uint32_t *buf) noexcept nogil
    uint8_t random_buffered_bounded_uint8(bitgen_t *bitgen_state,
                                          uint8_t off, uint8_t rng,
                                          uint8_t mask, bint use_masked,
                                          int *bcnt, uint32_t *buf) noexcept nogil
    np.npy_bool random_buffered_bounded_bool(bitgen_t *bitgen_state,
                                             np.npy_bool off, np.npy_bool rng,
                                             np.npy_bool mask, bint use_masked,
                                             int *bcnt, uint32_t *buf) noexcept nogil

    void random_bounded_uint64_fill(bitgen_t *bitgen_state,
                                    uint64_t off, uint64_t rng, np.npy_intp cnt,
                                    bint use_masked,
                                    uint64_t *out) noexcept nogil
    void random_bounded_uint32_fill(bitgen_t *bitgen_state,
                                    uint32_t off, uint32_t rng, np.npy_intp cnt,
                                    bint use_masked,
                                    uint32_t *out) noexcept nogil
    void random_bounded_uint16_fill(bitgen_t *bitgen_state,
                                    uint16_t off, uint16_t rng, np.npy_intp cnt,
                                    bint use_masked,
                                    uint16_t *out) noexcept nogil
    void random_bounded_uint8_fill(bitgen_t *bitgen_state,
                                   uint8_t off, uint8_t rng, np.npy_intp cnt,
                                   bint use_masked,
                                   uint8_t *out) noexcept nogil
    void random_bounded_bool_fill(bitgen_t *bitgen_state,
                                  np.npy_bool off, np.npy_bool rng, np.npy_intp cnt,
                                  bint use_masked,
                                  np.npy_bool *out) noexcept nogil

    void random_multinomial(bitgen_t *bitgen_state, int64_t n, int64_t *mnix,
                            double *pix, np.npy_intp d, binomial_t *binomial) noexcept nogil
    void random_wishart_large_df(bitgen_t *bitgen_state, int64_t df, np.npy_intp dim, np.npy_intp num, double *w, double *n) noexcept nogil
    void random_long_double_fill(bitgen_t* bitgen_state, np.npy_intp cnt, long double *out) noexcept nogil
    int random_long_double_size() noexcept nogil
    long double random_long_double(bitgen_t *bitgen_state) noexcept nogil
