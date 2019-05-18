#ifndef _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_
#define _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_


#include "../distributions/distributions.h"

typedef struct aug_brng {
  brng_t *basicrng;
  int has_gauss;
  double gauss;
} aug_brng_t;

extern double legacy_gauss(aug_brng_t *aug_state);
extern double legacy_standard_exponential(aug_brng_t *aug_state);
extern double legacy_pareto(aug_brng_t *aug_state, double a);
extern double legacy_weibull(aug_brng_t *aug_state, double a);
extern double legacy_power(aug_brng_t *aug_state, double a);
extern double legacy_gamma(aug_brng_t *aug_state, double shape, double scale);
extern double legacy_pareto(aug_brng_t *aug_state, double a);
extern double legacy_weibull(aug_brng_t *aug_state, double a);
extern double legacy_chisquare(aug_brng_t *aug_state, double df);
extern double legacy_noncentral_chisquare(aug_brng_t *aug_state, double df,
                                          double nonc);

extern double legacy_noncentral_f(aug_brng_t *aug_state, double dfnum,
                                  double dfden, double nonc);
extern double legacy_wald(aug_brng_t *aug_state, double mean, double scale);
extern double legacy_lognormal(aug_brng_t *aug_state, double mean,
                               double sigma);
extern double legacy_standard_t(aug_brng_t *aug_state, double df);
extern int64_t legacy_negative_binomial(aug_brng_t *aug_state, double n,
                                        double p);
extern double legacy_standard_cauchy(aug_brng_t *state);
extern double legacy_beta(aug_brng_t *aug_state, double a, double b);
extern double legacy_f(aug_brng_t *aug_state, double dfnum, double dfden);
extern double legacy_normal(aug_brng_t *aug_state, double loc, double scale);
extern double legacy_standard_gamma(aug_brng_t *aug_state, double shape);
extern double legacy_exponential(aug_brng_t *aug_state, double scale);

extern int64_t legacy_negative_binomial(aug_brng_t *aug_state, double n,
                                        double p);
extern int64_t legacy_random_hypergeometric(brng_t *brng_state,
                                            int64_t good, int64_t bad,
                                            int64_t sample);
extern int64_t legacy_random_logseries(brng_t *brng_state, double p);
extern int64_t legacy_random_poisson(brng_t *brng_state, double lam);
extern int64_t legacy_random_zipf(brng_t *brng_state, double a);
extern int64_t legacy_random_geometric(brng_t *brng_state, double p);

void legacy_random_bounded_uint64_fill(aug_brng_t *aug_state, uint64_t off,
                                       uint64_t rng, npy_intp cnt,
                                       uint64_t *out);
void legacy_random_bounded_uint32_fill(aug_brng_t *aug_state, uint32_t off,
                                       uint32_t rng, npy_intp cnt,
                                       uint32_t *out);
void legacy_random_bounded_uint16_fill(aug_brng_t *aug_state, uint16_t off,
                                       uint16_t rng, npy_intp cnt,
                                       uint16_t *out);
void legacy_random_bounded_uint8_fill(aug_brng_t *aug_state, uint8_t off,
                                      uint8_t rng, npy_intp cnt, uint8_t *out);
void legacy_random_bounded_bool_fill(aug_brng_t *aug_state, npy_bool off,
                                     npy_bool rng, npy_intp cnt, npy_bool *out);

void legacy_random_multinomial(brng_t *brng_state, RAND_INT_TYPE n,
                               RAND_INT_TYPE *mnix, double *pix, npy_intp d,
                               binomial_t *binomial);
#endif
