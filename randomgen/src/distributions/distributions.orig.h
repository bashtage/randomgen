#ifndef _RANDOMDGEN__DISTRIBUTIONS_H_
#define _RANDOMDGEN__DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "../common/randomgen_config_numpy.h"
#include "../common/randomgen_config.h"

/*
 * RAND_INT_TYPE is used to share integer generators with RandomState which
 * used long in place of int64_t. If changing a distribution that uses
 * RAND_INT_TYPE, then the original unmodified copy must be retained for
 * use in RandomState by copying to the legacy distributions source file.
 */
#ifdef RANDOMGEN_LEGACY
#define RAND_INT_TYPE long
#define RAND_INT_MAX LONG_MAX
#else
#define RAND_INT_TYPE int64_t
#define RAND_INT_MAX INT64_MAX
#endif

#ifdef _WIN32
#if _MSC_VER == 1500

static NPY_INLINE int64_t llabs(int64_t x) {
  int64_t o;
  if (x < 0) {
    o = -x;
  } else {
    o = x;
  }
  return o;
}
#endif
#endif

#ifdef DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR extern
#endif

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? x : y)
#define MAX(x, y) (((x) > (y)) ? x : y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

typedef struct s_binomial_t {
  int has_binomial; /* !=0: following parameters initialized for binomial */
  double psave;
  int64_t nsave;
  double r;
  double q;
  double fm;
  int64_t m;
  double p1;
  double xm;
  double xl;
  double xr;
  double c;
  double laml;
  double lamr;
  double p2;
  double p3;
  double p4;
} binomial_t;

/* Inline generators for internal use */
static NPY_INLINE uint32_t next_uint32(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint32(bitgen_state->state);
}

static NPY_INLINE uint64_t next_uint64(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint64(bitgen_state->state);
}

static NPY_INLINE float next_float(bitgen_t *bitgen_state) {
  return (next_uint32(bitgen_state) >> 9) * (1.0f / 8388608.0f);
}

static NPY_INLINE double next_double(bitgen_t *bitgen_state) {
  return bitgen_state->next_double(bitgen_state->state);
}

/* Bounded generators */
static NPY_INLINE uint64_t gen_mask(uint64_t max) {
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}

/* Generate 16 bit random numbers using a 32 bit buffer. */
static NPY_INLINE uint16_t buffered_uint16(bitgen_t *bitgen_state, int *bcnt,
                                           uint32_t *buf) {
  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 1;
  } else {
    buf[0] >>= 16;
    bcnt[0] -= 1;
  }

  return (uint16_t)buf[0];
}

/* Generate 8 bit random numbers using a 32 bit buffer. */
static NPY_INLINE uint8_t buffered_uint8(bitgen_t *bitgen_state, int *bcnt,
                                         uint32_t *buf) {
  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 3;
  } else {
    buf[0] >>= 8;
    bcnt[0] -= 1;
  }

  return (uint8_t)buf[0];
}

/* Static `masked rejection` function called by random_bounded_uint64(...) */
static NPY_INLINE uint64_t bounded_masked_uint64(bitgen_t *bitgen_state,
                                                 uint64_t rng, uint64_t mask) {
  uint64_t val;

  while ((val = (next_uint64(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* Static `masked rejection` function called by
 * random_buffered_bounded_uint32(...) */
static NPY_INLINE uint32_t
buffered_bounded_masked_uint32(bitgen_t *bitgen_state, uint32_t rng,
                               uint32_t mask, int *bcnt, uint32_t *buf) {
  /*
   * The buffer and buffer count are not used here but are included to allow
   * this function to be templated with the similar uint8 and uint16
   * functions
   */

  uint32_t val;

  while ((val = (next_uint32(bitgen_state) & mask)) > rng)
    ;

  return val;
}

/* Static `masked rejection` function called by
 * random_buffered_bounded_uint16(...) */
static NPY_INLINE uint16_t
buffered_bounded_masked_uint16(bitgen_t *bitgen_state, uint16_t rng,
                               uint16_t mask, int *bcnt, uint32_t *buf) {
  uint16_t val;

  while ((val = (buffered_uint16(bitgen_state, bcnt, buf) & mask)) > rng)
    ;

  return val;
}

/* Static `masked rejection` function called by
 * random_buffered_bounded_uint8(...) */
static NPY_INLINE uint8_t buffered_bounded_masked_uint8(bitgen_t *bitgen_state,
                                                        uint8_t rng,
                                                        uint8_t mask, int *bcnt,
                                                        uint32_t *buf) {
  uint8_t val;

  while ((val = (buffered_uint8(bitgen_state, bcnt, buf) & mask)) > rng)
    ;

  return val;
}

static NPY_INLINE npy_bool buffered_bounded_bool(bitgen_t *bitgen_state,
                                                 npy_bool off, npy_bool rng,
                                                 npy_bool mask, int *bcnt,
                                                 uint32_t *buf) {
  if (rng == 0)
    return off;
  if (!(bcnt[0])) {
    buf[0] = next_uint32(bitgen_state);
    bcnt[0] = 31;
  } else {
    buf[0] >>= 1;
    bcnt[0] -= 1;
  }
  return (buf[0] & 0x00000001UL) != 0;
}

DECLDIR float random_float(bitgen_t *bitgen_state);
DECLDIR double random_double(bitgen_t *bitgen_state);
DECLDIR int random_long_double_size(void);
DECLDIR long double random_long_double(bitgen_t *bitgen_state);
DECLDIR void random_long_double_fill(bitgen_t *bitgen_state, npy_intp cnt,
                                     long double *out);
DECLDIR void random_double_fill(bitgen_t *bitgen_state, npy_intp cnt,
                                double *out);

DECLDIR int64_t random_positive_int64(bitgen_t *bitgen_state);
DECLDIR int32_t random_positive_int32(bitgen_t *bitgen_state);
DECLDIR int64_t random_positive_int(bitgen_t *bitgen_state);
DECLDIR uint64_t random_uint(bitgen_t *bitgen_state);

DECLDIR double random_standard_exponential(bitgen_t *bitgen_state);
DECLDIR void random_standard_exponential_fill(bitgen_t *bitgen_state,
                                              npy_intp cnt, double *out);
DECLDIR float random_standard_exponential_f(bitgen_t *bitgen_state);
DECLDIR double random_standard_exponential_zig(bitgen_t *bitgen_state);
DECLDIR void random_standard_exponential_zig_fill(bitgen_t *bitgen_state,
                                                  npy_intp cnt, double *out);
DECLDIR float random_standard_exponential_zig_f(bitgen_t *bitgen_state);

/*
DECLDIR double random_gauss(bitgen_t *bitgen_state);
DECLDIR float random_gauss_f(bitgen_t *bitgen_state);
*/
DECLDIR double random_gauss_zig(bitgen_t *bitgen_state);
DECLDIR float random_gauss_zig_f(bitgen_t *bitgen_state);
DECLDIR void random_gauss_zig_fill(bitgen_t *bitgen_state, npy_intp cnt,
                                   double *out);

/*
DECLDIR double random_standard_gamma(bitgen_t *bitgen_state, double shape);
DECLDIR float random_standard_gamma_f(bitgen_t *bitgen_state, float shape);
*/
DECLDIR double random_standard_gamma_zig(bitgen_t *bitgen_state, double shape);
DECLDIR float random_standard_gamma_zig_f(bitgen_t *bitgen_state, float shape);

/*
DECLDIR double random_normal(bitgen_t *bitgen_state, double loc, double scale);
*/
DECLDIR double random_normal_zig(bitgen_t *bitgen_state, double loc,
                                 double scale);

DECLDIR double random_gamma(bitgen_t *bitgen_state, double shape, double scale);
DECLDIR float random_gamma_float(bitgen_t *bitgen_state, float shape,
                                 float scale);

DECLDIR double random_exponential(bitgen_t *bitgen_state, double scale);
DECLDIR double random_uniform(bitgen_t *bitgen_state, double lower,
                              double range);
DECLDIR double random_beta(bitgen_t *bitgen_state, double a, double b);
DECLDIR double random_chisquare(bitgen_t *bitgen_state, double df);
DECLDIR double random_f(bitgen_t *bitgen_state, double dfnum, double dfden);
DECLDIR double random_standard_cauchy(bitgen_t *bitgen_state);
DECLDIR double random_pareto(bitgen_t *bitgen_state, double a);
DECLDIR double random_weibull(bitgen_t *bitgen_state, double a);
DECLDIR double random_power(bitgen_t *bitgen_state, double a);
DECLDIR double random_laplace(bitgen_t *bitgen_state, double loc, double scale);
DECLDIR double random_gumbel(bitgen_t *bitgen_state, double loc, double scale);
DECLDIR double random_logistic(bitgen_t *bitgen_state, double loc,
                               double scale);
DECLDIR double random_lognormal(bitgen_t *bitgen_state, double mean,
                                double sigma);
DECLDIR double random_rayleigh(bitgen_t *bitgen_state, double mode);
DECLDIR double random_standard_t(bitgen_t *bitgen_state, double df);
DECLDIR double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                           double nonc);
DECLDIR double random_noncentral_f(bitgen_t *bitgen_state, double dfnum,
                                   double dfden, double nonc);
DECLDIR double random_wald(bitgen_t *bitgen_state, double mean, double scale);
DECLDIR double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa);
DECLDIR double random_triangular(bitgen_t *bitgen_state, double left,
                                 double mode, double right);

DECLDIR RAND_INT_TYPE random_poisson(bitgen_t *bitgen_state, double lam);
DECLDIR RAND_INT_TYPE random_negative_binomial(bitgen_t *bitgen_state, double n,
                                               double p);
DECLDIR RAND_INT_TYPE random_binomial(bitgen_t *bitgen_state, double p,
                                      RAND_INT_TYPE n, binomial_t *binomial);
DECLDIR RAND_INT_TYPE random_logseries(bitgen_t *bitgen_state, double p);
DECLDIR RAND_INT_TYPE random_geometric_search(bitgen_t *bitgen_state, double p);
DECLDIR RAND_INT_TYPE random_geometric_inversion(bitgen_t *bitgen_state,
                                                 double p);
DECLDIR RAND_INT_TYPE random_geometric(bitgen_t *bitgen_state, double p);
DECLDIR RAND_INT_TYPE random_zipf(bitgen_t *bitgen_state, double a);
DECLDIR int64_t random_hypergeometric(bitgen_t *bitgen_state, int64_t good,
                                      int64_t bad, int64_t sample);

DECLDIR uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max);

/* Generate random uint64 numbers in closed interval [off, off + rng]. */
DECLDIR uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off,
                                       uint64_t rng, uint64_t mask,
                                       bool use_masked);

/* Generate random uint32 numbers in closed interval [off, off + rng]. */
DECLDIR uint32_t random_buffered_bounded_uint32(bitgen_t *bitgen_state,
                                                uint32_t off, uint32_t rng,
                                                uint32_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);
DECLDIR uint16_t random_buffered_bounded_uint16(bitgen_t *bitgen_state,
                                                uint16_t off, uint16_t rng,
                                                uint16_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);
DECLDIR uint8_t random_buffered_bounded_uint8(bitgen_t *bitgen_state,
                                              uint8_t off, uint8_t rng,
                                              uint8_t mask, bool use_masked,
                                              int *bcnt, uint32_t *buf);
DECLDIR npy_bool random_buffered_bounded_bool(bitgen_t *bitgen_state,
                                              npy_bool off, npy_bool rng,
                                              npy_bool mask, bool use_masked,
                                              int *bcnt, uint32_t *buf);

DECLDIR void random_bounded_uint64_fill(bitgen_t *bitgen_state, uint64_t off,
                                        uint64_t rng, npy_intp cnt,
                                        bool use_masked, uint64_t *out);
DECLDIR void random_bounded_uint32_fill(bitgen_t *bitgen_state, uint32_t off,
                                        uint32_t rng, npy_intp cnt,
                                        bool use_masked, uint32_t *out);
DECLDIR void random_bounded_uint16_fill(bitgen_t *bitgen_state, uint16_t off,
                                        uint16_t rng, npy_intp cnt,
                                        bool use_masked, uint16_t *out);
DECLDIR void random_bounded_uint8_fill(bitgen_t *bitgen_state, uint8_t off,
                                       uint8_t rng, npy_intp cnt,
                                       bool use_masked, uint8_t *out);
DECLDIR void random_bounded_bool_fill(bitgen_t *bitgen_state, npy_bool off,
                                      npy_bool rng, npy_intp cnt,
                                      bool use_masked, npy_bool *out);

DECLDIR void random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                                RAND_INT_TYPE *mnix, double *pix, npy_intp d,
                                binomial_t *binomial);
DECLDIR void random_wishart_large_df(bitgen_t *bitgen_state, int64_t df,
                                     npy_intp dim, npy_intp num, double *w,
                                     double *n);

#ifdef __cplusplus
}
#endif

#endif
