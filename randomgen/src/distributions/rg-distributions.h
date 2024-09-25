#ifndef _RANDOMDGEN__DISTRIBUTIONS_H_
#define _RANDOMDGEN__DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/random/distributions.h"
#include "../common/randomgen_config_numpy.h"
#include "../common/randomgen_config.h"

/*
#ifdef DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR extern
#endif
*/

/* Inline generators for internal use */
static NPY_INLINE uint32_t rg_next_uint32(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint32(bitgen_state->state);
}


static NPY_INLINE uint64_t rg_next_uint64(bitgen_t *bitgen_state) {
  return bitgen_state->next_uint64(bitgen_state->state);
}

/* Inline generators for internal use */
static NPY_INLINE float rg_next_float(bitgen_t *bitgen_state) {
  return (rg_next_uint32(bitgen_state) >> 9) * (1.0f / 8388608.0f);
}

static NPY_INLINE double rg_next_double(bitgen_t *bitgen_state) {
  return bitgen_state->next_double(bitgen_state->state);
}


DECLDIR void random_double_fill(bitgen_t *bitgen_state, npy_intp cnt,
                                double *out);
DECLDIR float random_float(bitgen_t *bitgen_state);
DECLDIR int random_long_double_size(void);
DECLDIR long double random_long_double(bitgen_t *bitgen_state);
DECLDIR void random_long_double_fill(bitgen_t *bitgen_state, npy_intp cnt,
                                     long double *out);
DECLDIR void random_wishart_large_df(bitgen_t *bitgen_state, int64_t df,
                                     npy_intp dim, npy_intp num, double *w,
                                     double *n);


#ifdef __cplusplus
}
#endif

#endif
