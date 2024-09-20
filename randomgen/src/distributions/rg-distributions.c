#include "numpy/random/distributions.h"
#include "rg-distributions.h"
// #include "loggam.h"
// #include "ziggurat.h"
// #include "ziggurat_constants.h"
#include <float.h>
#include <stdio.h>

#if defined(_MSC_VER) && defined(_WIN64)
#include <intrin.h>
#endif

/// ,
void random_double_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out) {
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = rg_next_double(bitgen_state);
  }
}

float random_float(bitgen_t *bitgen_state) { return rg_next_float(bitgen_state); }

int random_long_double_size() {
    return (int)sizeof(long double);
}

static inline long double rg_next_long_double(bitgen_t *bitgen_state) {
#if ((LDBL_MANT_DIG+0) == 53) && ((LDBL_MAX_EXP+0) == 1024)
   /* C double/IEEE 754 binary64  */
   return rg_next_double(bitgen_state);
#elif ((LDBL_MANT_DIG+0) == 64) && ((LDBL_MAX_EXP+0) == 16384)
   /* x87 extended */
   /* Constant is 2**64 */
   return rg_next_uint64(bitgen_state) * (1.0L / 18446744073709551616.0L);
#elif ((LDBL_MANT_DIG+0) == 106) && ((LDBL_MAX_EXP+0) == 1024)
   /* IBM extended double */
   int64_t a = rg_next_uint64(bitgen_state) >> 11, b = rg_next_uint64(bitgen_state) >> 11;
   /* Constants are 2^53 and 2^106 */
   return (a * 9007199254740992.0L + b) / 81129638414606681695789005144064.0L;
#elif ((LDBL_MANT_DIG+0) == 113) && ((LDBL_MAX_EXP+0) == 16384)
    /* IEEE 754-2008 binary128 */
    int64_t a = rg_next_uint64(bitgen_state) >> 7, b = rg_next_uint64(bitgen_state) >> 8;
    /* Constants are 2^56 and 2^113 */
    return (a * 72057594037927936.0L + b) / 10384593717069655257060992658440192.0L;
#else
#error("Unknown value for LDBL_MANT_DIG")
#endif
}

long double random_long_double(bitgen_t *bitgen_state){
    return rg_next_long_double(bitgen_state);
}

void random_long_double_fill(bitgen_t *bitgen_state, npy_intp cnt, long double *out){
   for (int i=0; i < cnt; i++){
      out[i] = rg_next_long_double(bitgen_state);
   }
}

void random_wishart_large_df(bitgen_t *bitgen_state, int64_t df, npy_intp dim,
                             npy_intp num, double *w, double *n) {
  /*
   * Odell, P. L. , and A. H. Feiveson (1966) A numerical procedure to
   * generate a sample covariance matrix. Jour. Amer. Stat. Assoc. 61, 199–203
   */
  const npy_intp dim2 = dim * dim;
  npy_intp i, j, k, r;
  double vi;
  for (k = 0; k < num; k++) {
    for (i = 0; i < dim; i++) {
      vi = random_chisquare(bitgen_state, (double)(df-i));
      for (j = i; j < dim; j++) {
        if (i == j) {
          *(w + k * dim2 + i * dim + j) = vi;
        } else {
          *(n + dim * i + j) = random_standard_normal(bitgen_state);
          *(w + k * dim2 + i * dim + j) = *(n + dim * i + j) * sqrt(vi);
        }
        for (r = 0; r < i; r++) {
          *(w + k * dim2 + i * dim + j) += *(n + dim * r + i) * *(n + dim * r + j);
        }
        *(w + k * dim2 + j * dim + i) = *(w + k * dim2 + i * dim + j);
      }
    }
  }
}

int main() {
return 0;
}