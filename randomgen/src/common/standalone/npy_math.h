#ifndef __NPY_MATH_C99_H_
#define __NPY_MATH_C99_H_

#include "../randomgen_config.h"

#include "npy_common.h"

#include <math.h>

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#define npy_isnan(x) _isnan((x))
#else
#define npy_isnan(x) isnan(x)
#endif

NPY_INLINE static float __npy_nanf(void) {
  const union {
    npy_uint32 __i;
    float __f;
  } __bint = {0x7fc00000UL};
  return __bint.__f;
}

#define NPY_NANF __npy_nanf()

#define NPY_NAN ((npy_double)NPY_NANF)

#endif
