#ifndef _RANDOMDGEN__CONFIG_NUMPY_H_
#define _RANDOMDGEN__CONFIG_NUMPY_H_

#ifndef RANDOMGEN_STANDALONE

#include "Python.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"

#else

#include "standalone/aligned_malloc.h"
#include "standalone/npy_common.h"
#include "standalone/npy_math.h"
#include "standalone/python.h"

#endif

#ifndef NPY_MEMALIGN

#define NPY_MEMALIGN 64 /* 16 for SSE2, 32 for AVX, 64 for Xeon Phi */

#endif

#endif
