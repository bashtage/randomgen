#ifndef _RANDOMDGEN__ALIGNED_MALLOC_H_
#define _RANDOMDGEN__ALIGNED_MALLOC_H_

#include "../randomgen_config.h"

#include <malloc.h>
#include <stdlib.h>
#include <string.h>

static INLINE void *_aligned_calloc(size_t n, size_t size, size_t alignment) {

  void *p = 0;

  size_t asize = n * size;

  p = _aligned_malloc(asize, alignment);

  if (p) {
    memset(p, 0, asize);
  }

  return p;
}

#define malloc_aligned(a) _aligned_calloc(1, (a), NPY_MEMALIGN)
#define calloc_aligned(a, b) _aligned_calloc((a), (b), NPY_MEMALIGN)
#define free_aligned(a) _aligned_free(a)

#endif
