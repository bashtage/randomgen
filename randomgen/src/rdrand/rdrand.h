#ifndef _RANDOMDGEN__RDRAND_H_
#define _RANDOMDGEN__RDRAND_H_

#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/inttypes.h"
#define INLINE __forceinline
#else
#include <inttypes.h>
#define INLINE __inline __forceinline
#endif
#else
#include <inttypes.h>
#define INLINE inline
#endif

#undef HAVE_IMMINTRIN
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER) && defined(_WIN32)
#if _MSC_VER >= 1900
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#else
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#endif

typedef struct s_rdrand_state {
  int status;
} rdrand_state;


int rdrand_capable(void);


static INLINE uint64_t rdrand_next64(rdrand_state* state){
#if defined(HAVE_IMMINTRIN)
    uint64_t val;
#if defined(__x86_64__) || defined(_M_X64)
    state->status &= _rdrand64_step((long long unsigned int *)&val);
#else
    uint32_t low, high;
    state->status &= _rdrand32_step(&low);
    state->status &= _rdrand32_step(&high);
    val = ((uint64_t)high)<< 32 | low;
#endif
    return val;
#else
    return UINT64_MAX;
#endif
}

static INLINE uint32_t rdrand_next32(rdrand_state* state){
#if defined(HAVE_IMMINTRIN)
    uint32_t val;
    state->status &= _rdrand32_step(&val);
    return val;
#else
    return UINT32_MAX;
#endif
}

#endif /* _RANDOMDGEN__RDRAND_H_ */