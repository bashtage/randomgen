#ifndef _RANDOMDGEN__RDRAND_H_
#define _RANDOMDGEN__RDRAND_H_

#include "../common/randomgen_config.h"

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