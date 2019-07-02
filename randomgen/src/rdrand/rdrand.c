#include "rdrand.h"
#include "../common/cpu_features.h"

#define RANDOMGEN_USE_RDRAND 30

extern INLINE uint64_t rdrand_next64(rdrand_state* state);
extern INLINE uint32_t rdrand_next32(rdrand_state* state);

int rdrand_capable(void)
{
    int flags[32];
    feature_flags(flags);
    return flags[RANDOMGEN_USE_RDRAND];
}
