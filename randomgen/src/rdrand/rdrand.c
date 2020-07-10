#include "rdrand.h"
#include "../common/cpu_features.h"

#define RANDOMGEN_USE_RDRAND 30

extern INLINE int rdrand_fill_buffer(rdrand_state *state);
extern INLINE int rdrand_next64(rdrand_state *state, uint64_t *val);

int rdrand_capable(void) {
#if defined(__RDRND__) && __RDRND__
    int flags[32];
    feature_flags(flags, RANDOMGEN_ECX);
    return flags[RANDOMGEN_USE_RDRAND];
#else
    return 0;
#endif
}
