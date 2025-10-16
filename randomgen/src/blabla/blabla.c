#include "blabla.h"
#include "../common/cpu_features.h"

int RANDOMGEN_USE_AVX2;
uint64_t blabla_next64(blabla_state_t* state);
uint32_t blabla_next32(blabla_state_t* state);
double blabla_next_double(blabla_state_t* state);

#if defined(__AVX2__) && __AVX2__
#define BLABLA_FEATURE_REG RANDOMGEN_EBX
#define BLABLA_FEATURE_FLAG 5
#else
#define BLABLA_FEATURE_REG 0
#define BLABLA_FEATURE_FLAG 0
#endif

extern int blabla_avx2_capable(void)
{
#if !defined(RG_DISABLE_BLABLA_AVX2)
    RANDOMGEN_USE_AVX2 = avx2_capable();
#else
    RANDOMGEN_USE_AVX2 = 0;
#endif
    return RANDOMGEN_USE_AVX2;
}

void blabla_use_avx2(int flag) { RANDOMGEN_USE_AVX2 = flag; }

void blabla_seed(blabla_state_t* state, uint64_t seedval[2], uint64_t stream[2], uint64_t ctr[2])
{
    blabla_use_avx2(blabla_avx2_capable());
    state->ctr[0] = ctr[0];
    state->ctr[1] = ctr[1];
    // ctr[1] = 0;
    state->block_idx[0] = -1ULL;
    state->block_idx[1] = 0;
    // block_idx[1] = -1ULL; // Block is assumed to be uninitialized.
    state->keysetup[0] = seedval[0];
    state->keysetup[1] = stream[0];
    state->keysetup[2] = seedval[1];
    state->keysetup[3] = stream[1];
    state->rounds = R;
    state->has_uint32 = 0;
    state->uinteger = -1;
    for (int i = 0; i < 16; i++) {
        state->block[i] = 0;
    }
}

void blabla_advance(blabla_state_t* state, uint64_t delta[2])
{
    uint64_t carry, ctr_0 = state->ctr[0];
    state->ctr[0] += delta[0];
    carry = (state->ctr[0] < ctr_0) ? 1 : 0;
    state->ctr[1] += delta[1] + carry;
}
