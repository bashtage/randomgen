#include "speck-128.h"
#include "../common/cpu_features.h"

#define SSE41_FEATURE_FLAG 19

int RANDOMGEN_USE_SSE41;

int speck_sse41_capable(void)
{
#if defined(__SSSE3__) && __SSSE3__
    int flags[32];
    feature_flags(flags, RANDOMGEN_ECX);
    RANDOMGEN_USE_SSE41 = flags[SSE41_FEATURE_FLAG];
    return RANDOMGEN_USE_SSE41;
#else
    RANDOMGEN_USE_SSE41 = 0;
    return 0;
#endif
}

void speck_seed(speck_state_t *state, uint64_t seed[4])
{
    int i;
    speck_sse41_capable();
    speck_expandkey_128x256(seed, state->round_key);
    state->offset = SPECK_BUFFER_SZ;
    for (i = 0; i < SPECK_CTR_SZ; i++)
    {
        state->ctr[i].u64[0] = i;
        state->ctr[i].u64[1] = 0;
    }
}

void speck_use_sse41(int val) { RANDOMGEN_USE_SSE41 = val; }

void speck_set_counter(speck_state_t *state, uint64_t *ctr)
{
    int carry, i;
    for (i = 0; i < SPECK_CTR_SZ; i++)
    {
        state->ctr[i].u64[0] = ctr[0] + i;
        carry = state->ctr[i].u64[0] < ctr[0];
        state->ctr[i].u64[1] = ctr[1] + carry;
    }
}

void speck_advance(speck_state_t *state, uint64_t *step)
{
    uint64_t low;
    uint64_t adj_step[2];
    int new_offset;
    int i;
    if (state->offset == SPECK_BUFFER_SZ)
    {
        /* Force update and reset the offset to simplify */
        speck_next64(state);
        state->offset = 0;
    }
    /* Handle odd with buffer update */
    state->offset = state->offset + 8 * (step[0] % 2);
    adj_step[0] = (step[0] / 2) + ((step[1] % 2) << 63);
    adj_step[1] = (step[1] / 2) + ((step[2] % 2) << 63);
    /* Early return if no counter change */
    if ((adj_step[0] == 0) && (adj_step[1] == 0))
    {
        return;
    }
    /* Update the counters to new **next** values */
    for (i = 0; i < SPECK_CTR_SZ; i++)
    {
        /* Add with carry */
        low = state->ctr[i].u64[0];
        state->ctr[i].u64[0] += adj_step[0];
        state->ctr[i].u64[1] += adj_step[1] + (state->ctr[i].u64[0] < low);
        /* Now subtract to get the previous counter, with carry */
        low = state->ctr[i].u64[0];
        state->ctr[i].u64[0] -= SPECK_CTR_SZ;
        state->ctr[i].u64[1] -= (state->ctr[i].u64[0] > low);
    }
    /* Force update */
    new_offset = state->offset;
    state->offset = SPECK_BUFFER_SZ;
    speck_next64(state);
    /* Reset the offset */
    state->offset = new_offset;
}
