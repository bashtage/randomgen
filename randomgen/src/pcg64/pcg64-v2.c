#include "pcg64-v2.h"

extern INLINE void pcg64_initialize(pcg64_random_t *rng, pcg128_t initstate, pcg128_t initseq,
                                    int cheap_multiplier);

void pcg64_advance_r(pcg64_random_t *rng, pcg128_t delta, int cheap_multiplier) {
    if (cheap_multiplier == 0) {
        rng->state = pcg_advance_lcg_128(rng->state, delta, PCG_DEFAULT_MULTIPLIER_128, rng->inc);
    } else {
        rng->state = pcg_advance_lcg_128(rng->state, delta, PCG_CHEAP_MULTIPLIER_128, rng->inc);
    }
}

extern void pcg64_advance(pcg64_state_t *state, uint64_t *step, int cheap_multiplier) {
    pcg128_t delta;
#ifndef PCG_EMULATED_128BIT_MATH
    delta = (((pcg128_t)step[0]) << 64) | step[1];
#else
    delta.high = step[0];
    delta.low = step[1];
#endif
    pcg64_advance_r(state->pcg_state, delta, cheap_multiplier);
}

extern void pcg64_set_seed(pcg64_state_t *state, uint64_t *seed, uint64_t *inc,
                           int cheap_multiplier) {
    pcg128_t s, i;
#ifndef PCG_EMULATED_128BIT_MATH
    s = (((pcg128_t)seed[0]) << 64) | seed[1];
    i = (((pcg128_t)inc[0]) << 64) | inc[1];
#else
    s.high = seed[0];
    s.low = seed[1];
    i.high = inc[0];
    i.low = inc[1];
#endif
    pcg64_initialize(state->pcg_state, s, i, cheap_multiplier);
}

extern void pcg64_get_state(pcg64_state_t *state, uint64_t *state_arr, int *use_dxsm,
                            int *has_uint32, uint32_t *uinteger) {
    /*
     * state_arr contains state.high, state.low, inc.high, inc.low
     *    which are interpreted as the upper 64 bits (high) or lower
     *    64 bits of a uint128_t variable
     *
     */
    state_arr[0] = PCG_HIGH(state->pcg_state->state);
    state_arr[1] = PCG_LOW(state->pcg_state->state);
    state_arr[2] = PCG_HIGH(state->pcg_state->inc);
    state_arr[3] = PCG_LOW(state->pcg_state->inc);
    has_uint32[0] = state->has_uint32;
    uinteger[0] = state->uinteger;
    use_dxsm[0] = state->use_dxsm;
}

extern void pcg64_set_state(pcg64_state_t *state, uint64_t *state_arr, int use_dxsm, int has_uint32,
                            uint32_t uinteger) {
    /*
     * state_arr contains state.high, state.low, inc.high, inc.low
     *    which are interpreted as the upper 64 bits (high) or lower
     *    64 bits of a uint128_t variable
     *
     */
    state->pcg_state->state = PCG_128BIT_CONSTANT(state_arr[0], state_arr[1]);
    state->pcg_state->inc = PCG_128BIT_CONSTANT(state_arr[2], state_arr[3]);
    state->use_dxsm = use_dxsm & 0x1;
    state->has_uint32 = has_uint32;
    state->uinteger = uinteger;
}
