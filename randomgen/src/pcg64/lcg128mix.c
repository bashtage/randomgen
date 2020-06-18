#include "lcg128mix.h"

extern INLINE uint64_t lcg128mix_next64(lcg128mix_state_t *state);
extern INLINE uint32_t lcg128mix_next32(lcg128mix_state_t *state);

extern void lcg128mix_set_state(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[],
                                   uint64_t multiplier[]) {
    rng->state = PCG_128BIT_CONSTANT(state[0], state[1]);
    rng->inc = PCG_128BIT_CONSTANT(inc[0], inc[1]);
    rng->multiplier = PCG_128BIT_CONSTANT(multiplier[0], multiplier[1]);
}

extern void lcg128mix_get_state(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[],
                                   uint64_t multiplier[]) {
    state[0] = PCG_HIGH(rng->state);
    state[1] = PCG_LOW(rng->state);
    inc[0] = PCG_HIGH(rng->inc);
    inc[1] = PCG_LOW(rng->inc);
    multiplier[0] = PCG_HIGH(rng->multiplier);
    multiplier[1] = PCG_LOW(rng->multiplier);
}

extern void lcg128mix_seed(lcg128mix_random_t *rng, uint64_t state[], uint64_t inc[],
                              uint64_t multiplier[]) {
    rng->multiplier = PCG_128BIT_CONSTANT(multiplier[0], multiplier[1]);
    pcg128_t initstate, initinc;
    initstate = PCG_128BIT_CONSTANT(state[0], state[1]);
    initinc = PCG_128BIT_CONSTANT(inc[0], inc[1]);
    lcg128mix_initialize(rng, initstate, initinc);
}

static void lcg128mix_advance_r(lcg128mix_random_t *rng, pcg128_t delta) {
    rng->state = pcg_advance_lcg_128(rng->state, delta, rng->multiplier, rng->inc);
}

extern void lcg128mix_advance(lcg128mix_state_t *rng, uint64_t step[]) {
    pcg128_t delta = PCG_128BIT_CONSTANT(step[0], step[1]);
    lcg128mix_advance_r(rng->pcg_state, delta);
}
