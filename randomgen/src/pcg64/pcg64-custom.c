#include "pcg64-custom.h"

extern INLINE uint64_t pcg64_custom_next64(pcg64_custom_state_t *state);
extern INLINE uint32_t pcg64_custom_next32(pcg64_custom_state_t *state);

extern void pcg64_custom_set_state(pcg64_custom_random_t *rng, uint64_t state[], uint64_t inc[],
                                   uint64_t multiplier[]) {
    rng->state = PCG_128BIT_CONSTANT(state[0], state[1]);
    rng->inc = PCG_128BIT_CONSTANT(inc[0], inc[1]);
    rng->multiplier = PCG_128BIT_CONSTANT(multiplier[0], multiplier[1]);
}

extern void pcg64_custom_get_state(pcg64_custom_random_t *rng, uint64_t state[], uint64_t inc[],
                                   uint64_t multiplier[]) {
    state[0] = PCG_HIGH(rng->state);
    state[1] = PCG_LOW(rng->state);
    inc[0] = PCG_HIGH(rng->inc);
    inc[1] = PCG_LOW(rng->inc);
    multiplier[0] = PCG_HIGH(rng->multiplier);
    multiplier[1] = PCG_LOW(rng->multiplier);
}

extern void pcg64_custom_seed(pcg64_custom_random_t *rng, uint64_t state[], uint64_t inc[],
                              uint64_t multiplier[]) {
    rng->multiplier = PCG_128BIT_CONSTANT(multiplier[0], multiplier[1]);
    pcg128_t initstate, initinc;
    initstate = PCG_128BIT_CONSTANT(state[0], state[1]);
    initinc = PCG_128BIT_CONSTANT(inc[0], inc[1]);
    pcg64_custom_initialize(rng, initstate, initinc);
}

static void pcg64_custom_advance_r(pcg64_custom_random_t *rng, pcg128_t delta) {
    rng->state = pcg_advance_lcg_128(rng->state, delta, rng->multiplier, rng->inc);
}

extern void pcg64_custom_advance(pcg64_custom_state_t *rng, uint64_t step[]) {
    pcg128_t delta = PCG_128BIT_CONSTANT(step[0], step[1]);
    pcg64_custom_advance_r(rng->pcg_state, delta);
}
