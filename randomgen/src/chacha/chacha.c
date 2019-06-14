#include "chacha.h"
#include <stdio.h>


extern INLINE uint32_t chacha_next32(chacha_state_t *state);

extern INLINE uint64_t chacha_next64(chacha_state_t *state);

extern INLINE double chacha_next_double(chacha_state_t *state);

void chacha_seed(chacha_state_t *state, uint64_t *seedval, uint64_t *stream) {
    state->ctr[0] = state->ctr[1] = 0;
    // Using a 128-bit seed.
    state->keysetup[0] = seedval[0] & 0xffffffffu;
    state->keysetup[1] = seedval[0] >> 32;
    state->keysetup[2] = seedval[1] & 0xffffffffu;
    state->keysetup[3] = seedval[1] >> 32;
    // Using a 128-bit stream.
    state->keysetup[4] = stream[0] & 0xffffffffu;
    state->keysetup[5] = stream[0] >> 32;
    state->keysetup[6] = stream[1] & 0xffffffffu;
    state->keysetup[7] = stream[1] >> 32;
}


void chacha_advance(chacha_state_t *state, uint64_t *delta) {
    int carry, idx = state->ctr[0] % 16;
    uint64_t orig;
    orig = state->ctr[0];
    state->ctr[0] += delta[0];
    carry = state->ctr[0] < orig;
    state->ctr[1] += (delta[1] + carry);
    if ((idx + delta[0] >= 16 || delta[1]) && state->ctr[0] % 16 != 0) generate_block(state);
}
