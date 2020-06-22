#include "efiix64.h"

extern INLINE uint64_t efiix64_next64(efiix64_state_t *state);
extern INLINE uint32_t efiix64_next32(efiix64_state_t *state);

static uint64_t arbee_raw64(arbee_state_t *state) {
    uint64_t e = state->a + rotate(state->b, 45);
    state->a = state->b ^ rotate(state->c, 13);
    state->b = state->c + rotate(state->d, 37);
    state->c = e + state->d + state->i++;
    state->d = e + state->a;
    return state->d;
}

static void arbee_mix(arbee_state_t *state) {
    for (int x = 0; x < 12; x++) {
        arbee_raw64(state);
    }
}

static void arbee_seed(arbee_state_t *state, uint64_t seed1, uint64_t seed2, uint64_t seed3,
                uint64_t seed4) {
    state->a = seed1;
    state->b = seed2;
    state->c = seed3;
    state->d = seed4;
    state->i = 1;
    arbee_mix(state);
}

void efiix64_seed(efiix64_state_t *state, uint64_t seed[4]) {
    arbee_state_t seeder;
    uint64_t s1 = seed[0], s2 = seed[1], s3 = seed[2], s4 = seed[3];
    arbee_seed(&seeder, s1, s2, s3, s4);
    for (unsigned long w = 0; w < INDIRECTION_SIZE; w++) {
        state->indirection_table[w] = arbee_raw64(&seeder);
    }
    state->i = arbee_raw64(&seeder);
    for (unsigned long w = 0; w < ITERATION_SIZE; w++) {
        state->iteration_table[(w + state->i) % ITERATION_SIZE] = arbee_raw64(&seeder);
    }
    state->a = arbee_raw64(&seeder);
    state->b = arbee_raw64(&seeder);
    state->c = arbee_raw64(&seeder);
    for (unsigned long w = 0; w < 64; w++) {
        efiix64_raw64(state);
    }
    arbee_raw64(&seeder);
    s1 += arbee_raw64(&seeder);
    s2 += arbee_raw64(&seeder);
    s3 += arbee_raw64(&seeder);
    arbee_seed(&seeder, s1 ^ state->a, s2 ^ state->b, s3 ^ state->c, ~s4);
    for (unsigned long w = 0; w < INDIRECTION_SIZE; w++) {
        state->indirection_table[w] ^= arbee_raw64(&seeder);
        ;
    }
    for (unsigned long w = 0; w < ITERATION_SIZE + 16; w++) {
        efiix64_raw64(state);
    }
}
