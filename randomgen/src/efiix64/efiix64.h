#ifndef EFIIX64_H_INCLUDED
#define EFIIX64_H_INCLUDED 1

#include "../common/randomgen_config.h"

#define ITERATION_SIZE_L2 5
#define ITERATION_SIZE 32 /* 1 << ITERATION_SIZE_L2 */
#define INDIRECTION_SIZE_L2 4
#define INDIRECTION_SIZE 16 /* 1 << INDIRECTION_SIZE_L2 */

static INLINE uint64_t rotate(uint64_t x, int k) {
#ifdef _MSC_VER
  return _rotl64(x, k);
#else
  return (x << k) | (x >> (64 - k));
#endif
}

typedef struct EFIIX_STATE_T {
  uint64_t indirection_table[INDIRECTION_SIZE];
  uint64_t iteration_table[ITERATION_SIZE];
  uint64_t i, a, b, c;
  int has_uint32;
  uint32_t uinteger;
} efiix64_state_t;

typedef struct ARBEE_STATE_T {
  uint64_t a, b, c, d, e, i;
} arbee_state_t;

static INLINE uint64_t efiix64_raw64(efiix64_state_t *state) {
  uint64_t iterated = state->iteration_table[state->i % ITERATION_SIZE];
  uint64_t indirect = state->indirection_table[state->c % INDIRECTION_SIZE];
  state->indirection_table[state->c % INDIRECTION_SIZE] = iterated + state->a;
  state->iteration_table[state->i % ITERATION_SIZE] = indirect;
  uint64_t old = state->a ^ state->b;

  state->a = state->b + state->i++;
  state->b = state->c + indirect;
  state->c = old + rotate(state->c, 25);
  return state->b ^ iterated;
}

static INLINE uint64_t efiix64_next64(efiix64_state_t *state) {
  return efiix64_raw64(state);
}

static INLINE uint32_t efiix64_next32(efiix64_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = efiix64_raw64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)next;
}

void efiix64_seed(efiix64_state_t *state, uint64_t seed[4]);

#endif /* EFIIX64_H_INCLUDED */