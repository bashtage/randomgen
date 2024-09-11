inline static uint64_t squares64(uint64_t ctr, uint64_t key) {
    uint64_t t, x, y, z;
    y = x = ctr * key; z = y + key;
    x = x*x + y; x = (x>>32) | (x<<32); /* round 1 */
    x = x*x + z; x = (x>>32) | (x<<32); /* round 2 */
    x = x*x + y; x = (x>>32) | (x<<32); /* round 3 */
    t = x = x*x + z; x = (x>>32) | (x<<32); /* round 4 */
    return t ^ ((x*x + y) >> 32); /* round 5 */
}


#ifndef SQUARES_H_INCLUDED
#define SQUARES_H_INCLUDED 1

/*
 * A C implementation of the Squares PRNG of Widynski
 *
 * Dual NCSA/BSD Licensed
 *
 * Copyright (c) 2024 Kevin Sheppard
 *
 */

#include "../common/randomgen_config.h"

struct SQUARES_STATE_T {
  uint64_t ctr;
  uint64_t key;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct SQUARES_STATE_T squares_state_t;

static INLINE uint64_t squares_next64(squares_state_t *state) {
    uint64_t t, x, y, z;
    y = x = state->ctr * state->key; z = y + state->key;
    x = x*x + y; x = (x>>32) | (x<<32); /* round 1 */
    x = x*x + z; x = (x>>32) | (x<<32); /* round 2 */
    x = x*x + y; x = (x>>32) | (x<<32); /* round 3 */
    t = x = x*x + z; x = (x>>32) | (x<<32); /* round 4 */
    return t ^ ((x*x + y) >> 32); /* round 5 */
}

static INLINE uint32_t squares64_next32(squares_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = squares_next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE double squares_next_double(squares_state_t *state) {
    return (squares_next64(state) >> 11) * (1.0 / 9007199254740992.0);
}

void squares_seed(squares_state_t *state, uint64_t *seed);

#endif /* SQUARES_H_INCLUDED */
