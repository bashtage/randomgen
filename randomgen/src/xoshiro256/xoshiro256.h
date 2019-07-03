#ifndef _RANDOMDGEN__XOSHIRO256_H_
#define _RANDOMDGEN__XOSHIRO256_H_

#include "../common/randomgen_config.h"

typedef struct XOSHIRO256_STATE_T {
  uint64_t s[4];
  int has_uint32;
  uint32_t uinteger;
} xoshiro256_state_t;

static INLINE uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static INLINE uint64_t xoshiro256_next(uint64_t *s) {
  const uint64_t result_starstar = rotl(s[1] * 5, 7) * 9;
  const uint64_t t = s[1] << 17;

  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];

  s[2] ^= t;

  s[3] = rotl(s[3], 45);

  return result_starstar;
}

static INLINE uint64_t
xoshiro256_next64(xoshiro256_state_t *state) {
  return xoshiro256_next(&state->s[0]);
}

static INLINE uint32_t
xoshiro256_next32(xoshiro256_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = xoshiro256_next(&state->s[0]);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void xoshiro256_jump(xoshiro256_state_t *state);

#endif
