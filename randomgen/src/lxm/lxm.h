#ifndef _RANDOMDGEN__LXM_H_
#define _RANDOMDGEN__LXM_H_

#include <stdint.h>

/* a in a * s + b
 * https://nuclear.llnl.gov/CNP/rng/rngman/node4.html
 */
#define LCG_MULT 2862933555777941757ULL
#define LCG_ADD 3037000493ULL

struct LXM_STATE_T {
  uint64_t x[4]; /* xorshift */
  uint64_t lcg_state;    /* LCG */
  uint64_t b;    /* LCG add value: default 3037000493 */
  int has_uint32;
  uint32_t uinteger;
};

typedef struct LXM_STATE_T lxm_state_t;

static inline uint64_t rotl(const uint64_t x, int k) {
  /* https://prng.di.unimi.it/xoshiro256plus.c */
  return (x << k) | (x >> (64 - k));
}

/* Using David Stafford best parameters
 * https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
 */

static inline uint64_t murmur_hash3(uint64_t key) {
  /* https://prng.di.unimi.it/splitmix64.c */
  key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9;
  key = (key ^ (key >> 27)) * 0x94d049bb133111eb;
  return key ^ (key >> 31);
}

static inline void xorshift(lxm_state_t *state) {
  /* https://prng.di.unimi.it/xoshiro256plus.c */
  const uint64_t t = state->x[1] << 17;

  state->x[2] ^= state->x[0];
  state->x[3] ^= state->x[1];
  state->x[1] ^= state->x[2];
  state->x[0] ^= state->x[3];
  state->x[2] ^= t;
  state->x[3] = rotl(state->x[3], 45);
}

static inline void lcg(lxm_state_t *state) {
  /* https://nuclear.llnl.gov/CNP/rng/rngman/node4.html */
  state->lcg_state = LCG_MULT * state->lcg_state + state->b;
}

static inline uint64_t lxm_next64(lxm_state_t *state) {
  uint64_t next_val = murmur_hash3(state->x[0] + state->lcg_state);
  lcg(state);
  xorshift(state);
  return next_val;
}

static inline uint32_t lxm_next32(lxm_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = lxm_next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void lxm_jump(lxm_state_t *state);

#endif