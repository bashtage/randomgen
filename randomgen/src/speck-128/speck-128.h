#ifndef _RANDOMDGEN__SPECK128_H_
#define _RANDOMDGEN__SPECK128_H_

#include "../common/randomgen_config.h"

#include <string.h>

#define SPECK_UNROLL 12
#define SPECK_BUFFER_SZ 8 * SPECK_UNROLL
#define SPECK_ROUNDS 34 /* Only correct for 128x256 */
#define SPECK_CTR_SZ SPECK_UNROLL / 2

static INLINE void TF83(uint64_t *x, uint64_t *y, const uint64_t k) {
  x[0] = ((x[0] >> 8) | (x[0] << (64 - 8)));
  x[0] += y[0];
  x[0] ^= k;
  y[0] = ((y[0] << 3) | (y[0] >> (64 - 3)));
  y[0] ^= x[0];
}

#define ROTL64(x, r) (((x) << (r)) | (x >> (64 - (r))))
#define ROTR64(x, r) (((x) >> (r)) | ((x) << (64 - (r))))
#define ER64(x, y, s) (x = (ROTR64(x, 8) + y) ^ (s), y = ROTL64(y, 3) ^ x)

static INLINE void speck_encrypt(uint64_t c[2], const uint64_t p[2],
                                 const uint64_t k[SPECK_ROUNDS]) {
  int i;
  c[0] = p[0];
  c[1] = p[1];

  // Don't unroll this loop. Things slow down.
  for (i = 0; i < SPECK_ROUNDS; ++i) {
    ER64(c[0], c[1], k[i]);
  }
}

static INLINE void speck_expandkey_128x256(uint64_t key[4],
                                           uint64_t round_key[SPECK_ROUNDS]) {
  uint64_t i, D, C, B, A;
  D = key[3];
  C = key[2];
  B = key[1];
  A = key[0];
  for (i = 0; i < 33; i += 3) {
    round_key[i] = A;
    ER64(B, A, i);
    round_key[i + 1] = A;
    ER64(C, A, i + 1);
    round_key[i + 2] = A;
    ER64(D, A, i + 2);
  }
  round_key[33] = A;
}

union SPEC_T {
#if (defined(HAVE_SSE2) && HAVE_SSE2)
  __m128i m128;
#endif
  uint64_t u64[2];
};
typedef union SPEC_T spec_t;

struct SPECK_STATE_T {
  spec_t ctr[SPECK_CTR_SZ];
  uint8_t buffer[SPECK_BUFFER_SZ];
  uint64_t round_key[SPECK_ROUNDS];
  int offset;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct SPECK_STATE_T speck_state_t;

static INLINE void advance_counter(speck_state_t *state) {
  uint64_t low;
  int i;
  for (i = 0; i < SPECK_CTR_SZ; i++) {
    low = state->ctr[i].u64[0];
    state->ctr[i].u64[0] += SPECK_CTR_SZ;
    if (state->ctr[i].u64[0] < low)
      state->ctr[i].u64[1]++;
  }
}

static INLINE void generate_block(speck_state_t *state) {
  uint64_t *buffer = (uint64_t *)state->buffer;
  uint64_t *ctr = (uint64_t *)state->ctr;
  speck_encrypt(buffer + 0, ctr + 0, state->round_key);
  speck_encrypt(buffer + 2, ctr + 2, state->round_key);
  speck_encrypt(buffer + 4, ctr + 4, state->round_key);
  speck_encrypt(buffer + 6, ctr + 6, state->round_key);
  speck_encrypt(buffer + 8, ctr + 8, state->round_key);
  speck_encrypt(buffer + 10, ctr + 10, state->round_key);
  advance_counter(state);
  state->offset = 0;
}

static INLINE void generate_block_fast(speck_state_t *state) {
  int i;
  uint64_t *buffer;
  memcpy(&state->buffer, &state->ctr, sizeof(state->buffer));

  buffer = (uint64_t *)state->buffer;
  for (i = 0; i < SPECK_ROUNDS; ++i) {
    ER64(buffer[1], buffer[0], state->round_key[i]);
    ER64(buffer[3], buffer[2], state->round_key[i]);
    ER64(buffer[5], buffer[4], state->round_key[i]);
    ER64(buffer[7], buffer[6], state->round_key[i]);
    ER64(buffer[9], buffer[8], state->round_key[i]);
    ER64(buffer[11], buffer[10], state->round_key[i]);
  }
  advance_counter(state);
  state->offset = 0;
}

static INLINE uint64_t speck_next64(speck_state_t *state) {
  uint64_t output;
  if
    UNLIKELY((state->offset == SPECK_BUFFER_SZ)) generate_block_fast(state);
  memcpy(&output, &state->buffer[state->offset], sizeof(output));
  state->offset += sizeof(output);
  return output;
}

static INLINE uint32_t speck_next32(speck_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = speck_next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void speck_seed(speck_state_t *state, uint64_t seed[4]);
void speck_set_counter(speck_state_t *state, uint64_t *ctr);
void speck_advance(speck_state_t *state, uint64_t *step);

#endif /* _RANDOMDGEN__SPECK128_H_ */