#ifndef _RANDOMDGEN__SPECK128_H_
#define _RANDOMDGEN__SPECK128_H_

#include "../common/randomgen_config.h"
#include "../common/randomgen_immintrin.h"

#include "speck-128-common.h"
#if defined(__SSSE3__) && __SSSE3__
#include "speck-128-sse.h"
#endif

extern int RANDOMGEN_USE_SSE41;

static INLINE void TF83(uint64_t *x, uint64_t *y, const uint64_t k) {
  x[0] = ((x[0] >> 8) | (x[0] << (64 - 8)));
  x[0] += y[0];
  x[0] ^= k;
  y[0] = ((y[0] << 3) | (y[0] >> (64 - 3)));
  y[0] ^= x[0];
}

#if !defined(ROTL64)
#define ROTL64(x, r) (((x) << (r)) | (x >> (64 - (r))))
#endif
#if !defined(ROTR64)
#define ROTR64(x, r) (((x) >> (r)) | ((x) << (64 - (r))))
#endif
#if !defined(ER64)
#define ER64(x, y, s) (x = (ROTR64(x, 8) + y) ^ (s), y = ROTL64(y, 3) ^ x)
#endif

static INLINE void speck_encrypt(uint64_t c[2], const uint64_t p[2],
                                 const speck_t *k, const int rounds) {
  int i;
  c[0] = p[0];
  c[1] = p[1];

  // Don't unroll this loop. Things slow down.
  for (i = 0; i < rounds; ++i) {
    ER64(c[0], c[1], k[i].u64[0]);
  }
}

static INLINE void speck_expandkey_128x256(uint64_t key[4], speck_t *round_key) {
  uint64_t i, D, C, B, A;
  D = key[3];
  C = key[2];
  B = key[1];
  A = key[0];
  /* Set both u64[0] and u64[1] so it is ready for SSE */
  for (i = 0; i < 33; i += 3) {
    round_key[i].u64[0] = A;
    round_key[i].u64[1] = A;
    ER64(B, A, i);
    round_key[i + 1].u64[0] = A;
    round_key[i + 1].u64[1] = A;
    ER64(C, A, i + 1);
    round_key[i + 2].u64[0] = A;
    round_key[i + 2].u64[1] = A;
    ER64(D, A, i + 2);
  }
  round_key[33].u64[0] = A;
  round_key[33].u64[1] = A;
}

static INLINE void generate_block(speck_state_t *state) {
  uint64_t *buffer = (uint64_t *)state->buffer;
  uint64_t *ctr = (uint64_t *)state->ctr;
  speck_encrypt(buffer + 0, ctr + 0, state->round_key, state->rounds);
  speck_encrypt(buffer + 2, ctr + 2, state->round_key, state->rounds);
  speck_encrypt(buffer + 4, ctr + 4, state->round_key, state->rounds);
  speck_encrypt(buffer + 6, ctr + 6, state->round_key, state->rounds);
  speck_encrypt(buffer + 8, ctr + 8, state->round_key, state->rounds);
  speck_encrypt(buffer + 10, ctr + 10, state->round_key, state->rounds);
#if SPECK_UNROLL==16
  speck_encrypt(buffer + 12, ctr + 12, state->round_key, state->rounds);
  speck_encrypt(buffer + 14, ctr + 14, state->round_key, state->rounds);
#endif
  advance_counter(state);
  state->offset = 0;
}

static INLINE void generate_block_fast(speck_state_t *state) {
  int i;
  uint64_t *buffer;
  memcpy(&state->buffer, &state->ctr, sizeof(state->buffer));

  buffer = (uint64_t *)state->buffer;
  for (i = 0; i < state->rounds; ++i) {
    ER64(buffer[1], buffer[0], state->round_key[i].u64[0]);
    ER64(buffer[3], buffer[2], state->round_key[i].u64[0]);
    ER64(buffer[5], buffer[4], state->round_key[i].u64[0]);
    ER64(buffer[7], buffer[6], state->round_key[i].u64[0]);
    ER64(buffer[9], buffer[8], state->round_key[i].u64[0]);
    ER64(buffer[11], buffer[10], state->round_key[i].u64[0]);
#if SPECK_UNROLL==16
    ER64(buffer[13], buffer[12], state->round_key[i].u64[0]);
    ER64(buffer[15], buffer[14], state->round_key[i].u64[0]);
#endif
  }
  advance_counter(state);
  state->offset = 0;
}

static INLINE uint64_t speck_next64(speck_state_t *state) {
  uint64_t output;
  if
    UNLIKELY((state->offset == SPECK_BUFFER_SZ)) {
#if defined(__SSSE3__) && __SSSE3__
      if (RANDOMGEN_USE_SSE41 == 1) {
        generate_block_ssse3(state);
      } else {
#endif
        generate_block_fast(state);
#if defined(__SSSE3__) && __SSSE3__
      }
#endif
    }
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

int speck_sse41_capable(void);
void speck_use_sse41(int val);
void speck_seed(speck_state_t *state, uint64_t seed[4]);
void speck_set_counter(speck_state_t *state, uint64_t *ctr);
void speck_advance(speck_state_t *state, uint64_t *step);

#endif /* _RANDOMDGEN__SPECK128_H_ */