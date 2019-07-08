#ifndef _RANDOMGEN_SPECK_SSE_H
#define _RANDOMGEN_SPECK_SSE_H 1

#include "speck-128-common.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#define u128 __m128i
#define SET _mm_set_epi64x
#define LOW _mm_unpacklo_epi64
#define HIGH _mm_unpackhi_epi64
#define LD(ip) _mm_loadu_si128((u128 *)(ip))
#define ST(ip, X) _mm_storeu_si128((u128 *)(ip), X)
#define SL _mm_slli_epi64
#define SR _mm_srli_epi64
#define ADD _mm_add_epi64
#define XOR _mm_xor_si128

#define SHFL _mm_shuffle_epi8
#define R8 SET(0x080f0e0d0c0b0a09LL, 0x0007060504030201LL)
#define ROR8(X) (SHFL(X, R8))
#define ROL(X, r) (XOR(SL(X, r), SR(X, (64 - r))))

#if !defined(ROTL64)
#define ROTL64(x, r) (((x) << (r)) | (x >> (64 - (r))))
#endif
#if !defined(ROTR64)
#define ROTR64(x, r) (((x) >> (r)) | ((x) << (64 - (r))))
#endif
#if !defined(ER64)
#define ER64(x, y, s) (x = (ROTR64(x, 8) + y) ^ (s), y = ROTL64(y, 3) ^ x)
#endif

#define Rx6(X, Y, k)                                                           \
  (X[0] = ADD(ROR8(X[0]), Y[0]), X[0] = XOR(X[0], k.m128),                     \
   Y[0] = ROL(Y[0], 3), Y[0] = XOR(Y[0], X[0]), X[1] = ADD(ROR8(X[1]), Y[1]),  \
   X[1] = XOR(X[1], k.m128), Y[1] = ROL(Y[1], 3), Y[1] = XOR(Y[1], X[1]),      \
   X[2] = ADD(ROR8(X[2]), Y[2]), X[2] = XOR(X[2], k.m128),                     \
   Y[2] = ROL(Y[2], 3), Y[2] = XOR(Y[2], X[2]))

#define Rx8(X, Y, k)                                                           \
  (X[0] = ADD(ROR8(X[0]), Y[0]), X[0] = XOR(X[0], k.m128),                     \
   Y[0] = ROL(Y[0], 3), Y[0] = XOR(Y[0], X[0]), X[1] = ADD(ROR8(X[1]), Y[1]),  \
   X[1] = XOR(X[1], k.m128), Y[1] = ROL(Y[1], 3), Y[1] = XOR(Y[1], X[1]),      \
   X[2] = ADD(ROR8(X[2]), Y[2]), X[2] = XOR(X[2], k.m128),                     \
   Y[2] = ROL(Y[2], 3), Y[2] = XOR(Y[2], X[2]), X[3] = ADD(ROR8(X[3]), Y[3]),  \
   X[3] = XOR(X[3], k.m128), Y[3] = ROL(Y[3], 3), Y[3] = XOR(Y[3], X[3]))

static inline void speck_expandkey_128x256_sse4(uint64_t key[], u128 *round_key) {
  uint64_t i, D, C, B, A;
  D = key[3];
  C = key[2];
  B = key[1];
  A = key[0];

  for (i = 0; i < 33; i += 3) {
    round_key[i] = SET(A, A);
    ER64(B, A, i);
    round_key[i + 1] = SET(A, A);
    ER64(C, A, i + 1);
    round_key[i + 2] = SET(A, A);
    ER64(D, A, i + 2);
  }
  round_key[33] = SET(A, A);
}

static inline void LOAD(uint8_t pt[], u128 *X, u128 *Y) {
  u128 R, S;
  R = LD(pt);
  S = LD(pt + 16);
  X[0] = HIGH(R, S);
  Y[0] = LOW(R, S);
  R = LD(pt + 32);
  S = LD(pt + 48);
  X[1] = HIGH(R, S);
  Y[1] = LOW(R, S);
  R = LD(pt + 64);
  S = LD(pt + 80);
  X[2] = HIGH(R, S);
  Y[2] = LOW(R, S);
#if SPECK_UNROLL == 16
  R = LD(pt + 96);
  S = LD(pt + 112);
  X[3] = HIGH(R, S);
  Y[3] = LOW(R, S);
#elif SPECK_UNROLL != 12
#error "SSE Path does not exist"
#endif
}

static inline void STORE(uint8_t ct[], u128 *X, u128 *Y) {
  u128 R, S;
  R = LOW(Y[0], X[0]);
  ST(ct, R);
  S = HIGH(Y[0], X[0]);
  ST(ct + 16, S);
  R = LOW(Y[1], X[1]);
  ST(ct + 32, R);
  S = HIGH(Y[1], X[1]);
  ST(ct + 48, S);
  R = LOW(Y[2], X[2]);
  ST(ct + 64, R);
  S = HIGH(Y[2], X[2]);
  ST(ct + 80, S);
#if SPECK_UNROLL == 16
  R = LOW(Y[3], X[3]);
  ST(ct + 96, R);
  S = HIGH(Y[3], X[3]);
  ST(ct + 112, S);
#elif SPECK_UNROLL != 12
#error "SSE Path does not exist"
#endif
}

static inline void speck_128x256_encrypt_sse(uint8_t buffer[],
                                             const speck_t *round_key,
                                             const int rounds) {
  int i;
  u128 X[SPECK_UNROLL / 4], Y[SPECK_UNROLL / 4];
  LOAD(buffer, X, Y);
  for (i = 0; i < rounds; i++) {
#if SPECK_UNROLL == 16
    Rx8(X, Y, round_key[i]);
#elif SPECK_UNROLL == 12
    Rx6(X, Y, round_key[i]);
#else
#error "SSE Path does not exist"
#endif
  }
  STORE(buffer, X, Y);
}

static inline void generate_block_ssse3(speck_state_t *state) {
  uint8_t *buffer;
  memcpy(&state->buffer, &state->ctr, sizeof(state->buffer));
  buffer = (uint8_t *)state->buffer;
  speck_128x256_encrypt_sse(buffer, state->round_key, state->rounds);
  advance_counter(state);
  state->offset = 0;
}

#endif /* _RANDOMGEN_SPECK_SSE_H */
