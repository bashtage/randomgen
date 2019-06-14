/***
 * Daniel Baker has adapted this code to C++:
 * https://github.com/dnbaker/hll/blob/master/aesctr.h
 * He reports that UNROLL_COUNT of 8 (rather than 4) is measurably
 * but not enormously faster.
 ***/

#ifndef AESCTR_H
#define AESCTR_H
// #ifdef __AES__
// contributed by Samuel Neves

#undef HAVE_IMMINTRIN
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER) && defined(_WIN32)
#if _MSC_VER >= 1900
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#else
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#endif

#ifdef _WIN32
#define ALIGN_WINDOWS __declspec(align(16))
#define ALIGN_GCC_CLANG
#if _MSC_VER == 1500
#include "../common/inttypes.h"
#define INLINE __forceinline
#else
#include <inttypes.h>
#define INLINE __inline __forceinline
#endif
#else
#define ALIGN_WINDOWS
#define ALIGN_GCC_CLANG __attribute__((aligned(16)))
#include <inttypes.h>
#define INLINE inline
#endif


#include <stddef.h>
#include <string.h>

#define AESCTR_UNROLL 4
#define AESCTR_ROUNDS 10

#ifdef _WIN32
#define UNLIKELY(x) ((x))
#else
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif

union AES128_T {
#if defined(HAVE_SSE2)
  __m128i m128;
#endif
  uint64_t u64[2];
  uint32_t u32[4];
  uint8_t u8[16];
};

typedef union AES128_T aes128_t;

struct AESCTR_STATE_T {
  ALIGN_WINDOWS aes128_t ctr[AESCTR_UNROLL] ALIGN_GCC_CLANG;
  ALIGN_WINDOWS aes128_t seed[AESCTR_ROUNDS + 1] ALIGN_GCC_CLANG;
  ALIGN_WINDOWS uint8_t state[16 * AESCTR_UNROLL] ALIGN_GCC_CLANG;
  size_t offset;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct AESCTR_STATE_T aesctr_state_t;

#if defined(HAVE_SSE2)
#define AES_ROUND(rcon, index)                                                 \
  do {                                                                         \
    __m128i k2 = _mm_aeskeygenassist_si128(k, rcon);                           \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3, 3, 3, 3)));      \
    state->seed[index].m128 = k;                                               \
  } while (0)
#endif

static INLINE void aesctr_seed_r(aesctr_state_t *state, uint64_t *seed) {
  /*static const uint8_t rcon[] = {
      0x8d, 0x01, 0x02, 0x04,
      0x08, 0x10, 0x20, 0x40,
      0x80, 0x1b, 0x36
  };*/
#if defined(HAVE_SSE2)
  __m128i k = _mm_set_epi64x(seed[1], seed[0]);
  state->seed[0].m128 = k;
  // D. Lemire manually unrolled following loop since _mm_aeskeygenassist_si128
  // requires immediates

  /*for(int i = 1; i <= AESCTR_ROUNDS; ++i)
  {
      __m128i k2 = _mm_aeskeygenassist_si128(k, rcon[i]);
      k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
      k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
      k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
      k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3,3,3,3)));
      state->seed[i] = k;
  }*/
  AES_ROUND(0x01, 1);
  AES_ROUND(0x02, 2);
  AES_ROUND(0x04, 3);
  AES_ROUND(0x08, 4);
  AES_ROUND(0x10, 5);
  AES_ROUND(0x20, 6);
  AES_ROUND(0x40, 7);
  AES_ROUND(0x80, 8);
  AES_ROUND(0x1b, 9);
  AES_ROUND(0x36, 10);

  for (int i = 0; i < AESCTR_UNROLL; ++i) {
    state->ctr[i].m128 = _mm_set_epi64x(0, i);
  }
  state->offset = 16 * AESCTR_UNROLL;
#endif
}

#undef AES_ROUND

static INLINE uint64_t aesctr_r(aesctr_state_t *state) {
#if defined(HAVE_SSE2)
  if (UNLIKELY(state->offset >= 16 * AESCTR_UNROLL)) {
    __m128i work[AESCTR_UNROLL];
    for (int i = 0; i < AESCTR_UNROLL; ++i) {
      work[i] = _mm_xor_si128(state->ctr[i].m128, state->seed[0].m128);
    }
    for (int r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
      const __m128i subkey = state->seed[r].m128;
      for (int i = 0; i < AESCTR_UNROLL; ++i) {
        work[i] = _mm_aesenc_si128(work[i], subkey);
      }
    }
    for (int i = 0; i < AESCTR_UNROLL; ++i) {
      state->ctr[i].m128 =
          _mm_add_epi64(state->ctr[i].m128, _mm_set_epi64x(0, AESCTR_UNROLL));
      _mm_storeu_si128((__m128i *)&state->state[16 * i],
                       _mm_aesenclast_si128(work[i],
                                            state->seed[AESCTR_ROUNDS].m128));
    }
    state->offset = 0;
  }
  uint64_t output = 0;
  memcpy(&output, &state->state[state->offset], sizeof(output));
  state->offset += sizeof(output);
  return output;
#else
  return 0;
#endif
}

static INLINE uint64_t aes_next64(aesctr_state_t *state) {
  return aesctr_r(state);
}

static INLINE uint32_t aes_next32(aesctr_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = aesctr_r(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE double aes_next_double(aesctr_state_t *state) {
  return (aesctr_r(state) >> 11) * (1. / (UINT64_C(1) << 53));
}

extern void aesctr_seed(aesctr_state_t *state, uint64_t *seed);
extern void aesctr_set_counter(aesctr_state_t *state, uint64_t *counter);
extern void aesctr_set_seed_counter(aesctr_state_t *state, uint64_t *seed,
                                    uint64_t *counter);
extern void aesctr_get_seed_counter(aesctr_state_t *state, uint64_t *seed,
                                    uint64_t *counter);
extern int aes_capable(void);
extern void aesctr_advance(aesctr_state_t *state, uint64_t *step);
//#endif // __AES__
#endif
