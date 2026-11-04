#ifndef _RANDOMDGEN__MWCX_H_
#define _RANDOMDGEN__MWCX_H_

/*
 * Scrambled multiply-with-carry generators MWC192X, MWC128X, MWC128X/32 and
 * MWC64X/32, translated from the reference C code by Sebastiano Vigna
 * (https://github.com/vigna/MWCPRNG/tree/main/prngs), which is released
 * under the following license:
 *
 * Written in 2023-2026 by Sebastiano Vigna (vigna@acm.org)
 *
 * To the extent possible under law, the author has dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted.
 *
 * Each step computes t = a * x_{n-k} + c_{n-1}; the new digit x_n is the
 * lower half of t, and the new carry c_n its upper half. The output is the
 * most recent digit xored with the carry (the scrambler X). The jump
 * functions are implemented in Python using modular arithmetic: see mwcx.pyx.
 */

#include "../common/randomgen_config.h"

#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_ARM64))
#include <intrin.h>
#endif

#if defined(__SIZEOF_INT128__) && !defined(RANDOMGEN_MWCX_FORCE_PORTABLE)
#define RANDOMGEN_MWCX_UINT128 1
#elif defined(_MSC_VER) && defined(_M_AMD64) &&                                \
    !defined(RANDOMGEN_MWCX_FORCE_PORTABLE)
#pragma intrinsic(_umul128)
#define RANDOMGEN_MWCX_UMUL128 1
#elif defined(_MSC_VER) && defined(_M_ARM64) &&                                \
    !defined(RANDOMGEN_MWCX_FORCE_PORTABLE)
#pragma intrinsic(__umulh)
#define RANDOMGEN_MWCX_UMULH 1
#endif

#define MWC192X_A 0xff4e24dd418f5433ULL
#define MWC128X_A 0xff6da236274e252eULL
#define MWC128X32_A 0xffcccc66U
#define MWC64X32_A 0xfff145f8U

typedef struct MWC192X_STATE_T {
  uint64_t x;
  uint64_t y;
  uint64_t c;
  int has_uint32;
  uint32_t uinteger;
} mwc192x_state_t;

typedef struct MWC128X_STATE_T {
  uint64_t x;
  uint64_t c;
  int has_uint32;
  uint32_t uinteger;
} mwc128x_state_t;

typedef struct MWC128X32_STATE_T {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t c;
} mwc128x32_state_t;

typedef struct MWC64X32_STATE_T {
  uint32_t x;
  uint32_t c;
} mwc64x32_state_t;

/* Computes a * x + c, storing the upper 64 bits in *hi and returning the
   lower 64 bits. The result fits in 128 bits as a, x, c < 2^64. */
static INLINE uint64_t mwcx_mul_add(uint64_t a, uint64_t x, uint64_t c,
                                    uint64_t *hi) {
#if defined(RANDOMGEN_MWCX_UINT128)
  const __uint128_t t = (__uint128_t)a * x + c;
  *hi = (uint64_t)(t >> 64);
  return (uint64_t)t;
#else
  uint64_t lo, h;
#if defined(RANDOMGEN_MWCX_UMUL128)
  lo = _umul128(a, x, &h);
#elif defined(RANDOMGEN_MWCX_UMULH)
  lo = a * x;
  h = __umulh(a, x);
#else
  /* Portable 32-bit schoolbook multiplication */
  const uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
  const uint64_t x0 = x & 0xffffffffULL, x1 = x >> 32;
  const uint64_t p00 = a0 * x0, p01 = a0 * x1, p10 = a1 * x0, p11 = a1 * x1;
  const uint64_t mid = (p00 >> 32) + (p10 & 0xffffffffULL) + p01;
  lo = (mid << 32) | (p00 & 0xffffffffULL);
  h = p11 + (p10 >> 32) + (mid >> 32);
#endif
  lo += c;
  h += lo < c;
  *hi = h;
  return lo;
#endif
}

static INLINE uint64_t mwc192x_next64(mwc192x_state_t *state) {
  const uint64_t result = state->y ^ state->c;
  uint64_t hi;
  const uint64_t lo = mwcx_mul_add(MWC192X_A, state->x, state->c, &hi);
  state->x = state->y;
  state->y = lo;
  state->c = hi;
  return result;
}

static INLINE uint32_t mwc192x_next32(mwc192x_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = mwc192x_next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE uint64_t mwc128x_next64(mwc128x_state_t *state) {
  const uint64_t result = state->x ^ state->c;
  uint64_t hi;
  const uint64_t lo = mwcx_mul_add(MWC128X_A, state->x, state->c, &hi);
  state->x = lo;
  state->c = hi;
  return result;
}

static INLINE uint32_t mwc128x_next32(mwc128x_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = mwc128x_next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE uint32_t mwc128x32_next32(mwc128x32_state_t *state) {
  const uint32_t result = state->z ^ state->c;
  const uint64_t t = MWC128X32_A * (uint64_t)state->x + state->c;
  state->x = state->y;
  state->y = state->z;
  state->z = (uint32_t)t;
  state->c = (uint32_t)(t >> 32);
  return result;
}

static INLINE uint64_t mwc128x32_next64(mwc128x32_state_t *state) {
  const uint64_t hi = mwc128x32_next32(state);
  return hi << 32 | mwc128x32_next32(state);
}

static INLINE double mwc128x32_next_double(mwc128x32_state_t *state) {
  const int32_t a = mwc128x32_next32(state) >> 5;
  const int32_t b = mwc128x32_next32(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

static INLINE uint32_t mwc64x32_next32(mwc64x32_state_t *state) {
  const uint32_t result = state->x ^ state->c;
  const uint64_t t = MWC64X32_A * (uint64_t)state->x + state->c;
  state->x = (uint32_t)t;
  state->c = (uint32_t)(t >> 32);
  return result;
}

static INLINE uint64_t mwc64x32_next64(mwc64x32_state_t *state) {
  const uint64_t hi = mwc64x32_next32(state);
  return hi << 32 | mwc64x32_next32(state);
}

static INLINE double mwc64x32_next_double(mwc64x32_state_t *state) {
  const int32_t a = mwc64x32_next32(state) >> 5;
  const int32_t b = mwc64x32_next32(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

#endif
