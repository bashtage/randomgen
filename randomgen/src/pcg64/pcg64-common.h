#ifndef PCG64_COMMON_H_INCLUDED
#define PCG64_COMMON_H_INCLUDED 1

#include "../common/randomgen_config.h"
#include "pcg64-common.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && defined(_M_AMD64) && _M_AMD64
#include <intrin.h>
#pragma intrinsic(_umul128)
#endif

#if defined(__GNUC_GNU_INLINE__) && !defined(__cplusplus)
#error Nonstandard GNU inlining semantics. Compile with -std=c99 or better.
#endif

typedef uint64_t (*pcg_output_func_t)(uint64_t high, uint64_t low);

#if defined(__SIZEOF_INT128__) && !defined(PCG_FORCE_EMULATED_128BIT_MATH)

typedef __uint128_t pcg128_t;

#else

#define PCG_EMULATED_128BIT_MATH 1

typedef struct PCG128_T {
  uint64_t high;
  uint64_t low;
} pcg128_t;

#endif

#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
static INLINE pcg128_t PCG_128BIT_CONSTANT(uint64_t high, uint64_t low) {
  pcg128_t result;
  result.high = high;
  result.low = low;
  return result;
}
#else
#define PCG_128BIT_CONSTANT(high, low) (((pcg128_t)(high) << 64) + low)
#endif

#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
#define PCG_HIGH(a) a.high
#else
#define PCG_HIGH(a) (uint64_t)(a >> 64)
#endif

#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
#define PCG_LOW(a) a.low
#else
#define PCG_LOW(a) (uint64_t) a
#endif

static INLINE pcg128_t pcg128_add(pcg128_t a, pcg128_t b) {
  pcg128_t result;
#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  result.low = a.low + b.low;
  result.high = a.high + b.high + (result.low < b.low);
#else
  result = a + b;
#endif
  return result;
}

static INLINE void _pcg_mult64(uint64_t x, uint64_t y, uint64_t *z1,
                               uint64_t *z0) {
#if defined _WIN32 && _MSC_VER >= 1900 && _M_AMD64
  z0[0] = _umul128(x, y, z1);
#else
  uint64_t x0, x1, y0, y1;
  uint64_t w0, w1, w2, t;
  /* Lower 64 bits are straightforward clock-arithmetic. */
  *z0 = x * y;

  x0 = x & 0xFFFFFFFFULL;
  x1 = x >> 32;
  y0 = y & 0xFFFFFFFFULL;
  y1 = y >> 32;
  w0 = x0 * y0;
  t = x1 * y0 + (w0 >> 32);
  w1 = t & 0xFFFFFFFFULL;
  w2 = t >> 32;
  w1 += x0 * y1;
  *z1 = x1 * y1 + w2 + (w1 >> 32);
#endif
}

static INLINE pcg128_t pcg128_mult(pcg128_t a, pcg128_t b) {
  pcg128_t result;
#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  uint64_t h1;

  h1 = a.high * b.low + a.low * b.high;
  _pcg_mult64(a.low, b.low, &(result.high), &(result.low));
  result.high += h1;
#else
  result = a * b;
#endif
  return result;
}

static INLINE uint64_t pcg_rotr_64(uint64_t value, unsigned int rot) {
#ifdef _WIN32
  return _rotr64(value, rot);
#else
  return (value >> rot) | (value << ((-rot) & 63));
#endif
}

static INLINE uint64_t pcg_output_dxsm(uint64_t high, uint64_t low,
                                       const uint64_t dxsm_multiplier) {
  uint64_t hi = high;
  uint64_t lo = low;
  lo |= 1;
  hi ^= hi >> 32;
  hi *= dxsm_multiplier;
  hi ^= hi >> 48;
  hi *= lo;
  return hi;
}

static INLINE uint64_t pcg_output_xsl_rr(uint64_t high, uint64_t low) {
  return pcg_rotr_64(high ^ low, high >> 58u);
}

static INLINE uint64_t pcg_output_upper(uint64_t high, uint64_t low) {
  return high;
}

static INLINE uint64_t pcg_output_lower(uint64_t high, uint64_t low) {
  return low;
}

static INLINE uint64_t pcg_output_murmur3(uint64_t high, uint64_t low) {
  uint64_t z = (high ^ (high >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                             pcg128_t cur_plus);

#endif /* PCG64_COMMON_H_INCLUDED */