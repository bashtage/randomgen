/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     https://www.pcg-random.org
 *
 * Relicensed MIT in May 2019
 *
 * The MIT License
 *
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef PCG64_H_INCLUDED
#define PCG64_H_INCLUDED 1

#include "../common/randomgen_config.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && defined(_M_AMD64) && _M_AMD64
#include <intrin.h>
#pragma intrinsic(_umul128)
#endif

#if __GNUC_GNU_INLINE__ && !defined(__cplusplus)
#error Nonstandard GNU inlining semantics. Compile with -std=c99 or better.
#endif

#if __cplusplus
extern "C" {
#endif

#if defined(__SIZEOF_INT128__) && !defined(PCG_FORCE_EMULATED_128BIT_MATH)
typedef __uint128_t pcg128_t;
#define PCG_128BIT_CONSTANT(high, low) (((pcg128_t)(high) << 64) + low)
#else
typedef struct {
  uint64_t high;
  uint64_t low;
} pcg128_t;

static INLINE pcg128_t PCG_128BIT_CONSTANT(uint64_t high, uint64_t low) {
  pcg128_t result;
  result.high = high;
  result.low = low;
  return result;
}

#define PCG_EMULATED_128BIT_MATH 1
#endif

typedef struct {
  pcg128_t state;
} pcg_state_128;

typedef struct {
  pcg128_t state;
  pcg128_t inc;
} pcg_state_setseq_128;

#define PCG_DEFAULT_MULTIPLIER_HIGH 2549297995355413924ULL
#define PCG_DEFAULT_MULTIPLIER_LOW 4865540595714422341ULL

#define PCG_DEFAULT_MULTIPLIER_128                                             \
  PCG_128BIT_CONSTANT(PCG_DEFAULT_MULTIPLIER_HIGH, PCG_DEFAULT_MULTIPLIER_LOW)
#define PCG_DEFAULT_INCREMENT_128                                              \
  PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)
#define PCG_CHEAP_MULTIPLIER_HIGH 0ULL
#define PCG_CHEAP_MULTIPLIER_LOW 0xda942042e4dd58b5ULL
#define PCG_CHEAP_MULTIPLIER_128                                               \
  PCG_128BIT_CONSTANT(0ULL, 0xda942042e4dd58b5ULL)

static INLINE uint64_t pcg_rotr_64(uint64_t value, unsigned int rot) {
#ifdef _WIN32
  return _rotr64(value, rot);
#else
  return (value >> rot) | (value << ((-rot) & 63));
#endif
}

static INLINE uint64_t pcg_output_dxsm(uint64_t high, uint64_t low) {
  uint64_t hi = high;
  uint64_t lo = low;

  lo |= 1;
  hi ^= hi >> 32;
  hi *= 0xda942042e4dd58b5ULL;
  hi ^= hi >> 48;
  hi *= lo;
  return hi;
}

#ifdef PCG_EMULATED_128BIT_MATH

static INLINE pcg128_t pcg128_add(pcg128_t a, pcg128_t b) {
  pcg128_t result;

  result.low = a.low + b.low;
  result.high = a.high + b.high + (result.low < b.low);
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
  uint64_t h1;
  pcg128_t result;

  h1 = a.high * b.low + a.low * b.high;
  _pcg_mult64(a.low, b.low, &(result.high), &(result.low));
  result.high += h1;
  return result;
}

static INLINE void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng) {
  rng->state =
      pcg128_add(pcg128_mult(rng->state, PCG_DEFAULT_MULTIPLIER_128), rng->inc);
}

static INLINE void pcg_cm_step_r(pcg_state_setseq_128 *rng) {
  // TODO: Fix pcg128_mult to not use the high (Maybe)
  rng->state =
      pcg128_add(pcg128_mult(rng->state, PCG_CHEAP_MULTIPLIER_128), rng->inc);
}

static INLINE uint64_t pcg_output_xsl_rr_128_64(pcg128_t state) {
  return pcg_rotr_64(state.high ^ state.low, state.high >> 58u);
}

static INLINE void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq,
                                            int cheap_multiplier) {
  rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
  rng->inc.high = initseq.high << 1u;
  rng->inc.high |= initseq.low >> 63u;
  rng->inc.low = (initseq.low << 1u) | 1u;
  if (cheap_multiplier == 0) {
    pcg_setseq_128_step_r(rng);
  } else {
    pcg_cm_step_r(rng);
  }
  rng->state = pcg128_add(rng->state, initstate);
  if (cheap_multiplier == 0) {
    pcg_setseq_128_step_r(rng);
  } else {
    pcg_cm_step_r(rng);
  }
}

static INLINE uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128 *rng, int use_dxsm) {
#if defined _WIN32 && _MSC_VER >= 1900 && _M_AMD64
  uint64_t h1;
  pcg128_t product;

  /* Manually inline the multiplication and addition using intrinsics */
  h1 = rng->state.high * PCG_DEFAULT_MULTIPLIER_LOW +
       rng->state.low * PCG_DEFAULT_MULTIPLIER_HIGH;
  product.low =
      _umul128(rng->state.low, PCG_DEFAULT_MULTIPLIER_LOW, &(product.high));
  product.high += h1;
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
  if (use_dxsm != 0) {
    return pcg_output_dxsm(rng->state.high, rng->state.low);
  } else {
    return _rotr64(rng->state.high ^ rng->state.low, rng->state.high >> 58u);
  }
#else
  pcg_setseq_128_step_r(rng);
  if (use_dxsm != 0) {
    return pcg_output_dxsm(rng->state.high, rng->state.low);
  } else {
    return pcg_output_xsl_rr_128_64(rng->state);
  }
#endif
}

static INLINE uint64_t pcg_cm_dxsm_random_r(pcg_state_setseq_128 *rng) {
#if defined _WIN32 && _MSC_VER >= 1900 && _M_AMD64
  uint64_t h1, out;
  pcg128_t product;

  out = pcg_output_dxsm(rng->state.high, rng->state.low);
  /* Manually inline the multiplication and addition using intrinsics */

  h1 = rng->state.high * PCG_CHEAP_MULTIPLIER_LOW + rng->state.low * 0;
  product.low =
      _umul128(rng->state.low, PCG_CHEAP_MULTIPLIER_LOW, &(product.high));
  product.high += h1;
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
  return out;
#else
  uint64_t out = pcg_output_dxsm(rng->state.high, rng->state.low);
  pcg_cm_step_r(rng);
  return out;
#endif
}

#else /* PCG_EMULATED_128BIT_MATH */

static INLINE void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng) {
  rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_128 + rng->inc;
}

static INLINE void pcg_cm_step_r(pcg_state_setseq_128 *rng) {
  rng->state = rng->state * PCG_CHEAP_MULTIPLIER_LOW + rng->inc;
}

static INLINE uint64_t pcg_output_xsl_rr_128_64(pcg128_t state) {
  return pcg_rotr_64(((uint64_t)(state >> 64u)) ^ (uint64_t)state,
                     state >> 122u);
}

static INLINE uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128 *rng, int use_dxsm) {
  pcg_setseq_128_step_r(rng);
  if (use_dxsm != 0) {
    uint64_t hi = rng->state >> 64, lo = rng->state & 0xFFFFFFFFFFFFFFFFULL;
    return pcg_output_dxsm(hi, lo);
  } else {
    return pcg_output_xsl_rr_128_64(rng->state);
  }
}

static INLINE uint64_t pcg_cm_dxsm_random_r(pcg_state_setseq_128 *rng) {
  uint64_t out, hi = rng->state >> 64, lo = rng->state & 0xFFFFFFFFFFFFFFFFULL;
  out = pcg_output_dxsm(hi, lo);
  pcg_cm_step_r(rng);
  return out;
}

static INLINE void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq,
                                            int cheap_multiplier) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  if (cheap_multiplier == 0) {
    pcg_setseq_128_step_r(rng);
  } else {
    pcg_cm_step_r(rng);
  }
  rng->state += initstate;
  if (cheap_multiplier == 0) {
    pcg_setseq_128_step_r(rng);
  } else {
    pcg_cm_step_r(rng);
  }
}

#endif /* PCG_EMULATED_128BIT_MATH */

extern pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta,
                                    pcg128_t cur_mult, pcg128_t cur_plus);

static INLINE void pcg_setseq_128_advance_r(pcg_state_setseq_128 *rng,
                                            pcg128_t delta,
                                            int cheap_multiplier) {
  if (cheap_multiplier == 0) {
    rng->state = pcg_advance_lcg_128(rng->state, delta,
                                     PCG_DEFAULT_MULTIPLIER_128, rng->inc);
  } else {
    rng->state = pcg_advance_lcg_128(rng->state, delta,
                                     PCG_CHEAP_MULTIPLIER_128, rng->inc);
  }
}

typedef pcg_state_setseq_128 pcg64_random_t;
#define pcg64_random_r pcg_setseq_128_xsl_rr_64_random_r
#define pcg64_srandom_r pcg_setseq_128_srandom_r
#define pcg64_advance_r pcg_setseq_128_advance_r
#define pcg64_dxsm_random_r pcg_cm_dxsm_random_r

#if __cplusplus
}
#endif

typedef struct PCG64_STATE_T {
  pcg64_random_t *pcg_state;
  int use_dxsm;
  int has_uint32;
  uint32_t uinteger;
} pcg64_state_t;

static INLINE uint64_t pcg64_next64(pcg64_state_t *state) {
  return pcg64_random_r(state->pcg_state, state->use_dxsm);
}

static INLINE uint64_t pcg64_cm_dxsm_next64(pcg64_state_t *state) {
  return pcg64_dxsm_random_r(state->pcg_state);
}

static INLINE uint32_t pcg64_next32(pcg64_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = pcg64_random_r(state->pcg_state, state->use_dxsm);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE uint32_t pcg64_cm_dxsm_next32(pcg64_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = pcg64_dxsm_random_r(state->pcg_state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void pcg64_advance(pcg64_state_t *state, uint64_t *step, int cheap_multiplier);

void pcg64_set_seed(pcg64_state_t *state, uint64_t *seed, uint64_t *inc,
                    int cheap_multiplier);

void pcg64_get_state(pcg64_state_t *state, uint64_t *state_arr, int *use_dxsm,
                     int *has_uint32, uint32_t *uinteger);

void pcg64_set_state(pcg64_state_t *state, uint64_t *state_arr, int use_dxsm,
                     int has_uint32, uint32_t uinteger);

#endif /* PCG64_H_INCLUDED */
