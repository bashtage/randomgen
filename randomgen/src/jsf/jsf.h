#ifndef JSF_H_INCLUDED
#define JSF_H_INCLUDED 1

/*
 * A C++ implementation of a Bob Jenkins Small Fast (Noncryptographic) PRNGs
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 Melissa E. O'Neill
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* Based on code published by Bob Jenkins in 2007, adapted for C from C++ */

#include "../common/randomgen_config.h"

typedef union JSF_UINT_T {
  uint64_t u64;
  uint32_t u32;
} jsf_uint_t;

struct JSF_STATE_T {
  jsf_uint_t a;
  jsf_uint_t b;
  jsf_uint_t c;
  jsf_uint_t d;
  int p;
  int q;
  int r;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct JSF_STATE_T jsf_state_t;

static INLINE uint64_t rotate64(uint64_t x, int k) {
#ifdef _MSC_VER
  return _rotl64(x, k);
#else
  return (x << k) | (x >> (64 - k));
#endif
}

static INLINE uint32_t rotate32(uint32_t x, int k) {
#ifdef _MSC_VER
  return _rotl(x, k);
#else
  return (x << k) | (x >> (32 - k));
#endif
}

static INLINE uint64_t next64(jsf_state_t *state) {
  uint64_t e;
  e = state->a.u64 - rotate64(state->b.u64, state->p);
  state->a.u64 = state->b.u64 ^ rotate64(state->c.u64, state->q);
  state->b.u64 = state->c.u64 +
                 (state->r ? rotate64(state->d.u64, state->r) : state->d.u64);
  state->c.u64 = state->d.u64 + e;
  state->d.u64 = e + state->a.u64;
  return state->d.u64;
}

static INLINE uint32_t next32(jsf_state_t *state) {
  uint32_t e;
  e = state->a.u32 - rotate32(state->b.u32, state->p);
  state->a.u32 = state->b.u32 ^ rotate32(state->c.u32, state->q);
  state->b.u32 = state->c.u32 +
                 (state->r ? rotate32(state->d.u32, state->r) : state->d.u32);
  state->c.u32 = state->d.u32 + e;
  state->d.u32 = e + state->a.u32;
  return state->d.u32;
}

static INLINE uint64_t jsf64_next64(jsf_state_t *state) {
    return next64(state);
}
static INLINE uint32_t jsf64_next32(jsf_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = next64(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}
static INLINE double jsf64_next_double(jsf_state_t *state) {
    return (next64(state) >> 11) * (1.0 / 9007199254740992.0);
}
static INLINE uint64_t jsf32_next64(jsf_state_t *state) {
    return (uint64_t)next32(state) << 32 | next32(state);
}
static INLINE uint32_t jsf32_next32(jsf_state_t *state) {
    return next32(state);
}
static INLINE double jsf32_next_double(jsf_state_t *state) {
  int32_t a = next32(state) >> 5, b = next32(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

void jsf64_seed(jsf_state_t *state, uint64_t *seed, int size);
void jsf32_seed(jsf_state_t *state, uint32_t *seed, int size);

#endif /* JSF_H_INCLUDED */
