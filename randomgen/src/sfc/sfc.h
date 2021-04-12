#ifndef SFC_H_INCLUDED
#define SFC_H_INCLUDED 1

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

struct SFC_STATE_T {
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint64_t w;
  uint64_t k;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct SFC_STATE_T sfc_state_t;

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

static INLINE uint64_t next64(sfc_state_t *state) {
    enum {LROT = 24, RSHIFT = 11, LSHIFT = 3};
    const uint64_t out = state->a + state->b + state->w;
    state->w += state->k;
    state->a = state->b ^ (state->b >> RSHIFT);
    state->b = state->c + (state->c << LSHIFT);
    state->c = ((state->c << LROT) | (state->c >> (64 - LROT))) + out;
    return out;
}

static INLINE uint64_t sfc_next64(sfc_state_t *state) {
    return next64(state);
}
static INLINE uint32_t sfc_next32(sfc_state_t *state) {
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

void sfc_seed(sfc_state_t *state, uint64_t *seed, uint64_t w, uint64_t k);


#endif /* SFC_H_INCLUDED */
