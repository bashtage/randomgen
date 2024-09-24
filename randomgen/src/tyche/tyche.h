// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER
#include "../common/randomgen_config.h"

#ifndef _RANDOMGEN_TYCHE_H_
#define _RANDOMGEN_TYCHE_H_

const uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;

struct TYCHE_STATE_T {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
};

typedef struct TYCHE_STATE_T tyche_state_t;


inline uint32_t rotl(uint32_t value, unsigned int x) {
  return (value << x) | (value >> (32 - x));
}

inline void mix2(tyche_state_t *state) {
    state->b = rotl(state->b, 7) ^ state->c;
    state->c -= state->d;
    state->d = rotl(state->d, 8) ^ state->a;
    state->a -= state->b;
    state->b = rotl(state->b, 12) ^ state->c;
    state->c -= state->d;
    state->d = rotl(state->d, 16) ^ state->a;
    state->a -= state->b;
}

inline void mix(tyche_state_t *state){
     state->a += state->b;
     state->d = rotl(state->d ^ state->a, 16);
     state->c += state->d;
     state->b = rotl(state->b ^ state->c, 12);
     state->a += state->b;
     state->d = rotl(state->d ^ state->a, 8);
     state->c += state->d;
     state->b = rotl(state->b ^ state->c, 7);
  }

#define MIX mix

static inline uint32_t tyche_next(tyche_state_t *state) {
    MIX(state);
    return state->b;
}

static inline uint32_t tyche_next32(tyche_state_t *state) {
    return tyche_next(state);
}

static inline uint64_t tyche_next64(tyche_state_t *state) {
      return ((uint64_t)tyche_next(state) << 32) | tyche_next(state);
 }

static inline double tyche_next_double(tyche_state_t *state) {
      int32_t a = next_uint32(state) >> 5, b = next_uint32(state) >> 6;
      return (a * 67108864.0 + b) / 9007199254740992.0;
}

void tyche_seed(tyche_state_t *state, uint64_t seed, uint32_t ctr);

#endif  // _RANDOMGEN_TYCHE_H_
