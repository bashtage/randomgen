/*
 * Romu Pseudorandom Number Generators
 *
 * Copyright 2020 Mark A. Overton
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
 * ------------------------------------------------------------------------------------------------
 *
 * Website: romu-random.org
 * Paper:   https://arxiv.org/abs/2002.11331
 *
 * Copy and paste the generator you want from those below.
 * To compile, you will need to #include <stdint.h> and use the ROTL definition
 * below.
 */

#ifndef _RANDOMDGEN__ROMU_H_
#define _RANDOMDGEN__ROMU_H_

#include "../common/randomgen_config.h"

struct ROMU_STATE_T {
  uint64_t w;
  uint64_t x;
  uint64_t y;
  uint64_t z;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct ROMU_STATE_T romu_state_t;

#ifdef _MSC_VER
#define ROTL(d, lrot) _rotl64(d, lrot)
#else
#define ROTL(d, lrot) ((d << (lrot)) | (d >> (64 - (lrot))))
#endif

static INLINE uint64_t romuQuad_random(romu_state_t *state) {
  uint64_t wp = state->w, xp = state->x, yp = state->y, zp = state->z;
  state->w = 15241094284759029579u * zp; // a-mult
  state->x = zp + ROTL(wp, 52);          // b-rotl, c-add
  state->y = yp - xp;                    // d-sub
  state->z = yp + wp;                    // e-add
  state->z = ROTL(state->z, 19);         // f-rotl
  return xp;
}

static INLINE uint64_t romuTrio_random(romu_state_t *state) {
  uint64_t xp = state->x, yp = state->y, zp = state->z;
  state->x = 15241094284759029579u * zp;
  state->y = yp - xp;
  state->y = ROTL(state->y, 12);
  state->z = zp - yp;
  state->z = ROTL(state->z, 44);
  return xp;
}

static INLINE uint64_t romuquad_next64(romu_state_t *state) {
  return romuQuad_random(state);
}

static INLINE uint64_t romutrio_next64(romu_state_t *state) {
  return romuTrio_random(state);
}

static INLINE uint32_t romuquad_next32(romu_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = romuQuad_random(state);
  state->has_uint32 = 1;
  state->uinteger = next >> 32;
  return (uint32_t)next;
}

static INLINE uint32_t romutrio_next32(romu_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = romuTrio_random(state);
  state->has_uint32 = 1;
  state->uinteger = next >> 32;
  return (uint32_t)next;
}

void romu_seed(romu_state_t *state, uint64_t w, uint64_t x, uint64_t y,
               uint64_t z, int quad);

#endif /* _RANDOMDGEN__ROMU_H_ */
