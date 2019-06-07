#ifndef _RANDOMDGEN__PHILOX4X32_H_
#define _RANDOMDGEN__PHILOX4X32_H_

#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/inttypes.h"
#define INLINE __forceinline
#else
#include <inttypes.h>
#define INLINE __inline __forceinline
#endif
#else
#include <inttypes.h>
#define INLINE inline
#endif

#define PHILOX_BUFFER_SIZE 4

struct r123array2x32 {
  uint32_t v[2];
};
struct r123array4x32 {
  uint32_t v[4];
};

static INLINE uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t *hip) {
  /* TODO: consider _mulx_u32, requires new CPU. Also __emulu */
  uint64_t product = ((uint64_t)a) * ((uint64_t)b);
  *hip = product >> 32;
  return (uint32_t)product;
}

static INLINE struct r123array2x32 _philox4x32bumpkey(struct r123array2x32 key) {
  key.v[0] += ((uint32_t)0x9E3779B9);
  key.v[1] += ((uint32_t)0xBB67AE85);
  return key;
}
static INLINE struct r123array4x32 _philox4x32round(struct r123array4x32 ctr,
                                             struct r123array2x32 key) {
  uint32_t hi0;
  uint32_t hi1;
  uint32_t lo0 = mulhilo32(((uint32_t)0xD2511F53), ctr.v[0], &hi0);
  uint32_t lo1 = mulhilo32(((uint32_t)0xCD9E8D57), ctr.v[2], &hi1);
  struct r123array4x32 out = {
      {hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
  return out;
}

enum r123_enum_philox4x32 { philox4x32_rounds = 10 };
typedef struct r123array4x32 philox4x32_ctr_t;
typedef struct r123array2x32 philox4x32_key_t;
typedef struct r123array2x32 philox4x32_ukey_t;

static INLINE philox4x32_ctr_t philox4x32_R(unsigned int R, philox4x32_ctr_t ctr,
                                     philox4x32_key_t key) {
  if (R > 0) {
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 1) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 2) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 3) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 4) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 5) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 6) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 7) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 8) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 9) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 10) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 11) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 12) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 13) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 14) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  if (R > 15) {
    key = _philox4x32bumpkey(key);
    ctr = _philox4x32round(ctr, key);
  }
  return ctr;
}


typedef struct s_philox4x32_state {
  philox4x32_ctr_t *ctr;
  philox4x32_key_t *key;
  int buffer_pos;
  uint32_t buffer[PHILOX_BUFFER_SIZE];
} philox4x32_state;

static INLINE uint32_t philox4x32_next(philox4x32_state *state) {
  uint32_t out;
  int i;
  philox4x32_ctr_t ct;

  if (state->buffer_pos < PHILOX_BUFFER_SIZE) {
    out = state->buffer[state->buffer_pos];
    state->buffer_pos++;
    return out;
  }
  /* generate 4 new uint32_t */
  state->ctr->v[0]++;
  /* Handle carry */
  if (state->ctr->v[0] == 0) {
    state->ctr->v[1]++;
    if (state->ctr->v[1] == 0) {
      state->ctr->v[2]++;
      if (state->ctr->v[2] == 0) {
        state->ctr->v[3]++;
      }
    }
  }
  ct = philox4x32_R(philox4x32_rounds, *state->ctr, *state->key);
  out = ct.v[0];
  state->buffer_pos = 1;
  for (i = 1; i < 4; i++) {
    state->buffer[i] = ct.v[i];
  }
  return out;
}

static INLINE uint64_t philox4x32_next64(philox4x32_state *state) {
  return ((uint64_t)philox4x32_next(state) << 32) | philox4x32_next(state);
}

static INLINE uint32_t philox4x32_next32(philox4x32_state *state) {
  return philox4x32_next(state);
}

static INLINE double philox4x32_next_double(philox4x32_state *state) {
  int32_t a = philox4x32_next(state) >> 5, b = philox4x32_next(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern void philox4x32_advance(uint32_t *step, philox4x32_state *state);

#endif /* _RANDOMDGEN__PHILOX4X32_H_ */

