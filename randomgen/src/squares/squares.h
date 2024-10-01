#include "../common/randomgen_config.h"

#ifndef _RANDOMGEN_SQUARES_H_
#define _RANDOMGEN_SQUARES_H_

struct SQUARES_STATE_T {
  uint64_t key;
  uint64_t counter;
  int has_uint32;
  uint32_t uinteger;
};
typedef struct SQUARES_STATE_T squares_state_t;

inline static uint32_t squares32(squares_state_t *state) {
  uint64_t x, y, z;

  y = x = state->counter * state->key;
  z = y + state->key;
  state->counter++;
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  x = x * x + z;
  x = (x >> 32) | (x << 32);
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  return (x * x + z) >> 32;
}

inline static uint64_t squares64(squares_state_t *state) {
  uint64_t t, x, y, z;
  y = x = state->counter * state->key;
  z = y + state->key;
  state->counter++;
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  x = x * x + z;
  x = (x >> 32) | (x << 32);
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  t = x = x * x + z;
  x = (x >> 32) | (x << 32);
  return t ^ ((x * x + y) >> 32);
}

inline static uint64_t squares_next64(squares_state_t *state) {
  return squares64(state);
}
inline static uint32_t squares_next32(squares_state_t *state) {
  uint64_t next;
   if (state -> has_uint32) {
        state -> has_uint32 = 0;
        return state -> uinteger;
    }
  next = squares64(state);
  state->has_uint32 = 1;
  state->uinteger = next >> 32;
  return (uint32_t)next;
}

inline static double squares_next_double(squares_state_t *state) {
  return (squares64(state) >> 11) * (1.0 / 9007199254740992.0);
}
inline static uint64_t squares_32_next64(squares_state_t *state) {
  return squares32(state) | (((uint64_t)squares32(state)) << 32);
}

inline static uint32_t squares_32_next32(squares_state_t *state) {
  return squares32(state);
}
inline static double squares_32_next_double(squares_state_t *state) {
  int32_t a = squares32(state) >> 5, b = squares32(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

#endif