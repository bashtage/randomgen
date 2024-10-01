#include <inttypes.h>

#ifndef _SQUARES_H_
#define _SQUARES_H_


inline static uint32_t squares32(uint64_t key, uint64_t ctr) {
  uint64_t x, y, z;

  y = x = ctr * key;
  z = y + key;
  ctr++;
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  x = x * x + z;
  x = (x >> 32) | (x << 32);
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  return (x * x + z) >> 32;
}

inline static uint64_t squares64(uint64_t key, uint64_t ctr) {
  uint64_t t, x, y, z;
  y = x = ctr * key;
  z = y + key;
  ctr++;
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

#endif