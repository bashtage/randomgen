/* Author: Lucas Clemente Vella
 * Source code placed into public domain. */

#include "util.orig.h"

uint32_t
rotl(uint32_t x, unsigned int n)
{
  return (x << n) | (x >> (32-n));
}

uint32_t
pack_littleendian(const uint8_t *v)
{
#ifdef LITTLE_ENDIAN
  return *((uint32_t*)v);
#else
#error "Only use on LE"
	return (uint32_t)v[3] << 24
      | (uint32_t)v[2] << 16
      | (uint32_t)v[1] << 8
      | (uint32_t)v[0];
#endif
}

void
unpack_littleendian(uint32_t value, uint8_t *v)
{
#ifdef LITTLE_ENDIAN
  *((uint32_t*)v) = value;
#else
#error "Only use on LE"
  int i;
  for(i = 0; i < 4; ++i)
    v[i] = value >> (i * 8);
#endif
}

size_t
min(size_t a, size_t b)
{
  return (a < b) ? a : b;
}

int
is_aligned(const void *ptr)
{
  return ((unsigned)ptr & 3u) == 0; /* Multiple of 4 */
}
