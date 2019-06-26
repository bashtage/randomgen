/* Author: Lucas Clemente Vella
 * Source code placed into public domain. */

#pragma once

#include <inttypes.h>
#include <stddef.h>

#ifdef UNALIGNED_ACCESS_ALLOWED
#define UNALIGNED_ACCESS 1
#else
#define UNALIGNED_ACCESS 0
#endif

uint32_t rotl(uint32_t x, unsigned int n);

uint32_t pack_littleendian(const uint8_t *v);
void unpack_littleendian(uint32_t value, uint8_t *v);

size_t min(size_t a, size_t b);

int is_aligned(const void *ptr);
