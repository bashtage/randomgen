/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
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
 */

/* This code provides a mechanism for getting external randomness for
 * seeding purposes.  Usually, it's just a wrapper around reading
 * /dev/random.
 *
 * Alas, because not every system provides /dev/random, we need a fallback.
 * We also need to try to test whether or not to use the fallback.
 */

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "entropy.h"
#ifdef _WIN32
/* Windows */
#include <sys/timeb.h>
#include <time.h>
#include <windows.h>

#include <bcrypt.h>
#else
/* Unix */
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#endif

#ifndef IS_UNIX
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) ||               \
                         (defined(__APPLE__) && defined(__MACH__)))
#define IS_UNIX 1
#else
#define IS_UNIX 0
#endif
#endif

// If HAVE_DEV_RANDOM is set, we use that value, otherwise we guess
#ifndef HAVE_DEV_RANDOM
#define HAVE_DEV_RANDOM IS_UNIX
#endif

#if HAVE_DEV_RANDOM
#include <fcntl.h>
#include <unistd.h>
#endif

/* Largest single request made to the system, in bytes */
#define ENTROPY_MAX_CHUNK ((size_t)1 << 30)

#if HAVE_DEV_RANDOM
/* entropy_getbytes(dest, size):
 *     Use /dev/random to get some external entropy for seeding purposes.
 *
 * Note:
 *     If reading /dev/random fails (which ought to never happen), it returns
 *     false, otherwise it returns true.  If it fails, you could instead call
 *     fallback_entropy_getbytes which always succeeds.
 */

bool entropy_getbytes(void *dest, size_t size) {
  unsigned char *out = (unsigned char *)dest;
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0)
    return false;
  /* read can return fewer bytes than requested or be interrupted by a signal */
  while (size > 0) {
    size_t chunk = size < ENTROPY_MAX_CHUNK ? size : ENTROPY_MAX_CHUNK;
    ssize_t sz = read(fd, out, chunk);
    if (sz < 0) {
      if (errno == EINTR)
        continue;
      close(fd);
      return false;
    }
    if (sz == 0) {
      close(fd);
      return false;
    }
    out += sz;
    size -= (size_t)sz;
  }
  return close(fd) == 0;
}
#endif

#ifdef _WIN32
bool entropy_getbytes(void *dest, size_t size) {
  unsigned char *out = (unsigned char *)dest;

  /* The length argument of BCryptGenRandom is a ULONG */
  while (size > 0) {
    ULONG chunk = (ULONG)(size < ENTROPY_MAX_CHUNK ? size : ENTROPY_MAX_CHUNK);
    if (!BCRYPT_SUCCESS(BCryptGenRandom(NULL, out, chunk,
                                        BCRYPT_USE_SYSTEM_PREFERRED_RNG))) {
      return false;
    }
    out += chunk;
    size -= chunk;
  }
  return true;
}
#endif

/* Thomas Wang 32/64 bits integer hash function */
uint32_t entropy_hash_32(uint32_t key) {
  key += ~(key << 15);
  key ^= (key >> 10);
  key += (key << 3);
  key ^= (key >> 6);
  key += ~(key << 11);
  key ^= (key >> 16);
  return key;
}

uint64_t entropy_hash_64(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

uint32_t entropy_randombytes(void) {

#ifndef _WIN32
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return entropy_hash_32(getpid()) ^ entropy_hash_32(tv.tv_sec) ^
         entropy_hash_32(tv.tv_usec) ^ entropy_hash_32(clock());
#else
  uint32_t out = 0;
  int64_t counter;
  struct _timeb tv;
  _ftime_s(&tv);
  out = entropy_hash_32(GetCurrentProcessId()) ^
        entropy_hash_32((uint32_t)tv.time) ^ entropy_hash_32(tv.millitm) ^
        entropy_hash_32(clock());
  if (QueryPerformanceCounter((LARGE_INTEGER *)&counter) != 0)
    out ^= entropy_hash_32((uint32_t)(counter & 0xffffffff));
  return out;
#endif
}

bool entropy_fallback_getbytes(void *dest, size_t size) {
  unsigned char *out = (unsigned char *)dest;
  while (size > 0) {
    uint32_t word = entropy_randombytes();
    size_t n = size < sizeof(word) ? size : sizeof(word);
    memcpy(out, &word, n);
    out += n;
    size -= n;
  }
  return true;
}

void entropy_fill(void *dest, size_t size) {
  bool success;
  success = entropy_getbytes(dest, size);
  if (!success) {
    entropy_fallback_getbytes(dest, size);
  }
}
