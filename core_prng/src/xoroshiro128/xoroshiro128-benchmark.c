/*
 * cl xoroshiro128-benchmark.c xoroshiro128plus.orig.c \
 * ../splitmix64/splitmix64.c /Ox
 * Measure-Command { .\xoroshiro128-benchmark.exe }
 *
 * gcc -O3 xoroshiro128-benchmark.c xoroshiro128plus.orig.c \
 * ../splitmix64/splitmix64.c -o  xoroshiro128-benchmark time
 * ./xoroshiro128-benchmark
 *
 */
#include "../splitmix64/splitmix64.h"
#include "xoroshiro128plus.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000000000

int main() {
  uint64_t sum = 0;
  uint64_t seed = 0xDEADBEAF;
  s[0] = splitmix64_next(&seed);
  s[1] = splitmix64_next(&seed);
  int i;
  for (i = 0; i < N; i++) {
    sum += next();
  }
  printf("0x%" PRIx64 "\n", sum);
}
