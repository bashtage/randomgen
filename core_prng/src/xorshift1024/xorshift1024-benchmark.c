/*
 * cl xorshift1024-benchmark.c xorshift2014.orig.c
 *     ../splitmix64/splitmix64.c /Ox
 *
 * Measure-Command { .\xorshift1024-benchmark.exe }
 *
 * gcc -O3 xorshift1024-benchmark.c xorshift2014.orig.c /
 * ../splitmix64/splitmix64.c -o xorshift1024-benchmark time
 * ./xoroshiro128-benchmark
 *
 */
#include "../splitmix64/splitmix64.h"
#include "xorshift1024.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000000000

int main() {
  uint64_t sum = 0;
  uint64_t seed = 0xDEADBEAF;
  int i;
  for (i = 0; i < 16; i++) {
    s[i] = splitmix64_next(&seed);
  }
  p = 0;
  for (i = 0; i < N; i++) {
    sum += next();
  }
  printf("0x%" PRIx64 "\n", sum);
}
