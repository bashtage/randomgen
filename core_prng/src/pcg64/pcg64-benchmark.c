/*
 * cl pcg64-benchmark.c pcg64.c ../splitmix64/splitmix64.c /Ox
 * Measure-Command { .\xoroshiro128-benchmark.exe }
 *
 * gcc pcg64-benchmark.c pcg64.c ../splitmix64/splitmix64.c -O3 -o
 *    pcg64-benchmark
 * time ./pcg64-benchmark
 */
#include "../splitmix64/splitmix64.h"
#include "pcg64.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000000000

int main() {
  pcg64_random_t rng;
  uint64_t sum = 0;
  uint64_t seed = 0xDEADBEAF;
  int i;
  rng.state.high = splitmix64_next(&seed);
  rng.state.low = splitmix64_next(&seed);
  rng.inc.high = 0;
  rng.inc.low = 1;

  for (i = 0; i < N; i++) {
    sum += pcg64_random_r(&rng);
  }
  printf("0x%" PRIx64 "\n", sum);
}
