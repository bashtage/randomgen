/*
 * cl mt19937-benchmark.c mt19937.c /Ox
 * Measure-Command { .\mt19937-benchmark.exe }
 *
 * gcc mt19937-benchmark.c mt19937.c -O3 -o mt19937-benchmark
 * time ./mt19937-benchmark
 */
#include "mt19937.h"
#include <inttypes.h>

#define N 1000000000

int main() {
  int i;
  uint32_t seed = 0x0;
  uint64_t sum;
  mt19937_state state;
  mt19937_seed(&state, seed);
  for (i = 0; i < N; i++) {
    sum += mt19937_next64(&state);
  }
  printf("0x%" PRIx64 "\n", sum);
}
