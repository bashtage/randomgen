/*
 *
 * cl dsfmt-benchmark.c dSFMT.c /Ox -DHAVE_SSE2
 * Measure-Command { .\dsfmt-benchmark.exe }
 *
 *
 */
#include "dSFMT.h"
#include <inttypes.h>

#define N 1000000000

int main() {
  int i, j;
  uint32_t seed = 0xDEADBEAF;
  uint64_t total = 0, sum = 0;
  dsfmt_t state;
  double buffer[DSFMT_N64];

  uint64_t out;
  uint64_t *tmp;
  dsfmt_init_gen_rand(&state, seed);
  for (i = 0; i < N / (DSFMT_N64 / 2); i++) {
    dsfmt_fill_array_close_open(&state, &buffer[0], DSFMT_N64);
    for (j = 0; j < DSFMT_N64; j += 2) {
      tmp = (uint64_t *)&buffer[j];
      out = (*tmp >> 16) << 32;
      tmp = (uint64_t *)&buffer[j + 1];
      out |= (*tmp >> 16) & 0xffffffff;
      sum += out;
      total++;
    }
  }
  printf("0x%" PRIx64 ", total: %" PRIu64 "\n", sum, total);
}
