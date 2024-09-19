/*
 * cl dSFMT-test-gen.c dSFMT.c dSFMT-jump.c -DHAVE_SSE2 -DDSFMT_MEXP=19937 /O2
 *
 * gcc dSFMT-test-gen.c dSFMT.c  dSFMT-jump.c -DHAVE_SSE2 -DDSFMT_MEXP=19937 -o dSFMT
 */

#include <inttypes.h>
#include <stdio.h>
#include "dsfmt.h"
#include "dsfmt-test-data-seed.h"


int main(void) {
  int i;
  double d;
  uint64_t *temp;
  uint32_t seed = 0UL;
  dsfmt_t state;
  dsfmt_init_by_array(&state, seed_seq_0, 2 * DSFMT_N64);
  double out[1000];
  dsfmt_fill_array_close1_open2(&state, out, 1000);

  FILE *fp;
  fp = fopen("dSFMT-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, %" PRIu32 "\n", seed);
  for (i = 0; i < 1000; i++) {
    d = out[i];
    temp = (uint64_t *)&d;
    fprintf(fp, "%d, %" PRIu64 "\n", i, *temp);
    if (i==999) {
        printf("%d, %" PRIu64 "\n", i, *temp);
    }
  }
  fclose(fp);

  seed = 0xDEADBEAFUL;
  dsfmt_init_by_array(&state, seed_seq_deadbeaf, 2 * DSFMT_N64);
  dsfmt_fill_array_close1_open2(&state, out, 1000);
  fp = fopen("dSFMT-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, %" PRIu32 "\n", seed);
  for (i = 0; i < 1000; i++) {
    d = out[i];
    temp = (uint64_t *)&d;
    fprintf(fp, "%d, %" PRIu64 "\n", i, *temp);
    if (i==999) {
        printf("%d, %" PRIu64 "\n", i, *temp);
    }
  }
  fclose(fp);
}
