/*
* cl sfmt-test-gen.c sfmt.c sfmt-jump.c -DHAVE_SSE2 -DSFMT_MEXP=19937 /O2
* gcc sfmt-test-gen.c sfmt.c sfmt-jump.c -DHAVE_SSE2 -DSFMT_MEXP=19937 -o sfmt /O2
*/
#include "sfmt.h"
#include "sfmt-test-data-seed.h"
#include <inttypes.h>
#include <stdio.h>


int main(void) {
  int i;
  uint64_t *temp;
  uint32_t seed = 0UL;
  sfmt_t state;
  printf("seed: %" PRIu32 "\n", SFMT_N64);
  sfmt_init_by_array(&state, (uint32_t *)&seed_seq_0, 2 * SFMT_N64);
  uint64_t out[1000];
  sfmt_fill_array64(&state, out, 1000);

  FILE *fp;
  fp = fopen("sfmt-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, %" PRIu32 "\n", seed);
  for (i = 0; i < 1000; i++) {
    fprintf(fp, "%d, %" PRIu64 "\n", i, out[i]);
    if (i == 999) {
      printf("%d, %" PRIu64 "\n", i, out[i]);
    }
    printf("%d, %" PRIu64 "\n", i, out[i]);
  }
  fclose(fp);

  seed = 0xDEADBEAFUL;
  sfmt_init_by_array(&state, (uint32_t *)&seed_seq_deadbeaf, 2 * SFMT_N64);
  sfmt_fill_array64(&state, out, 1000);
  fp = fopen("sfmt-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, %" PRIu32 "\n", seed);
  for (i = 0; i < 1000; i++) {
    fprintf(fp, "%d, %" PRIu64 "\n", i, out[i]);
    if (i == 999) {
      printf("%d, %" PRIu64 "\n", i, out[i]);
    }
  }
  fclose(fp);
}
