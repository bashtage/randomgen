/*
*
* gcc SFMT-test-gen.c SFMT.c -DHAVE_SSE2 -DSFMT_MEXP=19937 -o SFMT
*/
#include "sfmt.h"
#include <inttypes.h>
#include <stdio.h>


int main(void) {
  int i;
  uint64_t *temp;
  uint32_t seed = 1UL;
  sfmt_t state;
  sfmt_init_gen_rand(&state, seed);
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
    printf("%d, %" PRIu64 "\n", i, out[i]);
  }
  fclose(fp);

  seed = 123456789UL;
  sfmt_init_gen_rand(&state, seed);
  sfmt_fill_array64(&state, out, 1000);
  fp = fopen("sfmt-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, %" PRIu32 "\n", seed);
  for (i = 0; i < 1000; i++) {
    fprintf(fp, "%d, %" PRIu64 "\n", i, out[i]);
    printf("%d, %" PRIu64 "\n", i, out[i]);
  }
  fclose(fp);
}
