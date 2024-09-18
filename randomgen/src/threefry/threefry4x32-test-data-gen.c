/*
 * Generate testing csv files
 *
 *  cl threefry-test-data-gen.c /Ox
 *  threefry-test-data-gen.exe
 *
 *  gcc threefry-test-data-gen.c  /O2 -o threefry-test-data-gen
 *  ./threefry-test-data-gen
 *
 * Requires the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "Random123/threefry.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  threefry4x32_key_t ctr = {{0, 0, 0, 0}};
  uint32_t state, seed = 0xDEADBEAF;
  /* SeedSequence(0xDEADBEAF).generate_state(4, dtype=np.uint32) */
  uint32_t seed_seq[4] = {3575046730, 1345399395, 3393510519, 1080278155};
  state = seed;
  threefry4x32_ctr_t key = {{0}};
  threefry4x32_ctr_t out;
  uint32_t store[N];
  int i, j;
  for (i = 0; i < 4; i++) {
    key.v[i] = seed_seq[i];
  }
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = threefry4x32_R(threefry4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  FILE *fp;
  fp = fopen("threefry4x32-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx32 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx32 "\n", i, store[i]);
    }
  }
  fclose(fp);


  state = seed = 0;
  /* SeedSequence(0).generate_state(4, dtype=np.uint32) */
  key.v[0] = 2968811710;
  key.v[1] = 3677149159;
  key.v[2] = 745650761;
  key.v[3] = 2884920346;
  ctr.v[0] = 0;
  ctr.v[1] = 0;
  ctr.v[2] = 0;
  ctr.v[3] = 0;

  for (i = 0; i < N / 4; i++) {
    ctr.v[0]++;
    out = threefry4x32_R(threefry4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  fp = fopen("threefry4x32-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx32 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx32 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
