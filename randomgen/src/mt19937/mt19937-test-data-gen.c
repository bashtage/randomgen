/*
 * Generate testing csv files
 *
 * cl mt19937-test-data-gen.c randomkit.c -IC:\anaconda\envs\randomgen\Lib\site-packages\numpy\_core\include -IC:\anaconda\envs\randomgen\include Advapi32.lib Kernel32.lib C:\anaconda\envs\randomgen\libs\python312.lib  -DRK_NO_WINCRYPT=1
 *
 */
#include "randomkit.h"
#include <inttypes.h>
#include <stdio.h>
#include "mt119937-test-data-seed.h"
#define N 1000

int main() {
  uint64_t sum = 0;
  uint32_t seed = 0xDEADBEAF;
  int i;
  rk_state state;
  uint64_t store[N];
  rk_init_by_array(&state, (uint32_t *)&seed_seq_deadbeaf, 624);
  for (i = 0; i < N; i++) {
    store[i] = (uint64_t)rk_random(&state);
  }

  FILE *fp;
  fp = fopen("mt19937-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  seed = 0;
  rk_init_by_array(&state, (uint32_t *)&seed_seq_0, 624);
  for (i = 0; i < N; i++) {
    store[i] = (uint64_t)rk_random(&state);
  }
  fp = fopen("mt19937-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx32 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
