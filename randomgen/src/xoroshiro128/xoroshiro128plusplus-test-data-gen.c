/*
 * Generate testing csv files
 *
 *  cl xoroshiro128plusplus-test-data-gen.c xoroshiro128plusplus.orig.c  /Ox
 *  xoroshiro128plusplus-test-data-gen.exe
 *
 *  gcc xoroshiro128plusplus-test-data-gen.c xoroshiro128plusplus.orig.c /
 *  -o xoroshiro128plusplus-test-data-gen
 *  ./xoroshiro128plusplus-test-data-gen
 *
 */

#include "xoroshiro128plus.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main()
{
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  int i;
  /* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint64) */
  s[0] = 5778446405158232650;
  s[1] = 4639759349701729399;
  uint64_t store[N];
  for (i = 0; i < N; i++)
  {
    store[i] = next();
  }

  FILE *fp;
  fp = fopen("xoroshiro128plusplus-testset-1.csv", "w");
  if (fp == NULL)
  {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++)
  {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999)
    {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  seed = state = 0;
  for (i = 0; i < 2; i++)
  /* SeedSequence(0).generate_state(2, dtype=np.uint64) */
  s[0] = 15793235383387715774;
  s[1] = 12390638538380655177;
  for (i = 0; i < N; i++)
  {
    store[i] = next();
  }
  fp = fopen("xoroshiro128plusplus-testset-2.csv", "w");
  if (fp == NULL)
  {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++)
  {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999)
    {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
