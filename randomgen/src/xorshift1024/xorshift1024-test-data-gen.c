/*
 * Generate testing csv files
 *
 *  cl xorshift1024-test-data-gen.c xorshift1024.orig.c /Ox
 * xorshift1024-test-data-gen.exe *
 *
 *  gcc xorshift1024-test-data-gen.c xorshift1024.orig.c /
 *  -o xorshift1024-test-data-gen
 *  ./xorshift1024-test-data-gen
 *
 */

#include "xorshift1024.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  uint64_t sum = 0;
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  int i;
  /* SeedSequence(0xDEADBEAF).generate_state(16, dtype=np.uint64) */
  s[0] = 5778446405158232650;
  s[1] = 4639759349701729399;
  s[2] = 13222832537653397986;
  s[3] = 2330059127936092250;
  s[4] = 6380887635277085283;
  s[5] = 2943025801430425506;
  s[6] = 16158800551411432655;
  s[7] = 4467384082323269519;
  s[8] = 4163226376263453348;
  s[9] = 16628552531038748367;
  s[10] = 17616013123752890245;
  s[11] = 17578598327112781894;
  s[12] = 438609640508191089;
  s[13] = 13797137212871506356;
  s[14] = 17329687526801996224;
  s[15] = 4335059551211669809;
  p = 0;
  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = next();
  }

  FILE *fp;
  fp = fopen("xorshift1024-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  seed = state = 0;
  /* SeedSequence(0).generate_state(16, dtype=np.uint64) */
  s[0] = 15793235383387715774;
  s[1] = 12390638538380655177;
  s[2] = 2361836109651742017;
  s[3] = 3188717715514472916;
  s[4] = 648184599915300350;
  s[5] = 6643206648905449565;
  s[6] = 2726452650616012281;
  s[7] = 7074207863174652740;
  s[8] = 2839732472023434771;
  s[9] = 1324431917831166204;
  s[10] = 12426324003838119764;
  s[11] = 13754663673472703591;
  s[12] = 11773959661440143617;
  s[13] = 16838540509216247236;
  s[14] = 15387639255561118537;
  s[15] = 18285287097764728708;

  p = 0;
  for (i = 0; i < N; i++) {
    store[i] = next();
  }
  fp = fopen("xorshift1024-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
