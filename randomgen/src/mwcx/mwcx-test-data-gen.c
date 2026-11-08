/*
 * Generate the testing csv files of the MWCX generators using the reference
 * C code (https://github.com/vigna/MWCPRNG/tree/main/prngs). Compile once for
 * each generator, defining one of MWC192X, MWC128X, MWC128X32 or MWC64X32:
 *
 *  gcc -O2 -DMWC192X -I/path/to/MWCPRNG/prngs mwcx-test-data-gen.c \
 *  -o mwcx-test-data-gen
 *  ./mwcx-test-data-gen
 *
 * The state is set from the output of SeedSequence as randomgen does: the
 * value used for the carry is read as a fixed-point fraction and multiplied
 * by the size of the range of valid carries.
 *
 * The program also prints the states produced by the reference jump
 * functions, which are used in test_direct.py.
 */

#include <inttypes.h>
#include <stdio.h>

#define N 1000

#if defined(MWC192X)
#include "MWC192X.c"
#define NAME "mwc192x"
#define FMT "0x%" PRIx64
typedef uint64_t out_t;
/* SeedSequence(0xDEADBEAF).generate_state(3, dtype=np.uint64) */
static const uint64_t seed1[] = {5778446405158232650ULL, 4639759349701729399ULL,
                                 13222832537653397986ULL};
/* SeedSequence(0).generate_state(3, dtype=np.uint64) */
static const uint64_t seed2[] = {15793235383387715774ULL,
                                 12390638538380655177ULL, 2361836109651742017ULL};
static void set_state(const uint64_t *s) {
  x = s[0], y = s[1];
  c = 1 + (uint64_t)((__uint128_t)s[2] * (MWC_A2 - 2) >> 64);
}
static void print_state(const char *label) {
  printf("%s: 0x%" PRIx64 ", 0x%" PRIx64 ", 0x%" PRIx64 "\n", label, x, y, c);
}
static void do_jump_n(void) {
  const uint64_t n[] = {0x9e3779b97f4a7c15ULL, 0x0123456789abcdefULL, 0};
  jump_n(n);
}
#elif defined(MWC128X)
#include "MWC128X.c"
#define NAME "mwc128x"
#define FMT "0x%" PRIx64
typedef uint64_t out_t;
/* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint64) */
static const uint64_t seed1[] = {5778446405158232650ULL, 4639759349701729399ULL};
/* SeedSequence(0).generate_state(2, dtype=np.uint64) */
static const uint64_t seed2[] = {15793235383387715774ULL,
                                 12390638538380655177ULL};
static void set_state(const uint64_t *s) {
  x = s[0];
  c = 1 + (uint64_t)((__uint128_t)s[1] * (MWC_A1 - 2) >> 64);
}
static void print_state(const char *label) {
  printf("%s: 0x%" PRIx64 ", 0x%" PRIx64 "\n", label, x, c);
}
static void do_jump_n(void) {
  const uint64_t n[] = {0x9e3779b97f4a7c15ULL, 0x0123456789abcdefULL};
  jump_n(n);
}
#elif defined(MWC128X32)
#include "MWC128X_32.c"
#define NAME "mwc128x32"
#define FMT "0x%" PRIx32
typedef uint32_t out_t;
/* SeedSequence(0xDEADBEAF).generate_state(4, dtype=np.uint32) */
static const uint32_t seed1[] = {3575046730U, 1345399395U, 3393510519U,
                                 1080278155U};
/* SeedSequence(0).generate_state(4, dtype=np.uint32) */
static const uint32_t seed2[] = {2968811710U, 3677149159U, 745650761U,
                                 2884920346U};
static void set_state(const uint32_t *s) {
  x = s[0], y = s[1], z = s[2];
  c = 1 + (uint32_t)((uint64_t)s[3] * (MWC_A3 - 2) >> 32);
}
static void print_state(const char *label) {
  printf("%s: 0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32 "\n",
         label, x, y, z, c);
}
static void do_jump_n(void) {
  const uint64_t n[] = {0x9e3779b97f4a7c15ULL, 0x0123456789abcdefULL};
  jump_n(n);
}
#elif defined(MWC64X32)
#include "MWC64X_32.c"
#define NAME "mwc64x32"
#define FMT "0x%" PRIx32
typedef uint32_t out_t;
/* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint32) */
static const uint32_t seed1[] = {3575046730U, 1345399395U};
/* SeedSequence(0).generate_state(2, dtype=np.uint32) */
static const uint32_t seed2[] = {2968811710U, 3677149159U};
static void set_state(const uint32_t *s) {
  x = s[0];
  c = 1 + (uint32_t)((uint64_t)s[1] * (MWC_A1 - 2) >> 32);
}
static void print_state(const char *label) {
  printf("%s: 0x%" PRIx32 ", 0x%" PRIx32 "\n", label, x, c);
}
static void do_jump_n(void) { jump_n(0x9e3779b97f4a7c15ULL); }
#else
#error "Define one of MWC192X, MWC128X, MWC128X32 or MWC64X32"
#endif

static int write_csv(const char *filename, uint64_t seed) {
  int i;
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    const out_t v = next();
    fprintf(fp, "%d, " FMT "\n", i, v);
    if (i == N - 1) {
      printf("%d, " FMT "\n", i, v);
    }
  }
  fclose(fp);
  return 0;
}

int main(void) {
  set_state(seed1);
  if (write_csv(NAME "-testset-1.csv", 0xDEADBEAF))
    return -1;
  set_state(seed2);
  if (write_csv(NAME "-testset-2.csv", 0))
    return -1;

  /* States reached from the first seed by the jump functions */
  set_state(seed1);
  jump();
  print_state("jump");
  jump();
  print_state("jump twice");
  set_state(seed1);
  long_jump();
  print_state("long_jump");
  set_state(seed1);
  do_jump_n();
  print_state("jump_n");
  return 0;
}
