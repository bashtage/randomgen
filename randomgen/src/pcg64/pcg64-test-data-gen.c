/*
 * Generate testing csv files
 *
 * GCC only
 *
 * gcc  pcg64-test-data-gen.c pcg64.orig.c -o
 * pgc64-test-data-gen
 */

#include "pcg64.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
    pcg64_random_t rng;
    uint64_t state;
    __uint128_t temp;
    uint64_t seed = 0ULL;
    // First 4 values produced by SeedSequence(0)
    __uint128_t seed_seq =
      PCG_128BIT_CONSTANT(15793235383387715774ULL, 12390638538380655177ULL);
    __uint128_t inc =
      PCG_128BIT_CONSTANT(2361836109651742017ULL, 3188717715514472916ULL);
    int i;
    uint64_t store[N];
    pcg64_srandom_r(&rng, seed_seq, inc);
    printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));
    printf("%" PRIx64 "\n", (uint64_t)rng.state);
    printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));
    printf("%" PRIx64 "\n", (uint64_t)rng.inc);
    for (i = 0; i < N; i++) {
        store[i] = pcg64_random_r(&rng);
    }

    FILE *fp;
    fp = fopen("pcg64-testset-1.csv", "w");
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

    seed = 0xDEADBEAFULL;
    // First values produced by SeedSequence(0xDEADBEAF)
    seed_seq = PCG_128BIT_CONSTANT(5778446405158232650ULL, 4639759349701729399ULL);
    inc = PCG_128BIT_CONSTANT(13222832537653397986ULL, 2330059127936092250ULL);
    pcg64_srandom_r(&rng, seed_seq, inc);
    printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));
    printf("%" PRIx64 "\n", (uint64_t)rng.state);
    printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));
    printf("%" PRIx64 "\n", (uint64_t)rng.inc);
    for (i = 0; i < N; i++) {
        store[i] = pcg64_random_r(&rng);
    }
    fp = fopen("pcg64-testset-2.csv", "w");
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
