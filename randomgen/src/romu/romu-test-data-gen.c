/*
 * Generate testing csv files
 *
 *  cl romu-test-data-gen.c /Ox
 *  romu-test-data-gen.exe
 *
 *  gcc romu-test-data-gen.c -o romu-test-data-gen
 *  ./romu-test-data-gen
 *
 */

#include "romu-orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000
#define BURN 10

int main() {
    int i;

    uint64_t seed = 0x0ULL;
    wState = 15793235383387715774ULL;
    xState = 12390638538380655177ULL;
    yState = 2361836109651742017ULL;
    zState = 3188717715514472916ULL;

    uint64_t store[N];
    for (i = 0; i < BURN; i++) {
        romuQuad_random();
    }
    for (i = 0; i < N; i++) {
        store[i] = romuQuad_random();
    }

    FILE *fp;
    fp = fopen("romuquad-testset-1.csv", "w");
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

    seed = 0xDEADBEEFULL;
    wState = 10671498545779160169ULL;
    xState = 17039977206943430958ULL;
    yState = 8098813118336512226ULL;
    zState = 451580776527170015ULL;

    for (i = 0; i < BURN; i++) {
        romuQuad_random();
    }
    for (i = 0; i < N; i++) {
        store[i] = romuQuad_random();
    }

    fp = fopen("romuquad-testset-2.csv", "w");
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

    seed = 0x0ULL;
    wState = 15793235383387715774ULL;
    xState = 12390638538380655177ULL;
    yState = 2361836109651742017ULL;
    zState = 3188717715514472916ULL;

    for (i = 0; i < BURN; i++) {
        romuTrio_random();
    }
    for (i = 0; i < N; i++) {
        store[i] = romuTrio_random();
    }

    fp = fopen("romutrio-testset-1.csv", "w");
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

    seed = 0xDEADBEEFULL;
    wState = 10671498545779160169ULL;
    xState = 17039977206943430958ULL;
    yState = 8098813118336512226ULL;
    zState = 451580776527170015ULL;
    for (i = 0; i < BURN; i++) {
        romuTrio_random();
    }
    for (i = 0; i < N; i++) {
        store[i] = romuTrio_random();
    }

    fp = fopen("romutrio-testset-2.csv", "w");
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
