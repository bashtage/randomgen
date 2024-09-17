/*
 * Generate testing csv files
 *
 *  cl aesctr-test-data-gen.c /Ox -D__AES__
 *  aesctr-test-data-gen.exe
 *
 *  gcc aesctr-test-data-gen.c -maes -o aesctr-test-data-gen
 *  ./aesctr-test-data-gen
 *
 *
 */

#include "aesctr.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main()
{
    aesctr_state state;
    uint64_t sm_state, seed = 0xDEADBEAF;
    sm_state = seed;
    /* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint64) */
    uint64_t initial_seed[2] = {5778446405158232650, 4639759349701729399};
    int i;
    for (i = 0; i < 2; i++) {
        printf("state %d: 0x%" PRIx64 "\n", i, initial_seed[i]);
    }

    uint64_t store[N];
    aesctr_seed_r(&state, &initial_seed[0]);
    for (i = 0; i < N; i++)
    {
        store[i] = aesctr_r(&state);
    }

    FILE *fp;
    fp = fopen("aesctr-testset-1.csv", "w");
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

    seed = 0;
    sm_state = seed;
    /* SeedSequence(0).generate_state(2, dtype=np.uint64) */
    initial_seed[0] = 15793235383387715774;
    initial_seed[1] = 12390638538380655177;
    for (i = 0; i < 2; i++) {
        printf("state %d: 0x%" PRIx64 "\n", i, initial_seed[i]);
    }
    aesctr_seed_r(&state, &initial_seed[0]);
    for (i = 0; i < N; i++)
    {
        store[i] = aesctr_r(&state);
    }

    fp = fopen("aesctr-testset-2.csv", "w");
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
