/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <https://creativecommons.org/publicdomain/zero/1.0/>. */

#include "xoshiro256.h"
#include <stdio.h>
#include <time.h>

#define N 1000000000
int main()
{
    xoshiro256_state_t state;
    state.s[0] = 12349123LL;
    state.s[1] = 9812839737LL;
    state.s[2] = 1289983092813174LL;
    state.s[3] = 12899830928131741LL;
    uint64_t sum = 0;
    int i;
    clock_t begin = clock();
    for (int i = 0; i < N; i++)
    {
        sum += xoshiro256_next64(&state);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, (uint64_t)N);
    printf("%" PRIu64 " randoms per second\n",
           (uint64_t)((double)N / time_spent));
    printf("%0.3f ms to produce 1,000,000 draws \n",
           1000. * (1000000.0 * (time_spent / (double)N)));
}
