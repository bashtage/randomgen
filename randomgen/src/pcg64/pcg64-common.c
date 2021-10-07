#include "pcg64-common.h"

#if !defined(PCG_EMULATED_128BIT_MATH) || !(PCG_EMULATED_128BIT_MATH)

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult, pcg128_t cur_plus) {
    pcg128_t acc_mult = 1u;
    pcg128_t acc_plus = 0u;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta /= 2;
    }
    return acc_mult * state + acc_plus;
}

#else

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult, pcg128_t cur_plus) {
    pcg128_t acc_mult = PCG_128BIT_CONSTANT(0u, 1u);
    pcg128_t acc_plus = PCG_128BIT_CONSTANT(0u, 0u);
    while ((delta.high > 0) || (delta.low > 0)) {
        if (delta.low & 1) {
            acc_mult = pcg128_mult(acc_mult, cur_mult);
            acc_plus = pcg128_add(pcg128_mult(acc_plus, cur_mult), cur_plus);
        }
        cur_plus = pcg128_mult(pcg128_add(cur_mult, PCG_128BIT_CONSTANT(0u, 1u)), cur_plus);
        cur_mult = pcg128_mult(cur_mult, cur_mult);
        delta.low = (delta.low >> 1) | (delta.high << 63);
        delta.high >>= 1;
    }
    return pcg128_add(pcg128_mult(acc_mult, state), acc_plus);
}

#endif
